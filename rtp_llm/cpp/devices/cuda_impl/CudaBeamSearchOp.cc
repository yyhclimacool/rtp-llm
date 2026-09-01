#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include "3rdparty/trt_beam_search/beamSearch.h"
#include "3rdparty/trt_beam_search/beamSearchKernels.h"

using namespace std;
namespace rtp_llm {

BeamSearchOutput CudaDevice::sampleBeamSearch(const BeamSearchParams& params) {
    // TODO(zhangjianning.zjn): make input_lengths_out and sequence_lengths_out computation optional
    // TODO(zhangjianning.zjn): make token_ids_out computation optional

    const int batch_size     = params.logits.shape()[0];
    const int beam_width_in  = params.logits.shape()[1];
    const int beam_width_out = params.num_beams_out != 0 ? params.num_beams_out : beam_width_in;
    const int vocab_size     = params.logits.shape()[2];
    const int max_seq_len    = params.token_ids->shape()[2];
    RTP_LLM_PROFILE_SCOPE_DYNAMIC(
        "cuda.beam_search(batch=%d,beam_in=%d,beam_out=%d,vocab=%d,max_seq=%d,reuse_logits=%d,reuse_buffers=%d)",
        batch_size,
        beam_width_in,
        beam_width_out,
        vocab_size,
        max_seq_len,
        static_cast<int>(params.reuse_logits_buffer),
        static_cast<int>(params.reuse_output_buffers));
    // TODO(zhangjianning.zjn): check the shape of params
    RTP_LLM_CHECK_WITH_INFO((vocab_size > 2 * beam_width_in),
                            "cuda beam search op need vocab_size[%d] > beam_width_in[%d] * 2",
                            vocab_size,
                            beam_width_in);
    RTP_LLM_CHECK_WITH_INFO((vocab_size > 2 * beam_width_out),
                            "cuda beam search op need vocab_size[%d] > beam_width_out[%d] * 2",
                            vocab_size,
                            beam_width_out);

#define DISPATCH_TYPE(T, T_EXPR, ...)                                                                                  \
    do {                                                                                                               \
        switch (T_EXPR) {                                                                                              \
            case DataType::TYPE_FP16: {                                                                                \
                using T = half;                                                                                        \
                (__VA_ARGS__)();                                                                                       \
            } break;                                                                                                   \
            case DataType::TYPE_FP32: {                                                                                \
                using T = float;                                                                                       \
                (__VA_ARGS__)();                                                                                       \
            } break;                                                                                                   \
            default:                                                                                                   \
                RTP_LLM_CHECK_WITH_INFO(                                                                               \
                    false, "cuda beam search op does not support dtype[%d]", params.logits.type());                    \
        }                                                                                                              \
    } while (0)

#define DISPATCH_BOOL(BOOL, BOOL_EXPR, ...)                                                                            \
    do {                                                                                                               \
        if (BOOL_EXPR) {                                                                                               \
            constexpr bool BOOL = true;                                                                                \
            (__VA_ARGS__)();                                                                                           \
        } else {                                                                                                       \
            constexpr bool BOOL = false;                                                                               \
            (__VA_ARGS__)();                                                                                           \
        }                                                                                                              \
    } while (0)

    // compute log softmax for probability calculation
    at::Tensor logits_tsr = Buffer2torchTensor(params.logits, false);
    at::Tensor log_softmax_logits_tsr;
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("cuda.beam_log_softmax(elements=%zu,inplace=%d)",
                                      params.logits.size(),
                                      static_cast<int>(params.reuse_logits_buffer));
        if (params.reuse_logits_buffer) {
            // The sampler no longer needs the logits after beam search. Reuse that storage for log probabilities so
            // log_softmax does not allocate another [batch, beam, vocab] tensor.
            at::_log_softmax_out(logits_tsr, logits_tsr, -1, false);
            log_softmax_logits_tsr = logits_tsr;
        } else {
            log_softmax_logits_tsr = logits_tsr.log_softmax(-1);
        }
    }

    // beam search heuristic
    tensorrt_llm::BeamSearchConfig config;
    {
        RTP_LLM_PROFILE_SCOPE("cuda.beam_configure");
        DISPATCH_TYPE(T, params.logits.type(), [&]() {
            config = tensorrt_llm::configureBeamSearch<T>(batch_size, beam_width_in, beam_width_out, vocab_size);
        });
    }

    const size_t output_batch_size = static_cast<size_t>(batch_size) * beam_width_out;
    const size_t token_ids_size    = output_batch_size * max_seq_len;

    if (params.reuse_output_buffers) {
        const auto tooSmall = [](const BufferPtr& buffer, DataType type, size_t size) {
            return !buffer || buffer->type() != type || buffer->size() < size;
        };
        const bool should_grow =
            tooSmall(beam_search_buffer_cache_.workspace, DataType::TYPE_BYTES, config.mWorkspaceSize)
            || tooSmall(beam_search_buffer_cache_.token_ids, DataType::TYPE_INT32, token_ids_size)
            || tooSmall(beam_search_buffer_cache_.beam_indices, DataType::TYPE_INT32, output_batch_size)
            || tooSmall(beam_search_buffer_cache_.output_ids, DataType::TYPE_INT32, output_batch_size)
            || tooSmall(beam_search_buffer_cache_.sequence_lengths, DataType::TYPE_INT32, output_batch_size)
            || (config.mVBWS
                && (tooSmall(beam_search_buffer_cache_.input_lengths, DataType::TYPE_INT32, output_batch_size)
                    || tooSmall(beam_search_buffer_cache_.cum_log_probs, DataType::TYPE_FP32, output_batch_size)));
        if (should_grow) {
            // Release the previous cache as a unit before growing. This bounds retained memory to one beam-search
            // configuration and avoids a transient peak containing both old and new workspaces.
            beam_search_buffer_cache_ = {};
        }
    }

    const auto cachedBufferView = [this](BufferPtr& cache, DataType type, size_t size, const char* tag) {
        if (!cache || cache->type() != type || cache->size() < size) {
            cache = allocateBuffer({type, {size}, AllocationType::DEVICE}, {tag});
        }
        return cache->slice(0, size);
    };

    BufferPtr workspace;
    if (params.reuse_output_buffers) {
        workspace = cachedBufferView(
            beam_search_buffer_cache_.workspace, DataType::TYPE_BYTES, config.mWorkspaceSize, "beam_search_workspace");
    } else {
        workspace = allocateBuffer({DataType::TYPE_BYTES, {config.mWorkspaceSize}, AllocationType::DEVICE},
                                   {"beam_search_workspace"});
    }
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("cuda.beam_clear_workspace(bytes=%zu)", workspace->sizeBytes());
        cudaMemsetAsync(workspace->data(), 0, workspace->sizeBytes(), stream_);
    }

    BufferPtr token_ids_out;
    BufferPtr beam_indices;
    BufferPtr output_ids;
    BufferPtr input_lengths_out;
    BufferPtr sequence_lengths_out;
    BufferPtr cum_log_probs_out;
    if (params.reuse_output_buffers) {
        token_ids_out = cachedBufferView(
            beam_search_buffer_cache_.token_ids, DataType::TYPE_INT32, token_ids_size, "token_ids_out");
        beam_indices = cachedBufferView(
            beam_search_buffer_cache_.beam_indices, DataType::TYPE_INT32, output_batch_size, "beam_indices");
        output_ids = cachedBufferView(
            beam_search_buffer_cache_.output_ids, DataType::TYPE_INT32, output_batch_size, "output_ids");
        sequence_lengths_out = cachedBufferView(beam_search_buffer_cache_.sequence_lengths,
                                                DataType::TYPE_INT32,
                                                output_batch_size,
                                                "sequence_lengths_out");
        if (config.mVBWS) {
            input_lengths_out = cachedBufferView(
                beam_search_buffer_cache_.input_lengths, DataType::TYPE_INT32, output_batch_size, "input_length_out");
            cum_log_probs_out = cachedBufferView(
                beam_search_buffer_cache_.cum_log_probs, DataType::TYPE_FP32, output_batch_size, "cum_log_probs_out");
        } else {
            input_lengths_out = params.input_lengths;
            cum_log_probs_out = params.cum_log_probs;
        }

        token_ids_out->updateShape(
            {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out), static_cast<size_t>(max_seq_len)});
        beam_indices->updateShape({static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)});
        output_ids->updateShape({static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)});
        sequence_lengths_out->updateShape({static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)});
        if (config.mVBWS) {
            input_lengths_out->updateShape({static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)});
            cum_log_probs_out->updateShape({static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)});
        }
    } else {
        token_ids_out = allocateBuffer(
            {DataType::TYPE_INT32,
             {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out), static_cast<size_t>(max_seq_len)},
             AllocationType::DEVICE},
            {"token_ids_out"});
        beam_indices         = allocateBuffer({DataType::TYPE_INT32,
                                               {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)},
                                               AllocationType::DEVICE},
                                              {"beam_indices"});
        output_ids           = allocateBuffer({DataType::TYPE_INT32,
                                               {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)},
                                               AllocationType::DEVICE},
                                              {"output_ids"});
        input_lengths_out    = config.mVBWS ?
                                   allocateBuffer({DataType::TYPE_INT32,
                                                   {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)},
                                                   AllocationType::DEVICE},
                                                  {"input_length_out"}) :
                                   params.input_lengths;
        sequence_lengths_out = allocateBuffer({DataType::TYPE_INT32,
                                               {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)},
                                               AllocationType::DEVICE},
                                              {"sequence_lengths_out"});
        cum_log_probs_out    = config.mVBWS ?
                                   allocateBuffer({DataType::TYPE_FP32,
                                                   {static_cast<size_t>(batch_size), static_cast<size_t>(beam_width_out)},
                                                   AllocationType::DEVICE},
                                                  {"cum_log_probs_out"}) :
                                   params.cum_log_probs;
    }

    // set BeamHypotheses
    tensorrt_llm::kernels::BeamHypotheses BH;
    // basic scalar
    BH.bVBWS         = config.mVBWS;
    BH.nMaxBatchSize = batch_size;
    BH.nBatchSize    = batch_size;
    BH.nBeamWidthIn  = beam_width_in;
    BH.nBeamWidthOut = beam_width_out;
    BH.nMaxSeqLen    = max_seq_len;
    BH.nVocabSize    = vocab_size;
    BH.nVPart        = config.mVPart;
    // buffer size
    BH.nByteMaxSharedMemoryPerBlock = config.mByteMaxSharedMemoryPerBlock;
    BH.nByteSharedMemoryStage1      = config.mByteSharedMemoryStage1;
    BH.nByteSharedMemoryStage3      = config.mByteSharedMemoryStage3;
    // input and ouput ptr
    BH.inputLengthsIn     = params.input_lengths->data<int>();
    BH.inputLengthsOut    = input_lengths_out->data<int>();
    BH.sequenceLengthsIn  = params.sequence_lengths->data<int>();
    BH.sequenceLengthsOut = sequence_lengths_out->data<int>();
    BH.cumLogProbsIn      = params.cum_log_probs->data<float>();
    BH.cumLogProbsOut     = cum_log_probs_out->data<float>();
    BH.tokenIdsIn         = params.token_ids->data<int>();
    BH.tokenIdsOut        = token_ids_out->data<int>();
    BH.parentIdsPtr       = beam_indices->data<int>();
    BH.outputIdsPtr       = output_ids->data<int>();

    check_cuda_error();

    // invoke beam search kernel
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("cuda.beam_invoke_topk(v2=%d,vpart=%zu,workspace=%zu)",
                                      static_cast<int>(config.mV2),
                                      config.mVPart,
                                      workspace->sizeBytes());
        DISPATCH_TYPE(T, params.logits.type(), [&]() {
            DISPATCH_BOOL(IS_V2, config.mV2, [&]() {
                tensorrt_llm::kernels::invokeTopkBeamSearch<T, IS_V2>(
                    static_cast<T*>(log_softmax_logits_tsr.data_ptr()), nullptr, workspace->data(), BH, stream_);
            });
        });
    }

    check_cuda_error();

    return BeamSearchOutput({std::move(token_ids_out),
                             std::move(input_lengths_out),
                             std::move(sequence_lengths_out),
                             std::move(cum_log_probs_out),
                             std::move(beam_indices)});
}

}  // namespace rtp_llm