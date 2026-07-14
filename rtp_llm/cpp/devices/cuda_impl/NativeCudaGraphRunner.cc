#include "rtp_llm/cpp/devices/cuda_impl/NativeCudaGraphRunner.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace rtp_llm {

CudaGraphExecutor::CudaGraphExecutor(CudaDevice* device, std::shared_ptr<at::cuda::CUDAStream> stream):
    ExecutorBase(),
    device_(device),
    graph_(std::make_shared<at::cuda::CUDAGraph>()),
    capture_stream_(std::move(stream)) {
    check_cuda_value(cudaEventCreateWithFlags(&input_ready_event_, cudaEventDisableTiming));
    check_cuda_value(cudaEventCreateWithFlags(&output_ready_event_, cudaEventDisableTiming));
}

CudaGraphExecutor::~CudaGraphExecutor() {
    if (input_ready_event_) {
        cudaEventDestroy(input_ready_event_);
    }
    if (output_ready_event_) {
        cudaEventDestroy(output_ready_event_);
    }
}

void CudaGraphExecutor::replay() {
    auto caller_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    if (caller_stream.stream() == capture_stream_->stream()) {
        graph_->replay();
        return;
    }

    // Preserve dependencies in both directions without synchronizing the host:
    // prior caller work -> graph replay -> subsequent caller work.
    check_cuda_value(cudaEventRecord(input_ready_event_, caller_stream.stream()));
    check_cuda_value(cudaStreamWaitEvent(capture_stream_->stream(), input_ready_event_, 0));

    device_->setStream(capture_stream_->stream());
    at::cuda::setCurrentCUDAStream(*capture_stream_);
    try {
        graph_->replay();
        check_cuda_value(cudaEventRecord(output_ready_event_, capture_stream_->stream()));
        check_cuda_value(cudaStreamWaitEvent(caller_stream.stream(), output_ready_event_, 0));
    } catch (...) {
        device_->setStream(caller_stream.stream());
        at::cuda::setCurrentCUDAStream(caller_stream);
        throw;
    }

    device_->setStream(caller_stream.stream());
    at::cuda::setCurrentCUDAStream(caller_stream);
}

void CudaGraphExecutor::captureBegin() {
    origin_stream_ = std::make_shared<at::cuda::CUDAStream>(at::cuda::getCurrentCUDAStream(at::cuda::current_device()));
    // Route both torch tensor allocations and RTP-LLM BufferManager allocations to the
    // graph-safe private mempool for the whole captured forward. PyTorch's CUDAGraph will
    // additionally flip TorchCudaAllocator into pool mode via beginAllocateToPool, but we set
    // this flag first so that device_->allocateBuffer() calls inside forward() also stay private.
    device_->nativeGraphBeginCapture();
    device_->setStream(capture_stream_->stream());
    at::cuda::setCurrentCUDAStream(*capture_stream_);
    device_->syncAndCheck();
    graph_->capture_begin();
    CaptureCheck::in_cuda_graph_capture = true;
}

void CudaGraphExecutor::captureEnd() {
    graph_->capture_end();
    CaptureCheck::in_cuda_graph_capture = false;
    device_->syncAndCheck();
    device_->setStream(origin_stream_->stream());
    at::cuda::setCurrentCUDAStream(*origin_stream_);
    device_->nativeGraphEndCapture();
    device_->registerARGraphBuffers();
}

INSTANTIATE_CUDAGRAPH_RUNNER(GptModelInputs, GptModelOutputs)

std::shared_ptr<NativeGraphRunner> CudaDevice::getNativeGraphRunner() {
    return std::make_shared<NativeCudaGraphRunner<GptModelInputs, GptModelOutputs>>(this);
}

}  // namespace rtp_llm
