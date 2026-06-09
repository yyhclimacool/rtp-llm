#include "rtp_llm/cpp/metrics/PerfEstimator.h"
#include <algorithm>

namespace rtp_llm {

PerfEstimator::PerfEstimator(const PerfEstimatorConfig& config): config_(config) {
    if (config_.moe_style == 1) {
        num_moe_layers_   = config_.num_layers;
        num_dense_layers_ = 0;
    } else if (config_.moe_style == 2) {
        num_moe_layers_   = static_cast<int64_t>(config_.moe_layer_index.size());
        num_dense_layers_ = config_.num_layers - num_moe_layers_;
    } else {
        num_moe_layers_   = 0;
        num_dense_layers_ = config_.num_layers;
    }

    ffn_mult_ = config_.is_gated_activation ? 3 : 2;

    q_per_gpu_  = std::max<int64_t>(1, config_.head_num / config_.tp_size);
    kv_per_gpu_ = std::max<int64_t>(1, config_.kv_head_num / config_.tp_size);

    layers_per_gpu_       = config_.num_layers / config_.pp_size;
    dense_layers_per_gpu_ = num_dense_layers_ / config_.pp_size;
    moe_layers_per_gpu_   = num_moe_layers_ / config_.pp_size;

    inter_size_per_gpu_ = config_.inter_size / config_.tp_size;
    vocab_per_gpu_      = config_.vocab_size / config_.tp_size;
}

int64_t PerfEstimator::computeAttentionFlops(const BatchPerfContext& ctx) const {
    int64_t D  = config_.hidden_size;
    int64_t d  = config_.size_per_head;
    int64_t q  = q_per_gpu_;
    int64_t kv = kv_per_gpu_;
    int64_t L  = layers_per_gpu_;
    int64_t T  = ctx.total_tokens();
    int64_t TC = ctx.total_token_context_product();

    int64_t qkv_proj = 2LL * T * D * (q + 2 * kv) * d * L;
    int64_t attn_qk  = 2LL * q * TC * d * L;
    int64_t attn_av  = 2LL * q * TC * d * L;
    int64_t out_proj = 2LL * T * D * q * d * L;

    return qkv_proj + attn_qk + attn_av + out_proj;
}

int64_t PerfEstimator::computeFfnFlops(const BatchPerfContext& ctx) const {
    int64_t D = config_.hidden_size;
    int64_t T = ctx.total_tokens();

    int64_t flops = 0;

    if (dense_layers_per_gpu_ > 0) {
        flops += 2LL * D * ffn_mult_ * inter_size_per_gpu_ * T * dense_layers_per_gpu_;
    }

    if (moe_layers_per_gpu_ > 0 && config_.moe_k > 0) {
        int64_t moe_inter        = config_.moe_inter_size / config_.tp_size;
        int64_t activated_tokens = T * config_.moe_k;
        flops += 2LL * D * ffn_mult_ * moe_inter * activated_tokens * moe_layers_per_gpu_;
    }

    return flops;
}

int64_t PerfEstimator::computeUnembedFlops(const BatchPerfContext& ctx) const {
    if (!config_.has_lm_head) {
        return 0;
    }
    int64_t D = config_.hidden_size;
    int64_t V = vocab_per_gpu_;
    int64_t T = ctx.num_logits_tokens();
    return 2LL * T * D * V;
}

int64_t PerfEstimator::computeReadBytes(const BatchPerfContext& ctx) const {
    int64_t D  = config_.hidden_size;
    int64_t d  = config_.size_per_head;
    int64_t q  = q_per_gpu_;
    int64_t kv = kv_per_gpu_;
    int64_t L  = layers_per_gpu_;
    int64_t T  = ctx.total_tokens();
    int     wb = config_.weight_byte_size;
    int     ab = config_.activation_byte_size;
    int     cb = config_.kv_cache_byte_size;

    int64_t bytes = 0;

    // Attention weights: QKV + output projection
    bytes += static_cast<int64_t>(D * (q + 2 * kv) * d * wb * L);
    bytes += static_cast<int64_t>(q * d * D * wb * L);

    // Attention activations input
    bytes += T * D * ab * L;      // QKV input
    bytes += T * q * d * ab * L;  // output proj input

    // Prefill attention reads: Q, K, V all from activations
    if (ctx.prefill_tokens > 0) {
        bytes += (ctx.prefill_tokens * q + 2LL * ctx.prefill_tokens * kv) * d * ab * L;
    }

    // Decode attention reads: Q from activations, K/V from KV cache
    if (ctx.decode_tokens > 0) {
        bytes += ctx.decode_tokens * q * d * ab * L;
        bytes += 2LL * ctx.decode_token_context_product * kv * d * cb * L;
    }

    // Dense FFN weights
    if (dense_layers_per_gpu_ > 0) {
        bytes += static_cast<int64_t>(D * ffn_mult_ * inter_size_per_gpu_ * wb * dense_layers_per_gpu_);
        bytes += T * D * ab * dense_layers_per_gpu_;
    }

    // MoE FFN weights (assume all activated experts loaded)
    if (moe_layers_per_gpu_ > 0 && config_.moe_k > 0) {
        int64_t moe_inter     = config_.moe_inter_size / config_.tp_size;
        int64_t num_activated = std::min(T * config_.moe_k, config_.expert_num / config_.tp_size);
        bytes += static_cast<int64_t>(D * ffn_mult_ * moe_inter * num_activated * wb * moe_layers_per_gpu_);
        bytes += T * config_.moe_k * D * ab * moe_layers_per_gpu_;
    }

    // Unembed
    if (config_.has_lm_head) {
        int64_t T_logits = ctx.num_logits_tokens();
        bytes += static_cast<int64_t>(D * vocab_per_gpu_ * wb);
        bytes += T_logits * D * ab;
    }

    return bytes;
}

int64_t PerfEstimator::computeWriteBytes(const BatchPerfContext& ctx) const {
    int64_t D  = config_.hidden_size;
    int64_t d  = config_.size_per_head;
    int64_t q  = q_per_gpu_;
    int64_t kv = kv_per_gpu_;
    int64_t L  = layers_per_gpu_;
    int64_t T  = ctx.total_tokens();
    int     ab = config_.activation_byte_size;
    int     cb = config_.kv_cache_byte_size;

    int64_t bytes = 0;

    // Attention QKV output
    bytes += T * (q + 2 * kv) * d * ab * L;
    // KV cache writes
    bytes += 2LL * T * kv * d * cb * L;
    // Attention output projection
    bytes += T * D * ab * L;

    // Dense FFN output
    if (dense_layers_per_gpu_ > 0) {
        bytes += T * D * ab * dense_layers_per_gpu_;
    }

    // MoE FFN output
    if (moe_layers_per_gpu_ > 0 && config_.moe_k > 0) {
        bytes += T * config_.moe_k * D * ab * moe_layers_per_gpu_;
    }

    // Unembed output
    if (config_.has_lm_head) {
        int64_t T_logits = ctx.num_logits_tokens();
        bytes += T_logits * vocab_per_gpu_ * ab;
    }

    return bytes;
}

RtpLLMPerfMetricsCollector PerfEstimator::estimate(const BatchPerfContext& ctx) const {
    RtpLLMPerfMetricsCollector collector;

    if (ctx.total_tokens() == 0) {
        return collector;
    }

    collector.num_flops       = computeAttentionFlops(ctx) + computeFfnFlops(ctx) + computeUnembedFlops(ctx);
    collector.num_read_bytes  = computeReadBytes(ctx);
    collector.num_write_bytes = computeWriteBytes(ctx);

    return collector;
}

}  // namespace rtp_llm
