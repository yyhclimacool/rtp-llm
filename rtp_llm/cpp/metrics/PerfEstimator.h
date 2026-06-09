#pragma once

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <cstdint>
#include <vector>

namespace rtp_llm {

struct PerfEstimatorConfig {
    int64_t num_layers    = 0;
    int64_t hidden_size   = 0;
    int64_t head_num      = 0;
    int64_t kv_head_num   = 0;
    int64_t size_per_head = 0;

    int64_t inter_size          = 0;
    bool    is_gated_activation = true;

    int64_t              moe_style      = 0;
    int64_t              expert_num     = 0;
    int64_t              moe_k          = 0;
    int64_t              moe_inter_size = 0;
    std::vector<int64_t> moe_layer_index;

    int64_t vocab_size  = 0;
    bool    has_lm_head = true;

    int64_t tp_size = 1;
    int64_t pp_size = 1;

    int weight_byte_size     = 2;
    int activation_byte_size = 2;
    int kv_cache_byte_size   = 2;
};

struct BatchPerfContext {
    int64_t prefill_tokens                = 0;
    int64_t prefill_requests              = 0;
    int64_t prefill_token_context_product = 0;

    int64_t decode_tokens                = 0;
    int64_t decode_token_context_product = 0;

    int64_t total_tokens() const {
        return prefill_tokens + decode_tokens;
    }
    int64_t total_token_context_product() const {
        return prefill_token_context_product + decode_token_context_product;
    }
    int64_t num_logits_tokens() const {
        return prefill_requests + decode_tokens;
    }
};

class PerfEstimator {
public:
    explicit PerfEstimator(const PerfEstimatorConfig& config);

    RtpLLMPerfMetricsCollector estimate(const BatchPerfContext& ctx) const;

private:
    int64_t computeAttentionFlops(const BatchPerfContext& ctx) const;
    int64_t computeFfnFlops(const BatchPerfContext& ctx) const;
    int64_t computeUnembedFlops(const BatchPerfContext& ctx) const;

    int64_t computeReadBytes(const BatchPerfContext& ctx) const;
    int64_t computeWriteBytes(const BatchPerfContext& ctx) const;

    PerfEstimatorConfig config_;
    int64_t             num_dense_layers_     = 0;
    int64_t             num_moe_layers_       = 0;
    int64_t             q_per_gpu_            = 0;
    int64_t             kv_per_gpu_           = 0;
    int64_t             layers_per_gpu_       = 0;
    int64_t             dense_layers_per_gpu_ = 0;
    int64_t             moe_layers_per_gpu_   = 0;
    int64_t             inter_size_per_gpu_   = 0;
    int64_t             vocab_per_gpu_        = 0;
    int                 ffn_mult_             = 3;
};

}  // namespace rtp_llm
