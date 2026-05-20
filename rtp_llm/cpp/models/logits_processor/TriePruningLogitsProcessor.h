#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/GlobalTrieCSR.h"

namespace rtp_llm {

struct PerRequestTrieFilter {
    std::vector<uint32_t> sorted_leaves;
    uint64_t              root_children_active[64] = {};
    bool                  root_active_built        = false;

    void build(const std::vector<std::string>& whitelist_ad_ids, const AdIdMapping& mapping, const GlobalTrieCSR& trie);
    void buildRootActive(const GlobalTrieCSR& trie);
};

struct StreamTriePruningInfo {
    int32_t  input_length          = 0;
    int32_t  current_output_length = 0;
    bool     is_beam_search        = false;
    uint32_t current_node_id       = 0;

    std::shared_ptr<PerRequestTrieFilter> filter;

    StreamTriePruningInfo() = default;
    StreamTriePruningInfo(int32_t                               input_length,
                          int32_t                               current_output_length,
                          bool                                  is_beam_search,
                          std::shared_ptr<PerRequestTrieFilter> filter):
        input_length(input_length),
        current_output_length(current_output_length),
        is_beam_search(is_beam_search),
        filter(std::move(filter)) {}

    StreamTriePruningInfo copy() const {
        StreamTriePruningInfo info;
        info.input_length          = input_length;
        info.current_output_length = current_output_length;
        info.is_beam_search        = is_beam_search;
        info.current_node_id       = current_node_id;
        info.filter                = filter;  // shared, not deep-copied
        return info;
    }
};

class TriePruningLogitsProcessor: public BaseLogitsProcessor {
public:
    TriePruningLogitsProcessor() = default;
    explicit TriePruningLogitsProcessor(std::vector<StreamTriePruningInfo> infos);
    virtual ~TriePruningLogitsProcessor() {}

public:
    static std::shared_ptr<TriePruningLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                         int32_t                        num);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

public:
    size_t size() const {
        return infos_.size();
    }
    const std::vector<StreamTriePruningInfo>& infos() const {
        return infos_;
    }

    void insert(std::shared_ptr<TriePruningLogitsProcessor> others) {
        if (others != nullptr) {
            infos_.insert(infos_.end(), others->infos_.begin(), others->infos_.end());
        }
    }

private:
    std::vector<StreamTriePruningInfo> infos_;
};

using TriePruningLogitsProcessorPtr = std::shared_ptr<TriePruningLogitsProcessor>;

}  // namespace rtp_llm
