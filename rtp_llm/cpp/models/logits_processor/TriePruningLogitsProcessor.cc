#include "rtp_llm/cpp/models/logits_processor/TriePruningLogitsProcessor.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void PerRequestTrieFilter::build(const std::vector<std::string>& whitelist_ad_ids,
                                 const AdIdMapping&              mapping,
                                 const GlobalTrieCSR&            trie) {
    sorted_leaves.clear();
    sorted_leaves.reserve(whitelist_ad_ids.size());

    for (const auto& ad_id : whitelist_ad_ids) {
        auto leaf_idx = mapping.find(ad_id);
        if (leaf_idx.has_value()) {
            sorted_leaves.push_back(leaf_idx.value());
        }
    }

    std::sort(sorted_leaves.begin(), sorted_leaves.end());
    auto last = std::unique(sorted_leaves.begin(), sorted_leaves.end());
    sorted_leaves.erase(last, sorted_leaves.end());

    buildRootActive(trie);
}

void PerRequestTrieFilter::buildRootActive(const GlobalTrieCSR& trie) {
    std::memset(root_children_active, 0, sizeof(root_children_active));
    root_active_built = false;

    if (trie.num_nodes == 0) {
        return;
    }

    uint32_t offset = trie.node_children_offset[0];
    uint16_t count  = trie.node_children_count[0];

    auto ptr = sorted_leaves.begin();
    for (uint16_t i = 0; i < count; ++i) {
        int32_t  token_id = trie.all_children_token_ids[offset + i];
        uint32_t child_id = trie.all_children_node_ids[offset + i];
        uint32_t child_lo = trie.node_leaf_lo[child_id];
        uint32_t child_hi = trie.node_leaf_hi[child_id];

        while (ptr != sorted_leaves.end() && *ptr < child_lo) {
            ++ptr;
        }
        if (ptr != sorted_leaves.end() && *ptr < child_hi) {
            root_children_active[token_id >> 6] |= (1ULL << (token_id & 63));
        }
    }
    root_active_built = true;
}

TriePruningLogitsProcessor::TriePruningLogitsProcessor(std::vector<StreamTriePruningInfo> infos):
    infos_(std::move(infos)) {}

std::shared_ptr<TriePruningLogitsProcessor>
TriePruningLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input, int32_t num) {
    auto store = GlobalTrieStore::instance();
    if (!store->initSuccess()) {
        return nullptr;
    }

    const auto& config = generate_input->generate_config;
    if (config->trie_whitelist_ad_ids.empty()) {
        return nullptr;
    }

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build(config->trie_whitelist_ad_ids, store->mapping(), store->trie());

    if (filter->sorted_leaves.empty()) {
        return nullptr;
    }

    const bool is_beam_search = config->hasNumBeams() || config->num_return_sequences > 1;

    auto processor = std::make_shared<TriePruningLogitsProcessor>();
    for (int32_t i = 0; i < num; ++i) {
        processor->infos_.emplace_back(generate_input->inputLength(), 0, is_beam_search, filter);
    }
    return processor;
}

void TriePruningLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    const size_t batch_size = finish_idx - start_idx;
    RTP_LLM_CHECK(batch_size == size());

    auto store = GlobalTrieStore::instance();
    if (!store->initSuccess()) {
        return;
    }
    const auto& trie = store->trie();

    auto   batch_logits = inputs.logits.narrow(0, start_idx, batch_size);
    size_t vocab_size   = batch_logits.size(1);

    // Track which beams need masking vs which should pass through
    std::vector<bool>                needs_mask(batch_size, false);
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        auto& info = infos_[i];
        if (!info.filter || info.filter->sorted_leaves.empty()) {
            continue;
        }

        uint32_t current_node_id = info.current_node_id;
        needs_mask[i]            = true;

        // Case 1: leaf node - only allow end_token_id
        if (trie.node_is_leaf[current_node_id]) {
            batch_candidate_token_ids[i] = {static_cast<size_t>(trie.end_token_id)};
            continue;
        }

        // Case 2: root node with precomputed L1 bitmap
        if (current_node_id == 0 && info.filter->root_active_built) {
            for (size_t t = 0; t < vocab_size; ++t) {
                if (info.filter->root_children_active[t >> 6] & (1ULL << (t & 63))) {
                    batch_candidate_token_ids[i].push_back(t);
                }
            }
            continue;
        }

        // Case 3: internal node - merge scan
        const auto& leaves  = info.filter->sorted_leaves;
        uint32_t    offset  = trie.node_children_offset[current_node_id];
        uint16_t    count   = trie.node_children_count[current_node_id];
        uint32_t    node_lo = trie.node_leaf_lo[current_node_id];

        auto ptr = std::lower_bound(leaves.begin(), leaves.end(), node_lo);

        for (uint16_t c = 0; c < count; ++c) {
            int32_t  token_id = trie.all_children_token_ids[offset + c];
            uint32_t child_id = trie.all_children_node_ids[offset + c];
            uint32_t child_lo = trie.node_leaf_lo[child_id];
            uint32_t child_hi = trie.node_leaf_hi[child_id];

            while (ptr != leaves.end() && *ptr < child_lo) {
                ++ptr;
            }
            if (ptr != leaves.end() && *ptr < child_hi) {
                batch_candidate_token_ids[i].push_back(static_cast<size_t>(token_id));
            }
        }
    }

    // Check if any beam needs masking
    bool need_process = false;
    for (size_t i = 0; i < batch_size; ++i) {
        if (needs_mask[i]) {
            need_process = true;
            break;
        }
    }
    if (!need_process) {
        return;
    }

    // For beams that don't need masking, fill all tokens as candidates
    // to avoid generateVocabMask masking them (all-ones row = all masked)
    for (size_t i = 0; i < batch_size; ++i) {
        if (!needs_mask[i]) {
            batch_candidate_token_ids[i].resize(vocab_size);
            for (size_t t = 0; t < vocab_size; ++t) {
                batch_candidate_token_ids[i][t] = t;
            }
        }
    }

    auto batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
}

void TriePruningLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTriePruningInfo> new_infos;
    new_infos.reserve(src_batch_indices.size());
    for (int src_idx : src_batch_indices) {
        RTP_LLM_CHECK(static_cast<size_t>(src_idx) < infos_.size());
        new_infos.push_back(infos_[src_idx].copy());
    }
    infos_ = std::move(new_infos);
}

void TriePruningLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(size() == static_cast<size_t>(new_tokens.size(0)));

    auto store = GlobalTrieStore::instance();
    if (!store->initSuccess()) {
        return;
    }
    const auto& trie = store->trie();

    for (size_t i = 0; i < size(); ++i) {
        auto& info = infos_[i];
        if (!info.filter || info.filter->sorted_leaves.empty()) {
            continue;
        }

        const int64_t offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        const int64_t stride = new_tokens.size(1);
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            int32_t token_id = new_tokens.data_ptr<int>()[i * stride + j + offset];

            // Skip end_token_id - no transition needed
            if (token_id == trie.end_token_id) {
                continue;
            }

            info.current_node_id = trie.transition(info.current_node_id, token_id);
        }

        info.current_output_length += num_new_tokens;
    }
}

}  // namespace rtp_llm
