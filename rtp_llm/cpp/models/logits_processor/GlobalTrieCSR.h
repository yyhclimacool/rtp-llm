#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

struct GlobalTrieCSR {
    std::vector<int32_t>  all_children_token_ids;
    std::vector<uint32_t> all_children_node_ids;
    std::vector<uint32_t> node_children_offset;
    std::vector<uint16_t> node_children_count;
    std::vector<uint32_t> node_leaf_lo;
    std::vector<uint32_t> node_leaf_hi;
    std::vector<uint8_t>  node_is_leaf;

    uint32_t num_nodes      = 0;
    uint32_t num_leaves     = 0;
    int32_t  start_token_id = 0;
    int32_t  end_token_id   = 0;

    uint32_t transition(uint32_t current_node_id, int32_t token) const {
        uint32_t offset = node_children_offset[current_node_id];
        uint16_t count  = node_children_count[current_node_id];
        auto     begin  = all_children_token_ids.begin() + offset;
        auto     end    = begin + count;
        auto     it     = std::lower_bound(begin, end, token);
        if (it == end || *it != token) {
            return current_node_id;
        }
        uint32_t idx = it - begin;
        return all_children_node_ids[offset + idx];
    }
};

struct AdIdMapping {
    std::vector<std::pair<std::string, uint32_t>> sorted_entries;

    std::optional<uint32_t> find(const std::string& ad_id) const {
        auto it = std::lower_bound(
            sorted_entries.begin(),
            sorted_entries.end(),
            ad_id,
            [](const std::pair<std::string, uint32_t>& entry, const std::string& id) { return entry.first < id; });
        if (it != sorted_entries.end() && it->first == ad_id) {
            return it->second;
        }
        return std::nullopt;
    }
};

class GlobalTrieStore {
public:
    static std::shared_ptr<GlobalTrieStore>& instance() {
        static auto inst = std::shared_ptr<GlobalTrieStore>(new GlobalTrieStore());
        return inst;
    }

    bool init(const std::string& ckpt_path, const std::string& sid_trie_config);
    bool initSuccess() const {
        return init_success_;
    }

    const GlobalTrieCSR& trie() const {
        return trie_;
    }
    const AdIdMapping& mapping() const {
        return mapping_;
    }

private:
    GlobalTrieStore()                                  = default;
    GlobalTrieStore(const GlobalTrieStore&)            = delete;
    GlobalTrieStore& operator=(const GlobalTrieStore&) = delete;

    bool loadFromJson(const std::string& file_path);

private:
    std::mutex    mutex_;
    GlobalTrieCSR trie_;
    AdIdMapping   mapping_;
    bool          init_success_ = false;
};

}  // namespace rtp_llm
