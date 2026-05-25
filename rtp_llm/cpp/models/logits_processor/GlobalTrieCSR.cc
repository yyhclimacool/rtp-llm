#include "rtp_llm/cpp/models/logits_processor/GlobalTrieCSR.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

namespace {

struct TrieNode {
    std::unordered_map<int32_t, uint32_t>     children;         // token_id -> child node index (for Phase 1 insertion)
    std::vector<std::pair<int32_t, uint32_t>> sorted_children;  // sorted by token_id (built after Phase 1)
    std::string                               ad_id;            // only set for leaf nodes
    bool                                      is_leaf = false;
};

struct SidTrieJsonConfig: public autil::legacy::Jsonizable {
    int32_t start_token_id = 225;
    int32_t end_token_id   = 2;
    // Each entry: {"ad_id": "B0L6THWCFM", "sid_path": [782, 1501, 3, 256]}
    struct AdEntry: public autil::legacy::Jsonizable {
        std::string      ad_id;
        std::vector<int> sid_path;
        void             Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
            json.Jsonize("ad_id", ad_id, ad_id);
            json.Jsonize("sid_path", sid_path, sid_path);
        }
    };
    std::vector<AdEntry> ads;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("start_token_id", start_token_id, start_token_id);
        json.Jsonize("end_token_id", end_token_id, end_token_id);
        json.Jsonize("ads", ads, ads);
    }
};

}  // namespace

bool GlobalTrieStore::loadFromJson(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file) {
        RTP_LLM_LOG_WARNING("GlobalTrieStore: unable to open file [%s]", file_path.c_str());
        return false;
    }

    SidTrieJsonConfig config;
    try {
        std::ostringstream ss;
        ss << file.rdbuf();
        autil::legacy::FromJsonString(config, ss.str());
    } catch (autil::legacy::ExceptionBase& e) {
        RTP_LLM_LOG_WARNING("GlobalTrieStore: failed to parse JSON [%s]: %s", file_path.c_str(), e.what());
        return false;
    }

    if (config.ads.empty()) {
        RTP_LLM_LOG_WARNING("GlobalTrieStore: no ads in config [%s]", file_path.c_str());
        return false;
    }

    // Phase 1: Build tree structure using TrieNode objects
    std::vector<TrieNode> nodes;
    nodes.emplace_back();  // root node at index 0

    std::vector<std::pair<std::string, uint32_t>> ad_leaf_pairs;  // (ad_id, leaf_node_idx)

    for (const auto& ad : config.ads) {
        if (ad.sid_path.empty()) {
            continue;
        }
        uint32_t current = 0;
        for (int32_t token : ad.sid_path) {
            auto it = nodes[current].children.find(token);
            if (it == nodes[current].children.end()) {
                // 将当前 child 插入新节点
                uint32_t new_idx               = nodes.size();
                nodes[current].children[token] = new_idx;
                nodes.emplace_back();
                current = new_idx;
            } else {
                current = it->second;
            }
        }
        // path 遍历完了必然到达叶子节点
        nodes[current].is_leaf = true;
        nodes[current].ad_id   = ad.ad_id;
        ad_leaf_pairs.emplace_back(ad.ad_id, current);
    }

    uint32_t total_nodes = nodes.size();

    // Pre-sort children for all nodes (used by both Phase 2 and Phase 3)
    for (auto& node : nodes) {
        node.sorted_children.reserve(node.children.size());
        for (const auto& [token, child_idx] : node.children) {
            node.sorted_children.emplace_back(token, child_idx);
        }
        std::sort(node.sorted_children.begin(), node.sorted_children.end());
        node.children.clear();
    }

    // Phase 2: DFS to assign leaf indices and compute [leaf_lo, leaf_hi) intervals
    // Children are visited in sorted token_id order to ensure DFS-order leaf contiguity
    std::vector<uint32_t> leaf_lo(total_nodes, 0);
    std::vector<uint32_t> leaf_hi(total_nodes, 0);
    std::vector<uint8_t>  is_leaf(total_nodes, 0);

    // Map from node index in the TrieNode vector to its DFS-assigned leaf index
    std::vector<uint32_t> node_leaf_idx(total_nodes, UINT32_MAX);
    uint32_t              leaf_counter = 0;

    // Iterative DFS
    struct DfsFrame {
        uint32_t node_idx;
        size_t   child_pos;
    };

    std::vector<DfsFrame> stack;
    stack.push_back({0, 0});

    // Record leaf_lo at entry
    leaf_lo[0] = leaf_counter;

    while (!stack.empty()) {
        auto&       frame    = stack.back();
        const auto& children = nodes[frame.node_idx].sorted_children;

        if (frame.child_pos < children.size()) {
            uint32_t child_idx = children[frame.child_pos].second;
            frame.child_pos++;

            leaf_lo[child_idx] = leaf_counter;

            if (nodes[child_idx].is_leaf) {
                is_leaf[child_idx]       = 1;
                node_leaf_idx[child_idx] = leaf_counter;
                leaf_counter++;
                leaf_hi[child_idx] = leaf_counter;
            } else {
                stack.push_back({child_idx, 0});
            }
        } else {
            // All children processed, set leaf_hi
            leaf_hi[frame.node_idx] = leaf_counter;
            stack.pop_back();
        }
    }

    uint32_t total_leaves = leaf_counter;

    // Phase 3: Build CSR (Compressed Sparse Row) format
    // Count total children edges
    size_t total_edges = 0;
    for (const auto& node : nodes) {
        total_edges += node.sorted_children.size();
    }

    trie_.all_children_token_ids.resize(total_edges);
    trie_.all_children_node_ids.resize(total_edges);
    trie_.node_children_offset.resize(total_nodes);
    trie_.node_children_count.resize(total_nodes);
    trie_.node_leaf_lo   = std::move(leaf_lo);
    trie_.node_leaf_hi   = std::move(leaf_hi);
    trie_.node_is_leaf   = std::move(is_leaf);
    trie_.num_nodes      = total_nodes;
    trie_.num_leaves     = total_leaves;
    trie_.start_token_id = config.start_token_id;
    trie_.end_token_id   = config.end_token_id;

    uint32_t edge_offset = 0;
    for (uint32_t i = 0; i < total_nodes; ++i) {
        trie_.node_children_offset[i] = edge_offset;
        trie_.node_children_count[i]  = static_cast<uint16_t>(nodes[i].sorted_children.size());

        for (const auto& [token, child_idx] : nodes[i].sorted_children) {
            trie_.all_children_token_ids[edge_offset] = token;
            trie_.all_children_node_ids[edge_offset]  = child_idx;
            edge_offset++;
        }
    }

    // Phase 4: Build ad_id -> leaf_index mapping
    mapping_.sorted_entries.clear();
    mapping_.sorted_entries.reserve(ad_leaf_pairs.size());
    for (const auto& [ad_id, node_idx] : ad_leaf_pairs) {
        if (node_leaf_idx[node_idx] != UINT32_MAX) {
            mapping_.sorted_entries.emplace_back(ad_id, node_leaf_idx[node_idx]);
        }
    }
    std::sort(mapping_.sorted_entries.begin(), mapping_.sorted_entries.end());

    return true;
}

bool GlobalTrieStore::init(const std::string& ckpt_path, const std::string& sid_trie_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    init_success_ = false;

    if (sid_trie_config.empty()) {
        RTP_LLM_LOG_INFO("GlobalTrieStore: sid_trie_config is empty, skipping init");
        return false;
    }

    std::string file_path = ckpt_path + "/" + sid_trie_config;
    RTP_LLM_LOG_INFO("GlobalTrieStore: loading from [%s]", file_path.c_str());

    if (!loadFromJson(file_path)) {
        return false;
    }

    init_success_ = true;
    RTP_LLM_LOG_INFO("GlobalTrieStore: loaded successfully, %u nodes, %u leaves", trie_.num_nodes, trie_.num_leaves);
    return true;
}

}  // namespace rtp_llm
