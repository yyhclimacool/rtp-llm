#include "gtest/gtest.h"

#include <cmath>
#include <fstream>
#include <limits>

#include "rtp_llm/cpp/models/logits_processor/GlobalTrieCSR.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/TriePruningLogitsProcessor.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class TriePruningLogitsProcessorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        auto store = GlobalTrieStore::instance();
        store->init("./rtp_llm/cpp/models/logits_processor/test", "sid_trie_test.json");
        ASSERT_TRUE(store->initSuccess());
    }
};

// Test that GlobalTrieCSR loads correctly
TEST_F(TriePruningLogitsProcessorTest, testGlobalTrieLoad) {
    auto store = GlobalTrieStore::instance();
    ASSERT_TRUE(store->initSuccess());

    const auto& trie    = store->trie();
    const auto& mapping = store->mapping();

    // 7 ads total
    EXPECT_EQ(7u, trie.num_leaves);
    EXPECT_EQ(2, trie.end_token_id);

    // Check mapping has 7 entries
    EXPECT_EQ(7u, mapping.sorted_entries.size());

    // Verify ad_id lookup works
    EXPECT_TRUE(mapping.find("ad_1001").has_value());
    EXPECT_TRUE(mapping.find("ad_1007").has_value());
    EXPECT_FALSE(mapping.find("ad_9999").has_value());
}

// Test transition function
TEST_F(TriePruningLogitsProcessorTest, testTransition) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    // Start from root (0), transition with token 10
    uint32_t node = trie.transition(0, 10);
    EXPECT_NE(0u, node);
    EXPECT_FALSE(trie.node_is_leaf[node]);

    // From that node, transition with token 20
    uint32_t node2 = trie.transition(node, 20);
    EXPECT_NE(node, node2);
    EXPECT_FALSE(trie.node_is_leaf[node2]);

    // From node2, transition with token 30 -> leaf
    uint32_t leaf = trie.transition(node2, 30);
    EXPECT_TRUE(trie.node_is_leaf[leaf]);

    // Invalid transition should return same node
    uint32_t same = trie.transition(0, 999);
    EXPECT_EQ(0u, same);
}

// Test PerRequestTrieFilter building
TEST_F(TriePruningLogitsProcessorTest, testFilterBuild) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    PerRequestTrieFilter filter;
    // Whitelist only ads 1001 and 1004
    filter.build({"ad_1001", "ad_1004"}, store->mapping(), trie);

    EXPECT_EQ(2u, filter.sorted_leaves.size());
    EXPECT_TRUE(filter.root_active_built);

    // Root should have tokens 10 and 11 active (paths to 1001 and 1004)
    EXPECT_TRUE(filter.root_children_active[10 >> 6] & (1ULL << (10 & 63)));
    EXPECT_TRUE(filter.root_children_active[11 >> 6] & (1ULL << (11 & 63)));
    // Token 12 should NOT be active (ad 1007 not in whitelist)
    EXPECT_FALSE(filter.root_children_active[12 >> 6] & (1ULL << (12 & 63)));
}

// Test that filter with unknown ad_ids gracefully skips them
TEST_F(TriePruningLogitsProcessorTest, testFilterUnknownAdIds) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    PerRequestTrieFilter filter;
    filter.build({"ad_9999", "ad_8888", "ad_1001"}, store->mapping(), trie);

    // Only 1001 found
    EXPECT_EQ(1u, filter.sorted_leaves.size());
}

// Test process at root level - should mask non-reachable tokens
TEST_F(TriePruningLogitsProcessorTest, testProcessAtRoot) {
    auto store = GlobalTrieStore::instance();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001", "ad_1004"}, store->mapping(), store->trie());

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 1;
    const size_t vocab_size = 100;

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();

    // Tokens 10 and 11 should be allowed (paths to ads 1001 and 1004)
    EXPECT_FLOAT_EQ(data[10], 1.0f);
    EXPECT_FLOAT_EQ(data[11], 1.0f);
    // Token 12 should be masked (ad 1007 not in whitelist)
    EXPECT_TRUE(std::isinf(-data[12]) || data[12] <= -1e30f);
    // Token 0 should be masked (no path)
    EXPECT_TRUE(std::isinf(-data[0]) || data[0] <= -1e30f);
}

// Test process at internal node (after first transition)
TEST_F(TriePruningLogitsProcessorTest, testProcessAtInternalNode) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    // Whitelist only ad 1001: path [10, 20, 30]
    filter->build({"ad_1001"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);
    // Simulate having already transitioned with token 10
    infos[0].current_node_id = trie.transition(0, 10);

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 1;
    const size_t vocab_size = 100;

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();

    // At node for token 10, children are 20 and 21
    // Only ad 1001 in whitelist: path [10, 20, 30]
    // Token 20 should be allowed (leads to 1001)
    EXPECT_FLOAT_EQ(data[20], 1.0f);
    // Token 21 should be masked (leads to 1003, not in whitelist)
    EXPECT_TRUE(std::isinf(-data[21]) || data[21] <= -1e30f);
}

// Test process at leaf node - only end_token_id allowed
TEST_F(TriePruningLogitsProcessorTest, testProcessAtLeafNode) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);
    // Navigate to leaf: root -> 10 -> 20 -> 30
    uint32_t node = trie.transition(0, 10);
    node          = trie.transition(node, 20);
    node          = trie.transition(node, 30);
    ASSERT_TRUE(trie.node_is_leaf[node]);
    infos[0].current_node_id = node;

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 1;
    const size_t vocab_size = 100;

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();

    // Only end_token_id (2) should be allowed
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    // Everything else masked
    EXPECT_TRUE(std::isinf(-data[0]) || data[0] <= -1e30f);
    EXPECT_TRUE(std::isinf(-data[1]) || data[1] <= -1e30f);
    EXPECT_TRUE(std::isinf(-data[3]) || data[3] <= -1e30f);
    EXPECT_TRUE(std::isinf(-data[10]) || data[10] <= -1e30f);
}

// Test updateStatus transitions correctly
TEST_F(TriePruningLogitsProcessorTest, testUpdateStatus) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);
    EXPECT_EQ(0u, processor->infos()[0].current_node_id);

    // Simulate generating token 10
    auto tokens = torch::tensor({{10}}, torch::kInt32);
    processor->updateStatus(tokens, 1);

    uint32_t expected_node = trie.transition(0, 10);
    EXPECT_EQ(expected_node, processor->infos()[0].current_node_id);

    // Simulate generating token 20
    tokens = torch::tensor({{20}}, torch::kInt32);
    processor->updateStatus(tokens, 1);

    expected_node = trie.transition(expected_node, 20);
    EXPECT_EQ(expected_node, processor->infos()[0].current_node_id);
}

// Test updateMultiSeqStatus correctly copies state
TEST_F(TriePruningLogitsProcessorTest, testUpdateMultiSeqStatus) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001", "ad_1004"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);
    // Advance to some node
    infos[0].current_node_id = trie.transition(0, 10);

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    // Expand to 3 beams from beam 0
    processor->updateMultiSeqStatus({0, 0, 0});
    ASSERT_EQ(3u, processor->size());

    // All should share the same filter but have independent node state
    uint32_t expected = trie.transition(0, 10);
    EXPECT_EQ(expected, processor->infos()[0].current_node_id);
    EXPECT_EQ(expected, processor->infos()[1].current_node_id);
    EXPECT_EQ(expected, processor->infos()[2].current_node_id);

    // Advance beam 0 only
    auto tokens = torch::tensor({{20}, {21}, {20}}, torch::kInt32);
    processor->updateStatus(tokens, 1);

    // They should now diverge
    EXPECT_EQ(trie.transition(expected, 20), processor->infos()[0].current_node_id);
    EXPECT_EQ(trie.transition(expected, 21), processor->infos()[1].current_node_id);
    EXPECT_EQ(trie.transition(expected, 20), processor->infos()[2].current_node_id);
}

// Test fromGenerateInput returns nullptr when whitelist is empty
TEST_F(TriePruningLogitsProcessorTest, testFromGenerateInputEmpty) {
    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->trie_whitelist_ad_ids = {};
    generate_input->input_ids                              = torch::zeros({5}, torch::kInt32);

    auto p = TriePruningLogitsProcessor::fromGenerateInput(generate_input, 2);
    ASSERT_EQ(nullptr, p);
}

// Test fromGenerateInput returns valid processor when whitelist is provided
TEST_F(TriePruningLogitsProcessorTest, testFromGenerateInputValid) {
    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->trie_whitelist_ad_ids = {"ad_1001", "ad_1002", "ad_1003"};
    generate_input->input_ids                              = torch::zeros({5}, torch::kInt32);

    auto p = TriePruningLogitsProcessor::fromGenerateInput(generate_input, 3);
    ASSERT_NE(nullptr, p);
    ASSERT_EQ(3u, p->size());

    // All beams share the same filter
    EXPECT_EQ(p->infos()[0].filter.get(), p->infos()[1].filter.get());
    EXPECT_EQ(p->infos()[1].filter.get(), p->infos()[2].filter.get());
}

// Test full decode sequence for a single ad
TEST_F(TriePruningLogitsProcessorTest, testFullDecodeSequence) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    // Only allow ad 1004: path [11, 50, 60]
    filter->build({"ad_1004"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);
    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 1;
    const size_t vocab_size = 100;

    auto makeInputs = [&]() {
        SamplerInputs si;
        si.batch_size     = batch_size;
        si.batch_size_out = batch_size;
        si.vocab_size     = vocab_size;
        si.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return si;
    };

    // Step 0: at root, only token 11 should be allowed
    auto si = makeInputs();
    processor->process(si, 0, 1);
    auto logits_cpu = si.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(data[11], 1.0f);
    EXPECT_TRUE(std::isinf(-data[10]) || data[10] <= -1e30f);

    // Transition with token 11
    processor->updateStatus(torch::tensor({{11}}, torch::kInt32), 1);

    // Step 1: only token 50 should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    logits_cpu = si.logits.cpu();
    data       = logits_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(data[50], 1.0f);
    EXPECT_TRUE(std::isinf(-data[51]) || data[51] <= -1e30f);

    // Transition with token 50
    processor->updateStatus(torch::tensor({{50}}, torch::kInt32), 1);

    // Step 2: only token 60 should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    logits_cpu = si.logits.cpu();
    data       = logits_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(data[60], 1.0f);
    EXPECT_TRUE(std::isinf(-data[61]) || data[61] <= -1e30f);

    // Transition with token 60 -> leaf
    processor->updateStatus(torch::tensor({{60}}, torch::kInt32), 1);

    // Step 3: at leaf, only end_token_id (2) should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    logits_cpu = si.logits.cpu();
    data       = logits_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_TRUE(std::isinf(-data[0]) || data[0] <= -1e30f);
    EXPECT_TRUE(std::isinf(-data[60]) || data[60] <= -1e30f);
}

// =============================================================================
// A. GlobalTrieCSR structure correctness
// =============================================================================

// Verify CSR arrays are correctly built from sid_trie_test.json
// Trie structure (node IDs by insertion order):
//   node 0 (root): children {10→1, 11→7, 12→13}
//   node 1:  children {20→2, 21→5}
//   node 2:  children {30→3, 31→4}
//   node 3:  leaf (ad_1001)
//   node 4:  leaf (ad_1002)
//   node 5:  children {40→6}
//   node 6:  leaf (ad_1003)
//   node 7:  children {50→8, 51→11}
//   node 8:  children {60→9, 61→10}
//   node 9:  leaf (ad_1004)
//   node 10: leaf (ad_1005)
//   node 11: children {70→12}
//   node 12: leaf (ad_1006)
//   node 13: children {60→14}
//   node 14: children {80→15}
//   node 15: leaf (ad_1007)
TEST_F(TriePruningLogitsProcessorTest, testCSRStructure) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    // 16 nodes total: 1 root + 8 internal + 7 leaves
    EXPECT_EQ(16u, trie.num_nodes);
    EXPECT_EQ(7u, trie.num_leaves);

    // Root (node 0) has 3 children: tokens 10, 11, 12
    EXPECT_EQ(3, trie.node_children_count[0]);
    uint32_t root_off = trie.node_children_offset[0];
    EXPECT_EQ(10, trie.all_children_token_ids[root_off]);
    EXPECT_EQ(11, trie.all_children_token_ids[root_off + 1]);
    EXPECT_EQ(12, trie.all_children_token_ids[root_off + 2]);

    // Verify offset consistency: offset[i] + count[i] accounts for all edges
    for (uint32_t i = 0; i + 1 < trie.num_nodes; ++i) {
        // Offsets should be non-decreasing
        EXPECT_LE(trie.node_children_offset[i], trie.node_children_offset[i + 1]);
    }

    // Leaf nodes have 0 children
    for (uint32_t i = 0; i < trie.num_nodes; ++i) {
        if (trie.node_is_leaf[i]) {
            EXPECT_EQ(0, trie.node_children_count[i]);
        }
    }
}

// Verify DFS-order leaf contiguity invariant
TEST_F(TriePruningLogitsProcessorTest, testDFSLeafIntervals) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    // Root covers all leaves: [0, 7)
    EXPECT_EQ(0u, trie.node_leaf_lo[0]);
    EXPECT_EQ(7u, trie.node_leaf_hi[0]);

    // Every leaf has interval length 1
    for (uint32_t i = 0; i < trie.num_nodes; ++i) {
        if (trie.node_is_leaf[i]) {
            EXPECT_EQ(trie.node_leaf_lo[i] + 1, trie.node_leaf_hi[i]);
        }
    }

    // For every internal node, leaf_lo < leaf_hi
    for (uint32_t i = 0; i < trie.num_nodes; ++i) {
        if (!trie.node_is_leaf[i] && trie.node_children_count[i] > 0) {
            EXPECT_LT(trie.node_leaf_lo[i], trie.node_leaf_hi[i]);
        }
    }

    // Adjacent children have seamless intervals: c_i.leaf_hi == c_{i+1}.leaf_lo
    for (uint32_t i = 0; i < trie.num_nodes; ++i) {
        uint16_t count  = trie.node_children_count[i];
        uint32_t offset = trie.node_children_offset[i];
        for (uint16_t c = 0; c + 1 < count; ++c) {
            uint32_t child_a = trie.all_children_node_ids[offset + c];
            uint32_t child_b = trie.all_children_node_ids[offset + c + 1];
            EXPECT_EQ(trie.node_leaf_hi[child_a], trie.node_leaf_lo[child_b])
                << "Gap between sibling children at node " << i << ", children " << c << " and " << c + 1;
        }
    }
}

// Verify minimal trie: single ad (re-init the singleton with a temp JSON)
TEST_F(TriePruningLogitsProcessorTest, testSingleAdTrie) {
    std::string json = R"({"end_token_id": 2, "ads": [{"ad_id": "only", "sid_path": [100, 200]}]})";
    std::string path = "/tmp/sid_trie_single_test.json";
    std::ofstream(path) << json;

    auto store = GlobalTrieStore::instance();
    ASSERT_TRUE(store->init("/tmp", "sid_trie_single_test.json"));

    const auto& trie = store->trie();
    // root → node1 → node2(leaf): 3 nodes, 1 leaf
    EXPECT_EQ(3u, trie.num_nodes);
    EXPECT_EQ(1u, trie.num_leaves);
    EXPECT_EQ(0u, trie.node_leaf_lo[0]);
    EXPECT_EQ(1u, trie.node_leaf_hi[0]);

    EXPECT_TRUE(store->mapping().find("only").has_value());
    EXPECT_EQ(0u, store->mapping().find("only").value());

    // Restore original test data for subsequent tests
    store->init("./rtp_llm/cpp/models/logits_processor/test", "sid_trie_test.json");
}

// =============================================================================
// B. AdIdMapping
// =============================================================================

// Verify binary search returns correct leaf indices for all known ads
TEST_F(TriePruningLogitsProcessorTest, testAdIdMappingLookup) {
    auto        store   = GlobalTrieStore::instance();
    const auto& mapping = store->mapping();
    const auto& trie    = store->trie();

    // All 7 ads should be found
    std::vector<std::string> all_ads = {"ad_1001", "ad_1002", "ad_1003", "ad_1004", "ad_1005", "ad_1006", "ad_1007"};
    for (const auto& ad_id : all_ads) {
        auto idx = mapping.find(ad_id);
        ASSERT_TRUE(idx.has_value()) << "ad_id " << ad_id << " not found";
        EXPECT_LT(idx.value(), trie.num_leaves) << "leaf index out of range for " << ad_id;
    }

    // Non-existent ads should return nullopt
    EXPECT_FALSE(mapping.find("ad_0000").has_value());
    EXPECT_FALSE(mapping.find("").has_value());
    EXPECT_FALSE(mapping.find("ad_1008").has_value());
}

// Verify mapping is sorted by ad_id
TEST_F(TriePruningLogitsProcessorTest, testAdIdMappingSorted) {
    auto        store   = GlobalTrieStore::instance();
    const auto& mapping = store->mapping();

    for (size_t i = 0; i + 1 < mapping.sorted_entries.size(); ++i) {
        EXPECT_LT(mapping.sorted_entries[i].first, mapping.sorted_entries[i + 1].first)
            << "Entries not sorted at index " << i;
    }
}

// =============================================================================
// C. PerRequestTrieFilter
// =============================================================================

// Duplicate ad_ids should be deduplicated in sorted_leaves
TEST_F(TriePruningLogitsProcessorTest, testFilterDuplicateAdIds) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    PerRequestTrieFilter filter;
    filter.build({"ad_1001", "ad_1001", "ad_1002", "ad_1002", "ad_1002"}, store->mapping(), trie);

    EXPECT_EQ(2u, filter.sorted_leaves.size());
    // Should be sorted and unique
    EXPECT_LT(filter.sorted_leaves[0], filter.sorted_leaves[1]);
}

// All ads whitelisted
TEST_F(TriePruningLogitsProcessorTest, testFilterAllWhitelisted) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    PerRequestTrieFilter filter;
    filter.build({"ad_1001", "ad_1002", "ad_1003", "ad_1004", "ad_1005", "ad_1006", "ad_1007"}, store->mapping(), trie);

    EXPECT_EQ(7u, filter.sorted_leaves.size());
    EXPECT_TRUE(filter.root_active_built);

    // All 3 root children (tokens 10, 11, 12) should be active
    EXPECT_TRUE(filter.root_children_active[10 >> 6] & (1ULL << (10 & 63)));
    EXPECT_TRUE(filter.root_children_active[11 >> 6] & (1ULL << (11 & 63)));
    EXPECT_TRUE(filter.root_children_active[12 >> 6] & (1ULL << (12 & 63)));
}

// Large token_ids (>4096) should work with dynamic bitmap
TEST_F(TriePruningLogitsProcessorTest, testRootBitmapLargeTokenId) {
    auto store = GlobalTrieStore::instance();
    ASSERT_TRUE(store->init("./rtp_llm/cpp/models/logits_processor/test", "sid_trie_large_token_test.json"));

    const auto& trie    = store->trie();
    const auto& mapping = store->mapping();

    PerRequestTrieFilter filter;
    filter.build({"big_1", "big_2", "big_3"}, mapping, trie);

    EXPECT_EQ(3u, filter.sorted_leaves.size());
    EXPECT_TRUE(filter.root_active_built);

    // Bitmap should be large enough to hold token_id 150000
    size_t needed_slots = (150000 >> 6) + 1;
    EXPECT_GE(filter.root_children_active.size(), needed_slots);

    // Token 150000 should be active
    EXPECT_TRUE(filter.root_children_active[150000 >> 6] & (1ULL << (150000 & 63)));
    // Token 5000 should be active
    EXPECT_TRUE(filter.root_children_active[5000 >> 6] & (1ULL << (5000 & 63)));
    // Token 0 should NOT be active
    EXPECT_FALSE(filter.root_children_active[0] & 1ULL);

    // Restore original test data
    store->init("./rtp_llm/cpp/models/logits_processor/test", "sid_trie_test.json");
}

// =============================================================================
// D. TriePruningLogitsProcessor edge cases
// =============================================================================

// Null filter should pass through without masking
TEST_F(TriePruningLogitsProcessorTest, testProcessNullFilter) {
    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back();  // default: null filter

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 1;
    const size_t vocab_size = 100;

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();
    // All tokens should remain 1.0 (no masking)
    for (size_t t = 0; t < vocab_size; ++t) {
        EXPECT_FLOAT_EQ(data[t], 1.0f) << "Token " << t << " was unexpectedly masked";
    }
}

// Multiple beams at different trie positions
TEST_F(TriePruningLogitsProcessorTest, testProcessMultipleBeamsDifferentNodes) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001", "ad_1004"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    // Beam 0: at root
    infos.emplace_back(5, 0, false, filter);
    // Beam 1: at node after token 10 (should allow tokens 20, 21)
    infos.emplace_back(5, 0, false, filter);
    infos[1].current_node_id = trie.transition(0, 10);

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    const size_t batch_size = 2;
    const size_t vocab_size = 100;

    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();

    // Beam 0 (root): tokens 10, 11 allowed; token 12 masked (ad_1007 not in whitelist)
    EXPECT_FLOAT_EQ(data[10], 1.0f);
    EXPECT_FLOAT_EQ(data[11], 1.0f);
    EXPECT_TRUE(std::isinf(-data[12]) || data[12] <= -1e30f);

    // Beam 1 (after token 10): only token 20 allowed (ad_1001 path),
    // token 21 masked (ad_1003 not in whitelist)
    float* data1 = data + vocab_size;
    EXPECT_FLOAT_EQ(data1[20], 1.0f);
    EXPECT_TRUE(std::isinf(-data1[21]) || data1[21] <= -1e30f);
}

// insert() merges two processors
TEST_F(TriePruningLogitsProcessorTest, testInsertMergesProcessors) {
    auto store = GlobalTrieStore::instance();

    auto filter1 = std::make_shared<PerRequestTrieFilter>();
    filter1->build({"ad_1001"}, store->mapping(), store->trie());

    auto filter2 = std::make_shared<PerRequestTrieFilter>();
    filter2->build({"ad_1004"}, store->mapping(), store->trie());

    std::vector<StreamTriePruningInfo> infos1;
    infos1.emplace_back(5, 0, false, filter1);

    std::vector<StreamTriePruningInfo> infos2;
    infos2.emplace_back(5, 0, false, filter2);
    infos2.emplace_back(5, 0, false, filter2);

    auto proc1 = std::make_shared<TriePruningLogitsProcessor>(infos1);
    auto proc2 = std::make_shared<TriePruningLogitsProcessor>(infos2);

    ASSERT_EQ(1u, proc1->size());
    proc1->insert(proc2);
    ASSERT_EQ(3u, proc1->size());

    // First info uses filter1, next two use filter2
    EXPECT_EQ(proc1->infos()[0].filter.get(), filter1.get());
    EXPECT_EQ(proc1->infos()[1].filter.get(), filter2.get());
    EXPECT_EQ(proc1->infos()[2].filter.get(), filter2.get());
}

// updateStatus with end_token_id should not change current_node_id
TEST_F(TriePruningLogitsProcessorTest, testUpdateStatusSkipsEndToken) {
    auto        store = GlobalTrieStore::instance();
    const auto& trie  = store->trie();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001"}, store->mapping(), trie);

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);
    // Navigate to leaf: root -> 10 -> 20 -> 30
    uint32_t node = trie.transition(0, 10);
    node          = trie.transition(node, 20);
    node          = trie.transition(node, 30);
    ASSERT_TRUE(trie.node_is_leaf[node]);
    infos[0].current_node_id = node;

    auto processor = std::make_shared<TriePruningLogitsProcessor>(infos);

    uint32_t leaf_node = processor->infos()[0].current_node_id;

    // Send end_token_id (2) — should NOT cause transition
    auto tokens = torch::tensor({{trie.end_token_id}}, torch::kInt32);
    processor->updateStatus(tokens, 1);

    EXPECT_EQ(leaf_node, processor->infos()[0].current_node_id);
}

// insert with nullptr should be a no-op
TEST_F(TriePruningLogitsProcessorTest, testInsertNullptr) {
    auto store = GlobalTrieStore::instance();

    auto filter = std::make_shared<PerRequestTrieFilter>();
    filter->build({"ad_1001"}, store->mapping(), store->trie());

    std::vector<StreamTriePruningInfo> infos;
    infos.emplace_back(5, 0, false, filter);

    auto proc = std::make_shared<TriePruningLogitsProcessor>(infos);
    ASSERT_EQ(1u, proc->size());

    proc->insert(nullptr);
    EXPECT_EQ(1u, proc->size());
}

}  // namespace rtp_llm
