#include "gtest/gtest.h"

#include <cmath>
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
    auto data = si.logits.cpu().data_ptr<float>();
    EXPECT_FLOAT_EQ(data[11], 1.0f);
    EXPECT_TRUE(std::isinf(-data[10]) || data[10] <= -1e30f);

    // Transition with token 11
    processor->updateStatus(torch::tensor({{11}}, torch::kInt32), 1);

    // Step 1: only token 50 should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    data = si.logits.cpu().data_ptr<float>();
    EXPECT_FLOAT_EQ(data[50], 1.0f);
    EXPECT_TRUE(std::isinf(-data[51]) || data[51] <= -1e30f);

    // Transition with token 50
    processor->updateStatus(torch::tensor({{50}}, torch::kInt32), 1);

    // Step 2: only token 60 should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    data = si.logits.cpu().data_ptr<float>();
    EXPECT_FLOAT_EQ(data[60], 1.0f);
    EXPECT_TRUE(std::isinf(-data[61]) || data[61] <= -1e30f);

    // Transition with token 60 -> leaf
    processor->updateStatus(torch::tensor({{60}}, torch::kInt32), 1);

    // Step 3: at leaf, only end_token_id (2) should be allowed
    si = makeInputs();
    processor->process(si, 0, 1);
    data = si.logits.cpu().data_ptr<float>();
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_TRUE(std::isinf(-data[0]) || data[0] <= -1e30f);
    EXPECT_TRUE(std::isinf(-data[60]) || data[60] <= -1e30f);
}

}  // namespace rtp_llm
