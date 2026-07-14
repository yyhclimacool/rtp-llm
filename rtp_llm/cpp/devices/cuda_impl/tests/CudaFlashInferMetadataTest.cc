#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"

namespace rtp_llm {
namespace {

TEST(CudaFlashInferMetadataTest, BuildsDecodeMetadataFromDeviceInputs) {
    constexpr int batch_size           = 3;
    constexpr int max_blocks_per_batch = 4;
    constexpr int tokens_per_block     = 4;

    auto options          = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto sequence_lengths = torch::tensor({0, 4, 8}, options);
    auto block_ids        = torch::tensor({{10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33}}, options);

    auto page_indptr   = torch::zeros({batch_size + 1}, options);
    auto qo_indptr     = torch::zeros({batch_size + 1}, options);
    auto batch_indices = torch::zeros({batch_size}, options);
    auto positions     = torch::zeros({batch_size}, options);
    auto kv_lens       = torch::zeros({batch_size}, options);
    auto last_page_len = torch::zeros({batch_size}, options);
    auto page_indices  = torch::full({batch_size * max_blocks_per_batch}, -1, options);

    invokePrepareFlashInferDecodeMetadata(page_indptr.data_ptr<int>(),
                                          qo_indptr.data_ptr<int>(),
                                          batch_indices.data_ptr<int>(),
                                          positions.data_ptr<int>(),
                                          kv_lens.data_ptr<int>(),
                                          last_page_len.data_ptr<int>(),
                                          page_indices.data_ptr<int>(),
                                          sequence_lengths.data_ptr<int>(),
                                          block_ids.data_ptr<int>(),
                                          batch_size,
                                          max_blocks_per_batch,
                                          tokens_per_block,
                                          0);
    check_cuda_value(cudaDeviceSynchronize());

    EXPECT_TRUE(torch::equal(page_indptr.cpu(), torch::tensor({0, 1, 3, 6}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(qo_indptr.cpu(), torch::tensor({0, 1, 2, 3}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(batch_indices.cpu(), torch::tensor({0, 1, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(positions.cpu(), torch::tensor({0, 4, 8}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(kv_lens.cpu(), torch::tensor({1, 5, 9}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(last_page_len.cpu(), torch::tensor({1, 1, 1}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(page_indices.slice(0, 0, 6).cpu(), torch::tensor({10, 20, 21, 30, 31, 32}, torch::kInt32)));
}

}  // namespace
}  // namespace rtp_llm
