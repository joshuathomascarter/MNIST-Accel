// test_tiling.cpp — Unit tests for tiling strategy
// =============================================================================
//
// Tests the tile computation engine: padding, grid calculation, tile descriptor
// generation, MNIST-specific layer plans, and BSR sparse tiling.
//
// =============================================================================
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"
#include "memory/buffer_manager.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>

using namespace accel::compute;
using namespace accel::memory;

// =============================================================================
// Test helpers
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { test_##name(); } } reg_##name; \
    static void test_##name()

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::fprintf(stderr, "  FAIL %s:%d: %s != %s  (%lld != %lld)\n", \
                     __FILE__, __LINE__, #a, #b, \
                     (long long)_a, (long long)_b); \
        tests_failed++; return; \
    } } while(0)

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        tests_failed++; return; \
    } } while(0)

#define PASS(name) do { \
    std::fprintf(stdout, "  PASS: %s\n", name); \
    tests_passed++; } while(0)

// =============================================================================
// Test: padTo16
// =============================================================================
TEST(padTo16_basic) {
    ASSERT_EQ(padTo16(1), 16u);
    ASSERT_EQ(padTo16(16), 16u);
    ASSERT_EQ(padTo16(17), 32u);
    ASSERT_EQ(padTo16(32), 32u);
    ASSERT_EQ(padTo16(0), 0u);
    ASSERT_EQ(padTo16(15), 16u);
    ASSERT_EQ(padTo16(256), 256u);
    ASSERT_EQ(padTo16(257), 272u);
    PASS("padTo16_basic");
}

// =============================================================================
// Test: computePaddedShape
// =============================================================================
TEST(padded_shape) {
    GEMMShape shape{32, 9, 9};
    auto padded = computePaddedShape(shape);
    ASSERT_EQ(padded.M_padded, 32u);   // ceil(32/16)*16 = 2*16 = 32
    ASSERT_EQ(padded.N_padded, 16u);   // ceil(9/16)*16 = 16
    ASSERT_EQ(padded.K_padded, 16u);   // ceil(9/16)*16 = 16
    PASS("padded_shape");
}

// =============================================================================
// Test: computeTileGrid
// =============================================================================
TEST(tile_grid) {
    PaddedShape padded{32, 16, 16};
    auto grid = computeTileGrid(padded);
    ASSERT_EQ(grid.num_m_tiles, 2u);
    ASSERT_EQ(grid.num_n_tiles, 1u);
    ASSERT_EQ(grid.num_k_tiles, 1u);
    ASSERT_EQ(grid.totalTiles(), 2u);
    PASS("tile_grid");
}

// =============================================================================
// Test: planDenseGEMM — small matrix
// =============================================================================
TEST(plan_small_gemm) {
    GEMMShape shape{16, 16, 16};
    auto plan = planDenseGEMM(shape, "test_1x1x1");

    ASSERT_EQ(plan.grid.num_m_tiles, 1u);
    ASSERT_EQ(plan.grid.num_n_tiles, 1u);
    ASSERT_EQ(plan.grid.num_k_tiles, 1u);
    ASSERT_EQ(plan.totalTiles(), 1u);
    ASSERT_EQ(plan.tiles.size(), 1u);

    auto& td = plan.tiles[0];
    ASSERT_EQ(td.m_idx, 0u);
    ASSERT_EQ(td.n_idx, 0u);
    ASSERT_EQ(td.k_idx, 0u);
    ASSERT_TRUE(td.is_first_k);
    ASSERT_TRUE(td.is_last_k);

    PASS("plan_small_gemm");
}

// =============================================================================
// Test: planDenseGEMM — conv1 shape (32×9 weights)
// =============================================================================
TEST(plan_conv1_shape) {
    // conv1: M=32, K=9, N=676 → padded M=32, K=16, N=688
    GEMMShape shape{32, 676, 9};
    auto plan = planDenseGEMM(shape, "conv1");

    ASSERT_EQ(plan.padded.M_padded, 32u);
    ASSERT_EQ(plan.padded.K_padded, 16u);
    ASSERT_EQ(plan.padded.N_padded, 688u);   // ceil(676/16)*16 = 43*16 = 688

    ASSERT_EQ(plan.grid.num_m_tiles, 2u);
    ASSERT_EQ(plan.grid.num_k_tiles, 1u);
    ASSERT_EQ(plan.grid.num_n_tiles, 43u);

    // Total = 2 × 43 × 1 = 86
    ASSERT_EQ(plan.totalTiles(), 86u);
    ASSERT_EQ(plan.tiles.size(), 86u);

    // All tiles should have is_first_k=true AND is_last_k=true (only 1 K tile)
    for (const auto& td : plan.tiles) {
        ASSERT_TRUE(td.is_first_k);
        ASSERT_TRUE(td.is_last_k);
    }

    PASS("plan_conv1_shape");
}

// =============================================================================
// Test: planDenseGEMM — multi-K tiling
// =============================================================================
TEST(plan_multi_k) {
    // M=16, K=48, N=16 → 3 K-tiles
    GEMMShape shape{16, 16, 48};
    auto plan = planDenseGEMM(shape, "multi_k");

    ASSERT_EQ(plan.grid.num_m_tiles, 1u);
    ASSERT_EQ(plan.grid.num_n_tiles, 1u);
    ASSERT_EQ(plan.grid.num_k_tiles, 3u);
    ASSERT_EQ(plan.totalTiles(), 3u);

    // Check K-tile flags
    ASSERT_TRUE(plan.tiles[0].is_first_k);
    ASSERT_TRUE(!plan.tiles[0].is_last_k);

    ASSERT_TRUE(!plan.tiles[1].is_first_k);
    ASSERT_TRUE(!plan.tiles[1].is_last_k);

    ASSERT_TRUE(!plan.tiles[2].is_first_k);
    ASSERT_TRUE(plan.tiles[2].is_last_k);

    PASS("plan_multi_k");
}

// =============================================================================
// Test: tile offsets
// =============================================================================
TEST(tile_offsets) {
    // Weight tile at (m=2, k=3) in a grid with 5 K-tiles, base=1000
    uint32_t off = weightTileOffset(2, 3, 5, 1000);
    // index = 2*5 + 3 = 13, offset = 1000 + 13*256 = 4328
    ASSERT_EQ(off, 1000u + 13u * 256u);

    // Activation tile at (k=1, n=4) with 7 N-tiles, base=0
    uint32_t act_off = activationTileOffset(1, 4, 7, 0);
    // index = 1*7 + 4 = 11, offset = 11*256 = 2816
    ASSERT_EQ(act_off, 11u * 256u);

    // Output tile at (m=0, n=2) with 3 N-tiles, base=0
    uint32_t out_off = outputTileOffset(0, 2, 3, 0);
    // index = 0*3 + 2 = 2, offset = 2*256*4 = 2048
    ASSERT_EQ(out_off, 2u * 256u * 4u);

    PASS("tile_offsets");
}

// =============================================================================
// Test: planDenseGEMM — fc1 weight shape (128×9216)
// =============================================================================
TEST(plan_fc1) {
    // fc1: M=128, K=9216, N=1 → padded M=128, K=9216, N=16
    GEMMShape shape{128, 1, 9216};
    auto plan = planDenseGEMM(shape, "fc1");

    ASSERT_EQ(plan.padded.M_padded, 128u);
    ASSERT_EQ(plan.padded.K_padded, 9216u);  // 9216 is already a multiple of 16
    ASSERT_EQ(plan.padded.N_padded, 16u);

    ASSERT_EQ(plan.grid.num_m_tiles, 8u);
    ASSERT_EQ(plan.grid.num_k_tiles, 576u);
    ASSERT_EQ(plan.grid.num_n_tiles, 1u);

    // Total tiles = 8 × 1 × 576 = 4608
    ASSERT_EQ(plan.totalTiles(), 4608u);

    PASS("plan_fc1");
}

// =============================================================================
// Test: planDenseGEMM — fc2 weight shape (10×128)
// =============================================================================
TEST(plan_fc2) {
    GEMMShape shape{10, 1, 128};
    auto plan = planDenseGEMM(shape, "fc2");

    ASSERT_EQ(plan.padded.M_padded, 16u);
    ASSERT_EQ(plan.padded.K_padded, 128u);    // 128 is already a multiple of 16
    ASSERT_EQ(plan.padded.N_padded, 16u);

    ASSERT_EQ(plan.grid.num_m_tiles, 1u);
    ASSERT_EQ(plan.grid.num_k_tiles, 8u);
    ASSERT_EQ(plan.grid.num_n_tiles, 1u);

    ASSERT_EQ(plan.totalTiles(), 8u);

    PASS("plan_fc2");
}

// =============================================================================
// Test: MNIST layer plans
// =============================================================================
TEST(mnist_all_layers) {
    auto plans = mnist::planAllLayers();
    ASSERT_EQ(plans.size(), 4u);

    // Verify layer names
    ASSERT_TRUE(plans[0].layer_name == "conv1");
    ASSERT_TRUE(plans[1].layer_name == "conv2");
    ASSERT_TRUE(plans[2].layer_name == "fc1");
    ASSERT_TRUE(plans[3].layer_name == "fc2");

    // Verify fc1 matches model_summary (4608 blocks = tiles along m,k)
    // fc1: M=128, K=9216, N=1 → 8 × 1 × 576 = 4608
    ASSERT_EQ(plans[2].totalTiles(), 4608u);

    // Verify fc2 matches model_summary (8 blocks)
    ASSERT_EQ(plans[3].totalTiles(), 8u);

    // Print all plans
    for (const auto& p : plans) {
        p.print(std::cout);
    }

    PASS("mnist_all_layers");
}

// =============================================================================
// Test: tile descriptor iteration order
// =============================================================================
TEST(tile_iteration_order) {
    // 2M × 3N × 4K tiles
    GEMMShape shape{32, 48, 64};
    auto plan = planDenseGEMM(shape);

    ASSERT_EQ(plan.grid.num_m_tiles, 2u);
    ASSERT_EQ(plan.grid.num_n_tiles, 3u);
    ASSERT_EQ(plan.grid.num_k_tiles, 4u);
    ASSERT_EQ(plan.tiles.size(), 24u);

    // Verify iteration order: M → N → K
    size_t idx = 0;
    for (uint32_t m = 0; m < 2; ++m) {
        for (uint32_t n = 0; n < 3; ++n) {
            for (uint32_t k = 0; k < 4; ++k) {
                ASSERT_EQ(plan.tiles[idx].m_idx, m);
                ASSERT_EQ(plan.tiles[idx].n_idx, n);
                ASSERT_EQ(plan.tiles[idx].k_idx, k);
                ++idx;
            }
        }
    }

    PASS("tile_iteration_order");
}

// =============================================================================
// Test: TilingPlan metrics
// =============================================================================
TEST(plan_metrics) {
    GEMMShape shape{32, 32, 32};
    auto plan = planDenseGEMM(shape, "metrics_test");

    // 2×2×2 = 8 tiles
    ASSERT_EQ(plan.totalTiles(), 8u);

    // Weight bytes: 2 * 2 * 256 = 1024
    ASSERT_EQ(plan.totalWeightBytes(), 1024u);

    // Activation bytes: 2 * 2 * 256 = 1024
    ASSERT_EQ(plan.totalActivationBytes(), 1024u);

    // Output bytes: 2 * 2 * 1024 = 4096
    ASSERT_EQ(plan.totalOutputBytes(), 4096u);

    // Total MACs: 32 * 32 * 32 = 32768
    ASSERT_EQ(plan.totalMACs(), 32768ull);

    PASS("plan_metrics");
}

// =============================================================================
// Test: planSparseGEMM — fully dense BSR (0% sparse) matches dense plan
// =============================================================================
TEST(sparse_fully_dense) {
    // Create a fully-dense BSR matrix: 2 block-rows × 3 block-cols, all NNZ
    BSRMatrix bsr;
    bsr.num_block_rows = 2;
    bsr.num_block_cols = 3;
    bsr.nnz_blocks = 6;  // All blocks present
    bsr.row_ptr = {0, 3, 6};
    bsr.col_idx = {0, 1, 2, 0, 1, 2};
    bsr.values.resize(6 * 256, 1);  // Dummy non-zero values

    GEMMShape shape{32, 16, 48};  // M=32, N=16, K=48

    auto sparse_plan = planSparseGEMM(bsr, shape, "full_dense_bsr");
    auto dense_plan  = planWeightStationaryGEMM(shape, "full_dense_ws");

    // Same number of tiles (no blocks pruned)
    ASSERT_EQ(sparse_plan.tiles.size(), dense_plan.tiles.size());
    ASSERT_TRUE(sparse_plan.is_sparse);
    ASSERT_EQ(sparse_plan.nnz_weight_blocks, 6u);

    // Sparsity should be 0%
    ASSERT_TRUE(sparse_plan.weight_sparsity() < 0.01f);
    ASSERT_TRUE(sparse_plan.sparseSpeedup() < 1.1);

    PASS("sparse_fully_dense");
}

// =============================================================================
// Test: planSparseGEMM — 50% sparse (diagonal blocks only)
// =============================================================================
TEST(sparse_50_percent) {
    // 3 block-rows × 3 block-cols, only diagonal blocks present
    BSRMatrix bsr;
    bsr.num_block_rows = 3;
    bsr.num_block_cols = 3;
    bsr.nnz_blocks = 3;  // Only 3 of 9 blocks
    bsr.row_ptr = {0, 1, 2, 3};
    bsr.col_idx = {0, 1, 2};     // Diagonal only
    bsr.values.resize(3 * 256, 1);

    GEMMShape shape{48, 16, 48};  // M=48, N=16, K=48

    auto plan = planSparseGEMM(bsr, shape, "diag_sparse");

    // Should have 3 NNZ blocks × 1 N-tile = 3 tiles (instead of 9)
    ASSERT_EQ(plan.tiles.size(), 3u);
    ASSERT_EQ(plan.nnz_weight_blocks, 3u);
    ASSERT_TRUE(plan.weight_sparsity() > 0.6f);  // 6/9 = 66.7% sparse

    // Each tile should be both first_k and last_k (1 NZ per row)
    for (const auto& td : plan.tiles) {
        ASSERT_TRUE(td.is_first_k);
        ASSERT_TRUE(td.is_last_k);
    }

    // Speedup should be ~3×
    ASSERT_TRUE(plan.sparseSpeedup() > 2.5);
    ASSERT_TRUE(plan.sparseSpeedup() < 3.5);

    PASS("sparse_50_percent");
}

// =============================================================================
// Test: planSparseGEMM — weight caching across N tiles
// =============================================================================
TEST(sparse_weight_caching) {
    // 1 block-row × 2 block-cols, both present, N=28 → 2 N-tiles
    BSRMatrix bsr;
    bsr.num_block_rows = 1;
    bsr.num_block_cols = 2;
    bsr.nnz_blocks = 2;
    bsr.row_ptr = {0, 2};
    bsr.col_idx = {0, 1};
    bsr.values.resize(2 * 256, 1);

    GEMMShape shape{16, 32, 32};  // N=32 → 2 N-tiles

    auto plan = planSparseGEMM(bsr, shape, "wgt_cache_sparse");

    // 2 NNZ × 2 N-tiles = 4 tiles
    ASSERT_EQ(plan.tiles.size(), 4u);

    // Loop order: m=0, k=col_idx[0]=0, n=0,1; k=col_idx[1]=1, n=0,1
    // Tile 0: m=0, k=0, n=0 → wgt_cached=false, is_first_k=true
    // Tile 1: m=0, k=0, n=1 → wgt_cached=true
    // Tile 2: m=0, k=1, n=0 → wgt_cached=false, is_last_k=true
    // Tile 3: m=0, k=1, n=1 → wgt_cached=true,  is_last_k=true
    ASSERT_TRUE(!plan.tiles[0].wgt_cached);
    ASSERT_TRUE(plan.tiles[0].is_first_k);
    ASSERT_TRUE(!plan.tiles[0].is_last_k);

    ASSERT_TRUE(plan.tiles[1].wgt_cached);

    ASSERT_TRUE(!plan.tiles[2].wgt_cached);
    ASSERT_TRUE(!plan.tiles[2].is_first_k);
    ASSERT_TRUE(plan.tiles[2].is_last_k);

    ASSERT_TRUE(plan.tiles[3].wgt_cached);
    ASSERT_TRUE(plan.tiles[3].is_last_k);

    PASS("sparse_weight_caching");
}

// =============================================================================
// Test: planSparseGEMM — weight offsets are BSR-packed (not dense layout)
// =============================================================================
TEST(sparse_weight_offsets) {
    // 2 block-rows × 4 block-cols, scattered NNZ
    BSRMatrix bsr;
    bsr.num_block_rows = 2;
    bsr.num_block_cols = 4;
    bsr.nnz_blocks = 3;
    bsr.row_ptr = {0, 1, 3};
    bsr.col_idx = {2, 0, 3};   // Row 0: col 2.  Row 1: col 0, col 3
    bsr.values.resize(3 * 256, 1);

    GEMMShape shape{32, 16, 64};
    auto plan = planSparseGEMM(bsr, shape, "offset_test", /*wgt_base=*/1000);

    // Weight offsets should be sequential in BSR order:
    //   NZ block 0 (row=0, col=2) → wgt_base + 0*256 = 1000
    //   NZ block 1 (row=1, col=0) → wgt_base + 1*256 = 1256
    //   NZ block 2 (row=1, col=3) → wgt_base + 2*256 = 1512
    ASSERT_EQ(plan.tiles[0].wgt_offset, 1000u);          // NZ 0, n=0
    ASSERT_EQ(plan.tiles[0].k_idx, 2u);                  // col_idx[0]=2
    ASSERT_EQ(plan.tiles[1].wgt_offset, 1000u + 256u);   // NZ 1, n=0
    ASSERT_EQ(plan.tiles[1].k_idx, 0u);                  // col_idx[1]=0
    ASSERT_EQ(plan.tiles[2].wgt_offset, 1000u + 2*256u); // NZ 2, n=0
    ASSERT_EQ(plan.tiles[2].k_idx, 3u);                  // col_idx[2]=3

    PASS("sparse_weight_offsets");
}

// =============================================================================
// Test: planSparseGEMM fc1-scale — 90% sparse fc1
// =============================================================================
TEST(sparse_fc1_90pct) {
    // Simulate fc1: M=128, K=9216, N=32 → padded M=128, K=9216
    // 8 block-rows × 576 block-cols = 4608 total blocks
    // At 90% sparsity: ~576 NNZ blocks

    uint32_t nbr = 8;
    uint32_t nbc = 576;
    uint32_t nnz_per_row = 58;  // ~10% of 576 = ~58 blocks per row

    BSRMatrix bsr;
    bsr.num_block_rows = nbr;
    bsr.num_block_cols = nbc;
    bsr.row_ptr.resize(nbr + 1);
    bsr.row_ptr[0] = 0;

    // Fill with evenly-spaced non-zero columns
    for (uint32_t r = 0; r < nbr; ++r) {
        bsr.row_ptr[r + 1] = bsr.row_ptr[r] + nnz_per_row;
        for (uint32_t i = 0; i < nnz_per_row; ++i) {
            bsr.col_idx.push_back(i * 10);  // Every 10th column
        }
    }
    bsr.nnz_blocks = static_cast<uint32_t>(bsr.col_idx.size());
    bsr.values.resize(bsr.nnz_blocks * 256, 42);  // Dummy values

    GEMMShape shape{128, 32, 9216};
    auto plan = planSparseGEMM(bsr, shape, "fc1_sparse90");

    // NNZ = 464, N-tiles = 32/16 = 2
    // Expected tiles = 464 × 2 = 928
    ASSERT_EQ(plan.nnz_weight_blocks, 464u);
    ASSERT_EQ(plan.tiles.size(), 928u);

    // Dense would be 4608 × 2 = 9216 tiles
    ASSERT_TRUE(plan.weight_sparsity() > 0.89f);
    ASSERT_TRUE(plan.sparseSpeedup() > 9.0);

    // Weight bytes: only 464 blocks × 256 = 118,784
    ASSERT_EQ(plan.totalWeightBytes(), 464u * 256u);

    plan.print(std::cout);

    PASS("sparse_fc1_90pct");
}

// =============================================================================
// Test: planSparseGEMM — empty row (all blocks pruned)
// =============================================================================
TEST(sparse_empty_row) {
    // 3 block-rows, row 1 is completely empty
    BSRMatrix bsr;
    bsr.num_block_rows = 3;
    bsr.num_block_cols = 2;
    bsr.nnz_blocks = 3;
    bsr.row_ptr = {0, 2, 2, 3};  // Row 1 has 0 NNZ
    bsr.col_idx = {0, 1, 1};     // Row 0: cols 0,1.  Row 2: col 1
    bsr.values.resize(3 * 256, 1);

    GEMMShape shape{48, 16, 32};
    auto plan = planSparseGEMM(bsr, shape, "empty_row");

    // 3 NNZ × 1 N-tile = 3 tiles (row 1 contributes nothing)
    ASSERT_EQ(plan.tiles.size(), 3u);

    // Tiles for row 0: m=0
    ASSERT_EQ(plan.tiles[0].m_idx, 0u);
    ASSERT_EQ(plan.tiles[1].m_idx, 0u);
    // Tile for row 2: m=2 (row 1 skipped entirely)
    ASSERT_EQ(plan.tiles[2].m_idx, 2u);

    PASS("sparse_empty_row");
}

// =============================================================================
// Test: TilingPlan sparse metrics
// =============================================================================
TEST(sparse_plan_metrics) {
    BSRMatrix bsr;
    bsr.num_block_rows = 2;
    bsr.num_block_cols = 4;
    bsr.nnz_blocks = 2;
    bsr.row_ptr = {0, 1, 2};
    bsr.col_idx = {1, 3};
    bsr.values.resize(2 * 256, 1);

    GEMMShape shape{32, 16, 64};
    auto plan = planSparseGEMM(bsr, shape, "metrics");

    // Dense would be 2 × 4 = 8 blocks, sparse has 2
    ASSERT_EQ(plan.dense_weight_blocks(), 8u);
    ASSERT_EQ(plan.nnz_weight_blocks, 2u);
    ASSERT_TRUE(std::abs(plan.weight_sparsity() - 0.75f) < 0.01f);
    ASSERT_TRUE(std::abs(plan.sparseSpeedup() - 4.0) < 0.1);

    // Weight bytes: 2 × 256 = 512 (not 8 × 256 = 2048)
    ASSERT_EQ(plan.totalWeightBytes(), 512u);

    PASS("sparse_plan_metrics");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::fprintf(stdout, "\n=== Tiling Unit Tests ===\n\n");

    // Tests are auto-registered by static constructors above.
    // Just report results.

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n\n",
                 tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
