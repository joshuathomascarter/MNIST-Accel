// test_end_to_end.cpp — End-to-end inference test against golden model
// =============================================================================
//
// Verifies the full software pipeline:
//   1. Dense INT8 GEMM → golden model vs tiled GEMM (no hardware)
//   2. BSR encode/decode roundtrip with GEMM verification
//   3. Golden MNIST inference with known weights
//   4. Tiling plan → manual tile-by-tile GEMM accumulation
//
// These tests run entirely in software — no /dev/mem or hardware required.
//
// =============================================================================
#include "compute/golden_model.hpp"
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"
#include "memory/buffer_manager.hpp"
#include "memory/address_map.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace accel::compute;
using namespace accel::memory;

static int tests_passed = 0;
static int tests_failed = 0;

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
// Helpers
// =============================================================================

/// Fill a matrix with deterministic pseudo-random INT8 values
static void fillRandom(int8_t* data, size_t count, int seed) {
    std::srand(seed);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int8_t>((std::rand() % 256) - 128);
    }
}

/// Fill bias with small INT32 values
static void fillBias(int32_t* data, size_t count, int seed) {
    std::srand(seed);
    for (size_t i = 0; i < count; ++i) {
        data[i] = (std::rand() % 200) - 100;
    }
}

// =============================================================================
// Test 1: Small GEMM — dense vs tiled must match
// =============================================================================
void test_dense_vs_tiled_small() {
    const uint32_t M = 4, N = 3, K = 5;
    std::vector<int8_t> A(M * K), B(K * N);
    fillRandom(A.data(), A.size(), 42);
    fillRandom(B.data(), B.size(), 43);

    auto C_dense = GoldenModel::gemmINT8(A.data(), B.data(), M, N, K);
    auto C_tiled = GoldenModel::tiledGemmINT8(A.data(), B.data(), M, N, K);

    ASSERT_EQ(C_dense.size(), (size_t)(M * N));
    ASSERT_EQ(C_tiled.size(), (size_t)(M * N));
    for (size_t i = 0; i < C_dense.size(); ++i) {
        ASSERT_EQ(C_dense[i], C_tiled[i]);
    }
    PASS("dense_vs_tiled_small");
}

// =============================================================================
// Test 2: GEMM matching for sizes exceeding 14×14 tile boundary
// =============================================================================
void test_dense_vs_tiled_multi_tile() {
    const uint32_t M = 28, N = 14, K = 42;  // 2×1×3 tiles
    std::vector<int8_t> A(M * K), B(K * N);
    fillRandom(A.data(), A.size(), 100);
    fillRandom(B.data(), B.size(), 101);

    auto C_dense = GoldenModel::gemmINT8(A.data(), B.data(), M, N, K);
    auto C_tiled = GoldenModel::tiledGemmINT8(A.data(), B.data(), M, N, K);

    ASSERT_EQ(C_dense.size(), (size_t)(M * N));
    for (size_t i = 0; i < C_dense.size(); ++i) {
        ASSERT_EQ(C_dense[i], C_tiled[i]);
    }
    PASS("dense_vs_tiled_multi_tile");
}

// =============================================================================
// Test 3: Non-aligned dimensions (requires padding in tiled path)
// =============================================================================
void test_dense_vs_tiled_non_aligned() {
    const uint32_t M = 17, N = 5, K = 23;  // odd sizes, not multiple of 14
    std::vector<int8_t> A(M * K), B(K * N);
    fillRandom(A.data(), A.size(), 200);
    fillRandom(B.data(), B.size(), 201);

    auto C_dense = GoldenModel::gemmINT8(A.data(), B.data(), M, N, K);
    auto C_tiled = GoldenModel::tiledGemmINT8(A.data(), B.data(), M, N, K);

    ASSERT_EQ(C_dense.size(), (size_t)(M * N));
    for (size_t i = 0; i < C_dense.size(); ++i) {
        ASSERT_EQ(C_dense[i], C_tiled[i]);
    }
    PASS("dense_vs_tiled_non_aligned");
}

// =============================================================================
// Test 4: BSR encode → decode → GEMM matches original
// =============================================================================
void test_bsr_roundtrip_gemm() {
    const uint32_t M = 14, K = 28;
    const uint32_t N = 1;
    std::vector<int8_t> W(M * K), A(K * N);
    fillRandom(W.data(), W.size(), 300);
    fillRandom(A.data(), A.size(), 301);

    // Original GEMM
    auto C_orig = GoldenModel::gemmINT8(W.data(), A.data(), M, N, K);

    // BSR encode → decode → re-GEMM
    BSREncoder encoder;
    auto bsr = encoder.encode(W.data(), M, K);
    auto W_decoded = encoder.decode(bsr, M, K);

    // Decoded weights should be identical (no sparsity loss for non-zero blocks)
    for (size_t i = 0; i < W.size(); ++i) {
        ASSERT_EQ(W[i], W_decoded[i]);
    }

    auto C_decoded = GoldenModel::gemmINT8(W_decoded.data(), A.data(), M, N, K);
    for (size_t i = 0; i < C_orig.size(); ++i) {
        ASSERT_EQ(C_orig[i], C_decoded[i]);
    }
    PASS("bsr_roundtrip_gemm");
}

// =============================================================================
// Test 5: Sparse weight matrix → BSR reduces block count
// =============================================================================
void test_sparse_bsr_block_savings() {
    const uint32_t M = 28, K = 28;  // 2×2 = 4 blocks
    std::vector<int8_t> W(M * K, 0);

    // Only fill the top-left 14×14 block
    for (uint32_t r = 0; r < 14; ++r)
        for (uint32_t c = 0; c < 14; ++c)
            W[r * K + c] = static_cast<int8_t>(r + c + 1);

    BSREncoder encoder;
    auto bsr = encoder.encode(W.data(), M, K);

    ASSERT_EQ(bsr.num_block_rows, 2u);
    ASSERT_EQ(bsr.num_block_cols, 2u);
    ASSERT_EQ(bsr.nnz_blocks, 1u);  // Only 1 out of 4 blocks is non-zero
    ASSERT_TRUE(bsr.sparsity() > 0.7f);

    PASS("sparse_bsr_block_savings");
}

// =============================================================================
// Test 6: Tiling plan produces correct tile descriptors
// =============================================================================
void test_tiling_plan_descriptors() {
    GEMMShape shape{28, 1, 14};  // 2 M-tiles, 1 N-tile, 1 K-tile
    auto plan = planDenseGEMM(shape, "test_layer");

    ASSERT_EQ(plan.grid.num_m_tiles, 2u);
    ASSERT_EQ(plan.grid.num_n_tiles, 1u);
    ASSERT_EQ(plan.grid.num_k_tiles, 1u);
    ASSERT_EQ(plan.totalTiles(), 2u);
    ASSERT_EQ(plan.tiles.size(), 2u);

    // First tile: m=0, n=0, k=0, is_first_k=true, is_last_k=true
    auto& t0 = plan.tiles[0];
    ASSERT_EQ(t0.m_idx, 0u);
    ASSERT_EQ(t0.n_idx, 0u);
    ASSERT_EQ(t0.k_idx, 0u);
    ASSERT_TRUE(t0.is_first_k);
    ASSERT_TRUE(t0.is_last_k);

    // Second tile: m=1
    auto& t1 = plan.tiles[1];
    ASSERT_EQ(t1.m_idx, 1u);
    ASSERT_TRUE(t1.is_first_k);
    ASSERT_TRUE(t1.is_last_k);

    PASS("tiling_plan_descriptors");
}

// =============================================================================
// Test 7: Tile-by-tile manual accumulation matches golden GEMM
// =============================================================================
void test_tiled_accumulation() {
    const uint32_t M = 14, N = 1, K = 28;  // 1 M, 1 N, 2 K tiles
    std::vector<int8_t> A(M * K), B(K * N);
    fillRandom(A.data(), A.size(), 500);
    fillRandom(B.data(), B.size(), 501);

    // Golden result
    auto C_golden = GoldenModel::gemmINT8(A.data(), B.data(), M, N, K);

    // Manual tile-by-tile accumulation (simulating hardware)
    uint32_t M_pad = padTo14(M), N_pad = padTo14(N), K_pad = padTo14(K);

    // Pad A [M×K] → [M_pad×K_pad]
    std::vector<int8_t> A_pad(M_pad * K_pad, 0);
    for (uint32_t r = 0; r < M; ++r)
        std::memcpy(&A_pad[r * K_pad], &A[r * K], K);

    // Pad B [K×N] → [K_pad×N_pad]
    std::vector<int8_t> B_pad(K_pad * N_pad, 0);
    for (uint32_t r = 0; r < K; ++r)
        std::memcpy(&B_pad[r * N_pad], &B[r * N], N);

    // Accumulate tile-by-tile
    std::vector<int32_t> C_acc(M_pad * N_pad, 0);
    uint32_t tm = M_pad / 14, tn = N_pad / 14, tk = K_pad / 14;

    for (uint32_t mi = 0; mi < tm; ++mi) {
        for (uint32_t ni = 0; ni < tn; ++ni) {
            for (uint32_t ki = 0; ki < tk; ++ki) {
                // Extract 14×14 tiles
                for (uint32_t r = 0; r < 14; ++r) {
                    for (uint32_t c = 0; c < 14; ++c) {
                        int32_t sum = 0;
                        for (uint32_t d = 0; d < 14; ++d) {
                            int8_t a_val = A_pad[(mi * 14 + r) * K_pad + (ki * 14 + d)];
                            int8_t b_val = B_pad[(ki * 14 + d) * N_pad + (ni * 14 + c)];
                            sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                        }
                        C_acc[(mi * 14 + r) * N_pad + (ni * 14 + c)] += sum;
                    }
                }
            }
        }
    }

    // Extract unpadded result
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            ASSERT_EQ(C_golden[r * N + c], C_acc[r * N_pad + c]);
        }
    }
    PASS("tiled_accumulation");
}

// =============================================================================
// Test 8: im2col + GEMM matches conv2d golden model
// =============================================================================
void test_im2col_conv() {
    // Small conv: 1 input channel, 2 output channels, 3×3 kernel, 5×5 input
    const uint32_t C_in = 1, C_out = 2, kH = 3, kW = 3;
    const uint32_t H_in = 5, W_in = 5, stride = 1, padding = 0;
    const uint32_t H_out = 3, W_out = 3;

    std::vector<int8_t> input(C_in * H_in * W_in);
    std::vector<int8_t> weight(C_out * C_in * kH * kW);
    std::vector<int32_t> bias(C_out, 0);

    fillRandom(input.data(), input.size(), 600);
    fillRandom(weight.data(), weight.size(), 601);

    auto output = GoldenModel::conv2d(input.data(), weight.data(), bias.data(),
                                       C_in, C_out, H_in, W_in, kH, kW,
                                       stride, padding);

    ASSERT_EQ(output.size(), (size_t)(C_out * H_out * W_out));

    // Manually verify first output element (channel 0, position (0,0))
    int32_t expected = 0;
    for (uint32_t ci = 0; ci < C_in; ++ci)
        for (uint32_t kr = 0; kr < kH; ++kr)
            for (uint32_t kc = 0; kc < kW; ++kc)
                expected += (int32_t)weight[0 * C_in * kH * kW + ci * kH * kW + kr * kW + kc] *
                            (int32_t)input[ci * H_in * W_in + kr * W_in + kc];
    expected += bias[0];

    ASSERT_EQ(output[0], expected);

    PASS("im2col_conv");
}

// =============================================================================
// Test 9: ReLU clamps negative values
// =============================================================================
void test_relu() {
    std::vector<int32_t> data = {-10, -1, 0, 1, 127, -128, 50};
    GoldenModel::reluINT32(data);

    ASSERT_EQ(data[0], 0);
    ASSERT_EQ(data[1], 0);
    ASSERT_EQ(data[2], 0);
    ASSERT_EQ(data[3], 1);
    ASSERT_EQ(data[4], 127);
    ASSERT_EQ(data[5], 0);
    ASSERT_EQ(data[6], 50);

    PASS("relu");
}

// =============================================================================
// Test 10: Max pooling
// =============================================================================
void test_max_pool() {
    // 1 channel, 4×4 input, pool_size=2, stride=2 → 2×2 output
    std::vector<int32_t> input = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    auto output = GoldenModel::maxPool2d(input.data(), 1, 4, 4, 2, 2);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_EQ(output[0], 6);   // max(1,2,5,6)
    ASSERT_EQ(output[1], 8);   // max(3,4,7,8)
    ASSERT_EQ(output[2], 14);  // max(9,10,13,14)
    ASSERT_EQ(output[3], 16);  // max(11,12,15,16)

    PASS("max_pool");
}

// =============================================================================
// Test 11: Argmax
// =============================================================================
void test_argmax() {
    std::vector<float> logits = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f, -1.0f};
    ASSERT_EQ(GoldenModel::argmax(logits), 3u);

    std::vector<float> logits2 = {10.0f, -5.0f, 3.0f};
    ASSERT_EQ(GoldenModel::argmax(logits2), 0u);

    PASS("argmax");
}

// =============================================================================
// Test 12: MNIST layer shapes are consistent
// =============================================================================
void test_mnist_shapes() {
    auto s1 = mnist::conv1Shape();
    ASSERT_EQ(s1.M, 32u);
    ASSERT_EQ(s1.K, 9u);

    auto s2 = mnist::conv2Shape();
    ASSERT_EQ(s2.M, 64u);
    ASSERT_EQ(s2.K, 288u);

    auto s3 = mnist::fc1Shape();
    ASSERT_EQ(s3.M, 128u);
    ASSERT_EQ(s3.K, 9216u);
    ASSERT_EQ(s3.N, 1u);

    auto s4 = mnist::fc2Shape();
    ASSERT_EQ(s4.M, 10u);
    ASSERT_EQ(s4.K, 128u);
    ASSERT_EQ(s4.N, 1u);

    // Verify all layers have plans
    auto plans = mnist::planAllLayers();
    ASSERT_EQ(plans.size(), 4u);

    // Verify total MAC counts are reasonable
    for (auto& p : plans) {
        ASSERT_TRUE(p.totalMACs() > 0);
        ASSERT_TRUE(p.totalTiles() > 0);
    }

    PASS("mnist_shapes");
}

// =============================================================================
// Test 13: Full pipeline — small GEMM through BSR + tiling + golden
// =============================================================================
void test_full_pipeline_small() {
    // 14×14 × 14×1 GEMM — single tile, simplest case
    const uint32_t M = 14, N = 1, K = 14;
    std::vector<int8_t> W(M * K), X(K * N);
    fillRandom(W.data(), W.size(), 777);
    fillRandom(X.data(), X.size(), 778);

    // 1. Golden GEMM
    auto C_golden = GoldenModel::gemmINT8(W.data(), X.data(), M, N, K);

    // 2. Tiled GEMM
    auto C_tiled = GoldenModel::tiledGemmINT8(W.data(), X.data(), M, N, K);

    // 3. BSR encode → verify density
    BSREncoder encoder;
    auto bsr = encoder.encode(W.data(), M, K);
    ASSERT_EQ(bsr.nnz_blocks, 1u);  // Single block, non-zero

    // 4. BSR decode → re-compute GEMM
    auto W_dec = encoder.decode(bsr, M, K);
    auto C_bsr = GoldenModel::gemmINT8(W_dec.data(), X.data(), M, N, K);

    // 5. Tiling plan
    auto plan = planDenseGEMM(GEMMShape{M, N, K}, "test");
    ASSERT_EQ(plan.totalTiles(), 1u);

    // All three paths must agree
    for (size_t i = 0; i < C_golden.size(); ++i) {
        ASSERT_EQ(C_golden[i], C_tiled[i]);
        ASSERT_EQ(C_golden[i], C_bsr[i]);
    }

    PASS("full_pipeline_small");
}

// =============================================================================
// Test 14: QuantParams output scale
// =============================================================================
void test_quant_params() {
    QuantParams qp;
    qp.scale_act = 0.02f;
    qp.scale_wgt = 0.01f;

    float scale = qp.outputScale();
    float expected = 0.02f * 0.01f;
    ASSERT_TRUE(std::fabs(scale - expected) < 1e-8f);

    // Per-channel override
    qp.per_channel_scales = {0.05f, 0.03f, 0.01f};
    ASSERT_TRUE(std::fabs(qp.outputScale(0) - 0.02f * 0.05f) < 1e-8f);
    ASSERT_TRUE(std::fabs(qp.outputScale(1) - 0.02f * 0.03f) < 1e-8f);
    ASSERT_TRUE(std::fabs(qp.outputScale(2) - 0.02f * 0.01f) < 1e-8f);

    PASS("quant_params");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::fprintf(stdout, "\n=== End-to-End Integration Tests ===\n\n");

    test_dense_vs_tiled_small();
    test_dense_vs_tiled_multi_tile();
    test_dense_vs_tiled_non_aligned();
    test_bsr_roundtrip_gemm();
    test_sparse_bsr_block_savings();
    test_tiling_plan_descriptors();
    test_tiled_accumulation();
    test_im2col_conv();
    test_relu();
    test_max_pool();
    test_argmax();
    test_mnist_shapes();
    test_full_pipeline_small();
    test_quant_params();

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n\n",
                 tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
