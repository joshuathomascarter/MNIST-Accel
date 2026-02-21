// test_bsr_encoder.cpp — Unit tests for BSR sparse matrix encoder
// =============================================================================
#include "compute/bsr_encoder.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace accel::compute;

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

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a); double _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "  FAIL %s:%d: %s != %s  (%f != %f)\n", \
                     __FILE__, __LINE__, #a, #b, _a, _b); \
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
// Test: Encode a fully dense 14×14 matrix → 1 block
// =============================================================================
void test_dense_single_block() {
    std::vector<int8_t> data(14 * 14, 1);  // All ones
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 14, 14);

    ASSERT_EQ(bsr.num_block_rows, 1u);
    ASSERT_EQ(bsr.num_block_cols, 1u);
    ASSERT_EQ(bsr.nnz_blocks, 1u);
    ASSERT_EQ(bsr.row_ptr.size(), 2u);
    ASSERT_EQ(bsr.row_ptr[0], 0u);
    ASSERT_EQ(bsr.row_ptr[1], 1u);
    ASSERT_EQ(bsr.col_idx.size(), 1u);
    ASSERT_EQ(bsr.col_idx[0], 0u);
    ASSERT_EQ(bsr.values.size(), 196u);
    ASSERT_NEAR(bsr.density(), 1.0f, 0.001f);

    PASS("dense_single_block");
}

// =============================================================================
// Test: Encode an all-zero matrix → 0 NNZ blocks
// =============================================================================
void test_all_zero() {
    std::vector<int8_t> data(28 * 28, 0);  // 2×2 blocks, all zero
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 28, 28);

    ASSERT_EQ(bsr.num_block_rows, 2u);
    ASSERT_EQ(bsr.num_block_cols, 2u);
    ASSERT_EQ(bsr.nnz_blocks, 0u);
    ASSERT_EQ(bsr.values.size(), 0u);
    ASSERT_NEAR(bsr.density(), 0.0f, 0.001f);

    PASS("all_zero");
}

// =============================================================================
// Test: Encode with padding (non-multiple of 14 dimensions)
// =============================================================================
void test_padding() {
    // 10×10 matrix → padded to 14×14 → 1 block row, 1 block col
    std::vector<int8_t> data(10 * 10, 42);
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 10, 10);

    ASSERT_EQ(bsr.num_block_rows, 1u);
    ASSERT_EQ(bsr.num_block_cols, 1u);
    ASSERT_EQ(bsr.nnz_blocks, 1u);

    // The block should contain the original 10×10 data + zero padding
    ASSERT_EQ(bsr.values.size(), 196u);

    // First 10 elements of first row should be 42
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(bsr.values[i], 42);
    }
    // Padding elements (cols 10–13) should be 0
    for (int i = 10; i < 14; ++i) {
        ASSERT_EQ(bsr.values[i], 0);
    }

    PASS("padding");
}

// =============================================================================
// Test: Sparse matrix (diagonal blocks only)
// =============================================================================
void test_sparse_diagonal() {
    // 28×28 matrix with only diagonal blocks non-zero
    std::vector<int8_t> data(28 * 28, 0);

    // Set block (0,0) and block (1,1) to non-zero
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j) {
            data[i * 28 + j] = 1;               // Block (0,0)
            data[(14 + i) * 28 + (14 + j)] = 2; // Block (1,1)
        }
    }

    BSREncoder encoder;
    auto bsr = encoder.encode(data, 28, 28);

    ASSERT_EQ(bsr.num_block_rows, 2u);
    ASSERT_EQ(bsr.num_block_cols, 2u);
    ASSERT_EQ(bsr.nnz_blocks, 2u);

    // row_ptr: [0, 1, 2] — 1 NZ block per row
    ASSERT_EQ(bsr.row_ptr[0], 0u);
    ASSERT_EQ(bsr.row_ptr[1], 1u);
    ASSERT_EQ(bsr.row_ptr[2], 2u);

    // col_idx: [0, 1] — block (0,0) and block (1,1)
    ASSERT_EQ(bsr.col_idx[0], 0u);
    ASSERT_EQ(bsr.col_idx[1], 1u);

    ASSERT_NEAR(bsr.density(), 0.5f, 0.001f);

    PASS("sparse_diagonal");
}

// =============================================================================
// Test: Roundtrip encode → decode
// =============================================================================
void test_roundtrip() {
    // Create a random-ish matrix
    uint32_t rows = 20, cols = 30;
    std::vector<int8_t> original(rows * cols);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<int8_t>((i * 7 + 13) % 251 - 125);
    }

    BSREncoder encoder;
    auto bsr = encoder.encode(original, rows, cols);
    auto decoded = encoder.decode(bsr, rows, cols);

    ASSERT_EQ(decoded.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        ASSERT_EQ(decoded[i], original[i]);
    }

    PASS("roundtrip");
}

// =============================================================================
// Test: MNIST conv1 dimensions (32×9 → 3 blocks)
// =============================================================================
void test_conv1_dims() {
    // 32×9 dense matrix → padded to 42×14 → 3×1 block grid
    std::vector<int8_t> data(32 * 9, 1);
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 32, 9);

    ASSERT_EQ(bsr.num_block_rows, 3u);  // 42/14
    ASSERT_EQ(bsr.num_block_cols, 1u);  // 14/14
    ASSERT_EQ(bsr.nnz_blocks, 3u);     // All blocks dense

    PASS("conv1_dims");
}

// =============================================================================
// Test: Zero threshold pruning
// =============================================================================
void test_zero_threshold() {
    // Create matrix with small values that should be pruned
    std::vector<int8_t> data(14 * 28);
    // Block (0,0): values in range [-2, 2]
    for (int i = 0; i < 14; ++i)
        for (int j = 0; j < 14; ++j)
            data[i * 28 + j] = static_cast<int8_t>((i + j) % 5 - 2);

    // Block (0,1): values large
    for (int i = 0; i < 14; ++i)
        for (int j = 14; j < 28; ++j)
            data[i * 28 + j] = 100;

    BSREncoder encoder;

    // Without threshold: both blocks non-zero
    auto bsr1 = encoder.encode(data, 14, 28);
    ASSERT_EQ(bsr1.nnz_blocks, 2u);

    // With threshold=2: first block gets pruned (all |vals| <= 2)
    encoder.setZeroThreshold(2);
    auto bsr2 = encoder.encode(data, 14, 28);
    ASSERT_EQ(bsr2.nnz_blocks, 1u);
    ASSERT_EQ(bsr2.col_idx[0], 1u);  // Only block (0,1) survives

    PASS("zero_threshold");
}

// =============================================================================
// Test: Pack for DMA
// =============================================================================
void test_pack() {
    std::vector<int8_t> data(14 * 14, 5);
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 14, 14);
    auto packed = encoder.pack(bsr);

    // Metadata = row_ptr (2 uint32 = 8 bytes) + col_idx (1 uint32 = 4 bytes) = 12
    ASSERT_EQ(packed.metadata.size(), 12u);

    // Weights = 196 bytes
    ASSERT_EQ(packed.weights.size(), 196u);

    PASS("pack");
}

// =============================================================================
// Test: BSRMatrix print
// =============================================================================
void test_print() {
    std::vector<int8_t> data(28 * 28, 3);
    BSREncoder encoder;
    auto bsr = encoder.encode(data, 28, 28);
    bsr.print(std::cout);
    PASS("print");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::fprintf(stdout, "\n=== BSR Encoder Unit Tests ===\n\n");

    test_dense_single_block();
    test_all_zero();
    test_padding();
    test_sparse_diagonal();
    test_roundtrip();
    test_conv1_dims();
    test_zero_threshold();
    test_pack();
    test_print();

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n\n",
                 tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
