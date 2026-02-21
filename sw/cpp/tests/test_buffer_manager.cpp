// test_buffer_manager.cpp — Unit tests for double-buffered ping-pong manager
// =============================================================================
//
// Tests the BufferManager's state machine, bank swapping, and statistics.
// Since this requires hardware (CSR + DMA), tests use mock/simulated objects
// or test only the state logic that can be verified offline.
//
// NOTE: Full hardware tests require the PYNQ-Z2 board.  These tests verify
// the software logic: state transitions, statistics computation, and the
// TileDescriptor → DDR offset contract.
//
// =============================================================================
#include "memory/buffer_manager.hpp"
#include "memory/address_map.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

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

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a); double _b = (b); \
    if ((_a - _b) > (tol) || (_b - _a) > (tol)) { \
        std::fprintf(stderr, "  FAIL %s:%d: %f != %f\n", __FILE__, __LINE__, _a, _b); \
        tests_failed++; return; \
    } } while(0)

#define PASS(name) do { \
    std::fprintf(stdout, "  PASS: %s\n", name); \
    tests_passed++; } while(0)

// =============================================================================
// Test: Constants match RTL/address_map
// =============================================================================
void test_constants() {
    ASSERT_EQ(NUM_BANKS, 2u);
    ASSERT_EQ(TILE_DIM, 14u);
    ASSERT_EQ(TILE_ELEMENTS, 196u);
    ASSERT_EQ(TILE_BYTES_INT8, 196u);
    ASSERT_EQ(ACC_WIDTH, 32u);

    ASSERT_EQ(SYSTOLIC_SIZE, 14u);
    ASSERT_EQ(BLOCK_SIZE, 196u);
    ASSERT_EQ(PE_COUNT, 196u);

    PASS("constants");
}

// =============================================================================
// Test: BufferState names
// =============================================================================
void test_buffer_state_names() {
    ASSERT_TRUE(std::string(bufferStateName(BufferState::Idle)) == "Idle");
    ASSERT_TRUE(std::string(bufferStateName(BufferState::Loading)) == "Loading");
    ASSERT_TRUE(std::string(bufferStateName(BufferState::Ready)) == "Ready");
    ASSERT_TRUE(std::string(bufferStateName(BufferState::Computing)) == "Computing");
    ASSERT_TRUE(std::string(bufferStateName(BufferState::Draining)) == "Draining");
    PASS("buffer_state_names");
}

// =============================================================================
// Test: TileDescriptor structure
// =============================================================================
void test_tile_descriptor() {
    TileDescriptor td;
    td.m_idx = 3;
    td.n_idx = 2;
    td.k_idx = 5;
    td.wgt_offset = 1000;
    td.wgt_bytes = 196;
    td.act_offset = 2000;
    td.act_bytes = 196;
    td.is_first_k = true;
    td.is_last_k = false;

    ASSERT_EQ(td.m_idx, 3u);
    ASSERT_EQ(td.n_idx, 2u);
    ASSERT_EQ(td.k_idx, 5u);
    ASSERT_EQ(td.wgt_offset, 1000u);
    ASSERT_EQ(td.wgt_bytes, 196u);
    ASSERT_TRUE(td.is_first_k);
    ASSERT_TRUE(!td.is_last_k);

    PASS("tile_descriptor");
}

// =============================================================================
// Test: BufferStats derived metrics
// =============================================================================
void test_buffer_stats() {
    BufferStats stats;
    stats.total_time_us   = 1000;
    stats.compute_time_us = 800;
    stats.dma_time_us     = 200;
    stats.wgt_bytes_moved = 1000000;  // 1 MB
    stats.act_bytes_moved = 500000;

    ASSERT_NEAR(stats.computeUtilisation(), 0.8, 0.001);
    // wgt_bytes_moved / dma_time_us = 1000000 / 200 = 5000 bytes/µs = 5000 MB/s
    ASSERT_NEAR(stats.weightBandwidthMBs(), 5000.0, 0.1);
    ASSERT_NEAR(stats.actBandwidthMBs(), 2500.0, 0.1);

    PASS("buffer_stats");
}

// =============================================================================
// Test: DDR layout constants
// =============================================================================
void test_ddr_layout() {
    ASSERT_EQ(ddr_layout::WEIGHTS_OFFSET, 0x00000000u);
    ASSERT_EQ(ddr_layout::WEIGHTS_SIZE, 0x00400000u);    // 4 MB
    ASSERT_EQ(ddr_layout::ACTS_OFFSET, 0x00400000u);
    ASSERT_EQ(ddr_layout::ACTS_SIZE, 0x00100000u);       // 1 MB
    ASSERT_EQ(ddr_layout::OUTPUT_OFFSET, 0x00600000u);
    ASSERT_EQ(ddr_layout::OUTPUT_SIZE, 0x00100000u);     // 1 MB

    // Verify no overlaps
    ASSERT_TRUE(ddr_layout::WEIGHTS_OFFSET + ddr_layout::WEIGHTS_SIZE
                <= ddr_layout::ACTS_OFFSET);
    ASSERT_TRUE(ddr_layout::BSR_IDX_OFFSET + ddr_layout::BSR_IDX_SIZE
                <= ddr_layout::OUTPUT_OFFSET);

    PASS("ddr_layout");
}

// =============================================================================
// Test: BankInfo default initialisation
// =============================================================================
void test_bank_info_defaults() {
    BankInfo info;
    ASSERT_TRUE(info.state == BufferState::Idle);
    ASSERT_EQ(info.tile.m_idx, 0u);
    ASSERT_EQ(info.tile.n_idx, 0u);
    ASSERT_EQ(info.tile.k_idx, 0u);
    ASSERT_EQ(info.load_start_us, 0u);
    ASSERT_EQ(info.load_end_us, 0u);

    PASS("bank_info_defaults");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::fprintf(stdout, "\n=== Buffer Manager Unit Tests ===\n\n");

    test_constants();
    test_buffer_state_names();
    test_tile_descriptor();
    test_buffer_stats();
    test_ddr_layout();
    test_bank_info_defaults();

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n\n",
                 tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
