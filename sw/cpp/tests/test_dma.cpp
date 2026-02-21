// test_dma.cpp — Unit tests for DMA controller (software-level)
// =============================================================================
//
// Tests the DMAController's address computations, TransferDescriptor handling,
// bank management, and statistics tracking.  Actual /dev/mem transfers are
// not tested here — those require the PYNQ-Z2 board (see test_end_to_end.cpp).
//
// =============================================================================
#include "memory/dma_controller.hpp"
#include "memory/address_map.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>

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
// Test: TransferDescriptor sizes and fields
// =============================================================================
void test_transfer_descriptor() {
    TransferDescriptor td;
    td.channel      = DMAChannel::Weight;
    td.src_addr     = 0x10000000;
    td.dst_addr     = 0;
    td.length_bytes = 196;
    td.bank         = 0;

    ASSERT_EQ(td.src_addr, 0x10000000u);
    ASSERT_EQ(td.length_bytes, 196u);
    ASSERT_EQ(td.bank, 0u);
    ASSERT_TRUE(td.channel == DMAChannel::Weight);

    td.channel = DMAChannel::Activation;
    ASSERT_TRUE(td.channel == DMAChannel::Activation);

    td.channel = DMAChannel::Output;
    ASSERT_TRUE(td.channel == DMAChannel::Output);

    PASS("transfer_descriptor");
}

// =============================================================================
// Test: TransferStats bandwidth computation
// =============================================================================
void test_transfer_stats() {
    TransferStats stats;
    stats.bytes_transferred = 1000;
    stats.elapsed_us        = 10;
    stats.bandwidth_mbps    = static_cast<double>(stats.bytes_transferred) /
                              static_cast<double>(stats.elapsed_us);
    // 1000 / 10 = 100 bytes/µs = 100 MB/s
    ASSERT_TRUE(stats.bandwidth_mbps > 99.0 && stats.bandwidth_mbps < 101.0);

    PASS("transfer_stats");
}

// =============================================================================
// Test: DMA channel enumeration
// =============================================================================
void test_dma_channels() {
    // Verify all three channels have distinct values
    ASSERT_TRUE(DMAChannel::Weight != DMAChannel::Activation);
    ASSERT_TRUE(DMAChannel::Weight != DMAChannel::Output);
    ASSERT_TRUE(DMAChannel::Activation != DMAChannel::Output);

    PASS("dma_channels");
}

// =============================================================================
// Test: DMA register offsets (verify address_map constants)
// =============================================================================
void test_dma_registers() {
    // Weight DMA CSRs
    ASSERT_EQ(REG_DMA_SRC_ADDR, 0x90u);
    ASSERT_EQ(REG_DMA_DST_ADDR, 0x94u);
    ASSERT_EQ(REG_DMA_XFER_LEN, 0x98u);
    ASSERT_EQ(REG_DMA_CTRL,     0x9Cu);

    // Activation DMA CSRs
    ASSERT_EQ(REG_ACT_DMA_SRC,  0xA0u);
    ASSERT_EQ(REG_ACT_DMA_LEN,  0xA4u);
    ASSERT_EQ(REG_ACT_DMA_CTRL, 0xA8u);

    // DMA control bits
    ASSERT_EQ(dma_ctrl::START_BIT, 1u);
    ASSERT_EQ(dma_ctrl::BUSY_BIT,  2u);
    ASSERT_EQ(dma_ctrl::DONE_BIT,  4u);

    PASS("dma_registers");
}

// =============================================================================
// Test: DDR layout addresses don't overlap
// =============================================================================
void test_ddr_regions_no_overlap() {
    // Weights: 0x000000..0x3FFFFF (4MB)
    uint32_t w_end = ddr_layout::WEIGHTS_OFFSET + ddr_layout::WEIGHTS_SIZE;
    ASSERT_EQ(w_end, 0x00400000u);

    // Activations: 0x400000..0x4FFFFF (1MB)
    uint32_t a_end = ddr_layout::ACTS_OFFSET + ddr_layout::ACTS_SIZE;
    ASSERT_EQ(a_end, 0x00500000u);

    // BSR ptrs: 0x500000..0x50FFFF (64KB)
    uint32_t bp_end = ddr_layout::BSR_PTR_OFFSET + ddr_layout::BSR_PTR_SIZE;
    // BSR idx: 0x510000..0x51FFFF (64KB)
    uint32_t bi_end = ddr_layout::BSR_IDX_OFFSET + ddr_layout::BSR_IDX_SIZE;

    // Output: 0x600000..0x6FFFFF (1MB)
    uint32_t o_end = ddr_layout::OUTPUT_OFFSET + ddr_layout::OUTPUT_SIZE;
    ASSERT_EQ(o_end, 0x00700000u);

    // Verify ordering
    ASSERT_TRUE(w_end <= ddr_layout::ACTS_OFFSET);
    ASSERT_TRUE(a_end <= ddr_layout::BSR_PTR_OFFSET);
    ASSERT_TRUE(bp_end <= ddr_layout::BSR_IDX_OFFSET);
    ASSERT_TRUE(bi_end <= ddr_layout::OUTPUT_OFFSET);

    PASS("ddr_regions_no_overlap");
}

// =============================================================================
// Test: Transfer descriptor for a typical weight tile
// =============================================================================
void test_weight_tile_descriptor() {
    // A single 14×14 INT8 weight tile is 196 bytes
    TransferDescriptor td;
    td.channel      = DMAChannel::Weight;
    td.src_addr     = ddr_layout::WEIGHTS_OFFSET;
    td.dst_addr     = 0;  // BRAM address set by hardware
    td.length_bytes = 14 * 14 * 1;  // INT8
    td.bank         = 0;

    ASSERT_EQ(td.length_bytes, 196u);

    // Second tile starts 196 bytes later
    TransferDescriptor td2;
    td2.channel      = DMAChannel::Weight;
    td2.src_addr     = ddr_layout::WEIGHTS_OFFSET + 196;
    td2.length_bytes = 196;
    td2.bank         = 1;  // ping-pong to other bank

    ASSERT_EQ(td2.src_addr, 196u);
    ASSERT_EQ(td2.bank, 1u);

    PASS("weight_tile_descriptor");
}

// =============================================================================
// Test: Activation tile descriptor
// =============================================================================
void test_activation_tile_descriptor() {
    TransferDescriptor td;
    td.channel      = DMAChannel::Activation;
    td.src_addr     = ddr_layout::ACTS_OFFSET;
    td.length_bytes = 14 * 14;  // 196 bytes
    td.bank         = 0;

    ASSERT_EQ(td.src_addr, 0x00400000u);
    ASSERT_EQ(td.length_bytes, 196u);

    PASS("activation_tile_descriptor");
}

// =============================================================================
// Test: Output tile descriptor
// =============================================================================
void test_output_tile_descriptor() {
    // Output tiles use INT32 accumulators: 14×14×4 = 784 bytes
    TransferDescriptor td;
    td.channel      = DMAChannel::Output;
    td.src_addr     = 0;  // from accumulator
    td.dst_addr     = ddr_layout::OUTPUT_OFFSET;
    td.length_bytes = 14 * 14 * 4;  // INT32
    td.bank         = 0;

    ASSERT_EQ(td.length_bytes, 784u);
    ASSERT_EQ(td.dst_addr, 0x00600000u);

    PASS("output_tile_descriptor");
}

// =============================================================================
// Test: DMAException
// =============================================================================
void test_dma_exception() {
    bool caught = false;
    try {
        throw DMAException("test DMA error");
    } catch (const DMAException& e) {
        caught = true;
        ASSERT_TRUE(std::string(e.what()) == "test DMA error");
    }
    ASSERT_TRUE(caught);

    PASS("dma_exception");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::fprintf(stdout, "\n=== DMA Controller Unit Tests ===\n\n");

    test_transfer_descriptor();
    test_transfer_stats();
    test_dma_channels();
    test_dma_registers();
    test_ddr_regions_no_overlap();
    test_weight_tile_descriptor();
    test_activation_tile_descriptor();
    test_output_tile_descriptor();
    test_dma_exception();

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n\n",
                 tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
