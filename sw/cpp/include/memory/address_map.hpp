// address_map.hpp - Hardware address definitions for CSR and memory regions
// Auto-generated from hw/rtl/control/csr.sv
#pragma once

#include <cstdint>

namespace accel {
namespace memory {

// =============================================================================
// Base Addresses (Pynq-Z2 PL Region)
// =============================================================================
constexpr uint32_t ACCEL_BASE_ADDR     = 0x43C00000;  // AXI-Lite slave base
constexpr uint32_t DDR_BASE_ADDR       = 0x00000000;  // DDR physical base (PS side)
constexpr uint32_t DDR_HIGH_ADDR       = 0x1FFFFFFF;  // 512MB DDR
constexpr uint32_t PL_DDR_OFFSET       = 0x10000000;  // Reserved region for accelerator data

// =============================================================================
// CSR Register Offsets (match RTL csr.sv)
// =============================================================================

// Control Register [2]=irq_en (RW), [1]=abort (W1P), [0]=start (W1P)
constexpr uint32_t REG_CTRL            = 0x00;

// Matrix Dimensions (M×K × K×N → M×N)
constexpr uint32_t REG_DIMS_M          = 0x04;
constexpr uint32_t REG_DIMS_N          = 0x08;
constexpr uint32_t REG_DIMS_K          = 0x0C;

// Tile Dimensions (for tiled execution)
constexpr uint32_t REG_TILES_TM        = 0x10;
constexpr uint32_t REG_TILES_TN        = 0x14;
constexpr uint32_t REG_TILES_TK        = 0x18;

// Current Tile Indices
constexpr uint32_t REG_INDEX_M         = 0x1C;
constexpr uint32_t REG_INDEX_N         = 0x20;
constexpr uint32_t REG_INDEX_K         = 0x24;

// Buffer Control [0]=wrA (RW), [1]=wrB (RW), [8]=rdA (RO), [9]=rdB (RO)
constexpr uint32_t REG_BUFF            = 0x28;

// Quantization Scales (float32 bit patterns)
constexpr uint32_t REG_SCALE_SA        = 0x2C;  // Activation scale
constexpr uint32_t REG_SCALE_SW        = 0x30;  // Weight scale

// Status [0]=busy(RO), [1]=done_tile(R/W1C), [9]=err_illegal(R/W1C)
constexpr uint32_t REG_STATUS          = 0x3C;

// =============================================================================
// Performance Monitor Registers (Read-Only)
// =============================================================================
constexpr uint32_t REG_PERF_TOTAL      = 0x40;  // Total cycles from start to done
constexpr uint32_t REG_PERF_ACTIVE     = 0x44;  // Cycles where busy was high
constexpr uint32_t REG_PERF_IDLE       = 0x48;  // Cycles where busy was low
constexpr uint32_t REG_PERF_DMA_BYTES  = 0x4C;  // Total DMA bytes transferred
constexpr uint32_t REG_PERF_BLOCKS     = 0x50;  // Non-zero BSR blocks computed
constexpr uint32_t REG_PERF_STALL      = 0x54;  // Scheduler busy, PEs idle

// =============================================================================
// Result Registers (Read-Only, captured on done)
// =============================================================================
constexpr uint32_t REG_RESULT_0        = 0x80;
constexpr uint32_t REG_RESULT_1        = 0x84;
constexpr uint32_t REG_RESULT_2        = 0x88;
constexpr uint32_t REG_RESULT_3        = 0x8C;

// =============================================================================
// Weight DMA Control Registers (BSR weights)
// =============================================================================
constexpr uint32_t REG_DMA_SRC_ADDR    = 0x90;  // Source address in DDR
constexpr uint32_t REG_DMA_DST_ADDR    = 0x94;  // Destination (buffer select in MSBs)
constexpr uint32_t REG_DMA_XFER_LEN    = 0x98;  // Transfer length in bytes
constexpr uint32_t REG_DMA_CTRL        = 0x9C;  // [0]=start(W1P), [1]=busy(RO), [2]=done(R/W1C)

// =============================================================================
// Activation DMA Control Registers
// =============================================================================
constexpr uint32_t REG_ACT_DMA_SRC     = 0xA0;  // Activation source address
constexpr uint32_t REG_ACT_DMA_LEN     = 0xA4;  // Activation transfer length
constexpr uint32_t REG_ACT_DMA_CTRL    = 0xA8;  // [0]=start(W1P), [2]=done(R/W1C)

constexpr uint32_t REG_DMA_BYTES_XFER  = 0xB8;  // Bytes transferred (RO)

// =============================================================================
// BSR (Block Sparse Row) Control Registers
// =============================================================================
constexpr uint32_t REG_BSR_CONFIG      = 0xC0;  // Enable/flags/version
constexpr uint32_t REG_BSR_NUM_BLOCKS  = 0xC4;  // Total non-zero blocks
constexpr uint32_t REG_BSR_BLOCK_ROWS  = 0xC8;  // Block rows (M/14)
constexpr uint32_t REG_BSR_BLOCK_COLS  = 0xCC;  // Block cols (K/14)
constexpr uint32_t REG_BSR_STATUS      = 0xD0;  // Ready/busy/done/error + processed
constexpr uint32_t REG_BSR_ERROR       = 0xD4;  // Error detail code (RO)
constexpr uint32_t REG_BSR_PTR_ADDR    = 0xD8;  // row_ptr DDR address
constexpr uint32_t REG_BSR_IDX_ADDR    = 0xDC;  // col_idx DDR address

// =============================================================================
// Control Register Bit Definitions
// =============================================================================
namespace ctrl {
    constexpr uint32_t START_BIT       = (1 << 0);  // W1P: Start execution
    constexpr uint32_t ABORT_BIT       = (1 << 1);  // W1P: Abort current op
    constexpr uint32_t IRQ_EN_BIT      = (1 << 2);  // RW: Enable interrupt
}

namespace status {
    constexpr uint32_t BUSY_BIT        = (1 << 0);  // RO: Core is busy
    constexpr uint32_t DONE_TILE_BIT   = (1 << 1);  // R/W1C: Tile complete
    constexpr uint32_t ERR_ILLEGAL_BIT = (1 << 9);  // R/W1C: Illegal command
}

namespace dma_ctrl {
    constexpr uint32_t START_BIT       = (1 << 0);  // W1P: Start DMA
    constexpr uint32_t BUSY_BIT        = (1 << 1);  // RO: DMA busy
    constexpr uint32_t DONE_BIT        = (1 << 2);  // R/W1C: DMA done
}

namespace bsr_status {
    constexpr uint32_t READY_BIT       = (1 << 0);  // RO: Ready for next block
    constexpr uint32_t BUSY_BIT        = (1 << 1);  // RO: Processing
    constexpr uint32_t DONE_BIT        = (1 << 2);  // R/W1C: All blocks done
    constexpr uint32_t ERROR_BIT       = (1 << 3);  // R/W1C: Error occurred
}

// =============================================================================
// Hardware Constants (14×14 Systolic Array)
// =============================================================================
constexpr uint32_t SYSTOLIC_SIZE       = 14;        // PE array dimension
constexpr uint32_t BLOCK_SIZE          = SYSTOLIC_SIZE * SYSTOLIC_SIZE;  // 196 elements
constexpr uint32_t PE_COUNT            = BLOCK_SIZE;
constexpr uint32_t WEIGHT_BITS         = 8;         // INT8 weights
constexpr uint32_t ACC_BITS            = 32;        // Accumulator width

// Buffer sizes (from RTL parameters)
constexpr uint32_t ACT_BUFFER_DEPTH    = 1024;      // Activation buffer entries
constexpr uint32_t WGT_BUFFER_DEPTH    = 4096;      // Weight buffer entries
constexpr uint32_t OUT_BUFFER_DEPTH    = 1024;      // Output buffer entries

// =============================================================================
// Memory Layout for Accelerator Data in DDR
// =============================================================================
namespace ddr_layout {
    constexpr uint32_t WEIGHTS_OFFSET  = 0x00000000;  // BSR weights start
    constexpr uint32_t WEIGHTS_SIZE    = 0x00400000;  // 4MB for weights
    constexpr uint32_t ACTS_OFFSET     = 0x00400000;  // Activations start
    constexpr uint32_t ACTS_SIZE       = 0x00100000;  // 1MB for activations
    constexpr uint32_t BSR_PTR_OFFSET  = 0x00500000;  // row_ptr array
    constexpr uint32_t BSR_PTR_SIZE    = 0x00010000;  // 64KB
    constexpr uint32_t BSR_IDX_OFFSET  = 0x00510000;  // col_idx array
    constexpr uint32_t BSR_IDX_SIZE    = 0x00010000;  // 64KB
    constexpr uint32_t OUTPUT_OFFSET   = 0x00600000;  // Output results
    constexpr uint32_t OUTPUT_SIZE     = 0x00100000;  // 1MB for outputs
}

} // namespace memory
} // namespace accel
