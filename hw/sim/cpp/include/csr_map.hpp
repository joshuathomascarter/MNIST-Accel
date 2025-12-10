/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                            CSR_MAP.HPP                                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Control and Status Register (CSR) address map definitions.              ║
 * ║  MUST MATCH: hw/rtl/control/csr.sv                                       ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  Register Map:                                                            ║
 * ║    0x00 - 0x3F: Control & Configuration                                  ║
 * ║    0x40 - 0x7F: Status & Performance                                     ║
 * ║    0x80 - 0xBF: DMA Configuration                                        ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef CSR_MAP_HPP
#define CSR_MAP_HPP

#include <cstdint>

namespace resnet_accel {
namespace csr {

// =============================================================================
// Register Offsets (must match hw/rtl/control/csr.sv)
// =============================================================================

// Control & Configuration (0x00 - 0x3F)
constexpr uint32_t CTRL         = 0x00;  // [2]=irq_en, [1]=abort(W1P), [0]=start(W1P)
constexpr uint32_t DIMS_M       = 0x04;  // Matrix dimension M (rows)
constexpr uint32_t DIMS_N       = 0x08;  // Matrix dimension N (cols)
constexpr uint32_t DIMS_K       = 0x0C;  // Matrix dimension K (inner)
constexpr uint32_t TILES_Tm     = 0x10;  // Tile size M
constexpr uint32_t TILES_Tn     = 0x14;  // Tile size N
constexpr uint32_t TILES_Tk     = 0x18;  // Tile size K
constexpr uint32_t INDEX_m      = 0x1C;  // Current tile index M
constexpr uint32_t INDEX_n      = 0x20;  // Current tile index N
constexpr uint32_t INDEX_k      = 0x24;  // Current tile index K
constexpr uint32_t BUFF         = 0x28;  // [0]=wrA, [1]=wrB, [8]=rdA(RO), [9]=rdB(RO)
constexpr uint32_t SCALE_Sa     = 0x2C;  // Activation scale (float32 bits)
constexpr uint32_t SCALE_Sw     = 0x30;  // Weight scale (float32 bits)

// Status (0x3C)
constexpr uint32_t STATUS       = 0x3C;  // [0]=busy(RO), [1]=done_tile(R/W1C), [9]=err_illegal(R/W1C)

// Performance Counters (0x40 - 0x5F) - Read Only
constexpr uint32_t PERF_TOTAL       = 0x40;  // Total cycles from start to done
constexpr uint32_t PERF_ACTIVE      = 0x44;  // Cycles where busy was high
constexpr uint32_t PERF_IDLE        = 0x48;  // Cycles where busy was low
constexpr uint32_t PERF_CACHE_HITS  = 0x4C;  // Metadata cache hits
constexpr uint32_t PERF_CACHE_MISSES = 0x50; // Metadata cache misses
constexpr uint32_t PERF_DECODE_COUNT = 0x54; // Metadata decode operations

// Result Registers (0x80 - 0x8F) - Read Only
constexpr uint32_t RESULT_0     = 0x80;  // c_out[0]
constexpr uint32_t RESULT_1     = 0x84;  // c_out[1]
constexpr uint32_t RESULT_2     = 0x88;  // c_out[2]
constexpr uint32_t RESULT_3     = 0x8C;  // c_out[3]

// BSR DMA Control (0x90 - 0xBF)
constexpr uint32_t DMA_SRC_ADDR     = 0x90;  // Weight source address in DDR
constexpr uint32_t DMA_DST_ADDR     = 0x94;  // Destination address (buffer select)
constexpr uint32_t DMA_XFER_LEN     = 0x98;  // Transfer length in bytes
constexpr uint32_t DMA_CTRL         = 0x9C;  // [0]=start(W1P), [1]=busy(RO), [2]=done(R/W1C)
constexpr uint32_t DMA_BYTES_XFERRED = 0xB8; // Bytes transferred (RO)

// Activation DMA Control (0xA0 - 0xAF)
constexpr uint32_t ACT_DMA_SRC_ADDR = 0xA0;  // Activation source address in DDR
constexpr uint32_t ACT_DMA_LEN      = 0xA4;  // Activation transfer length in bytes
constexpr uint32_t ACT_DMA_CTRL     = 0xA8;  // [0]=start(W1P), [1]=busy(RO), [2]=done(R/W1C)

// =============================================================================
// Control Register Bits (CTRL @ 0x00)
// =============================================================================
constexpr uint32_t CTRL_START       = (1 << 0);  // Start computation (W1P)
constexpr uint32_t CTRL_ABORT       = (1 << 1);  // Abort computation (W1P)
constexpr uint32_t CTRL_IRQ_EN      = (1 << 2);  // Interrupt enable

// =============================================================================
// Status Register Bits (STATUS @ 0x3C)
// =============================================================================
constexpr uint32_t STATUS_BUSY          = (1 << 0);  // Core is busy (RO)
constexpr uint32_t STATUS_DONE_TILE     = (1 << 1);  // Tile done (R/W1C)
constexpr uint32_t STATUS_ERR_ILLEGAL   = (1 << 9);  // Illegal command error (R/W1C)

// =============================================================================
// DMA Control Register Bits (DMA_CTRL @ 0x9C, ACT_DMA_CTRL @ 0xA8)
// =============================================================================
constexpr uint32_t DMA_CTRL_START   = (1 << 0);  // Start DMA (W1P)
constexpr uint32_t DMA_CTRL_BUSY    = (1 << 1);  // DMA busy (RO)
constexpr uint32_t DMA_CTRL_DONE    = (1 << 2);  // DMA done (R/W1C)

// =============================================================================
// Buffer Select Register Bits (BUFF @ 0x28)
// =============================================================================
constexpr uint32_t BUFF_WR_A        = (1 << 0);  // Write bank A select
constexpr uint32_t BUFF_WR_B        = (1 << 1);  // Write bank B select
constexpr uint32_t BUFF_RD_A        = (1 << 8);  // Read bank A select (RO)
constexpr uint32_t BUFF_RD_B        = (1 << 9);  // Read bank B select (RO)

// =============================================================================
// Hardware Constants
// =============================================================================
constexpr size_t SYSTOLIC_ROWS      = 16;
constexpr size_t SYSTOLIC_COLS      = 16;
constexpr size_t BLOCK_SIZE         = 16;    // BSR block size
constexpr size_t BLOCK_ELEMENTS     = BLOCK_SIZE * BLOCK_SIZE;  // 256

// =============================================================================
// Zynq-7020 Memory Map
// =============================================================================
constexpr uint64_t ACCEL_BASE_ADDR      = 0x43C00000;  // AXI-Lite CSR base
constexpr uint64_t DDR_BASE_ADDR        = 0x00000000;
constexpr uint64_t DDR_SIZE             = 0x40000000;  // 1GB

// Reserved regions for accelerator buffers
constexpr uint64_t ACT_BUFFER_BASE      = 0x10000000;  // 64MB for activations
constexpr uint64_t WGT_BUFFER_BASE      = 0x14000000;  // 64MB for weights
constexpr uint64_t OUT_BUFFER_BASE      = 0x18000000;  // 64MB for outputs
constexpr uint64_t BSR_BUFFER_BASE      = 0x1C000000;  // 64MB for BSR metadata
constexpr size_t   BUFFER_REGION_SIZE   = 0x04000000;  // 64MB each

} // namespace csr
} // namespace resnet_accel

#endif // CSR_MAP_HPP
