// BSR (Block Sparse Row) CSR definitions for software driver
// Also includes HYBRID SCHEDULER mode selection.
#pragma once

#include <cstdint>

namespace resnet_accel {
namespace csr {

// ---------------------------------------------------------------------------
// BSR CONFIGURATION REGISTERS (base offsets chosen in unused CSR space)
// These registers are 32-bit wide and intended to match RTL CSR map.
// ---------------------------------------------------------------------------

// BSR configuration/control
constexpr uint32_t BSR_CONFIG       = 0xC0; // R/W
constexpr uint32_t BSR_NUM_BLOCKS   = 0xC4; // R/W
constexpr uint32_t BSR_BLOCK_ROWS   = 0xC8; // R/W
constexpr uint32_t BSR_BLOCK_COLS   = 0xCC; // R/W

// BSR status and diagnostics
constexpr uint32_t BSR_STATUS       = 0xD0; // R/W1C (ready/busy/done/error + processed count)
constexpr uint32_t BSR_ERROR_CODE   = 0xD4; // RO

// Addresses pointing to BSR arrays in DRAM (physical addresses; low 32-bits)
constexpr uint32_t BSR_PTR_ADDR     = 0xD8; // row_ptr array
constexpr uint32_t BSR_IDX_ADDR     = 0xDC; // col_idx array

// ---------------------------------------------------------------------------
// BSR_CONFIG BIT DEFINITIONS
// ---------------------------------------------------------------------------
// Bit 0: SCHED_MODE - Scheduler Mode Select (HYBRID SCHEDULER)
//        0 = BSR Sparse Scheduler (bsr_scheduler.sv) - for sparse layers
//        1 = Dense GEMM Scheduler (scheduler.sv) - for dense layers like FC1
//
// Bits 1-2: Reserved for future use (verify, zero-skip were in older design)
//
// The hybrid scheduler allows runtime selection between:
//   - BSR mode: Processes sparse weights in Block Sparse Row format
//   - Dense mode: Traditional tiled GEMM for fully-connected layers
//
// Usage:
//   // For sparse conv layers (e.g., pruned Conv1, Conv2):
//   write_reg(BSR_CONFIG, BSR_CONFIG_MODE_BSR);
//
//   // For dense FC layers (e.g., FC1 which is 100% dense):
//   write_reg(BSR_CONFIG, BSR_CONFIG_MODE_DENSE);
// ---------------------------------------------------------------------------

/// SCHED_MODE bit: 0=BSR scheduler, 1=Dense scheduler
constexpr uint32_t BSR_CONFIG_SCHED_MODE    = (1u << 0);

/// Convenience aliases for scheduler mode selection
constexpr uint32_t BSR_CONFIG_MODE_BSR      = 0u;           // Use BSR sparse scheduler
constexpr uint32_t BSR_CONFIG_MODE_DENSE    = (1u << 0);    // Use Dense GEMM scheduler

/// Legacy bit definitions (kept for backward compatibility)
constexpr uint32_t BSR_CONFIG_ENABLE    = (1u << 0);  // Note: Now same as SCHED_MODE
constexpr uint32_t BSR_CONFIG_VERIFY    = (1u << 1);
constexpr uint32_t BSR_CONFIG_ZERO_SKIP = (1u << 2);

// BSR_STATUS bits
constexpr uint32_t BSR_STATUS_READY     = (1u << 0);
constexpr uint32_t BSR_STATUS_BUSY      = (1u << 1);
constexpr uint32_t BSR_STATUS_DONE      = (1u << 2);
constexpr uint32_t BSR_STATUS_ERROR     = (1u << 3);

} // namespace csr
} // namespace resnet_accel
