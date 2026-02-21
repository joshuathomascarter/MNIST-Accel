# ACCEL-v1 System Architecture

## Overview

ACCEL-v1 is a sparse CNN inference accelerator targeting the Xilinx Zynq-7020 (PYNQ-Z2). The design implements:

- **14×14 weight-stationary systolic array** (196 INT8 MACs per cycle)
- **BSR (Block Sparse Row) hardware scheduler** that skips zero-weight blocks
- **AXI4 DMA** for weight/activation data movement
- **AXI4-Lite CSR** interface for host control
- **Dual-clock architecture** — 50 MHz control, 200 MHz datapath
- **INT8 quantization** with per-channel scaling and saturation

---

## Dataflow

### Dense Path

Standard weight-stationary systolic GEMM. Weights are loaded once per tile into all PEs, activations stream through, and partial sums accumulate locally.

### Sparse Path (BSR)

The BSR scheduler reads metadata (row pointers + column indices) from on-chip BRAM to identify non-zero weight blocks. Only non-zero blocks are loaded into the systolic array.

```
Host (PS)
    │
    │  AXI4 DMA: BSR metadata + INT8 weight blocks
    ▼
┌──────────────────┐
│   BSR DMA        │  Loads row_ptr, col_idx, weight blocks
│   (bsr_dma.sv)   │  into on-chip BRAM
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  BSR Scheduler   │  FSM reads metadata, identifies non-zero blocks,
│  (bsr_scheduler) │  generates load/compute/drain commands
└──────┬───────────┘
       │
       ├─── Weight data ──▶ Weight Buffer (BRAM) ──▶ Systolic Array
       │
       └─── Activation addr ──▶ Act Buffer (BRAM) ──▶ Systolic Array
                                                          │
                                                          ▼
                                                    Output Accumulator
                                                    (INT32 → INT8 requantize)
                                                          │
                                                          ▼
                                                    Output DMA → DDR
```

### BSR Format

Block Sparse Row encodes only non-zero 14×14 blocks:

| Field | Content |
|-------|---------|
| `row_ptr[num_block_rows + 1]` | Cumulative count of non-zero blocks per row |
| `col_idx[nnz_blocks]` | Column index for each non-zero block |
| `data[nnz_blocks × 196]` | Flattened 14×14 INT8 weight blocks |

At 70% block sparsity, BSR stores only 30% of blocks, yielding ~3.3× compute and memory savings.

---

## Processing Element

Each PE holds one stationary weight and performs INT8×INT8→INT32 MAC:

```
    activation_in (INT8) ──▶ REG ──┬──▶ activation_out (to PE below)
                                   │
                                   ▼
    weight_reg (INT8, stationary)  × ──▶ + ──▶ acc_reg (INT32) ──▶ psum_out
                                        ↑
                                   psum_in (from PE left)
```

- **Weight loading**: Captured on `load_weight` assertion, held stationary
- **Compute**: `acc += weight_reg × activation_in` each cycle
- **Zero bypass**: MAC gated when either operand is zero (power savings)
- **Operand isolation**: Clock gating on inactive PEs

---

## Systolic Array (14×14)

196 PEs arranged in a 14-row × 14-column grid. Weight-stationary dataflow:

1. **Load phase** (14 cycles) — Weights loaded systolically, one row per cycle
2. **Compute phase** (K + 13 cycles) — Activations stream with row skew
3. **Drain phase** — Partial sums read from right edge of each row

Parameters:
- `N_ROWS = 14`, `N_COLS = 14` (matches Z7020 DSP budget: 196 of 220)
- `BLOCK_SIZE = 14` (BSR block dimensions match array size)

---

## Memory Subsystem

### Buffers

| Buffer | Size | Width | Latency | Notes |
|--------|------|-------|---------|-------|
| Activation buffer | 1 KB | 112-bit (14 × INT8) | 2 cycles | Double-buffered (ping-pong) |
| Weight buffer | 1 KB | 112-bit (14 × INT8) | 2 cycles | Single-port BRAM |
| Output accumulator | 196 × 32-bit | 32-bit | 1 cycle | Per-PE INT32 accumulation |
| BSR metadata BRAM | 256 entries | 32-bit | 2 cycles | Stores row_ptr + col_idx |

### DMA Engines

| Engine | Direction | Width | Function |
|--------|-----------|-------|----------|
| `act_dma.sv` | DDR → Act Buffer | 64-bit AXI4 | Burst activation loading |
| `bsr_dma.sv` | DDR → Wgt Buffer + Meta BRAM | 64-bit AXI4 | BSR metadata + weight loading |
| `out_dma.sv` | Accum → DDR | 64-bit AXI4 | Result write-back |

---

## Control

### CSR Register Map (AXI4-Lite)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | RW | Start, reset, IRQ enable |
| 0x04 | STATUS | RO | Busy, done, error flags |
| 0x08–0x1C | Addr Config | RW | DDR base addresses (BSR, act, output) |
| 0x20 | TILE_CFG | RW | M, N, K dimensions |
| 0x2C–0x30 | Perf Counters | RO | Total cycles, stall cycles |
| 0x80+ | BSR Config | RW | Scheduler mode, BSR base addr, block count |
| 0x90+ | DMA Config | RW | DMA source/dest addresses, control |

### BSR Scheduler FSM

```
IDLE → FETCH_ROW → FETCH_COL → LOAD_WGT → WAIT_WGT → STREAM_ACT → DRAIN → NEXT_BLOCK
  │                                                                            │
  └────────────────────── (all blocks processed) ─────────────────── DONE ◄────┘
```

States:
- **FETCH_ROW**: Read `row_ptr[r]` and `row_ptr[r+1]` from metadata BRAM
- **FETCH_COL**: Read `col_idx[blk]` for current block
- **LOAD_WGT**: Load 14×14 weight block into systolic array (14 cycles)
- **WAIT_WGT**: Allow weight propagation through systolic chain
- **STREAM_ACT**: Stream activation tile through array (14 + 13 cycles)
- **DRAIN**: Collect partial sums from array edge

### Dual-Clock Architecture

| Domain | Frequency | Modules |
|--------|-----------|---------|
| Control (slow) | 50 MHz | CSR, BSR scheduler, DMA control FSMs |
| Datapath (fast) | 200 MHz | Systolic array, MAC units, buffers |

CDC handled by `pulse_sync.sv` (toggle-based) and `sync_2ff.sv` (2-FF synchronizer).

---

## Host Interfaces

### AXI4-Lite (Control)
- 32-bit data, word-aligned registers
- Read/write latency: 1 cycle per register
- Used for: CSR configuration, status polling, performance counter reads

### AXI4 (Data)
- 64-bit data, burst transfers up to 256 bytes
- Connected to Zynq PS HP port for DDR access
- Used for: weight/activation DMA, output write-back

---

## Quantization Pipeline

1. INT8 activations × INT8 weights → INT32 partial sums (in systolic array)
2. INT32 accumulation across K-dimension tiles
3. Requantization: multiply by Q16.16 per-channel scale factor
4. ReLU activation (clamp negative to zero)
5. Saturation: clamp to [-128, +127] range
6. Output: INT8 result

---

## Target Resources (XC7Z020)

| Resource | Estimated | Available | Utilization |
|----------|-----------|-----------|-------------|
| LUTs | ~18,000 | 53,200 | 34% |
| FFs | ~12,000 | 106,400 | 11% |
| BRAM (36 Kb) | 64 | 140 | 46% |
| DSP48E1 | 196 | 220 | 89% |

---

## Module Hierarchy

```
accel_top.sv
├── axi_lite_slave.sv          (CSR interface)
├── csr.sv                     (register file)
├── bsr_scheduler.sv           (sparse block FSM)
├── act_dma.sv                 (activation DMA)
├── bsr_dma.sv                 (BSR weight/metadata DMA)
├── out_dma.sv                 (output DMA)
├── axi_dma_bridge.sv          (AXI arbiter)
├── act_buffer.sv              (activation BRAM)
├── wgt_buffer.sv              (weight BRAM)
├── systolic_array_sparse.sv   (14×14 PE array)
│   └── pe.sv × 196            (processing elements)
│       └── mac8.sv × 196      (INT8 MAC units)
├── output_accumulator.sv      (INT32 accum + requantize)
├── perf.sv                    (performance counters)
├── pulse_sync.sv              (CDC)
└── sync_2ff.sv                (CDC)
```