# Signal Breakdown — `accel_top.sv` Full Wiring Reference

> Every signal, every wire, every port — origin and destination.  
> Architecture: 14×14 INT8 BSR-only Sparse Systolic Array Accelerator  
> Target: Zynq xc7z020 (PYNQ-Z2), 100 MHz

---

## Table of Contents

1. [Top-Level Ports (External I/O)](#1-top-level-ports)
2. [Module Instantiation Map](#2-module-instantiation-map)
3. [Signal Groups & Wire Table](#3-signal-groups--wire-table)
   - [3.1 Clock & Reset](#31-clock--reset)
   - [3.2 AXI4 Master (DDR HP Port)](#32-axi4-master-ddr-hp-port)
   - [3.3 AXI4-Lite Slave (CSR GP Port)](#33-axi4-lite-slave-csr-gp-port)
   - [3.4 CSR Bus (axi_lite_slave ↔ csr)](#34-csr-bus)
   - [3.5 CSR Configuration Outputs](#35-csr-configuration-outputs)
   - [3.6 DMA Control Signals](#36-dma-control-signals)
   - [3.7 AXI DMA Bridge Internal (2:1 Arbiter)](#37-axi-dma-bridge-internal)
   - [3.8 Buffer Write Interfaces (DMA → BRAM)](#38-buffer-write-interfaces)
   - [3.9 Metadata BRAM Read (Scheduler → BRAMs)](#39-metadata-bram-read)
   - [3.10 Scheduler Control](#310-scheduler-control)
   - [3.11 BSR Scheduler → Systolic Array Control](#311-bsr-scheduler-outputs)
   - [3.12 Buffer Read Data (BRAM → Systolic Array)](#312-buffer-read-data)
   - [3.13 Systolic Array Output](#313-systolic-array-output)
   - [3.14 Performance Counters](#314-performance-counters)
   - [3.15 DMA Packer Internals](#315-dma-packer-internals)
   - [3.16 Status Aggregation](#316-status-aggregation)
4. [Module Port-by-Port Connection Table](#4-module-port-by-port-connection-table)
   - [4.1 axi_lite_slave](#41-u_axi_lite_slave)
   - [4.2 csr](#42-u_csr)
   - [4.3 axi_dma_bridge](#43-u_axi_dma_bridge)
   - [4.4 act_dma](#44-u_act_dma)
   - [4.5 bsr_dma](#45-u_bsr_dma)
   - [4.6 dma_pack_112 (act packer)](#46-u_act_packer)
   - [4.7 dma_pack_112 (wgt packer)](#47-u_wgt_packer)
   - [4.8 bsr_scheduler](#48-u_bsr_scheduler)
   - [4.9 systolic_array_sparse](#49-u_systolic_sparse)
   - [4.10 perf](#410-u_perf)
5. [Inline BRAMs](#5-inline-brams)
6. [Modules NOT Instantiated](#6-modules-not-instantiated)
7. [Data Flow Summary](#7-data-flow-summary)

---

## 1. Top-Level Ports

### Clock & Reset

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `clk` | in | 1 | 100 MHz from Zynq FCLK_CLK0 |
| `rst_n` | in | 1 | Active-low sync reset from proc_sys_reset |

### AXI4 Master — Read Address Channel (to DDR via HP port)

| Port | Dir | Width | Driven By |
|------|-----|-------|-----------|
| `m_axi_arid` | out | 4 | `u_axi_dma_bridge.m_arid` |
| `m_axi_araddr` | out | 32 | `u_axi_dma_bridge.m_araddr` |
| `m_axi_arlen` | out | 8 | `u_axi_dma_bridge.m_arlen` |
| `m_axi_arsize` | out | 3 | `u_axi_dma_bridge.m_arsize` |
| `m_axi_arburst` | out | 2 | `u_axi_dma_bridge.m_arburst` |
| `m_axi_arvalid` | out | 1 | `u_axi_dma_bridge.m_arvalid` |
| `m_axi_arready` | in | 1 | → `u_axi_dma_bridge.m_arready` |

### AXI4 Master — Read Data Channel (from DDR)

| Port | Dir | Width | Consumed By |
|------|-----|-------|-------------|
| `m_axi_rid` | in | 4 | → `u_axi_dma_bridge.m_rid` |
| `m_axi_rdata` | in | 64 | → `u_axi_dma_bridge.m_rdata` |
| `m_axi_rresp` | in | 2 | → `u_axi_dma_bridge.m_rresp` |
| `m_axi_rlast` | in | 1 | → `u_axi_dma_bridge.m_rlast` |
| `m_axi_rvalid` | in | 1 | → `u_axi_dma_bridge.m_rvalid` |
| `m_axi_rready` | out | 1 | `u_axi_dma_bridge.m_rready` |

### AXI4-Lite Slave — Write Channels (from Zynq GP0)

| Port | Dir | Width | Consumed By |
|------|-----|-------|-------------|
| `s_axi_awaddr` | in | 32 | → `u_axi_lite_slave.s_axi_awaddr[7:0]` |
| `s_axi_awprot` | in | 3 | → `u_axi_lite_slave.s_axi_awprot` |
| `s_axi_awvalid` | in | 1 | → `u_axi_lite_slave.s_axi_awvalid` |
| `s_axi_awready` | out | 1 | `u_axi_lite_slave.s_axi_awready` |
| `s_axi_wdata` | in | 32 | → `u_axi_lite_slave.s_axi_wdata` |
| `s_axi_wstrb` | in | 4 | → `u_axi_lite_slave.s_axi_wstrb` |
| `s_axi_wvalid` | in | 1 | → `u_axi_lite_slave.s_axi_wvalid` |
| `s_axi_wready` | out | 1 | `u_axi_lite_slave.s_axi_wready` |
| `s_axi_bresp` | out | 2 | `u_axi_lite_slave.s_axi_bresp` |
| `s_axi_bvalid` | out | 1 | `u_axi_lite_slave.s_axi_bvalid` |
| `s_axi_bready` | in | 1 | → `u_axi_lite_slave.s_axi_bready` |

### AXI4-Lite Slave — Read Channels

| Port | Dir | Width | Consumed By |
|------|-----|-------|-------------|
| `s_axi_araddr` | in | 32 | → `u_axi_lite_slave.s_axi_araddr[7:0]` |
| `s_axi_arprot` | in | 3 | → `u_axi_lite_slave.s_axi_arprot` |
| `s_axi_arvalid` | in | 1 | → `u_axi_lite_slave.s_axi_arvalid` |
| `s_axi_arready` | out | 1 | `u_axi_lite_slave.s_axi_arready` |
| `s_axi_rdata` | out | 32 | `u_axi_lite_slave.s_axi_rdata` |
| `s_axi_rresp` | out | 2 | `u_axi_lite_slave.s_axi_rresp` |
| `s_axi_rvalid` | out | 1 | `u_axi_lite_slave.s_axi_rvalid` |
| `s_axi_rready` | in | 1 | → `u_axi_lite_slave.s_axi_rready` |

### Status Outputs

| Port | Dir | Width | Expression |
|------|-----|-------|------------|
| `busy` | out | 1 | `act_dma_busy \| bsr_dma_busy \| sched_busy` |
| `done` | out | 1 | `sched_done` |
| `error` | out | 1 | `bsr_dma_error \| act_dma_error` |

---

## 2. Module Instantiation Map

```
accel_top
├── u_axi_lite_slave     (axi_lite_slave)      — AXI4-Lite → CSR bus
├── u_csr                (csr)                  — Config/status registers
├── u_axi_dma_bridge     (axi_dma_bridge)       — 2:1 AXI read arbiter
├── u_act_dma            (act_dma)              — DDR → act buffer DMA
├── u_bsr_dma            (bsr_dma)              — DDR → BSR BRAMs DMA
├── u_act_packer         (dma_pack_112)         — 64→112 bit act packer
├── u_wgt_packer         (dma_pack_112)         — 64→112 bit wgt packer
├── [inline] row_ptr_bram                       — 1024×32-bit metadata
├── [inline] col_idx_bram                       — 1024×16-bit metadata
├── [inline] wgt_block_bram                     — 1024×112-bit weight data
├── [inline] act_buffer_ram                     — 1024×112-bit activation data
├── [inline] BRAM read mux logic                — row_ptr vs col_idx select
├── u_bsr_scheduler      (bsr_scheduler)        — BSR sparse block traversal
├── u_systolic_sparse    (systolic_array_sparse) — 14×14 PE grid
└── u_perf               (perf)                 — Cycle counters
```

---

## 3. Signal Groups & Wire Table

### 3.1 Clock & Reset

| Signal | Width | Origin | Destination(s) |
|--------|-------|--------|-----------------|
| `clk` | 1 | Top-level input | All modules |
| `rst_n` | 1 | Top-level input | All modules |

### 3.2 AXI4 Master (DDR HP Port)

All 13 signals directly connect `u_axi_dma_bridge` master ports ↔ top-level ports. See [Section 1](#1-top-level-ports).

### 3.3 AXI4-Lite Slave (CSR GP Port)

All 19 signals directly connect `u_axi_lite_slave` ↔ top-level ports. See [Section 1](#1-top-level-ports).

### 3.4 CSR Bus

| Signal | Width | Origin | Destination |
|--------|-------|--------|-------------|
| `csr_wen` | 1 | `u_axi_lite_slave.csr_wen` | `u_csr.csr_wen` |
| `csr_ren` | 1 | `u_axi_lite_slave.csr_ren` | `u_csr.csr_ren` |
| `csr_addr` | 8 | `u_axi_lite_slave.csr_addr` | `u_csr.csr_addr` |
| `csr_wdata` | 32 | `u_axi_lite_slave.csr_wdata` | `u_csr.csr_wdata` |
| `csr_rdata` | 32 | `u_csr.csr_rdata` | `u_axi_lite_slave.csr_rdata` |

### 3.5 CSR Configuration Outputs

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `start_pulse` | 1 | `u_csr.start_pulse` | `sched_start` assign, `u_perf.start_pulse` | W1P start trigger |
| `abort_pulse` | 1 | `u_csr.abort_pulse` | `u_bsr_scheduler.abort` | Emergency stop |
| `cfg_M` | 32 | `u_csr.M` | (unused in BSR-only mode) | Matrix M dimension |
| `cfg_N` | 32 | `u_csr.N` | (unused in BSR-only mode) | Matrix N dimension |
| `cfg_K` | 32 | `u_csr.K` | (unused in BSR-only mode) | Matrix K dimension |
| `cfg_act_src_addr` | 32 | `u_csr.act_dma_src_addr` | `act_dma_src_addr` → `u_act_dma.src_addr` | Act DDR address |
| `cfg_act_xfer_len` | 32 | `u_csr.act_dma_len` | `act_dma_xfer_len` → `u_act_dma.transfer_length` | Act transfer bytes |
| `cfg_act_dma_start` | 1 | `u_csr.act_dma_start_pulse` | `act_dma_start` → `u_act_dma.start` | Act DMA trigger |
| `cfg_bsr_src_addr` | 32 | `u_csr.dma_src_addr` | `bsr_dma_src_addr` → `u_bsr_dma.src_addr` | BSR DDR base |
| `cfg_bsr_num_blocks` | 32 | `u_csr.bsr_num_blocks` | `u_bsr_dma.csr_total_blocks` | NNZ block count |
| `cfg_bsr_block_rows` | 32 | `u_csr.bsr_block_rows` | `u_bsr_dma.csr_num_rows`, `sched_MT[9:0]` | Block rows (M/14) |
| `cfg_bsr_block_cols` | 32 | `u_csr.bsr_block_cols` | `u_bsr_dma.csr_num_cols`, `sched_KT[11:0]` | Block cols (K/14) |
| `cfg_bsr_ptr_addr` | 32 | `u_csr.bsr_ptr_addr` | (unused, future) | BSR row_ptr DDR addr |
| `cfg_bsr_idx_addr` | 32 | `u_csr.bsr_idx_addr` | (unused, future) | BSR col_idx DDR addr |
| `cfg_bsr_config` | 32 | `u_csr.bsr_config` | (unused, sched_mode removed) | BSR config flags |

### 3.6 DMA Control Signals

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `act_dma_start` | 1 | assign `cfg_act_dma_start` | `u_act_dma.start` | Start act transfer |
| `act_dma_done` | 1 | `u_act_dma.done` | `u_csr.dma_done_in` (AND'd) | Act DMA complete |
| `act_dma_busy` | 1 | `u_act_dma.busy` | `busy` assign, `u_csr.dma_busy_in` (OR'd) | Act DMA active |
| `act_dma_error` | 1 | `u_act_dma.error` | `error` assign | Act DMA error |
| `act_dma_src_addr` | 32 | assign `cfg_act_src_addr` | `u_act_dma.src_addr` | DDR source |
| `act_dma_xfer_len` | 32 | assign `cfg_act_xfer_len` | `u_act_dma.transfer_length` | Byte count |
| `bsr_dma_start` | 1 | `u_csr.dma_start_pulse` | `u_bsr_dma.start` | Start BSR transfer |
| `bsr_dma_done` | 1 | `u_bsr_dma.done` | `u_csr.dma_done_in` (AND'd) | BSR DMA complete |
| `bsr_dma_busy` | 1 | `u_bsr_dma.busy` | `busy` assign, `u_csr.dma_busy_in` (OR'd) | BSR DMA active |
| `bsr_dma_error` | 1 | `u_bsr_dma.error` | `error` assign | BSR DMA error |
| `bsr_dma_src_addr` | 32 | assign `cfg_bsr_src_addr` | `u_bsr_dma.src_addr` | DDR source |

### 3.7 AXI DMA Bridge Internal

#### Act DMA ↔ Bridge (Slave 1)

| Signal | Width | Origin | Destination |
|--------|-------|--------|-------------|
| `act_arid` | 4 | `u_act_dma.m_axi_arid` | `u_axi_dma_bridge.s1_arid` |
| `act_araddr` | 32 | `u_act_dma.m_axi_araddr` | `u_axi_dma_bridge.s1_araddr` |
| `act_arlen` | 8 | `u_act_dma.m_axi_arlen` | `u_axi_dma_bridge.s1_arlen` |
| `act_arsize` | 3 | `u_act_dma.m_axi_arsize` | `u_axi_dma_bridge.s1_arsize` |
| `act_arburst` | 2 | `u_act_dma.m_axi_arburst` | `u_axi_dma_bridge.s1_arburst` |
| `act_arvalid` | 1 | `u_act_dma.m_axi_arvalid` | `u_axi_dma_bridge.s1_arvalid` |
| `act_arready` | 1 | `u_axi_dma_bridge.s1_arready` | `u_act_dma.m_axi_arready` |
| `act_rid` | 4 | `u_axi_dma_bridge.s1_rid` | `u_act_dma.m_axi_rid` |
| `act_rdata` | 64 | `u_axi_dma_bridge.s1_rdata` | `u_act_dma.m_axi_rdata` |
| `act_rresp` | 2 | `u_axi_dma_bridge.s1_rresp` | `u_act_dma.m_axi_rresp` |
| `act_rlast` | 1 | `u_axi_dma_bridge.s1_rlast` | `u_act_dma.m_axi_rlast` |
| `act_rvalid` | 1 | `u_axi_dma_bridge.s1_rvalid` | `u_act_dma.m_axi_rvalid` |
| `act_rready` | 1 | `u_act_dma.m_axi_rready` | `u_axi_dma_bridge.s1_rready` |

#### BSR DMA ↔ Bridge (Slave 0, priority)

| Signal | Width | Origin | Destination |
|--------|-------|--------|-------------|
| `bsr_arid` | 4 | `u_bsr_dma.m_axi_arid` | `u_axi_dma_bridge.s0_arid` |
| `bsr_araddr` | 32 | `u_bsr_dma.m_axi_araddr` | `u_axi_dma_bridge.s0_araddr` |
| `bsr_arlen` | 8 | `u_bsr_dma.m_axi_arlen` | `u_axi_dma_bridge.s0_arlen` |
| `bsr_arsize` | 3 | `u_bsr_dma.m_axi_arsize` | `u_axi_dma_bridge.s0_arsize` |
| `bsr_arburst` | 2 | `u_bsr_dma.m_axi_arburst` | `u_axi_dma_bridge.s0_arburst` |
| `bsr_arvalid` | 1 | `u_bsr_dma.m_axi_arvalid` | `u_axi_dma_bridge.s0_arvalid` |
| `bsr_arready` | 1 | `u_axi_dma_bridge.s0_arready` | `u_bsr_dma.m_axi_arready` |
| `bsr_rid` | 4 | `u_axi_dma_bridge.s0_rid` | `u_bsr_dma.m_axi_rid` |
| `bsr_rdata` | 64 | `u_axi_dma_bridge.s0_rdata` | `u_bsr_dma.m_axi_rdata` |
| `bsr_rresp` | 2 | `u_axi_dma_bridge.s0_rresp` | `u_bsr_dma.m_axi_rresp` |
| `bsr_rlast` | 1 | `u_axi_dma_bridge.s0_rlast` | `u_bsr_dma.m_axi_rlast` |
| `bsr_rvalid` | 1 | `u_axi_dma_bridge.s0_rvalid` | `u_bsr_dma.m_axi_rvalid` |
| `bsr_rready` | 1 | `u_bsr_dma.m_axi_rready` | `u_axi_dma_bridge.s0_rready` |

### 3.8 Buffer Write Interfaces

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `act_buf_we` | 1 | `u_act_dma.act_we` | `u_act_packer.dma_we` | Act DMA write strobe |
| `act_buf_waddr` | 32 | `u_act_dma.act_addr` | (unused after packer) | Act DMA byte address |
| `act_buf_wdata` | 64 | `u_act_dma.act_wdata` | `u_act_packer.dma_wdata` | Act DMA 64-bit data |
| `row_ptr_we` | 1 | `u_bsr_dma.row_ptr_we` | `row_ptr_bram` write | Row pointer store |
| `row_ptr_waddr` | 10 | `u_bsr_dma.row_ptr_addr` | `row_ptr_bram` write addr | Row pointer index |
| `row_ptr_wdata` | 32 | `u_bsr_dma.row_ptr_wdata` | `row_ptr_bram` write data | Row pointer value |
| `col_idx_we` | 1 | `u_bsr_dma.col_idx_we` | `col_idx_bram` write | Col index store |
| `col_idx_waddr` | 10 | `u_bsr_dma.col_idx_addr` | `col_idx_bram` write addr | Col index entry |
| `col_idx_wdata` | 16 | `u_bsr_dma.col_idx_wdata` | `col_idx_bram` write data | Col block index |
| `wgt_we` | 1 | `u_bsr_dma.wgt_we` | `u_wgt_packer.dma_we` | Wgt DMA write strobe |
| `wgt_waddr` | 17 | `u_bsr_dma.wgt_addr` | (unused after packer) | Wgt DMA byte address |
| `wgt_wdata` | 64 | `u_bsr_dma.wgt_wdata` | `u_wgt_packer.dma_wdata` | Wgt DMA 64-bit data |

### 3.9 Metadata BRAM Read

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `sched_meta_addr` | 32 | `u_bsr_scheduler.meta_raddr` | `row_ptr_bram` read addr, `col_idx_bram` read addr | Metadata address |
| `sched_meta_ren` | 1 | `u_bsr_scheduler.meta_ren` | BRAM read logic (drives `meta_rvalid_r`) | Read enable |
| `sched_meta_ready` | 1 | `u_bsr_scheduler.meta_ready` | (not consumed, scheduler output) | Scheduler ready flag |
| `meta_rvalid_r` | 1 | inline register (1-cycle delay of `sched_meta_ren`) | `u_bsr_scheduler.meta_rvalid` | Read data valid |
| `meta_rdata_r` | 32 | inline mux: `meta_is_col_idx_r ? col_idx_rdata_r : row_ptr_rdata_r` | `u_bsr_scheduler.meta_rdata` | Muxed read data |
| `meta_is_col_idx_r` | 1 | inline register (latches `sched_meta_addr[8]`) | `meta_rdata_r` mux select | Address decode |
| `row_ptr_rdata_r` | 32 | `row_ptr_bram` read data (1-cycle) | `meta_rdata_r` mux input | Row pointer value |
| `col_idx_rdata_r` | 16 | `col_idx_bram` read data (1-cycle) | `meta_rdata_r` mux input (zero-extended to 32) | Col index value |

### 3.10 Scheduler Control

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `sched_start` | 1 | assign `start_pulse` | `u_bsr_scheduler.start` | Begin BSR traversal |
| `sched_busy` | 1 | `u_bsr_scheduler.busy` | `busy` assign | Scheduler active |
| `sched_done` | 1 | `u_bsr_scheduler.done` | `done` assign, `u_perf.done_pulse`, `u_csr.core_done_tile_pulse` | All blocks processed |
| `sched_MT` | 10 | assign `cfg_bsr_block_rows[9:0]` | `u_bsr_scheduler.MT` | Block row count |
| `sched_KT` | 12 | assign `cfg_bsr_block_cols[11:0]` | `u_bsr_scheduler.KT` | Block col count |

### 3.11 BSR Scheduler Outputs

These signals are driven **directly** by `u_bsr_scheduler` (no mux — dense scheduler removed).

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `load_weight` | 1 | `u_bsr_scheduler.load_weight` | `u_systolic_sparse.load_weight` | Load weights into PEs |
| `pe_en` | 1 | `u_bsr_scheduler.pe_en` | `u_systolic_sparse.block_valid` | Enable MAC operations |
| `accum_en` | 1 | `u_bsr_scheduler.accum_en` | (future: output_accumulator) | Accumulation enable |
| `pe_clr` | 1 | `u_bsr_scheduler.pe_clr` | `u_systolic_sparse.clr` | Clear accumulators |
| `wgt_rd_en` | 1 | `u_bsr_scheduler.wgt_rd_en` | (gate on `wgt_block_bram` read) | Weight read strobe |
| `wgt_rd_addr` | 32 | `u_bsr_scheduler.wgt_addr` | `wgt_block_bram` read addr `[9:0]` | Weight BRAM address |
| `act_rd_en` | 1 | `u_bsr_scheduler.act_rd_en` | `act_buffer_ram` read gate | Act read strobe |
| `act_rd_addr` | 32 | `u_bsr_scheduler.act_addr` | `act_buffer_ram` read addr `[9:0]` | Act BRAM address |

### 3.12 Buffer Read Data

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `act_rd_data` | 112 | `act_rd_data_r` (inline BRAM read reg) | `u_systolic_sparse.a_in_flat` | 14×INT8 activations |
| `wgt_rd_data` | 112 | `wgt_block_rdata_r[111:0]` | `u_systolic_sparse.b_in_flat` | 14×INT8 weights |

### 3.13 Systolic Array Output

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `systolic_out_flat` | 6272 | `u_systolic_sparse.c_out_flat` | `u_csr.result_data` (bits [127:0] only) | 196×INT32 accumulators |

### 3.14 Performance Counters

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `perf_total_cycles` | 32 | `u_perf.total_cycles_count` | `u_csr.perf_total_cycles` | Total execution cycles |
| `perf_active_cycles` | 32 | `u_perf.active_cycles_count` | `u_csr.perf_active_cycles` | MAC-active cycles |
| `perf_idle_cycles` | 32 | `u_perf.idle_cycles_count` | `u_csr.perf_idle_cycles` | Stall/idle cycles |
| `perf_done` | 1 | `u_perf.measurement_done` | (unused, internal) | Measurement end pulse |

### 3.15 DMA Packer Internals

#### Activation Packer (`u_act_packer`)

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `act_pack_we` | 1 | `u_act_packer.buf_we` | `act_buffer_ram` write enable | Packed 112-bit write |
| `act_pack_waddr` | 10 | `u_act_packer.buf_waddr` | `act_buffer_ram` write address | Auto-incrementing |
| `act_pack_wdata` | 112 | `u_act_packer.buf_wdata` | `act_buffer_ram` write data | 14×INT8 packed |

#### Weight Packer (`u_wgt_packer`)

| Signal | Width | Origin | Destination | Purpose |
|--------|-------|--------|-------------|---------|
| `wgt_pack_we` | 1 | `u_wgt_packer.buf_we` | `wgt_block_bram` write enable | Packed 112-bit write |
| `wgt_pack_waddr` | 10 | `u_wgt_packer.buf_waddr` | `wgt_block_bram` write address | Auto-incrementing |
| `wgt_pack_wdata` | 112 | `u_wgt_packer.buf_wdata` | `wgt_block_bram` write data | 14×INT8 packed |

### 3.16 Status Aggregation

| Expression | Result Signal | Width | Consumers |
|------------|---------------|-------|-----------|
| `act_dma_busy \| bsr_dma_busy \| sched_busy` | `busy` | 1 | Top-level output, `u_csr.core_busy`, `u_perf.busy_signal` |
| `sched_done` | `done` | 1 | Top-level output |
| `bsr_dma_error \| act_dma_error` | `error` | 1 | Top-level output |

---

## 4. Module Port-by-Port Connection Table

### 4.1 `u_axi_lite_slave`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.s_axi_awaddr` | in | `s_axi_awaddr[CSR_ADDR_W-1:0]` |
| `.s_axi_awprot` | in | `s_axi_awprot` |
| `.s_axi_awvalid` | in | `s_axi_awvalid` |
| `.s_axi_awready` | out | `s_axi_awready` |
| `.s_axi_wdata` | in | `s_axi_wdata` |
| `.s_axi_wstrb` | in | `s_axi_wstrb` |
| `.s_axi_wvalid` | in | `s_axi_wvalid` |
| `.s_axi_wready` | out | `s_axi_wready` |
| `.s_axi_bresp` | out | `s_axi_bresp` |
| `.s_axi_bvalid` | out | `s_axi_bvalid` |
| `.s_axi_bready` | in | `s_axi_bready` |
| `.s_axi_araddr` | in | `s_axi_araddr[CSR_ADDR_W-1:0]` |
| `.s_axi_arprot` | in | `s_axi_arprot` |
| `.s_axi_arvalid` | in | `s_axi_arvalid` |
| `.s_axi_arready` | out | `s_axi_arready` |
| `.s_axi_rdata` | out | `s_axi_rdata` |
| `.s_axi_rresp` | out | `s_axi_rresp` |
| `.s_axi_rvalid` | out | `s_axi_rvalid` |
| `.s_axi_rready` | in | `s_axi_rready` |
| `.csr_addr` | out | `csr_addr` → `u_csr` |
| `.csr_wen` | out | `csr_wen` → `u_csr` |
| `.csr_ren` | out | `csr_ren` → `u_csr` |
| `.csr_wdata` | out | `csr_wdata` → `u_csr` |
| `.csr_rdata` | in | `csr_rdata` ← `u_csr` |
| `.axi_error` | out | (unconnected) |

### 4.2 `u_csr`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.csr_wen` | in | `csr_wen` ← `u_axi_lite_slave` |
| `.csr_ren` | in | `csr_ren` ← `u_axi_lite_slave` |
| `.csr_addr` | in | `csr_addr` ← `u_axi_lite_slave` |
| `.csr_wdata` | in | `csr_wdata` ← `u_axi_lite_slave` |
| `.csr_rdata` | out | `csr_rdata` → `u_axi_lite_slave` |
| `.core_busy` | in | `busy` (aggregated status) |
| `.core_done_tile_pulse` | in | `sched_done` ← `u_bsr_scheduler` |
| `.core_bank_sel_rd_A` | in | `1'b0` (unused in BSR mode) |
| `.core_bank_sel_rd_B` | in | `1'b0` (unused in BSR mode) |
| `.rx_illegal_cmd` | in | `1'b0` (tied off) |
| `.start_pulse` | out | `start_pulse` → `sched_start`, `u_perf` |
| `.abort_pulse` | out | `abort_pulse` → `u_bsr_scheduler.abort` |
| `.irq_en` | out | (unconnected) |
| `.M` | out | `cfg_M` (unused in BSR mode) |
| `.N` | out | `cfg_N` (unused in BSR mode) |
| `.K` | out | `cfg_K` (unused in BSR mode) |
| `.Tm`, `.Tn`, `.Tk` | out | (unconnected) |
| `.m_idx`, `.n_idx`, `.k_idx` | out | (unconnected) |
| `.bank_sel_wr_A/B` | out | (unconnected) |
| `.bank_sel_rd_A/B` | out | (unconnected) |
| `.Sa_bits`, `.Sw_bits` | out | (unconnected, future quantization) |
| `.perf_total_cycles` | in | `perf_total_cycles` ← `u_perf` |
| `.perf_active_cycles` | in | `perf_active_cycles` ← `u_perf` |
| `.perf_idle_cycles` | in | `perf_idle_cycles` ← `u_perf` |
| `.perf_cache_hits` | in | `32'd0` (placeholder) |
| `.perf_cache_misses` | in | `32'd0` (placeholder) |
| `.perf_decode_count` | in | `32'd0` (placeholder) |
| `.result_data` | in | `systolic_out_flat[127:0]` (first 4 accumulators) |
| `.dma_busy_in` | in | `act_dma_busy \| bsr_dma_busy` |
| `.dma_done_in` | in | `act_dma_done & bsr_dma_done` |
| `.dma_bytes_xferred_in` | in | `32'd0` (placeholder) |
| `.dma_src_addr` | out | `cfg_bsr_src_addr` → `bsr_dma_src_addr` |
| `.dma_dst_addr` | out | (unconnected) |
| `.dma_xfer_len` | out | (unconnected) |
| `.dma_start_pulse` | out | `bsr_dma_start` → `u_bsr_dma.start` |
| `.act_dma_src_addr` | out | `cfg_act_src_addr` → `act_dma_src_addr` |
| `.act_dma_len` | out | `cfg_act_xfer_len` → `act_dma_xfer_len` |
| `.act_dma_start_pulse` | out | `cfg_act_dma_start` → `act_dma_start` |
| `.bsr_config` | out | `cfg_bsr_config` (unused, sched_mode removed) |
| `.bsr_num_blocks` | out | `cfg_bsr_num_blocks` → `u_bsr_dma.csr_total_blocks` |
| `.bsr_block_rows` | out | `cfg_bsr_block_rows` → `u_bsr_dma.csr_num_rows`, `sched_MT` |
| `.bsr_block_cols` | out | `cfg_bsr_block_cols` → `u_bsr_dma.csr_num_cols`, `sched_KT` |
| `.bsr_ptr_addr` | out | `cfg_bsr_ptr_addr` (unused) |
| `.bsr_idx_addr` | out | `cfg_bsr_idx_addr` (unused) |

### 4.3 `u_axi_dma_bridge`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.s0_ar*` | in | `bsr_ar*` ← `u_bsr_dma` |
| `.s0_r*` (outputs) | out | `bsr_r*` → `u_bsr_dma` |
| `.s0_rready` | in | `bsr_rready` ← `u_bsr_dma` |
| `.s1_ar*` | in | `act_ar*` ← `u_act_dma` |
| `.s1_r*` (outputs) | out | `act_r*` → `u_act_dma` |
| `.s1_rready` | in | `act_rready` ← `u_act_dma` |
| `.m_ar*` (outputs) | out | `m_axi_ar*` → top-level |
| `.m_r*` (inputs) | in | `m_axi_r*` ← top-level |

### 4.4 `u_act_dma`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.start` | in | `act_dma_start` ← `cfg_act_dma_start` |
| `.src_addr` | in | `act_dma_src_addr` ← `cfg_act_src_addr` |
| `.transfer_length` | in | `act_dma_xfer_len` ← `cfg_act_xfer_len` |
| `.done` | out | `act_dma_done` |
| `.busy` | out | `act_dma_busy` |
| `.error` | out | `act_dma_error` |
| `.m_axi_ar*` | out | `act_ar*` → `u_axi_dma_bridge.s1_ar*` |
| `.m_axi_r*` (inputs) | in | `act_r*` ← `u_axi_dma_bridge.s1_r*` |
| `.act_we` | out | `act_buf_we` → `u_act_packer.dma_we` |
| `.act_addr` | out | `act_buf_waddr` (unused after packer) |
| `.act_wdata` | out | `act_buf_wdata` → `u_act_packer.dma_wdata` |

### 4.5 `u_bsr_dma`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.start` | in | `bsr_dma_start` ← `u_csr.dma_start_pulse` |
| `.src_addr` | in | `bsr_dma_src_addr` ← `cfg_bsr_src_addr` |
| `.csr_num_rows` | in | `cfg_bsr_block_rows` |
| `.csr_num_cols` | in | `cfg_bsr_block_cols` |
| `.csr_total_blocks` | in | `cfg_bsr_num_blocks` |
| `.done` | out | `bsr_dma_done` |
| `.busy` | out | `bsr_dma_busy` |
| `.error` | out | `bsr_dma_error` |
| `.m_axi_ar*` | out | `bsr_ar*` → `u_axi_dma_bridge.s0_ar*` |
| `.m_axi_r*` (inputs) | in | `bsr_r*` ← `u_axi_dma_bridge.s0_r*` |
| `.row_ptr_we` | out | `row_ptr_we` → `row_ptr_bram` write |
| `.row_ptr_addr` | out | `row_ptr_waddr` → `row_ptr_bram` write addr |
| `.row_ptr_wdata` | out | `row_ptr_wdata` → `row_ptr_bram` write data |
| `.col_idx_we` | out | `col_idx_we` → `col_idx_bram` write |
| `.col_idx_addr` | out | `col_idx_waddr` → `col_idx_bram` write addr |
| `.col_idx_wdata` | out | `col_idx_wdata` → `col_idx_bram` write data |
| `.wgt_we` | out | `wgt_we` → `u_wgt_packer.dma_we` |
| `.wgt_addr` | out | `wgt_waddr` (unused after packer) |
| `.wgt_wdata` | out | `wgt_wdata` → `u_wgt_packer.dma_wdata` |

### 4.6 `u_act_packer`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.dma_we` | in | `act_buf_we` ← `u_act_dma.act_we` |
| `.dma_wdata` | in | `act_buf_wdata` ← `u_act_dma.act_wdata` |
| `.buf_we` | out | `act_pack_we` → `act_buffer_ram` write enable |
| `.buf_waddr` | out | `act_pack_waddr` → `act_buffer_ram` write address |
| `.buf_wdata` | out | `act_pack_wdata` → `act_buffer_ram` write data |

### 4.7 `u_wgt_packer`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.dma_we` | in | `wgt_we` ← `u_bsr_dma.wgt_we` |
| `.dma_wdata` | in | `wgt_wdata` ← `u_bsr_dma.wgt_wdata` |
| `.buf_we` | out | `wgt_pack_we` → `wgt_block_bram` write enable |
| `.buf_waddr` | out | `wgt_pack_waddr` → `wgt_block_bram` write address |
| `.buf_wdata` | out | `wgt_pack_wdata` → `wgt_block_bram` write data |

### 4.8 `u_bsr_scheduler`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.start` | in | `sched_start` ← `start_pulse` |
| `.abort` | in | `abort_pulse` ← `u_csr` |
| `.busy` | out | `sched_busy` → status aggregation |
| `.done` | out | `sched_done` → status, perf, CSR |
| `.MT` | in | `sched_MT` ← `cfg_bsr_block_rows[9:0]` |
| `.KT` | in | `sched_KT` ← `cfg_bsr_block_cols[11:0]` |
| `.meta_raddr` | out | `sched_meta_addr` → BRAM read mux |
| `.meta_ren` | out | `sched_meta_ren` → BRAM read logic |
| `.meta_req_ready` | in | `1'b1` (BRAM always ready) |
| `.meta_rdata` | in | `meta_rdata_r` ← BRAM read mux |
| `.meta_rvalid` | in | `meta_rvalid_r` ← 1-cycle register |
| `.meta_ready` | out | `sched_meta_ready` (unused) |
| `.wgt_rd_en` | out | `wgt_rd_en` → (BRAM read gate) |
| `.wgt_addr` | out | `wgt_rd_addr` → `wgt_block_bram` read |
| `.act_rd_en` | out | `act_rd_en` → `act_buffer_ram` read |
| `.act_addr` | out | `act_rd_addr` → `act_buffer_ram` read |
| `.load_weight` | out | `load_weight` → `u_systolic_sparse` |
| `.pe_en` | out | `pe_en` → `u_systolic_sparse.block_valid` |
| `.accum_en` | out | `accum_en` (future: output_accumulator) |
| `.pe_clr` | out | `pe_clr` → `u_systolic_sparse.clr` |

### 4.9 `u_systolic_sparse`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.block_valid` | in | `pe_en` ← `u_bsr_scheduler` |
| `.load_weight` | in | `load_weight` ← `u_bsr_scheduler` |
| `.clr` | in | `pe_clr` ← `u_bsr_scheduler` |
| `.a_in_flat` | in | `act_rd_data` (112b) ← `act_buffer_ram` |
| `.b_in_flat` | in | `wgt_rd_data` (112b) ← `wgt_block_bram` |
| `.c_out_flat` | out | `systolic_out_flat` (6272b) → `u_csr.result_data[127:0]` |

### 4.10 `u_perf`

| Port | Direction | Connected To |
|------|-----------|-------------|
| `.clk` | in | `clk` |
| `.rst_n` | in | `rst_n` |
| `.start_pulse` | in | `start_pulse` ← `u_csr` |
| `.done_pulse` | in | `sched_done` ← `u_bsr_scheduler` |
| `.busy_signal` | in | `busy` (aggregated status) |
| `.meta_cache_hits` | in | `32'd0` (placeholder) |
| `.meta_cache_misses` | in | `32'd0` (placeholder) |
| `.meta_decode_cycles` | in | `32'd0` (placeholder) |
| `.total_cycles_count` | out | `perf_total_cycles` → `u_csr` |
| `.active_cycles_count` | out | `perf_active_cycles` → `u_csr` |
| `.idle_cycles_count` | out | `perf_idle_cycles` → `u_csr` |
| `.cache_hit_count` | out | (unconnected) |
| `.cache_miss_count` | out | (unconnected) |
| `.decode_count` | out | (unconnected) |
| `.measurement_done` | out | `perf_done` (unused) |

---

## 5. Inline BRAMs

These are behavioral dual-port RAMs inferred as BRAM primitives by Vivado.

| BRAM | Width | Depth | Write Source | Read Source | BRAM36 Est. |
|------|-------|-------|-------------|-------------|-------------|
| `row_ptr_bram` | 32 | 1024 | `u_bsr_dma` (row_ptr_we/waddr/wdata) | `sched_meta_addr[9:0]` | 1× |
| `col_idx_bram` | 16 | 1024 | `u_bsr_dma` (col_idx_we/waddr/wdata) | `col_idx_rd_addr` (offset from meta) | 0.5× |
| `wgt_block_bram` | 112 | 1024 | `u_wgt_packer` (wgt_pack_we/waddr/wdata) | `wgt_rd_addr[9:0]` | 3× |
| `act_buffer_ram` | 112 | 1024 | `u_act_packer` (act_pack_we/waddr/wdata) | `act_rd_addr[9:0]` | 3× |

**Total estimated BRAM36:** ~7.5 (of 140 available on xc7z020)

---

## 6. Modules NOT Instantiated

These modules exist in the repo but are **not instantiated** in `accel_top.sv`:

| Module | File | Reason |
|--------|------|--------|
| `scheduler` (dense) | `hw/rtl/control/scheduler.sv` | Removed — BSR-only architecture |
| `act_buffer` | `hw/rtl/buffer/act_buffer.sv` | Inline BRAM used instead (no ping-pong needed) |
| `wgt_buffer` | `hw/rtl/buffer/wgt_buffer.sv` | Inline BRAM used instead (no ping-pong needed) |
| `output_accumulator` | `hw/rtl/buffer/output_accumulator.sv` | Future: add for output DMA path |
| `pulse_sync` | `hw/rtl/host_iface/pulse_sync.sv` | Single-clock domain — no CDC needed |
| `sync_2ff` / `async_fifo` | `hw/rtl/host_iface/sync_2ff.sv` | Single-clock domain — no CDC needed |
| `mac8` | `hw/rtl/mac/mac8.sv` | Instantiated inside `pe.sv`, not directly in top |
| `pe` | `hw/rtl/systolic/pe.sv` | Instantiated inside `systolic_array_sparse.sv` |

---

## 7. Data Flow Summary

```
Host CPU (ARM Cortex-A9)
   │
   ├──── AXI-Lite GP0 ──→ u_axi_lite_slave ──→ u_csr
   │                                                │
   │                                    ┌───────────┼───────────────────┐
   │                                    │           │                   │
   │                              start_pulse   bsr_dma_start   act_dma_start
   │                                    │           │                   │
   │                                    ▼           ▼                   ▼
   │                            u_bsr_scheduler  u_bsr_dma          u_act_dma
   │                                    │           │                   │
   │                                    │      ┌────┼────┐              │
   │                                    │      │    │    │              │
   │                                    │   row_ptr col_idx  wgt_we   act_we
   │                                    │    BRAM   BRAM     │         │
   │     AXI4 HP ◄──── u_axi_dma_bridge ◄── u_bsr_dma ──┘   │         │
   │                       ▲                u_act_dma ────────┘         │
   │                       │                                            │
   DDR Memory              │                u_wgt_packer          u_act_packer
                           │                     │                     │
                           │              wgt_block_bram         act_buffer_ram
                           │                     │                     │
                           │                     └────────┬────────────┘
                           │                              │
                           │                    ┌─────────▼──────────┐
                           │                    │ u_systolic_sparse  │
                           │                    │   14×14 PE Grid    │
                           │                    │  196 DSP48E1s      │
                           │                    └─────────┬──────────┘
                           │                              │
                           │                    systolic_out_flat (6272b)
                           │                              │
                           │                    u_csr.result_data[127:0]
                           │                              │
                           │                    u_perf (cycle counters)
                           │                              │
                           └──────────────────────────────┘
```

### Execution Sequence

1. **Configure:** Host writes M, N, K, DMA addresses, BSR params to CSR via AXI-Lite
2. **Load BSR:** Host triggers `bsr_dma_start` → `u_bsr_dma` reads DDR, fills `row_ptr_bram`, `col_idx_bram`, weight data through `u_wgt_packer` → `wgt_block_bram`
3. **Load Activations:** Host triggers `act_dma_start` → `u_act_dma` reads DDR, data flows through `u_act_packer` → `act_buffer_ram`
4. **Compute:** Host writes `start_pulse` → `u_bsr_scheduler` traverses BSR structure:
   - Reads `row_ptr_bram` for block boundaries
   - Reads `col_idx_bram` for column positions
   - Drives `load_weight` → loads 14×14 weight block into PEs
   - Drives `pe_en` → streams 14 activation rows through systolic array
   - Drives `accum_en` → accumulates partial products
   - Repeats for all non-zero blocks
5. **Read Results:** Host reads `result_data` from CSR (first 4 accumulators) and performance counters
6. **Done:** `sched_done` → `done` output asserted

---

*Generated from accel_top.sv — BSR-only architecture, dense scheduler removed.*
