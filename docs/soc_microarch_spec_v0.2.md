# SoC Micro-Architecture Specification v0.2

**Project:** MNIST ACCEL-v1 — Zynq-7020 FPGA SoC  
**Date:** 2025-01-28  
**Status:** Draft — Month 1 Complete  
**Changes from v0.1:** Added DRAM controller subsystem (Section 6), updated
resource estimates, added power management section, HFT deterministic path.

---

## 1. Overview

A single-bus SoC integrating a RISC-V CPU core, AXI interconnect, memory subsystem,
DRAM controller, peripheral set, and a 16×16 INT8 sparse systolic-array accelerator.
Target FPGA: Xilinx Zynq-7020 (XC7Z020CLG484-1).

```
┌────────────────────────────────────────────────────────────────┐
│                          soc_top                               │
│                                                                │
│  ┌──────────┐     ┌──────────────────────────────┐             │
│  │ simple_  │ OBI │          obi_to_axi          │             │
│  │   cpu    │────▶│           bridge              │             │
│  └──────────┘     └───────────┬──────────────────┘             │
│                               │ AXI4-Lite M0                   │
│                   ┌───────────▼──────────────────┐             │
│                   │     axi_crossbar             │             │
│                   │   2 Masters × 8 Slaves       │             │
│                   └──┬──┬──┬──┬──┬──┬──┬──┬─────┘             │
│                   S0 S1 S2 S3 S4 S5 S6 S7                     │
│                   │  │  │  │  │                                │
│           ┌───────┘  │  │  │  └─ (DRAM ctrl)                   │
│           │     ┌────┘  │  │                                   │
│           ▼     ▼       ▼  ▼                                   │
│       boot_rom sram  periph accel_top                          │
│       (8 KB)  (32KB) _mux  (16×16 SA)                         │
│                      ┌──┬──┬──┐                                │
│                      │  │  │  │                                │
│                   UART TIM GPIO PLIC                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DRAM Controller (dram_ctrl_top)             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐   │   │
│  │  │Cmd Queue│ │Addr Dec │ │ Write Buf│ │ Refresh    │   │   │
│  │  │(16-deep)│ │RBC/BRC  │ │ (16-deep)│ │ Controller │   │   │
│  │  └────┬────┘ └─────────┘ └────┬─────┘ └──────┬─────┘   │   │
│  │       │                       │               │         │   │
│  │  ┌────▼───────────────────────▼───────────────▼─────┐   │   │
│  │  │            FR-FCFS Scheduler                      │   │   │
│  │  └────┬──┬──┬──┬──┬──┬──┬──┬─────────────────────── │   │   │
│  │       │  │  │  │  │  │  │  │                         │   │
│  │      BK0 BK1 BK2 BK3 BK4 BK5 BK6 BK7               │   │
│  │      (Bank FSM × 8)                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐                  │   │
│  │  │ Det. Mode    │  │ Power Model  │                  │   │
│  │  │ (HFT fixed)  │  │ (CKE ctrl)   │                  │   │
│  │  └──────────────┘  └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌────────────────────────────────┐                            │
│  │  HFT Modules                   │                            │
│  │  ETH MAC RX → UDP Parser       │                            │
│  │  → Async FIFO → FP ALU         │                            │
│  └────────────────────────────────┘                            │
└────────────────────────────────────────────────────────────────┘
```

## 2. Memory Map

| Slave | Base Address   | Size   | Module          | Description                    |
|-------|---------------|--------|-----------------|--------------------------------|
| S0    | 0x0000_0000   | 8 KB   | boot_rom        | Read-only boot firmware        |
| S1    | 0x1000_0000   | 32 KB  | sram_ctrl       | SRAM with byte-enable          |
| S2    | 0x2000_0000   | 64 KB  | periph_mux      | Peripheral sub-decode          |
| S3    | 0x3000_0000   | 16 MB  | accel_top       | Accelerator CSR + DMA          |
| S4    | 0x4000_0000   | 256 MB | dram_ctrl_top   | DRAM Controller                |
| S5-S7 | 0x5000_0000+  | —      | (unused)        | DECERR                         |

### Peripheral Sub-Map (within S2)

| Sub-offset | Base Address   | Module     | Description                  |
|-----------|---------------|------------|------------------------------|
| 0x00      | 0x2000_0000   | uart_ctrl  | UART TX/RX + FIFO            |
| 0x01      | 0x2001_0000   | timer_ctrl | 64-bit mtime/mtimecmp        |
| 0x02      | 0x2002_0000   | gpio_ctrl  | 8-bit GPIO with dir reg      |
| 0x03      | 0x2003_0000   | plic       | 32-source interrupt ctrl     |

### DRAM Address Space (within S4)

| Offset         | Description                             |
|----------------|-----------------------------------------|
| 0x4000_0000    | DDR3 data region (RBC or BRC interleave)|

DRAM CSR registers (future) will be at a separate sub-address space.

## 3. CPU

**Current:** `simple_cpu` — a minimal OBI-interface test CPU.  
**Planned (Month 2):** Ibex (rv32imc) via git submodule.

OBI Interface:
- 32-bit address, 32-bit data
- Single-cycle grant (combinational)
- Response on next rising edge

## 4. Interconnect

**Module:** `axi_crossbar`  
**Topology:** 2 masters × 8 slaves, round-robin arbitration  
**Protocol:** AXI4-Lite with ID fields (4-bit)  

Address decode (upper 4 bits of address):
- `4'h0` → S0 (Boot ROM)
- `4'h1` → S1 (SRAM)
- `4'h2` → S2 (Peripherals)
- `4'h3` → S3 (Accelerator)
- `4'h4` → S4 (DRAM Controller)
- Other  → S5+ (DECERR)

## 5. Peripherals

### 5.1 UART (`uart_ctrl`)
- 16-byte TX/RX FIFOs
- Configurable baud divisor register
- IRQ on RX FIFO non-empty
- Separate `uart_tx.sv` / `uart_rx.sv` for TX shift register and 16× oversampled RX

### 5.2 Timer (`timer_ctrl`)
- 64-bit `mtime` free-running counter
- 64-bit `mtimecmp` compare register
- Interrupt when `mtime >= mtimecmp`
- Atomic 64-bit read via hi-lo-hi pattern

### 5.3 GPIO (`gpio_ctrl`)
- 8-bit bidirectional GPIO
- Direction register (1=output, 0=input)
- 2-FF input synchronizer

### 5.4 PLIC (`plic`)
- 32 interrupt sources, 3-bit priority (0-7)
- Threshold register filters low-priority interrupts
- Claim/complete MMIO protocol
- Sources wired: UART_RX(1), Timer(4), GPIO(future), ETH(future), ACCEL(future)

## 6. DRAM Controller

**Module:** `dram_ctrl_top`  
**Target:** DDR3-1600 via Zynq-7020 PS DDR interface (200 MHz controller clock)

### 6.1 Sub-Module Architecture

| Module                     | Function                                    |
|----------------------------|---------------------------------------------|
| `dram_addr_decoder`        | AXI addr → bank/row/col (RBC or BRC mode)  |
| `dram_cmd_queue` (16-deep) | Command FIFO with age tracking              |
| `dram_write_buffer` (16)   | Buffered W-channel data for write path      |
| `dram_bank_fsm` × 8       | Per-bank timing state machine               |
| `dram_refresh_ctrl`        | tREFI periodic refresh with handshake       |
| `dram_scheduler_frfcfs`    | FR-FCFS: Refresh > Row-hit > FCFS           |
| `dram_deterministic_mode`  | Fixed-latency read padding (HFT path)      |
| `dram_power_model`         | CKE power-down + cycle counters             |

### 6.2 Timing Parameters (200 MHz)

| Parameter | Cycles | DDR3 Spec |
|-----------|--------|-----------|
| tRCD      | 3      | 13.75 ns  |
| tRP       | 3      | 13.75 ns  |
| tRAS      | 7      | 35 ns     |
| tRC       | 10     | 48.75 ns  |
| tRTP      | 2      | 7.5 ns    |
| tWR       | 3      | 15 ns     |
| tCAS (CL) | 3      | 13.75 ns  |
| tREFI     | 1560   | 7.8 µs   |
| tRFC      | 52     | 260 ns    |

### 6.3 Address Interleaving

Default: **RBC (Row-Bank-Column)** for streaming workloads.  
Switchable to BRC at runtime via CSR.  
See [addr_interleaving_analysis.md](../analysis/addr_interleaving_analysis.md).

### 6.4 Deterministic Mode

For HFT critical path: every read completes in exactly `FIXED_LATENCY` (16) cycles.  
Up to 4 outstanding reads. `err_deadline_miss` flag on overrun.  
Enabled via CSR bit.

## 7. Accelerator Interface

**Module:** `accel_top` (wired to S3)

- AXI4-Lite slave for CSR access (registers at offsets 0x00–0xF0)
- AXI4 master for DDR DMA (weight + activation load, result writeback)
- 16×16 INT8 weight-stationary systolic array (256 DSP48s)
- BSR sparse scheduler for compressed weights
- Double-buffered output BRAM, bypassable max-pool unit

Key CSR registers:
- 0x00: STATUS, 0x04: CONTROL, 0x10-0x1C: DMA addresses
- 0xE0-0xF0: Layer pipeline management

## 8. Interrupt Architecture

```
  IRQ Sources          PLIC            CPU
  ┌─────────┐    ┌────────────┐   ┌─────────┐
  │ UART RX ├───▶│ Priority   │   │         │
  │ Timer   ├───▶│ Arbitrate  ├──▶│ M-mode  │
  │ GPIO    ├───▶│ Threshold  │   │ ext IRQ │
  │ ETH RX  ├───▶│ Claim/     │   │         │
  │ ACCEL   ├───▶│ Complete   │   └─────────┘
  └─────────┘    └────────────┘
```

PLIC `irq_o` drives CPU's external interrupt input.  
ISR software flow: claim → dispatch → complete (see `fw/isr.c`).

## 9. HFT Subsystem

| Module            | Function                      |
|-------------------|-------------------------------|
| `eth_mac_rx`      | Ethernet MAC receive path     |
| `eth_udp_parser`  | UDP packet extraction         |
| `async_fifo`      | CDC FIFO (125→100 MHz)        |
| `fixedpoint_alu`  | 32-bit fixed-point arithmetic |

HFT data path: ETH RX → UDP Parser → Async FIFO → Fixed-Point ALU.  
DRAM access via deterministic mode for guaranteed latency.

## 10. Clocking

**Current:** Single clock domain (`clk` → `clk_core` = 100 MHz).  
**DRAM controller:** 200 MHz (will use MMCM or PLL for 2× multiply).  
**ETH RX:** 125 MHz domain, CDC via `async_fifo`.

## 11. Reset

Active-low `rst_n` → internally synchronized.  
All peripherals share `rst_core_n`.  
DRAM controller has independent reset sequence (init PHY before traffic).

## 12. Power Management

`dram_power_model` module:
- Tracks active/idle/power-down cycles
- CKE deasserted after 64 idle cycles (configurable)
- Wakeup latency: tXP = 3 cycles
- Self-refresh not implemented (requires PS DDRC cooperation)

## 13. Resource Estimates (Zynq-7020)

| Block                   | LUTs  | FFs   | DSPs | BRAM36 |
|-------------------------|-------|-------|------|--------|
| CPU (simple_cpu)        | 200   | 150   | 0    | 0      |
| OBI→AXI Bridge          | 150   | 100   | 0    | 0      |
| AXI Crossbar            | 600   | 400   | 0    | 0      |
| Boot ROM                | 50    | 0     | 0    | 2      |
| SRAM Controller         | 100   | 50    | 0    | 16     |
| Peripherals (UART+T+G+P)| 580  | 440   | 0    | 0      |
| Accelerator (16×16 SA)  | 5,930 | 4,210 | 256  | 6      |
| **DRAM Controller**     | **1,960** | **1,540** | **0** | **0** |
| HFT Modules             | 600   | 400   | 1    | 0      |
| SoC Glue                | 200   | 100   | 0    | 0      |
| **Total**               | **10,370** | **7,390** | **257** | **24** |
| **Zynq-7020 Available** | 53,200 | 106,400 | 220 | 140  |
| **Utilisation**         | **19.5%** | **6.9%** | **116.8%** | **17.1%** |

## 14. Firmware Stack

```
  ┌─────────────────────────┐
  │       main.c            │  Application
  ├─────────────────────────┤
  │  hal_uart / timer /     │  HAL Drivers
  │  gpio / plic / eth      │
  ├─────────────────────────┤
  │  isr.c                  │  Interrupt dispatch
  ├─────────────────────────┤
  │  startup.S + link.ld    │  Boot + memory layout
  └─────────────────────────┘
```

## 15. Month 1 → Month 2 Transition

### Completed (Month 1)
- All SoC peripherals created and wired
- 16×16 systolic array + BSR scheduler
- Full DRAM controller with 8 sub-modules
- HFT pipeline (ETH → UDP → FIFO → ALU)
- Cocotb test suite (20+ tests)
- Microarch specs + synthesis estimates

### Planned (Month 2)
- Replace `simple_cpu` with Ibex rv32imc
- Wire accelerator to S3 with full DMA path
- Wire DRAM controller to S4
- First Vivado synthesis & P&R
- Expanded test coverage with VCD tracing
