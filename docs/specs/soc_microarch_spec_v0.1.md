# SoC Micro-Architecture Specification v0.1

**Project:** MNIST ACCEL-v1 — Zynq-7020 FPGA SoC  
**Date:** 2026-03-26  
**Status:** Draft  

---

## 1. Overview

A single-bus SoC integrating a RISC-V CPU core, AXI interconnect, memory subsystem,
peripheral set, and a 16×16 INT8 sparse systolic-array accelerator.
Target FPGA: Xilinx Zynq-7020 (XC7Z020CLG484-1).

```
┌─────────────────────────────────────────────────────┐
│                     soc_top                         │
│                                                     │
│  ┌──────────┐     ┌──────────────────────────────┐  │
│  │ simple_  │ OBI │          obi_to_axi          │  │
│  │   cpu    │────▶│           bridge              │  │
│  └──────────┘     └───────────┬──────────────────┘  │
│                               │ AXI4-Lite M0        │
│                   ┌───────────▼──────────────────┐  │
│                   │     axi_crossbar             │  │
│                   │   2 Masters × 8 Slaves       │  │
│                   └──┬──┬──┬──┬──┬──┬──┬──┬─────┘  │
│                   S0 S1 S2 S3 S4 S5 S6 S7          │
│                   │  │  │  │                        │
│           ┌───────┘  │  │  └─ (ACCEL placeholder)   │
│           │     ┌────┘  │                           │
│           ▼     ▼       ▼                           │
│       boot_rom sram  periph_mux                     │
│       (8 KB)  (32KB) ┌──┬──┬──┐                     │
│                      │  │  │  │                     │
│                   UART TIM GPIO PLIC                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 2. Memory Map

| Slave | Base Address   | Size   | Module          | Description                    |
|-------|---------------|--------|-----------------|--------------------------------|
| S0    | 0x0000_0000   | 8 KB   | boot_rom        | Read-only boot firmware        |
| S1    | 0x1000_0000   | 32 KB  | sram_ctrl       | SRAM with byte-enable          |
| S2    | 0x2000_0000   | 64 KB  | periph_mux      | Peripheral sub-decode          |
| S3    | 0x3000_0000   | 16 MB  | accel_top (TBD) | Accelerator CSR + DMA          |
| S4-S7 | 0x4000_0000+  | —      | (unused)        | DECERR                         |

### Peripheral Sub-Map (within S2)

| Sub-offset | Base Address   | Module     | Description                  |
|-----------|---------------|------------|------------------------------|
| 0x00      | 0x2000_0000   | uart_ctrl  | UART TX/RX + FIFO            |
| 0x01      | 0x2001_0000   | timer_ctrl | 64-bit mtime/mtimecmp        |
| 0x02      | 0x2002_0000   | gpio_ctrl  | 8-bit GPIO with dir reg      |
| 0x03      | 0x2003_0000   | plic       | 32-source interrupt controller|

## 3. CPU

**Current:** `simple_cpu` — a minimal OBI-interface test CPU.  
**Planned:** Ibex (rv32imc) via git submodule.

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
- Other  → S4+ (DECERR)

## 5. Peripherals

### 5.1 UART (`uart_ctrl`)

- 16-byte TX/RX FIFOs
- Configurable baud divisor register
- IRQ on RX FIFO non-empty
- Separate `uart_tx.sv` / `uart_rx.sv` modules for TX shift register and 16× oversampled RX

### 5.2 Timer (`timer_ctrl`)

- 64-bit `mtime` free-running counter
- 64-bit `mtimecmp` compare register
- Interrupt when `mtime >= mtimecmp`
- Atomic 64-bit read via hi-lo-hi pattern

### 5.3 GPIO (`gpio_ctrl`)

- 8-bit bidirectional GPIO
- Direction register (1=output, 0=input)
- 2-FF input synchronizer
- Directly memory-mapped (no FIFO)

### 5.4 PLIC (`plic`)

- 32 interrupt sources, 3-bit priority (0-7)
- Threshold register filters low-priority interrupts
- Claim/complete MMIO protocol
- Sources wired: UART_RX(1), Timer(4), GPIO(future), ETH(future), ACCEL(future)

## 6. Accelerator Interface (Planned)

**Module:** `accel_top` (exists, to be wired to S3)

The accelerator exposes:
- AXI4-Lite slave for CSR access (registers at offsets 0x00–0xF0)
- AXI4 master for DDR DMA (weight + activation load, result writeback)

Key CSR registers:
- 0x00: STATUS, 0x04: CONTROL, 0x10-0x1C: DMA addresses
- 0xE0-0xF0: Layer pipeline management

## 7. Interrupt Architecture

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

## 8. Clocking

**Current:** Single clock domain (`clk` → `clk_core`).  
**Planned:** Separate domains for ETH RX (125 MHz) and system (100 MHz) with `async_fifo` CDC.

## 9. Reset

Active-low `rst_n` → internally synchronized.  
All peripherals share `rst_core_n`.

## 10. Resource Estimates (Zynq-7020)

| Block              | LUTs  | FFs   | DSPs | BRAM36 |
|--------------------|-------|-------|------|--------|
| simple_cpu         | ~200  | ~150  | 0    | 0      |
| obi_to_axi bridge | ~150  | ~100  | 0    | 0      |
| axi_crossbar       | ~800  | ~400  | 0    | 0      |
| boot_rom (8KB)     | ~50   | ~30   | 0    | 2      |
| sram_ctrl (32KB)   | ~50   | ~30   | 0    | 8      |
| uart_ctrl          | ~200  | ~150  | 0    | 0      |
| timer_ctrl         | ~100  | ~80   | 0    | 0      |
| gpio_ctrl          | ~60   | ~40   | 0    | 0      |
| plic               | ~300  | ~200  | 0    | 0      |
| accel_top (16×16)  | ~8500 | ~5200 | 256  | ~20    |
| **SoC Total**      | ~10400| ~6400 | 256  | ~30    |
| **Zynq-7020 Avail**| 53200 | 106400| 220  | 140    |
| **Utilization**    | ~20%  | ~6%   | 116% | ~21%   |

## 11. Firmware Stack

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

Toolchain: `riscv32-unknown-elf-gcc` (rv32imc, ilp32)

---

*This spec will be updated as DRAM controller and Ibex integration are completed.*
