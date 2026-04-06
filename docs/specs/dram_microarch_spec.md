# DRAM Controller Microarchitecture Specification v0.1

## 1. Overview

Custom DDR3 DRAM controller targeting the Zynq-7020 PS-side DDR3 interface.
Designed for deterministic, low-latency memory access for both the MNIST
accelerator DMA engine and the HFT ultra-low-latency path.

### Block Diagram

```
  AXI4 Slave                                         DRAM PHY
  ─────────┐                                    ┌──────────────
           │  ┌──────────┐   ┌──────────────┐  │
    AW/W ──┼─►│ Cmd Queue│──►│  FR-FCFS     │──┼─► ACT/RD/WR/PRE
    AR   ──┼─►│ (16-deep)│   │  Scheduler   │  │
           │  └──────────┘   └──────┬───────┘  │
           │  ┌──────────┐         │           │
    W data─┼─►│Write Buf │─────────┘           │
           │  └──────────┘                     │
           │  ┌──────────┐   ┌──────────────┐  │
    R data◄┼──│ Det Mode │◄──│ Bank FSM x8  │◄─┤── RDATA
           │  └──────────┘   └──────────────┘  │
           │                 ┌──────────────┐  │
           │                 │ Refresh Ctrl │──┼─► REF
           │                 └──────────────┘  │
           │                 ┌──────────────┐  │
           │                 │ Power Model  │──┼─► CKE
           │                 └──────────────┘  │
  ─────────┘                                    └──────────────
```

## 2. Module Inventory

| Module                       | File                            | LUTs | FFs | Function                         |
|------------------------------|---------------------------------|------|-----|----------------------------------|
| `dram_addr_decoder`          | `dram_addr_decoder.sv`          |   30 |   0 | AXI addr → bank/row/col         |
| `dram_cmd_queue`             | `dram_cmd_queue.sv`             |  200 | 200 | 16-entry command FIFO + age      |
| `dram_write_buffer`          | `dram_write_buffer.sv`          |  100 | 520 | 16-entry write data buffer       |
| `dram_bank_fsm` (×8)        | `dram_bank_fsm.sv`              |  960 | 400 | Per-bank timing state machine    |
| `dram_refresh_ctrl`          | `dram_refresh_ctrl.sv`          |   50 |  30 | tREFI counter + REF handshake    |
| `dram_scheduler_frfcfs`      | `dram_scheduler_frfcfs.sv`      |  400 | 150 | FR-FCFS command scheduler        |
| `dram_deterministic_mode`    | `dram_deterministic_mode.sv`    |   80 | 160 | Fixed-latency padding (HFT)      |
| `dram_power_model`           | `dram_power_model.sv`           |   40 |  30 | CKE/power-down + cycle counters  |
| `dram_ctrl_top`              | `dram_ctrl_top.sv`              |  100 |  50 | Top integration + AXI front-end  |
| **Total**                    |                                 |**1960**|**1540**|                              |

## 3. Timing Parameters

All timings in controller clock cycles at 200 MHz (5 ns period).

| Parameter | Cycles | DDR3-1600 Spec |
|-----------|--------|----------------|
| tRCD      | 3      | 13.75 ns       |
| tRP       | 3      | 13.75 ns       |
| tRAS      | 7      | 35 ns          |
| tRC       | 10     | 48.75 ns       |
| tRTP      | 2      | 7.5 ns         |
| tWR       | 3      | 15 ns          |
| tCAS (CL) | 3      | 13.75 ns       |
| tREFI     | 1560   | 7.8 µs         |
| tRFC      | 52     | 260 ns         |
| tXP       | 3      | 6 ns           |

## 4. Address Decoding

Two modes selectable at runtime via CSR:

### Mode 0: RBC (Row-Bank-Column) — Default
```
[31:28] unused | [27:14] ROW | [13:11] BANK | [10:1] COL | [0] BYTE
```

### Mode 1: BRC (Bank-Row-Column)
```
[31:28] unused | [27:25] BANK | [24:11] ROW | [10:1] COL | [0] BYTE
```

See [addr_interleaving_analysis.md](../analysis/addr_interleaving_analysis.md)
for performance analysis.

## 5. Scheduling Algorithm: FR-FCFS

Priority (highest first):
1. **Refresh** — tREFI deadline must be met
2. **Row-hit** commands — oldest first (FCFS among hits)
3. **Row-miss / bank-idle** commands — oldest first

The scheduler issues sub-commands through a multi-step FSM:
```
SCH_IDLE → SCH_ISSUE_PRE (if row miss, precharge needed)
         → SCH_ISSUE_ACT (activate target row)
         → SCH_ISSUE_RW  (issue READ or WRITE)
         → SCH_REFRESH   (all-bank refresh)
```

## 6. Write Path

1. AXI AW channel accepted → command enqueued in `dram_cmd_queue` with `rw=1`
2. AXI W channel data stored in `dram_write_buffer` (indexed by queue slot)
3. Scheduler selects write command → issues PRE/ACT/WRITE sequence
4. On WRITE issue, `dram_write_buffer` drains data → PHY wdata

## 7. Read Path

1. AXI AR accepted → command enqueued with `rw=0`
2. Scheduler issues PRE/ACT/READ sequence
3. PHY returns `rdata_valid` after CAS latency
4. Normal mode: `rdata` → AXI R channel immediately
5. Deterministic mode: data buffered in `dram_deterministic_mode`, released exactly
   at `FIXED_LATENCY` cycles from AR acceptance

## 8. Power Management

`dram_power_model` monitors controller activity:
- **Active**: commands in flight or pending
- **Idle**: no commands, idle counter running
- **Power-Down**: CKE deasserted after `IDLE_THRESHOLD` (default 64) cycles

Cycle counters (`cnt_active_cycles`, `cnt_idle_cycles`, `cnt_pd_cycles`)
exposed for simulation power estimation.

## 9. Deterministic Mode (HFT)

For latency-critical paths, `dram_deterministic_mode` pads variable-latency
DRAM reads to a fixed cycle count:
- **FIXED_LATENCY** = 16 (default) — worst-case + margin
- Up to **MAX_OUTSTANDING** = 4 reads in flight
- If data fails to arrive before the deadline: `err_deadline_miss` flag asserts
- Bypassable via `det_enable` CSR bit

## 10. Verification Plan

| Test                             | File                              | Coverage          |
|----------------------------------|-----------------------------------|-------------------|
| Bank FSM timing                  | `test_dram_bank_fsm.py`          | State transitions |
| FR-FCFS scheduling               | `test_dram_frfcfs.py`            | Priority, refresh |
| Write buffer drain               | `test_dram_write_drain.py`       | Fill, drain, reuse|
| Address interleaving             | `test_dram_addr_interleave.py`   | RBC/BRC decode    |
| Deterministic latency            | `test_dram_deterministic.py`     | Fixed timing, miss|

## 11. Resource Budget (Zynq-7020)

```
DRAM Controller total: ~1960 LUTs, ~1540 FFs, 0 DSP, 0 BRAM
Zynq-7020 capacity:    53200 LUTs
DRAM % of device:      3.7%
```

Combined with accelerator (~8400 LUTs) and SoC peripherals (~2000 LUTs),
total utilisation remains under 25% of available resources.
