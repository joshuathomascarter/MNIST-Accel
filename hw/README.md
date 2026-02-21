# ACCEL-v1 Hardware Design

> 14×14 Weight-Stationary Systolic Array — BSR Sparse Acceleration — Zynq-7020

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Zynq Z7020 (PL)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │              │     │                 ACCEL-v1 (PL)                   │   │
│  │   ARM PS     │     │  ┌───────────┐  ┌───────────┐  ┌────────────┐  │   │
│  │              │     │  │           │  │           │  │            │  │   │
│  │  ┌────────┐  │ AXI │  │  BSR DMA  │─▶│  Weight   │─▶│  14×14     │  │   │
│  │  │ Linux  │  │ HP  │  │  Engine   │  │  Buffer   │  │  Systolic  │  │   │
│  │  │ Driver │◀─┼─────┼─▶│           │  │  (BRAM)   │  │  Array     │  │   │
│  │  └────────┘  │     │  └───────────┘  └───────────┘  │            │  │   │
│  │              │     │                                 │  196 PEs   │  │   │
│  │  ┌────────┐  │ AXI │  ┌───────────┐  ┌───────────┐  │  INT8 MAC  │  │   │
│  │  │ Python │  │Lite │  │    CSR    │  │Activation │─▶│            │  │   │
│  │  │ PYNQ   │◀─┼─────┼─▶│  Control  │  │  Buffer   │  │            │  │   │
│  │  └────────┘  │     │  │           │  │  (BRAM)   │  │            │──┼───┼─▶ Output
│  │              │     │  └───────────┘  └───────────┘  └────────────┘  │   │
│  └──────────────┘     │                                                 │   │
│                       │  ┌───────────┐  ┌───────────────────────────┐  │   │
│        DDR3           │  │   BSR     │  │    Output Accumulator     │  │   │
│   ┌─────────────┐     │  │ Scheduler │  │    + ReLU + Requantize    │  │   │
│   │ Weights     │     │  │           │  │    (INT32 → INT8)         │  │   │
│   │ Activations │◀────┼──│           │◀─│                           │  │   │
│   │ Results     │     │  └───────────┘  └───────────────────────────┘  │   │
│   └─────────────┘     └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Systolic Array (14×14, Weight-Stationary)

```
                    Activations (broadcast down columns)
                    ↓     ↓     ↓     ↓           ↓
              ┌─────┬─────┬─────┬─────┬─ ─ ─┬─────┐
              │a[0] │a[1] │a[2] │a[3] │     │a[13]│
              └──┬──┴──┬──┴──┬──┴──┬──┴─ ─ ─┴──┬──┘
                 ↓     ↓     ↓     ↓           ↓
    ┌────┐    ┌─────┬─────┬─────┬─────┬─────┬─────┐
    │w[0]│───▶│PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[0]
    └────┘    │0,0  │0,1  │0,2  │0,3  │     │0,13 │
              └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┘
    ┌────┐       ↓     ↓     ↓     ↓           ↓
    │w[1]│───▶┌─────┬─────┬─────┬─────┬─────┬─────┐
    └────┘    │PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[1]
              │1,0  │1,1  │1,2  │1,3  │     │1,13 │
              └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┘
                 :     :     :     :           :
    ┌────┐       ↓     ↓     ↓     ↓           ↓
    │w[13]│──▶┌─────┬─────┬─────┬─────┬─────┬─────┐
    └────┘    │PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[13]
              │13,0 │13,1 │13,2 │13,3 │     │13,13│
              └─────┴─────┴─────┴─────┴─────┴─────┘

   Each PE: stores 1 weight (INT8), acc += w × a (INT8×INT8→INT32)
   Activations propagate down, partial sums accumulate right.
```

### Dataflow Timing

1. **Load phase** — Weights loaded systolically into PEs (14 cycles)
2. **Compute phase** — Activations stream through, MACs accumulate
3. **Drain phase** — Partial sums collected from right edge

```
Cycle:    1    2    3    4    5    ...   K+13
Row 0:   │ a0   a1   a2   a3   ...   aK-1     │  → psum[0]
Row 1:   │      a0   a1   a2   ...   aK-2     │  → psum[1]
Row 2:   │           a0   a1   ...   aK-3     │  → psum[2]
  :      │                                    │
Row 13:  │                          a0   ...  │  → psum[13]
          ◄──── K cycles + 13 skew cycles ────►
```

---

## Processing Element

```
                    ┌─────────────────────────────────┐
                    │           PE [row, col]         │
    activation_in ─▶│  ┌─────┐                        │
    (INT8)          │  │ REG │─┬──────────────────────┼──▶ activation_out
                    │  └─────┘ │                      │     (to PE below)
                    │          ↓                      │
                    │     ┌─────────┐                 │
    weight_in ─────▶│────▶│   ×     │ INT8 × INT8     │
    (INT8)          │     │  (MUL)  │ = INT16         │
                    │     └────┬────┘                 │
                    │          ↓                      │
                    │     ┌─────────┐   ┌─────────┐   │
    psum_in ───────▶│────▶│    +    │──▶│   REG   │───┼──▶ psum_out
    (INT32)         │     │  (ACC)  │   │ (INT32) │   │    (to PE right)
                    │     └─────────┘   └─────────┘   │
                    │                                 │
                    └─────────────────────────────────┘

    Latency: 1 cycle (fully pipelined)
    MAC unit: mac8.sv with zero-value bypass and operand isolation
```

---

## BSR Sparse Format

The BSR scheduler skips zero-weight blocks entirely. Metadata (row pointers + column indices) is stored in BRAM and read by the hardware FSM to determine which blocks to load.

### Memory Layout

```
Dense Matrix (example, 70% block-sparse):       BSR Encoding:

┌────┬────┬────┬────┐                           row_ptr[]:  [0, 2, 3, 5, 5]
│████│    │████│    │  Row 0: 2 non-zero         col_idx[]:  [0, 2, 1, 0, 3]
├────┼────┼────┼────┤                           data[]:     [Block(0,0), Block(0,2),
│    │████│    │    │  Row 1: 1 non-zero                     Block(1,1), Block(2,0),
├────┼────┼────┼────┤                                        Block(2,3)]
│████│    │    │████│  Row 2: 2 non-zero
├────┼────┼────┼────┤                           Each block: 14×14 = 196 INT8 values
│    │    │    │    │  Row 3: 0 non-zero
└────┴────┴────┴────┘

Sparsity savings:  70% sparse → 3.3× fewer blocks → 3.3× less compute
                   90% sparse → 10× fewer blocks  → ~9× effective speedup
```

---

## AXI Interface

### CSR Register Map (AXI4-Lite)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | RW | `[0]` START, `[1]` RESET, `[2]` IRQ_EN |
| 0x04 | STATUS | RO | `[0]` BUSY, `[1]` DONE, `[2]` ERROR |
| 0x08 | BSR_ADDR | RW | DDR base address for BSR weight data |
| 0x10 | ACT_ADDR | RW | DDR base address for activation data |
| 0x18 | OUT_ADDR | RW | DDR base address for output results |
| 0x20 | TILE_CFG | RW | `[15:0]` M, `[31:16]` N, `[47:32]` K |
| 0x2C | CYCLES | RO | Performance counter: total cycles |
| 0x30 | STALLS | RO | Performance counter: stall cycles |

### AXI4 Data Interface

- 64-bit data width, burst transfers up to 256B
- Read channel: weight and activation DMA from DDR
- Write channel: output results back to DDR

---

## RTL Module Hierarchy

```
hw/rtl/
├── top/
│   ├── accel_top.sv                 # Top-level integration
│   └── accel_top_dual_clk.sv       # Dual-clock wrapper (50/200 MHz)
├── systolic/
│   ├── pe.sv                        # Processing element
│   ├── systolic_array.sv           # Dense systolic array
│   └── systolic_array_sparse.sv    # Sparse-aware array (BSR)
├── mac/
│   └── mac8.sv                      # INT8 MAC with zero-bypass
├── buffer/
│   ├── act_buffer.sv                # Double-buffered activation BRAM
│   ├── wgt_buffer.sv               # Weight BRAM
│   └── output_accumulator.sv       # INT32 accumulation + requantize
├── dma/
│   ├── act_dma.sv                   # Activation DMA (AXI4 master)
│   ├── bsr_dma.sv                   # BSR metadata + weight loader
│   ├── out_dma.sv                   # Output DMA
│   └── dma_pack_112.sv             # 64-bit → 112-bit data packing
├── control/
│   ├── csr.sv                       # Control/status registers
│   ├── bsr_scheduler.sv            # Sparse block scheduler FSM
│   ├── pulse_sync.sv               # CDC pulse synchronizer
│   └── sync_2ff.sv                  # 2-FF synchronizer
├── host_iface/
│   ├── axi_lite_slave.sv           # AXI4-Lite CSR interface
│   └── axi_dma_bridge.sv          # AXI4 DMA arbiter
└── monitor/
    └── perf.sv                      # Performance counters
```

---

## Resource Estimates (XC7Z020)

| Resource | Estimated | Available | Utilization |
|----------|-----------|-----------|-------------|
| LUTs | ~18,000 | 53,200 | 34% |
| FFs | ~12,000 | 106,400 | 11% |
| BRAM (36 Kb) | 64 | 140 | 46% |
| DSP48E1 | 196 | 220 | 89% |

---

## Simulation

### Verilator

```bash
cd hw/sim
make
```

### Cocotb

```bash
cd hw/sim/cocotb
make
```

### SystemVerilog Testbenches

| Testbench | Module Under Test |
|-----------|-------------------|
| `accel_top_tb.sv` | Full integration |
| `accel_top_tb_full.sv` | Extended integration |
| `systolic_tb.sv` | Systolic array |
| `pe_tb.sv` | Single PE |
| `bsr_dma_tb.sv` | BSR DMA transfers |
| `output_accumulator_tb.sv` | Accumulator + requantize |
| `perf_tb.sv` | Performance counters |
| `integration_tb.sv` | Multi-module integration |
| `meta_decode_tb.sv` | Metadata decoder |
| `tb_axi_lite_slave_enhanced.sv` | AXI-Lite CSR |

---

## FPGA Deployment (PYNQ-Z2)

### Synthesis

```bash
vivado -mode batch -source tools/synthesize_vivado.tcl
```

### Block Design

```
┌─────────────────────────────────────────────────────┐
│  ┌──────────┐     ┌──────────────┐   ┌───────────┐  │
│  │ ZYNQ PS  │     │ AXI Intercon │   │ accel_top │  │
│  │ M_AXI_HP ─┼────▶│              │──▶│ S_AXI_LITE│  │
│  │ S_AXI_HP ◀┼─────│              │◀──│ M_AXI     │  │
│  │ FCLK_CLK ─┼─────────────────────▶│ clk       │  │
│  └──────────┘     └──────────────┘   └───────────┘  │
└─────────────────────────────────────────────────────┘
```

### Deploy

```bash
scp build/accel_top.bit xilinx@pynq:/home/xilinx/

# On the board:
python3 -c "
from pynq import Overlay
ol = Overlay('accel_top.bit')
print('Accelerator loaded')
"
```