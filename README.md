# ACCEL-v1: Sparse INT8 CNN Accelerator

<div align="center">

**14×14 Weight-Stationary Systolic Array · BSR Sparse Acceleration · Zynq-7020 FPGA**

![RTL](https://img.shields.io/badge/RTL-SystemVerilog-blue)
![Target](https://img.shields.io/badge/Target-Zynq%20Z7020-green)
![Status](https://img.shields.io/badge/Status-Simulation%20Verified-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Overview

ACCEL-v1 is a sparse neural network inference accelerator targeting the Xilinx Zynq-7020 (PYNQ-Z2). The design centers on a 14×14 weight-stationary systolic array with hardware-level Block Sparse Row (BSR) scheduling, INT8 quantization, and AXI4 DMA integration.

**Key capabilities:**
- 14×14 systolic array — 196 INT8 MACs per cycle, weight-stationary dataflow
- BSR sparse format — hardware scheduler skips zero-weight blocks entirely
- INT8 quantization — per-channel scaling, 0.2% accuracy loss on MNIST (98.7%)
- AXI4 DMA + AXI4-Lite CSR control interface
- Dual-clock architecture — 50 MHz control / 200 MHz datapath
- Full software stack — Python training/export, C++ host driver framework

**Status:** RTL simulation-verified (Verilator + cocotb). FPGA deployment pending.

---

## Performance Targets

| Metric | Value | Notes |
|--------|-------|-------|
| Peak Throughput | 39.2 GOPS | 196 MACs/cycle × 200 MHz |
| Sparse Speedup | 6–9× | vs. dense baseline at 70–90% block sparsity |
| Memory Reduction | 9.7× | BSR format (118 KB vs. 1.15 MB, MNIST FC1) |
| INT8 Accuracy | 98.7% | MNIST CNN, –0.2% from FP32 baseline |
| Power Target | 840 mW | Dual-clock + clock gating (vs. 2.0 W baseline) |

*Pending hardware validation after synthesis and FPGA deployment.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Zynq-7020 (PL)                          │
│                                                             │
│  Host (PS)                    Control                       │
│  ┌──────────┐  AXI4-Lite   ┌──────────────┐                │
│  │ ARM CPU  │◄────────────►│ CSR Block    │                │
│  │ / PYNQ   │              │ BSR Sched.   │                │
│  └────┬─────┘              │ Clock Gate   │                │
│       │                    └──────┬───────┘                │
│       │ AXI4 DMA                  │                         │
│       ▼                          ▼                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │ Act DMA  │  │ BSR DMA  │  │ 14×14 Systolic Array     │  │
│  │          │  │          │  │ (196 PEs, INT8×INT8→INT32)│  │
│  └────┬─────┘  └────┬─────┘  │ Weight-Stationary        │  │
│       ▼              ▼        │ Zero-Value Bypass        │  │
│  ┌──────────┐  ┌──────────┐  └────────────┬─────────────┘  │
│  │Act Buffer│  │Wgt Buffer│               │                │
│  │ (BRAM)   │  │ (BRAM)   │               ▼                │
│  └──────────┘  └──────────┘  ┌──────────────────────────┐  │
│                              │ Output Accumulator       │  │
│                              │ Requantize (INT32→INT8)  │  │
│                              │ ReLU + Saturation        │  │
│                              └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Parameter | Choice | Rationale |
|-----------|--------|-----------|
| Data type | INT8 | 4× memory reduction, minimal accuracy loss |
| Array size | 14×14 | Fits Z7020 DSP budget (196 of 220 DSP48E1s) |
| Block size | 14×14 | Matches array dimensions for BSR tiling |
| Dataflow | Weight-stationary | Minimizes weight reloads, maximizes data reuse |
| Sparse format | BSR | Sequential memory access, hardware-friendly metadata |
| Clock gating | BUFGCE | 810 mW estimated savings (40.5% reduction) |

### Resource Estimates (Zynq XC7Z020)

| Resource | Estimated | Available | Utilization |
|----------|-----------|-----------|-------------|
| LUTs | ~18,000 | 53,200 | 34% |
| FFs | ~12,000 | 106,400 | 11% |
| BRAM (36 Kb) | 64 | 140 | 46% |
| DSP48E1 | 196 | 220 | 89% |

---

## Repository Structure

```
├── hw/                              # Hardware design
│   ├── rtl/                         # Production RTL (21 modules, ~6,500 lines)
│   │   ├── top/                     #   Top-level: accel_top.sv, dual-clock wrapper
│   │   ├── systolic/                #   PE array: pe.sv, systolic_array_sparse.sv
│   │   ├── mac/                     #   Compute: mac8.sv (INT8 MAC, zero-bypass)
│   │   ├── control/                 #   FSMs: bsr_scheduler.sv, csr.sv, CDC
│   │   ├── dma/                     #   DMA engines: act_dma.sv, bsr_dma.sv
│   │   ├── buffer/                  #   BRAM buffers: act, weight, output accum.
│   │   ├── host_iface/              #   AXI4-Lite slave, AXI DMA bridge
│   │   └── monitor/                 #   Performance counters
│   └── sim/                         # Simulation & verification
│       ├── sv/                      #   SystemVerilog testbenches (~4,200 lines)
│       └── cocotb/                  #   Python-based AXI protocol tests
│
├── sw/                              # Software stack
│   ├── cpp/                         # C++ host driver framework (~6,000 lines)
│   │   ├── include/                 #   Headers: BSR encoder, DMA, buffer mgmt
│   │   ├── src/                     #   Implementation: golden models, tiling
│   │   ├── apps/                    #   MNIST inference, benchmarking
│   │   └── tests/                   #   Unit tests
│   └── ml_python/                   # Python ML tooling (~9,000 lines)
│       ├── training/                #   MNIST CNN training, BSR export (14×14)
│       ├── exporters/               #   PyTorch → INT8 weight conversion
│       ├── golden/                  #   Bit-exact reference models
│       ├── host/                    #   PYNQ driver, AXI simulation
│       ├── demo/                    #   MNIST digit classification demo
│       └── tests/                   #   Quantization & integration tests
│
├── data/                            # Model weights & test data
│   ├── bsr_export_14x14/           #   14×14 BSR weights (production format)
│   ├── int8/                        #   Per-channel INT8 quantized weights
│   ├── checkpoints/                 #   FP32 training checkpoint
│   └── MNIST/                       #   Raw MNIST dataset
│
├── docs/                            # Documentation
│   ├── architecture/                #   Architecture specs, dataflow, BSR format
│   ├── guides/                      #   Simulation, FPGA deployment, quantization
│   └── verification/                #   Test results, verification checklist
│
└── tools/                           # Build & CI scripts
    ├── build.sh                     #   Verilator build
    ├── test.sh                      #   Test runner
    └── synthesize_vivado.tcl        #   Vivado synthesis flow
```

---

## Getting Started

### Quick Start — MNIST Digit Classifier Demo

The fastest way to see ACCEL-v1 in action. This runs the INT8-quantized CNN model
that maps onto the 14×14 systolic array, using PyTorch on CPU to simulate what the
FPGA accelerator computes.

```bash
# 1. Clone and enter the project
git clone https://github.com/joshuathomascarter/ResNet-Accel-2.git
cd ResNet-Accel-2

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the interactive digit classifier
python3 sw/ml_python/demo/classify_digit.py
```

This opens a drawing canvas. Draw any digit (0–9), click **Classify**, and the model
returns a prediction with confidence. The pre-trained checkpoint
(`data/checkpoints/mnist_fp32.pt`) is included in the repository.

**Three usage modes:**

| Mode | Command | Description |
|------|---------|-------------|
| Interactive | `python3 sw/ml_python/demo/classify_digit.py` | Draw digits on a canvas, click Classify |
| Image file | `python3 sw/ml_python/demo/classify_digit.py digit.png` | Classify a digit from an image file |
| MNIST test | `python3 sw/ml_python/demo/classify_digit.py --test 50` | Run on N random MNIST test samples |

> **Note:** The interactive drawing mode requires `tkinter`, which is included with
> most Python installations. On macOS it ships with the Homebrew/system Python.
> On Ubuntu: `sudo apt install python3-tk`.

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | ML tooling, demo, cocotb tests |
| PyTorch | 1.9+ | Model inference & training |
| Verilator | 5.x+ | RTL simulation (optional) |
| CMake | 3.16+ | C++ build system (optional) |
| Vivado | 2021.1+ | FPGA synthesis (optional) |

```bash
# macOS
brew install python cmake verilator

# Ubuntu
sudo apt install python3 python3-pip python3-tk cmake verilator

# Python packages (all platforms)
pip install -r requirements.txt
```

### Build C++ Host Driver

```bash
cd sw/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Run RTL Simulation

```bash
# SystemVerilog testbenches via Verilator
cd hw/sim
make

# Cocotb AXI protocol tests
cd hw/sim/cocotb
make
```

### Export Quantized Weights

```bash
# Export INT8 BSR weights (14×14 block format)
cd sw/ml_python/training
python3 export_bsr_14x14.py --from-int8
```

---

## Running All Tests

Below are the commands to exercise every layer of the stack. All commands
assume you are in the project root (`ResNet-Accel-2/`).

### 1. Python ML & Golden Model Tests (pytest)

These tests verify BSR INT8 GEMM correctness, weight exporter output, MAC
golden models, edge-case saturation, CSR register packing, and the AXI
host tiler integration.

```bash
source .venv/bin/activate
cd sw/ml_python
python -m pytest tests/ -v
```

Individual test files:

| File | What it tests |
|------|---------------|
| `test_golden_models.py` | BSR INT8 GEMM vs. numpy reference |
| `test_exporters.py` | BSR export format and metadata |
| `test_mac.py` | MAC8 golden model (matches RTL semantics) |
| `test_edges.py` | Zero matrices, INT8 saturation, identity |
| `test_csr_pack.py` | CSR register pack/unpack round-trips |
| `test_integration.py` | End-to-end tiled GEMM, Config serialization, AXI tiler |

### 2. C++ Host Driver Tests (CMake / CTest)

```bash
cd sw/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

### 3. RTL Simulation — Verilator

Requires [Verilator 5.x+](https://verilator.org/guide/latest/install.html).

```bash
cd hw/sim
make          # builds + runs all SV testbenches
```

Individual testbenches: `make pe_tb`, `make systolic_tb`, `make accel_top_tb`, etc.

### 4. RTL Simulation — Cocotb

Requires `cocotb` and a Verilog simulator (Verilator or Icarus).

```bash
cd hw/sim/cocotb
make                        # default top-level AXI test
make -f Makefile.accel_top  # full accel_top cocotb test
```

### 5. Yosys Synthesis (Xilinx 7-Series)

Requires [Yosys](https://github.com/YosysHQ/yosys) (any recent build).

```bash
cd hw
bash yosys_run.sh
```

This preprocesses the RTL (strips SVA assertions), maps to Xilinx 7-series
primitives, and writes a gate-level netlist to `hw/reports/yosys_netlist.json`.

### 6. Vivado Synthesis (Optional — Zynq-7020)

Requires Xilinx Vivado 2021.1+.

```bash
vivado -mode batch -source tools/synthesize_vivado.tcl
```

### 7. MNIST Demo

```bash
python3 sw/ml_python/demo/classify_digit.py           # interactive (tkinter)
python3 sw/ml_python/demo/classify_digit.py --test 50  # batch MNIST test
```

### FPGA Deployment (Zynq-7020)

```bash
# 1. Synthesize
vivado -mode batch -source tools/synthesize_vivado.tcl

# 2. Deploy to PYNQ-Z2
scp build/accel_top.bit xilinx@pynq:/home/xilinx/

# 3. Run inference
python3 sw/ml_python/host/accel.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [hw/README.md](hw/README.md) | Hardware architecture, PE diagrams, dataflow timing |
| [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) | System-level design specification |
| [docs/architecture/SPARSITY_FORMAT.md](docs/architecture/SPARSITY_FORMAT.md) | BSR sparse format and hardware FSM |
| [docs/DEEP_DIVE.md](docs/DEEP_DIVE.md) | Performance analysis, MNIST layer breakdown |
| [docs/guides/SIMULATION_GUIDE.md](docs/guides/SIMULATION_GUIDE.md) | Verilator and cocotb test setup |
| [docs/guides/QUANTIZATION_PRACTICAL.md](docs/guides/QUANTIZATION_PRACTICAL.md) | INT8 quantization methodology |
| [docs/guides/POWER_ANALYSIS.md](docs/guides/POWER_ANALYSIS.md) | Clock gating and power optimization |
| [AUDIT.md](AUDIT.md) | Internal code audit with known issues |

---

## Design Notes

### BSR Sparse Dataflow

The accelerator uses Block Sparse Row format to skip zero-weight blocks at the hardware level. The BSR scheduler FSM reads row pointers and column indices from on-chip BRAM, loads only non-zero 14×14 weight blocks into the systolic array, and streams corresponding activation tiles. At 70% block sparsity, this yields ~3× effective throughput improvement; at 90% sparsity, ~9× improvement.

### Weight-Stationary Systolic Array

Weights are loaded once per tile and held fixed in each PE while activations stream through. This minimizes weight memory bandwidth (single load per K-dimension block) at the cost of activation broadcast. For CNN inference where weight reuse is high, this dataflow is well-matched.

### INT8 Quantization Pipeline

Per-channel symmetric quantization preserves accuracy:
1. **Training** — Standard FP32 MNIST CNN (98.9% accuracy)
2. **Calibration** — Per-channel scale factors computed from weight distributions
3. **Quantization** — FP32 → INT8 with per-channel scales stored alongside weights
4. **Hardware** — INT8×INT8→INT32 MAC accumulation, requantize with saturation on output

---

## Known Limitations

- FPGA synthesis and on-board validation not yet completed
- C++ host driver is a framework with partial stub implementations
- Dual-clock wrapper (`accel_top_dual_clk.sv`) has known compilation issues
- DMA data width (64-bit) requires multi-beat transfers for 14-wide activation vectors
- See [AUDIT.md](AUDIT.md) for a detailed internal code audit

---

## References

- Y. Chen et al., "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs," ISCA 2016
- S. Han et al., "EIE: Efficient Inference Engine on Compressed Deep Neural Networks," ISCA 2016
- N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017
- Xilinx, "7 Series DSP48E1 Slice User Guide" (UG479)
- ARM, "AMBA AXI and ACE Protocol Specification" (IHI 0022E)

---

## Author

**Josh Carter** — [GitHub](https://github.com/joshuathomascarter) · joshtcarter0710@gmail.com

## License

MIT — see [LICENSE](LICENSE) for details.
