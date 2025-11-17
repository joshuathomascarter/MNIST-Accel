# ACCEL-v1: Sparse Neural Network Accelerator

A high-performance FPGA-based accelerator for sparse neural networks using Block Sparse Row (BSR) format with INT8 quantization and row-stationary dataflow.

## 📁 Project Structure

```
ACCEL-v1/
├── accel/                    # Core accelerator implementation
│   ├── python/              # Host software & training
│   ├── data/                # Training data & weights
│   └── scripts/             # Helper scripts
│
├── rtl/                     # Verilog/SystemVerilog RTL
│   ├── top/                 # Top-level integration
│   ├── host_iface/          # AXI4-Lite + DMA communication
│   ├── systolic/            # Systolic array (sparse & dense)
│   ├── dma/                 # DMA engines (BSR & dense)
│   └── control/             # Control logic & CSRs
│
├── testbench/               # Verification infrastructure
│   ├── unit/                # Per-module testbenches
│   ├── integration/         # System-level tests
│   ├── cocotb/             # Python/Verilog co-simulation
│   └── verilator/          # C++ Verilator tests
│
├── docs/                    # Documentation
│   ├── architecture/        # Design documentation
│   ├── verification/        # Test & verification docs
│   ├── guides/             # How-to guides
│   └── project/            # Project management
│
├── scripts/                 # Build & test automation
│   ├── build.sh            # Unified build script
│   ├── test.sh             # Unified test runner
│   └── ci/                 # CI/CD scripts
│
└── build/                   # Generated files (gitignored)
    ├── sim/                 # Simulation outputs
    ├── synth/              # Synthesis outputs
    └── logs/               # Build & test logs
```

## 🚀 Quick Start

### Build Everything
```bash
./scripts/build.sh
```

### Run All Tests
```bash
./scripts/test.sh
```

### Run Specific Tests
```bash
./scripts/test.sh python     # Python AXI simulator
./scripts/test.sh verilog    # Verilog testbench
./scripts/test.sh cocotb     # Cocotb integration
```

## 🔧 Key Features

- **Sparse Acceleration**: BSR format with 8×8 blocks
- **INT8 Quantization**: Per-channel quantization for weights & activations
- **Row-Stationary Dataflow**: Optimized for sparse matrix operations
- **AXI4-Lite Interface**: CSR-based control from host
- **AXI4 Burst DMA**: High-bandwidth weight loading
- **Dual Communication**: UART (debug) + AXI (performance)

## 📚 Documentation

See [`docs/`](docs/) for complete documentation:
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [Verification Guide](docs/verification/VERIFICATION.md)
- [AXI Communication](docs/guides/COCOTB_TESTING_GUIDE.md)
- [Quantization Guide](docs/guides/QUANTIZATION_PRACTICAL.md)

## 🎯 Hardware Targets

- **Simulation**: Icarus Verilog, Verilator
- **FPGA**: Xilinx 7-series (Artix-7, Zynq)
- **Clock**: 100 MHz target

## 📊 Status

- ✅ RTL implementation complete
- ✅ Python AXI simulator (100% tests passing)
- ✅ Verilog testbench (82% tests passing)
- ✅ INT8 quantization training pipeline
- 🔄 FPGA synthesis & deployment

## 🤝 Contributing

This is a research/educational project. See individual module READMEs for implementation details.

## 📄 License

[Add license information]

---

**Author**: Joshua Carter  
**Repository**: https://github.com/joshuathomascarter/ACCEL-v1
