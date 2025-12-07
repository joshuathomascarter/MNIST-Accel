# ResNet-Accel Project Structure

This document describes the professional organization of the ResNet-Accel FPGA accelerator project.

## Directory Layout

```
ResNet-Accel/
├── hw/                           # Hardware Design & Verification
│   ├── rtl/                       # Register Transfer Level (RTL)
│   │   ├── buffer/                # FIFO, accumulator buffers
│   │   ├── control/               # Control path, scheduler, CSR
│   │   ├── dma/                   # DMA engines (activation, weight, BSR)
│   │   ├── host_iface/            # AXI interfaces to host
│   │   ├── mac/                   # Multiply-Accumulate (MAC8) unit
│   │   ├── meta/                  # Metadata decoder for sparse formats
│   │   ├── monitor/               # Performance counters
│   │   ├── systolic/              # Systolic array (PE grid)
│   │   └── top/                   # Top-level integration (accel_top.sv)
│   │
│   ├── sim/                       # Simulation & Testbenches
│   │   ├── sv/                    # SystemVerilog testbenches
│   │   ├── cpp/                   # C++ testbenches (Verilator)
│   │   ├── cocotb/                # Python-based testbenches (cocotb)
│   │   ├── Makefile               # Simulation build rules
│   │   └── README.md              # Simulation guide
│   │
│   ├── impl/                      # Implementation & Synthesis
│   │   ├── scripts/               # Vivado TCL scripts
│   │   └── README.md              # Implementation guide
│   │
│   └── constraints/               # Timing & I/O Constraints
│       └── *.xdc                  # Xilinx Design Constraint files
│
├── sw/                            # Software (Drivers, Models, Tests)
│   ├── exporters/                 # Model exporters (PyTorch → INT8)
│   ├── golden/                    # Golden reference models
│   ├── golden_models/             # Standalone golden models
│   ├── host/                      # Host-side driver & benchmarks
│   ├── host_axi/                  # AXI master simulator
│   ├── training/                  # Training utilities
│   ├── tests/                     # Python unit tests
│   ├── utils/                     # Utility functions
│   ├── INT8 quantization/         # Quantization scripts
│   ├── MNIST CNN/                 # MNIST training example
│   └── README.md                  # Software guide
│
├── docs/                          # Documentation
│   ├── architecture/              # Architecture specs
│   │   ├── ARCHITECTURE.md
│   │   ├── HOST_RS_TILER.md
│   │   ├── ROW_STATIONARY_DATAFLOW.md
│   │   └── SPARSITY_FORMAT.md
│   │
│   ├── guides/                    # Practical guides
│   │   ├── SIMULATION_GUIDE.md
│   │   ├── FPGA_DEPLOYMENT.md
│   │   ├── QUANTIZATION.md
│   │   └── POWER_ANALYSIS.md
│   │
│   ├── figs/                      # Diagrams & visuals
│   ├── project/                   # Project tracking docs
│   ├── verification/              # Test results & checklist
│   └── archive/                   # Deprecated docs
│
├── data/                          # Datasets & Weights
│   ├── models/                    # Pre-trained weights
│   ├── datasets/                  # MNIST, ImageNet, etc.
│   ├── fixtures/                  # Test fixtures
│   ├── int8/                      # Quantized weights
│   ├── bsr_export/                # Block-sparse weights
│   ├── checkpoints/               # Training checkpoints
│   └── MNIST/                     # MNIST raw data
│
├── tools/                         # Build Tools & Scripts
│   ├── build/                     # Build scripts (Vivado, Verilator)
│   ├── test/                      # Test runners
│   ├── ci/                        # CI/CD workflows
│   ├── run/                       # Execution scripts
│   ├── Makefile.verilator         # Verilator simulation
│   ├── build.sh                   # Main build script
│   ├── test.sh                    # Test runner
│   └── synthesize_vivado.tcl      # Vivado synthesis
│
├── build/                         # Generated Artifacts (gitignored)
│   ├── obj_dir/                   # Verilator output
│   ├── vivado/                    # Vivado project
│   └── logs/                      # Build logs
│
├── .github/                       # GitHub Configuration
│   └── workflows/                 # CI/CD pipelines
│
├── README.md                      # Project overview
├── STRUCTURE.md                   # This file
├── .gitignore                     # Git ignore rules
└── ACCEL-v1.code-workspace       # VS Code workspace config
```

## Key Components

### Hardware (hw/)
- **RTL Design**: Weight-stationary systolic array with INT8 quantization
- **Simulation**: Testbenches in SystemVerilog, C++, and Python (cocotb)
- **Constraints**: XDC files for timing and I/O mapping

### Software (sw/)
- **Exporters**: Convert PyTorch models to INT8 fixed-point
- **Golden Models**: Reference implementations for verification
- **Host Driver**: Communication with accelerator over AXI
- **Tests**: Unit tests and integration tests

### Documentation (docs/)
- Architecture specifications
- Design guides (quantization, sparsity, dataflow)
- Implementation guides
- Verification results

### Data (data/)
- Datasets (MNIST, imagenet fixtures)
- Pre-trained weights (FP32, INT8)
- Test fixtures for verification

### Tools (tools/)
- Build scripts for Vivado and Verilator
- Test runners and CI/CD integration
- Execution scripts for board deployment

## Workflow

### Development
```bash
# Simulate RTL
cd hw/sim
make -f Makefile.cocotb SIM=cocotb MODULE=test_systolic_array

# Run Python tests
cd sw/tests
python -m pytest test_*.py

# Build for FPGA
cd tools
./synthesize_vivado.tcl
```

### Deployment
```bash
# Build bitstream
cd hw/impl
vivado -mode batch -source ../../../tools/synthesize_vivado.tcl

# Run on board
cd sw/host
python accel.py --bitstream design.bit
```

## File Naming Conventions

- **RTL Files**: `*.sv` (SystemVerilog 2017+)
- **Testbenches**: `*_tb.sv` or `*_test.py`
- **Constraints**: `*.xdc`
- **Python**: `*.py` (PEP 8 compliant)
- **Build Scripts**: `Makefile`, `*.tcl`, `*.sh`

## Dependencies

- Verilog/SystemVerilog simulator (Vivado, Verilator, VCS)
- Python 3.8+
- PyTorch (for model export)
- cocotb (for Python testbenches)
- Vivado 2021.1+ (for FPGA implementation)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshuathomascarter/ResNet-Accel.git
   cd ResNet-Accel
   ```

2. **Run simulations**
   ```bash
   cd hw/sim
   make -f Makefile.cocotb
   ```

3. **Export a model**
   ```bash
   cd sw
   python exporters/export_conv.py --model resnet18 --output data/models/
   ```

4. **Synthesize for FPGA**
   ```bash
   cd tools
   ./build.sh
   ```

## Maintainers

- Josh Carter (@joshuathomascarter)

## License

See LICENSE file for details.
