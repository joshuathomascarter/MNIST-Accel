# Project Structure

```
ACCEL-v1/
├── hw/                              # Hardware (RTL + Verification)
│   ├── rtl/                         # Production SystemVerilog (21 modules)
│   │   ├── top/                     #   accel_top.sv, accel_top_dual_clk.sv
│   │   ├── systolic/                #   pe.sv, systolic_array.sv, systolic_array_sparse.sv
│   │   ├── mac/                     #   mac8.sv (INT8 MAC, zero-bypass)
│   │   ├── control/                 #   bsr_scheduler.sv, csr.sv, CDC primitives
│   │   ├── dma/                     #   act_dma.sv, bsr_dma.sv, out_dma.sv
│   │   ├── buffer/                  #   act_buffer.sv, wgt_buffer.sv, output_accumulator.sv
│   │   ├── host_iface/              #   axi_lite_slave.sv, axi_dma_bridge.sv
│   │   └── monitor/                 #   perf.sv (performance counters)
│   ├── rtl_synth/                   # Synthesis-ready copies (stripped assertions)
│   ├── sim/                         # Simulation
│   │   ├── sv/                      #   SystemVerilog testbenches (10 files)
│   │   └── cocotb/                  #   Python cocotb AXI tests
│   └── reports/                     # Synthesis reports
│
├── sw/                              # Software
│   ├── cpp/                         # C++ host driver framework
│   │   ├── include/                 #   Headers (compute, driver, memory, utils)
│   │   ├── src/                     #   Implementation
│   │   ├── apps/                    #   Applications (inference, benchmark)
│   │   └── tests/                   #   Unit tests
│   └── ml_python/                   # Python ML tooling
│       ├── training/                #   Model training, BSR export
│       ├── exporters/               #   Weight format converters
│       ├── golden/                  #   Bit-exact reference models
│       ├── golden_models/           #   Standalone golden models
│       ├── host/                    #   PYNQ driver, AXI simulation
│       ├── host_axi/               #   AXI CSR interface
│       ├── demo/                    #   MNIST classification demo
│       ├── tests/                   #   Python tests
│       ├── INT8 quantization/       #   Post-training quantization
│       └── MNIST CNN/               #   CNN training
│
├── data/                            # Model weights & datasets
│   ├── bsr_export_14x14/           #   14×14 BSR weights (row_ptr, col_idx, blocks)
│   ├── bsr_export/                  #   BSR export (28×28 input)
│   ├── int8/                        #   INT8 quantized weights + scales
│   ├── checkpoints/                 #   FP32 training checkpoints
│   └── MNIST/                       #   Raw MNIST dataset
│
├── docs/                            # Documentation
│   ├── architecture/                #   Architecture specs, dataflow, BSR format
│   ├── guides/                      #   Simulation, deployment, quantization guides
│   ├── verification/                #   Test results, verification checklist
│   └── figs/                        #   Diagrams
│
├── tools/                           # Build & CI
│   ├── build.sh                     #   Verilator build script
│   ├── test.sh                      #   Test runner
│   ├── synthesize_vivado.tcl        #   Vivado synthesis
│   └── run/                         #   Execution scripts
│
├── README.md                        # Project overview
├── STRUCTURE.md                     # This file
└── AUDIT.md                         # Internal code audit
```

## Conventions

- **RTL**: SystemVerilog 2017+, all modules in `hw/rtl/`
- **Testbenches**: `*_tb.sv` (SystemVerilog) or `test_*.py` (cocotb/pytest)
- **Parameters**: Array size and block size are 14×14 throughout
- **Data format**: INT8 weights/activations, INT32 accumulators, BSR sparse encoding
- **Build**: CMake for C++, Make for RTL simulation, Vivado TCL for synthesis