# Testbench Directory

This directory contains all verification code for ACCEL-v1, organized by language/framework.

## Directory Structure

```
testbench/
├── sv/           # Pure SystemVerilog testbenches (Verilator)
├── cpp/          # C++ Verilator test harnesses
├── cocotb/       # Python CocoTB tests
├── Makefile      # Top-level build
└── CMakeLists.txt
```

## Testbench Categories

### SystemVerilog (`sv/`)

Self-contained SV testbenches that run with Verilator `--timing --main`:

| File | Tests | Description |
|------|-------|-------------|
| `pe_tb.sv` | 6 | Processing element: MAC, weight load |
| `systolic_tb.sv` | 9 | 16×16 systolic array dataflow |
| `bsr_dma_tb.sv` | 5 | BSR DMA: multi-block, backpressure |
| `meta_decode_tb.sv` | 4 | BSR metadata parsing |
| `perf_tb.sv` | 3 | Performance counter verification |
| `output_accumulator_tb.sv` | 7 | Double-buffer, ReLU, quantization |
| `accel_top_tb.sv` | Basic | Simple top-level smoke test |
| `accel_top_tb_full.sv` | 27 | Comprehensive top-level tests |
| `integration_tb.sv` | 5 | Full datapath integration |

### C++ (`cpp/`)

Verilator C++ test harnesses for detailed control:

| File | Description |
|------|-------------|
| `test_mac8.cpp` | MAC unit exhaustive testing |
| `test_pe.cpp` | PE unit with random inputs |
| `test_csr.cpp` | CSR register edge cases |
| `test_systolic_array.cpp` | Array timing verification |
| `test_stress.cpp` | Random stress testing |
| `test_throughput.cpp` | Performance measurement |
| `test_latency.cpp` | Pipeline latency analysis |

### CocoTB (`cocotb/`)

Python-based AXI protocol tests:

| File | Description |
|------|-------------|
| `test_accel_top.py` | AXI-Lite CSR read/write |
| `cocotb_axi_master_test.py` | DMA protocol verification |

## Running Tests

### All Tests
```bash
cd /path/to/ACCEL-v1
./scripts/test.sh
```

### Individual SV Testbench
```bash
# Build and run pe_tb
verilator --sv --cc --exe --build --trace --timing --main \
    -Wall -Wno-fatal -Wno-MULTIDRIVEN \
    -I rtl/mac -I rtl/systolic \
    testbench/sv/pe_tb.sv \
    -o build/pe_tb

./build/pe_tb
```

### With Coverage
```bash
verilator --sv --cc --exe --build --trace --timing --main \
    --coverage --coverage-line \
    -Wall -Wno-fatal -Wno-MULTIDRIVEN \
    -I rtl/... \
    testbench/sv/accel_top_tb_full.sv \
    -o build/accel_top_tb_cov

./build/accel_top_tb_cov
verilator_coverage --annotate build/coverage_annotate coverage.dat
```

### CocoTB Tests
```bash
cd testbench/cocotb
make -f Makefile.cocotb
```

## Coverage Goals

| Target | Current | Goal |
|--------|---------|------|
| Line Coverage | 71% | 90% |
| Branch Coverage | ~60% | 80% |
| FSM Coverage | ~80% | 95% |

### Uncovered Areas

1. **CSR error paths** — Invalid burst types, write strobe patterns
2. **DMA timeout** — Error recovery, underflow handling
3. **AXI protocol edge cases** — Interleaved transactions

## Adding New Tests

### SystemVerilog
1. Create `testbench/sv/<module>_tb.sv`
2. Use `$display` for test output, `$finish` to end
3. Add to `scripts/test.sh`

### C++ Verilator
1. Create `testbench/cpp/test_<module>.cpp`
2. Include Verilator headers, instantiate DUT
3. Add to CMakeLists.txt

### CocoTB
1. Create `testbench/cocotb/test_<feature>.py`
2. Use cocotb decorators (`@cocotb.test()`)
3. Add to Makefile.cocotb
