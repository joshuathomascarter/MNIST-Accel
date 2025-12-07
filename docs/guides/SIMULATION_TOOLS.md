# Simulation Tools Guide

**Author**: Joshua Carter  
**Date**: November 19, 2025  
**Tools**: Verilator 5.0+, ModelSim, iverilog

---

## Overview

ACCEL-v1 supports three simulation workflows:

1. **Verilator** (preferred) - Fast C++ co-simulation, industry-standard linting
2. **ModelSim/Questa** - Full SystemVerilog support, waveform debugging
3. **iverilog** (legacy) - Lightweight open-source fallback

---

## 1. Verilator Workflow (Recommended)

### Installation

```bash
# Ubuntu/Debian
sudo apt install verilator

# Verify version (need 5.0+)
verilator --version
```

### Quick Start

```bash
# Lint all RTL files
make -f Makefile.verilator lint

# Build simulation executable
make -f Makefile.verilator sim

# Run simulation
make -f Makefile.verilator run

# View waveforms
gtkwave build/verilator/trace.vcd
```

### Coverage Analysis

```bash
# Build with coverage enabled
make -f Makefile.verilator coverage

# Run simulation
./build/verilator/Vaccel_top

# Generate coverage report
verilator_coverage logs/*.dat --annotate coverage_html/

# View in browser
firefox coverage_html/index.html
```

### Advantages

- ✅ **Fast**: 10-100× faster than event-driven simulators
- ✅ **Lint**: Industry-grade static analysis (catches 90% of bugs pre-sim)
- ✅ **Coverage**: Line, toggle, functional coverage built-in
- ✅ **C++ Integration**: Easy testbench development
- ✅ **CI/CD**: Automated regression testing

### Limitations

- ❌ No 4-state logic (X/Z) - catches uninitialized signals
- ❌ Limited SystemVerilog constructs (no `$random`, some assertions)
- ❌ Timing checks less accurate than gate-level sim

---

## 2. ModelSim Workflow

### Installation

```bash
# Intel/Altera Quartus includes ModelSim (free with license)
# Or purchase Mentor ModelSim SE/DE/PE

# Verify installation
vsim -version
```

### Compile Order

**Critical**: SystemVerilog modules must be compiled in dependency order.

```tcl
# ModelSim compile script (compile_modelsim.tcl)
vlib work

# 1. Leaf modules (no dependencies)
vlog -sv rtl/mac/mac8.sv
vlog -sv rtl/systolic/pe.sv
vlog -sv rtl/uart/uart_tx.sv
vlog -sv rtl/uart/uart_rx.sv
vlog -sv rtl/meta/meta_decode.sv

# 2. Buffers
vlog -sv rtl/buffer/act_buffer.sv
vlog -sv rtl/buffer/wgt_buffer.sv

# 3. Systolic array (depends on pe.sv)
vlog -sv rtl/systolic/systolic_array.sv
vlog -sv rtl/systolic/systolic_array_sparse.sv

# 4. Control modules
vlog -sv rtl/control/csr.sv
vlog -sv rtl/control/scheduler.sv
vlog -sv rtl/control/block_reorder_buffer.sv
vlog -sv rtl/control/bsr_scheduler.sv
vlog -sv rtl/control/multi_layer_buffer.sv

# 5. DMA
vlog -sv rtl/dma/dma_lite.sv
vlog -sv rtl/dma/bsr_dma.sv
vlog -sv rtl/dma/axi_dma_master.sv

# 6. Host interfaces
vlog -sv rtl/host_iface/axi_lite_slave.sv
vlog -sv rtl/host_iface/axi_lite_slave_v2.sv
vlog -sv rtl/host_iface/axi_dma_bridge.sv

# 7. Monitor
vlog -sv rtl/monitor/perf.sv

# 8. Top-level (depends on everything)
vlog -sv rtl/top/top_sparse.sv
vlog -sv rtl/top/accel_top.sv

# 9. Testbenches
vlog -sv testbench/integration/test_accel_top.cpp
```

### Run Simulation

```tcl
# Launch GUI
vsim -gui work.accel_top

# Add waveforms
add wave -radix hex /accel_top/*
add wave -radix unsigned /accel_top/systolic_inst/c_out_flat

# Run for 10 µs
run 10us

# Export waveform
write wave -format vcd sim.vcd
```

### Advantages

- ✅ **Full SystemVerilog**: All constructs supported
- ✅ **Debugging**: Interactive GUI, breakpoints, signal inspection
- ✅ **Assertions**: Complete SVA support with bind
- ✅ **Timing**: Accurate delay modeling

### Limitations

- ❌ **Slow**: Event-driven simulation (1000× slower than Verilator)
- ❌ **License Cost**: Commercial tool ($$$)
- ❌ **CI/CD**: Harder to automate than Verilator

---

## 3. iverilog Workflow (Legacy)

### Installation

```bash
sudo apt install iverilog gtkwave
```

### Compile & Run

```bash
# Compile (all files must be listed)
iverilog -g2012 -o build/sim.vvp \
    rtl/mac/mac8.sv \
    rtl/systolic/pe.sv \
    rtl/systolic/systolic_array.sv \
    rtl/buffer/act_buffer.sv \
    rtl/buffer/wgt_buffer.sv \
    rtl/control/scheduler.sv \
    rtl/top/accel_top.sv \
    testbench/tb_accel_top.v

# Run simulation
vvp build/sim.vvp

# View waveform
gtkwave build/accel_top.vcd
```

### Advantages

- ✅ **Free**: Truly open-source
- ✅ **Lightweight**: Small footprint
- ✅ **Portable**: Works on Linux/macOS/Windows

### Limitations

- ❌ **Incomplete SystemVerilog**: Many constructs unsupported (interfaces, classes)
- ❌ **Slow**: Slower than Verilator
- ❌ **No Lint**: Minimal static analysis
- ❌ **Limited Coverage**: No built-in coverage tools

---

## 4. Comparison Matrix

| Feature | Verilator | ModelSim | iverilog |
|---------|-----------|----------|----------|
| **Speed** | ⭐⭐⭐⭐⭐ (10-100×) | ⭐⭐ (baseline) | ⭐⭐⭐ (2-5×) |
| **SystemVerilog Support** | ⭐⭐⭐ (80%) | ⭐⭐⭐⭐⭐ (100%) | ⭐⭐ (40%) |
| **Lint Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Coverage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **Waveform Debug** | ⭐⭐⭐ (VCD only) | ⭐⭐⭐⭐⭐ (GUI) | ⭐⭐⭐ (GTKWave) |
| **Cost** | Free | $$$$ | Free |
| **CI/CD** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Recommendation**:
- **Development**: Verilator (fast iteration, lint catches bugs early)
- **Debug**: ModelSim (interactive GUI when Verilator fails)
- **CI/CD**: Verilator (regression tests, coverage)

---

## 5. Regression Testing Strategy

### Verilator + CTest

```cmake
# CMakeLists.txt
add_test(NAME systolic_basic
    COMMAND Vaccel_top --test=systolic_2x2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME sparse_bsr
    COMMAND Vaccel_top --test=sparse_90_percent
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Coverage requirement: 85% line coverage
set_tests_properties(systolic_basic PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED")
```

### GitHub Actions CI

```yaml
# .github/workflows/verilator-ci.yml
name: RTL Verification
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Verilator
        run: sudo apt install verilator
      - name: Lint RTL
        run: make -f Makefile.verilator lint

  simulate:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Simulation
        run: make -f Makefile.verilator coverage
      - name: Run Tests
        run: ./build/verilator/Vaccel_top
      - name: Check Coverage
        run: |
          verilator_coverage logs/*.dat --write filtered.dat
          coverage=$(grep -oP 'Coverage: \K[0-9.]+' filtered.dat)
          if (( $(echo "$coverage < 85" | bc -l) )); then
            echo "Coverage $coverage% below 85% threshold"
            exit 1
          fi
```

---

## 6. Known Issues & Workarounds

### Issue 1: Verilator doesn't support `interface`

**Workaround**: Use explicit port connections instead of interfaces.

```systemverilog
// ❌ Don't use (Verilator unsupported)
axi4_if axi_bus();

// ✅ Use explicit signals
logic [31:0] axi_awaddr;
logic        axi_awvalid;
logic        axi_awready;
```

### Issue 2: ModelSim compile order errors

**Symptom**: `Error: (vlog-2110) Illegal reference to net "signal_name"`

**Fix**: Ensure modules are compiled before they're instantiated. Use `vlog -work work +incdir+rtl` to add include paths.

### Issue 3: iverilog doesn't support `always_ff`

**Workaround**: Use `always @(posedge clk)` instead.

```systemverilog
// ❌ iverilog unsupported
always_ff @(posedge clk or negedge rst_n) begin

// ✅ Compatible with iverilog
always @(posedge clk or negedge rst_n) begin
```

---

## 7. Best Practices

1. **Lint first**: Run `make lint` before every commit
2. **Coverage goals**: Target 85% line coverage, 70% toggle coverage
3. **Waveform discipline**: Only save critical signals (reduces VCD size 100×)
4. **Test isolation**: Each module should have standalone testbench
5. **Assertion hygiene**: Use `ifdef FORMAL` guards for synthesis-incompatible SVA

---

## 8. References

- [Verilator User Guide](https://verilator.org/guide/latest/)
- [ModelSim User Manual](https://www.intel.com/content/www/us/en/docs/programmable/quartus-prime/current/sim/modelsim-support.html)
- [iverilog Documentation](https://steveicarus.github.io/iverilog/)
- [SystemVerilog LRM](https://ieeexplore.ieee.org/document/8299595)

---

**Next Steps**:
1. Run `make -f Makefile.verilator lint` - fix all warnings
2. Create C++ testbench in `testbench/verilator/tb_accel_top.cpp`
3. Set up GitHub Actions CI for automated regression
4. Target 90% coverage before hardware deployment
