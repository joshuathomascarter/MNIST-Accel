# Cocotb Testing Guide

## Overview

ACCEL-v1 uses [cocotb](https://www.cocotb.org/) for Python-driven RTL verification. Tests are located in `hw/sim/cocotb/` and drive the SystemVerilog DUT through AXI4-Lite CSR writes and AXI4 DMA transactions.

## Directory Structure

```
hw/sim/cocotb/
    Makefile                    # Default cocotb Makefile
    Makefile.accel_top          # Top-level integration Makefile
    Makefile.cocotb             # Common cocotb settings
    test_accel_top.py           # Top-level integration tests
    cocotb_axi_master_test.py   # AXI master interface tests
    sim_build/                  # Verilator build artifacts
    results.xml                 # JUnit test results
```

## Prerequisites

```bash
pip install cocotb cocotb-bus
```

Verilator (>= 5.0) or Icarus Verilog must be installed for simulation.

## Running Tests

### Full Test Suite

```bash
cd hw/sim/cocotb
make -f Makefile.accel_top
```

### Individual Test

```bash
cd hw/sim/cocotb
make -f Makefile.accel_top TESTCASE=test_csr_read_write
```

## Writing Tests

### Basic Structure

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

@cocotb.test()
async def test_csr_read_write(dut):
    """Verify CSR register read/write via AXI4-Lite."""
    clock = Clock(dut.clk, 5, units="ns")  # 200 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    await Timer(100, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Write TILE_CFG register (offset 0x20)
    await axi_lite_write(dut, 0x20, 0x000E000E)  # M=14, N=14

    # Read back
    val = await axi_lite_read(dut, 0x20)
    assert val == 0x000E000E, f"CSR mismatch: {val:#x}"
```

### AXI4-Lite Helper Functions

```python
async def axi_lite_write(dut, addr, data):
    """Write a 32-bit value to an AXI4-Lite register."""
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_wvalid.value = 1
    dut.s_axi_bready.value = 1
    await RisingEdge(dut.clk)
    while not dut.s_axi_bvalid.value:
        await RisingEdge(dut.clk)
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0

async def axi_lite_read(dut, addr):
    """Read a 32-bit value from an AXI4-Lite register."""
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 1
    await RisingEdge(dut.clk)
    while not dut.s_axi_rvalid.value:
        await RisingEdge(dut.clk)
    dut.s_axi_arvalid.value = 0
    return dut.s_axi_rdata.value.integer
```

### Makefile Configuration

```makefile
# Makefile.accel_top
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(shell find ../../rtl -name "*.sv")
TOPLEVEL = accel_top
MODULE = test_accel_top
SIM = verilator

include $(shell cocotb-config --makefiles)/Makefile.sim
```

## Test Categories

| Category | File | Description |
|----------|------|-------------|
| CSR | `test_accel_top.py` | Register read/write, status polling |
| DMA | `test_accel_top.py` | AXI4 burst transfers, activation/weight loading |
| BSR | `test_accel_top.py` | BSR scheduler, sparse block processing |
| AXI | `cocotb_axi_master_test.py` | AXI master protocol compliance |

## Analyzing Results

Test results are written to `results.xml` (JUnit format). Waveform traces are saved to `dump.vcd` and can be viewed with GTKWave:

```bash
gtkwave hw/sim/cocotb/dump.vcd
```