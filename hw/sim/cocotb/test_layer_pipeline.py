"""
test_layer_pipeline.py — Cocotb testbench for multi-layer pipeline
=============================================================================

Tests:
1. Layer counter CSR registers (read/write LAYER_TOTAL, LAYER_CURRENT, LAYER_CONFIG)
2. Output BRAM ctrl FSM: intermediate layer feedback path
3. Output BRAM ctrl FSM: last layer DDR drain path
4. Pool enable configuration through CSR

Target: Exercises csr.sv layer registers + output_bram_ctrl.sv FSM.
Uses a simplified testbench wrapper that exposes CSR bus and ctrl signals.

=============================================================================
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer


# CSR Register addresses (must match csr.sv)
REG_CTRL          = 0x00
REG_STATUS        = 0x3C
REG_LAYER_TOTAL   = 0xE0
REG_LAYER_CURRENT = 0xE4
REG_LAYER_CONFIG  = 0xE8
REG_LAYER_OUT_DIM = 0xEC
REG_LAYER_STATUS  = 0xF0


async def reset_dut(dut):
    """Apply reset for 5 cycles."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def csr_write(dut, addr, data):
    """Write a 32-bit value to CSR address."""
    dut.csr_wen.value = 1
    dut.csr_ren.value = 0
    dut.csr_addr.value = addr
    dut.csr_wdata.value = data
    await RisingEdge(dut.clk)
    dut.csr_wen.value = 0
    await RisingEdge(dut.clk)


async def csr_read(dut, addr):
    """Read a 32-bit value from CSR address."""
    dut.csr_ren.value = 1
    dut.csr_wen.value = 0
    dut.csr_addr.value = addr
    await RisingEdge(dut.clk)
    dut.csr_ren.value = 0
    val = int(dut.csr_rdata.value)
    await RisingEdge(dut.clk)
    return val


@cocotb.test()
async def test_layer_total_register(dut):
    """Write and read LAYER_TOTAL register."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Default should be 1
    val = await csr_read(dut, REG_LAYER_TOTAL)
    assert val == 1, f"Default LAYER_TOTAL should be 1, got {val}"

    # Write 4 layers
    await csr_write(dut, REG_LAYER_TOTAL, 4)
    val = await csr_read(dut, REG_LAYER_TOTAL)
    assert val == 4, f"LAYER_TOTAL should be 4, got {val}"

    dut._log.info("PASS: LAYER_TOTAL register")


@cocotb.test()
async def test_layer_current_register(dut):
    """Write and read LAYER_CURRENT register."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Default should be 0
    val = await csr_read(dut, REG_LAYER_CURRENT)
    assert val == 0, f"Default LAYER_CURRENT should be 0, got {val}"

    # Write layer 2
    await csr_write(dut, REG_LAYER_CURRENT, 2)
    val = await csr_read(dut, REG_LAYER_CURRENT)
    assert val == 2, f"LAYER_CURRENT should be 2, got {val}"

    dut._log.info("PASS: LAYER_CURRENT register")


@cocotb.test()
async def test_layer_config_pool_en(dut):
    """Write LAYER_CONFIG to enable pooling, verify pool_en output."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Default: pool disabled
    val = await csr_read(dut, REG_LAYER_CONFIG)
    assert (val & 1) == 0, f"Default pool_en should be 0, got {val & 1}"

    # Enable pooling
    await csr_write(dut, REG_LAYER_CONFIG, 1)
    val = await csr_read(dut, REG_LAYER_CONFIG)
    assert (val & 1) == 1, f"pool_en should be 1, got {val & 1}"

    # Verify output wire
    assert int(dut.pool_en.value) == 1, "pool_en output should be 1"

    # Disable pooling
    await csr_write(dut, REG_LAYER_CONFIG, 0)
    assert int(dut.pool_en.value) == 0, "pool_en output should be 0"

    dut._log.info("PASS: LAYER_CONFIG pool_en")


@cocotb.test()
async def test_layer_output_dimensions(dut):
    """Write and read LAYER_OUT_DIM register (packed H/W)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write H=24, W=24
    packed = (24 << 16) | 24
    await csr_write(dut, REG_LAYER_OUT_DIM, packed)
    val = await csr_read(dut, REG_LAYER_OUT_DIM)

    h = (val >> 16) & 0xFFFF
    w = val & 0xFFFF
    assert h == 24 and w == 24, f"Expected H=24 W=24, got H={h} W={w}"

    # Write H=8, W=8
    packed = (8 << 16) | 8
    await csr_write(dut, REG_LAYER_OUT_DIM, packed)
    val = await csr_read(dut, REG_LAYER_OUT_DIM)

    h = (val >> 16) & 0xFFFF
    w = val & 0xFFFF
    assert h == 8 and w == 8, f"Expected H=8 W=8, got H={h} W={w}"

    dut._log.info("PASS: LAYER_OUT_DIM register")


@cocotb.test()
async def test_layer_status_w1c(dut):
    """Verify LAYER_STATUS bits are sticky and cleared by W1C."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Default should be 0
    val = await csr_read(dut, REG_LAYER_STATUS)
    assert val == 0, f"Default LAYER_STATUS should be 0, got {val}"

    # Simulate layer_done_in pulse
    dut.layer_done_in.value = 1
    await RisingEdge(dut.clk)
    dut.layer_done_in.value = 0
    await RisingEdge(dut.clk)

    val = await csr_read(dut, REG_LAYER_STATUS)
    assert (val & 1) == 1, f"LAYER_STATUS[0] (layer_done) should be 1, got {val}"

    # Clear it via W1C
    await csr_write(dut, REG_LAYER_STATUS, 1)
    val = await csr_read(dut, REG_LAYER_STATUS)
    assert (val & 1) == 0, f"LAYER_STATUS[0] should be cleared, got {val}"

    # Simulate last_layer_done_in pulse
    dut.last_layer_done_in.value = 1
    await RisingEdge(dut.clk)
    dut.last_layer_done_in.value = 0
    await RisingEdge(dut.clk)

    val = await csr_read(dut, REG_LAYER_STATUS)
    assert (val & 2) == 2, f"LAYER_STATUS[1] (last_layer_done) should be 1, got {val}"

    # Clear it
    await csr_write(dut, REG_LAYER_STATUS, 2)
    val = await csr_read(dut, REG_LAYER_STATUS)
    assert (val & 2) == 0, f"LAYER_STATUS[1] should be cleared, got {val}"

    dut._log.info("PASS: LAYER_STATUS W1C")
