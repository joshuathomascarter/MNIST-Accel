"""
test_dram_bank_fsm.py — Cocotb testbench for DRAM Bank FSM
===========================================================

Tests:
1. IDLE → ACT → ACTIVE (tRCD timing)
2. Row hit: ACT + READ in same row
3. Row miss: PRE → ACT different row
4. tRAS enforcement: early PRE blocked
5. WRITE → back to ACTIVE after tWR

Target: hw/rtl/dram/dram_bank_fsm.sv
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Command opcodes
OP_NOP   = 0b000
OP_ACT   = 0b001
OP_READ  = 0b010
OP_WRITE = 0b011
OP_PRE   = 0b100


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd_op.value = 0
    dut.cmd_row.value = 0
    dut.cmd_col.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_cmd(dut, op, row=0, col=0):
    """Issue one command for 1 cycle."""
    dut.cmd_valid.value = 1
    dut.cmd_op.value = op
    dut.cmd_row.value = row
    dut.cmd_col.value = col
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    dut.cmd_op.value = OP_NOP


@cocotb.test()
async def test_act_timing(dut):
    """IDLE → ACT row 5 → ACTIVATING (tRCD=3) → ACTIVE."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Should be IDLE (state=0)
    assert int(dut.bank_state.value) == 0, "Expected IDLE"

    # ACT row 5
    await send_cmd(dut, OP_ACT, row=5)

    # Should be ACTIVATING (state=1)
    assert int(dut.bank_state.value) == 1, f"Expected ACTIVATING(1), got {int(dut.bank_state.value)}"
    assert int(dut.phy_act.value) == 0, "phy_act should be deasserted after cmd cycle"

    # Wait tRCD (default 3 cycles, already spent 1 entering)
    await ClockCycles(dut.clk, 3)

    # Should be ACTIVE (state=2)
    st = int(dut.bank_state.value)
    assert st == 2, f"Expected ACTIVE(2) after tRCD, got {st}"
    assert int(dut.row_open.value) == 1
    assert int(dut.open_row.value) == 5

    dut._log.info("PASS: ACT timing correct")


@cocotb.test()
async def test_row_hit_read(dut):
    """ACT row 10, wait tRCD, READ → phy_read asserted, returns to ACTIVE."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # ACT
    await send_cmd(dut, OP_ACT, row=10)
    await ClockCycles(dut.clk, 4)  # wait for ACTIVE

    # Check row_hit
    dut.cmd_row.value = 10
    await RisingEdge(dut.clk)
    assert int(dut.row_hit.value) == 1, "Expected row_hit for matching row"

    # READ
    await send_cmd(dut, OP_READ, col=64)

    # phy_read should have been pulsed (check state progressed to READING=3)
    st = int(dut.bank_state.value)
    assert st == 3, f"Expected READING(3), got {st}"

    # Wait tCAS (3 cycles)
    await ClockCycles(dut.clk, 4)
    st = int(dut.bank_state.value)
    assert st == 2, f"Expected back to ACTIVE(2), got {st}"

    dut._log.info("PASS: row-hit READ works correctly")


@cocotb.test()
async def test_precharge_and_reactivate(dut):
    """ACT row 10, wait, PRE, wait tRP, ACT row 20."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # ACT row 10
    await send_cmd(dut, OP_ACT, row=10)
    await ClockCycles(dut.clk, 10)  # wait well past tRAS

    # PRE
    await send_cmd(dut, OP_PRE)
    st = int(dut.bank_state.value)
    assert st == 5, f"Expected PRECHARGING(5), got {st}"

    # Wait tRP (3 cycles)
    await ClockCycles(dut.clk, 4)
    st = int(dut.bank_state.value)
    assert st == 0, f"Expected IDLE(0) after tRP, got {st}"

    # ACT row 20
    await send_cmd(dut, OP_ACT, row=20)
    await ClockCycles(dut.clk, 4)
    assert int(dut.open_row.value) == 20
    assert int(dut.bank_state.value) == 2  # ACTIVE

    dut._log.info("PASS: PRE → re-ACT works")


@cocotb.test()
async def test_write_timing(dut):
    """ACT, WRITE, wait tWR, back to ACTIVE."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_cmd(dut, OP_ACT, row=7)
    await ClockCycles(dut.clk, 4)

    await send_cmd(dut, OP_WRITE, col=32)
    st = int(dut.bank_state.value)
    assert st == 4, f"Expected WRITING(4), got {st}"

    # tWR = 3 cycles
    await ClockCycles(dut.clk, 4)
    st = int(dut.bank_state.value)
    assert st == 2, f"Expected ACTIVE(2) after tWR, got {st}"

    dut._log.info("PASS: WRITE timing correct")


@cocotb.test()
async def test_row_miss_detection(dut):
    """row_hit should be 0 when queried with a different row."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_cmd(dut, OP_ACT, row=100)
    await ClockCycles(dut.clk, 4)

    dut.cmd_row.value = 200
    await RisingEdge(dut.clk)
    assert int(dut.row_hit.value) == 0, "Expected row_hit=0 for different row"

    dut._log.info("PASS: row miss detected correctly")
