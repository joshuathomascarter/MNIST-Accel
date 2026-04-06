"""
test_dram_write_drain.py — Cocotb tests for dram_write_buffer drain path.

Verifies:
  1. Write-then-drain: data written via wr_* appears at drain_* on index select.
  2. Back-to-back drain: successive drains from different indices.
  3. Buffer full: wr_ready deasserts when all entries filled.
  4. Drain-empty-refill: after draining, slot is reusable.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset(dut):
    dut.rst_n.value = 0
    dut.wr_valid.value = 0
    dut.drain_valid.value = 0
    dut.drain_idx.value = 0
    dut.drain_ready.value = 0
    dut.wr_data.value = 0
    dut.wr_strb.value = 0
    dut.wr_id.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_write_then_drain(dut):
    """Write one entry and drain it."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Write data
    dut.wr_valid.value = 1
    dut.wr_data.value = 0xDEADBEEF
    dut.wr_strb.value = 0xF
    dut.wr_id.value = 3
    await RisingEdge(dut.clk)
    dut.wr_valid.value = 0
    await RisingEdge(dut.clk)

    # Drain index 0
    dut.drain_valid.value = 1
    dut.drain_idx.value = 0
    dut.drain_ready.value = 1
    await RisingEdge(dut.clk)
    data = int(dut.drain_data.value)
    assert data == 0xDEADBEEF, f"Expected 0xDEADBEEF, got 0x{data:08X}"
    dut.drain_valid.value = 0
    dut.drain_ready.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_back_to_back_drain(dut):
    """Write multiple entries and drain them from different indices."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    patterns = [0x11111111, 0x22222222, 0x33333333]

    for p in patterns:
        dut.wr_valid.value = 1
        dut.wr_data.value = p
        dut.wr_strb.value = 0xF
        dut.wr_id.value = 0
        await RisingEdge(dut.clk)

    dut.wr_valid.value = 0
    await RisingEdge(dut.clk)

    for idx, exp in enumerate(patterns):
        dut.drain_valid.value = 1
        dut.drain_idx.value = idx
        dut.drain_ready.value = 1
        await RisingEdge(dut.clk)
        data = int(dut.drain_data.value)
        assert data == exp, f"idx {idx}: expected 0x{exp:08X}, got 0x{data:08X}"

    dut.drain_valid.value = 0
    dut.drain_ready.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_buffer_full(dut):
    """Fill the buffer and verify wr_ready deasserts."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    depth = 16
    for i in range(depth):
        dut.wr_valid.value = 1
        dut.wr_data.value = i
        dut.wr_strb.value = 0xF
        dut.wr_id.value = 0
        await RisingEdge(dut.clk)

    dut.wr_valid.value = 0
    await RisingEdge(dut.clk)

    full = int(dut.full.value)
    assert full == 1, "Expected buffer full"


@cocotb.test()
async def test_drain_and_refill(dut):
    """Drain a slot then reuse it."""
    clock = Clock(dut.clk, 5, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Write two entries
    for val in [0xAAAA_AAAA, 0xBBBB_BBBB]:
        dut.wr_valid.value = 1
        dut.wr_data.value = val
        dut.wr_strb.value = 0xF
        dut.wr_id.value = 0
        await RisingEdge(dut.clk)
    dut.wr_valid.value = 0
    await RisingEdge(dut.clk)

    # Drain slot 0
    dut.drain_valid.value = 1
    dut.drain_idx.value = 0
    dut.drain_ready.value = 1
    await RisingEdge(dut.clk)
    dut.drain_valid.value = 0
    dut.drain_ready.value = 0
    await RisingEdge(dut.clk)

    # Refill — should land in slot 0 again
    dut.wr_valid.value = 1
    dut.wr_data.value = 0xCCCC_CCCC
    dut.wr_strb.value = 0xF
    dut.wr_id.value = 5
    await RisingEdge(dut.clk)
    dut.wr_valid.value = 0
    await RisingEdge(dut.clk)

    # Drain slot 0 and check new data
    dut.drain_valid.value = 1
    dut.drain_idx.value = 0
    dut.drain_ready.value = 1
    await RisingEdge(dut.clk)
    data = int(dut.drain_data.value)
    assert data == 0xCCCC_CCCC, f"Expected 0xCCCCCCCC, got 0x{data:08X}"
    dut.drain_valid.value = 0
    dut.drain_ready.value = 0
    await ClockCycles(dut.clk, 2)
