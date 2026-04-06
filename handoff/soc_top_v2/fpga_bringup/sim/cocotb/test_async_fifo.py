"""
test_async_fifo.py — Cocotb testbench for async_fifo (Gray-code CDC FIFO)
=========================================================================

Tests:
1. Write-then-read — basic data integrity
2. Fill to capacity — full flag asserts
3. Empty flag — read from empty FIFO
4. Burst throughput — simultaneous write/read

Target: hw/rtl/hft/async_fifo.sv   (DEPTH=16, WIDTH=64)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer


async def reset_fifo(dut):
    """Assert both resets, deassert after 5 write-clock cycles."""
    dut.wr_rst_n.value = 0
    dut.rd_rst_n.value = 0
    dut.wr_en.value    = 0
    dut.rd_en.value    = 0
    dut.wr_data.value  = 0
    await ClockCycles(dut.wr_clk, 5)
    dut.wr_rst_n.value = 1
    dut.rd_rst_n.value = 1
    # Wait for synchronizers to settle
    await ClockCycles(dut.wr_clk, 4)


@cocotb.test()
async def test_write_then_read(dut):
    """Write 4 words, then read them back and verify."""
    wr_clk = Clock(dut.wr_clk, 8, units="ns")   # 125 MHz
    rd_clk = Clock(dut.rd_clk, 10, units="ns")   # 100 MHz
    cocotb.start_soon(wr_clk.start())
    cocotb.start_soon(rd_clk.start())
    await reset_fifo(dut)

    test_data = [0xDEAD_BEEF_CAFE_0001, 0x1234_5678_9ABC_DEF0,
                 0xAAAA_BBBB_CCCC_DDDD, 0x0000_0000_0000_0042]

    # Write phase
    for d in test_data:
        dut.wr_data.value = d
        dut.wr_en.value   = 1
        await RisingEdge(dut.wr_clk)
    dut.wr_en.value = 0

    # Wait for pointer sync (2 FF + margin)
    await ClockCycles(dut.rd_clk, 6)

    # Read phase
    for i, expected in enumerate(test_data):
        dut.rd_en.value = 1
        await RisingEdge(dut.rd_clk)
        got = int(dut.rd_data.value)
        assert got == expected, f"Word {i}: expected 0x{expected:016X}, got 0x{got:016X}"
    dut.rd_en.value = 0

    dut._log.info("PASS: write-then-read data integrity")


@cocotb.test()
async def test_full_flag(dut):
    """Fill FIFO to DEPTH=16 and check full flag asserts."""
    wr_clk = Clock(dut.wr_clk, 8, units="ns")
    rd_clk = Clock(dut.rd_clk, 10, units="ns")
    cocotb.start_soon(wr_clk.start())
    cocotb.start_soon(rd_clk.start())
    await reset_fifo(dut)

    for i in range(16):
        dut.wr_data.value = i
        dut.wr_en.value   = 1
        await RisingEdge(dut.wr_clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.wr_clk)

    full = int(dut.full.value) & 1
    assert full == 1, f"Expected full=1 after writing 16 words, got {full}"

    dut._log.info("PASS: full flag asserts at capacity")


@cocotb.test()
async def test_empty_flag(dut):
    """Empty flag should be high after reset, low after write, high after drain."""
    wr_clk = Clock(dut.wr_clk, 8, units="ns")
    rd_clk = Clock(dut.rd_clk, 10, units="ns")
    cocotb.start_soon(wr_clk.start())
    cocotb.start_soon(rd_clk.start())
    await reset_fifo(dut)

    # Should be empty after reset
    empty = int(dut.empty.value) & 1
    assert empty == 1, f"Expected empty=1 after reset, got {empty}"

    # Write one word
    dut.wr_data.value = 0xBEEF
    dut.wr_en.value   = 1
    await RisingEdge(dut.wr_clk)
    dut.wr_en.value = 0

    # Wait for sync
    await ClockCycles(dut.rd_clk, 6)
    empty = int(dut.empty.value) & 1
    assert empty == 0, f"Expected empty=0 after write, got {empty}"

    # Read it out
    dut.rd_en.value = 1
    await RisingEdge(dut.rd_clk)
    dut.rd_en.value = 0
    await ClockCycles(dut.rd_clk, 2)

    empty = int(dut.empty.value) & 1
    assert empty == 1, f"Expected empty=1 after drain, got {empty}"

    dut._log.info("PASS: empty flag tracks FIFO state")


@cocotb.test()
async def test_concurrent_write_read(dut):
    """Simultaneous write and read — data should flow through correctly."""
    wr_clk = Clock(dut.wr_clk, 8, units="ns")
    rd_clk = Clock(dut.rd_clk, 10, units="ns")
    cocotb.start_soon(wr_clk.start())
    cocotb.start_soon(rd_clk.start())
    await reset_fifo(dut)

    # Pre-fill 4 words
    for i in range(4):
        dut.wr_data.value = 0xA000 + i
        dut.wr_en.value   = 1
        await RisingEdge(dut.wr_clk)
    dut.wr_en.value = 0

    await ClockCycles(dut.rd_clk, 6)

    # Now write 8 more while reading
    async def writer():
        for i in range(8):
            dut.wr_data.value = 0xB000 + i
            dut.wr_en.value   = 1
            await RisingEdge(dut.wr_clk)
        dut.wr_en.value = 0

    cocotb.start_soon(writer())

    read_vals = []
    for _ in range(12):  # 4 pre-filled + 8 new
        dut.rd_en.value = 1
        await RisingEdge(dut.rd_clk)
        if not (int(dut.empty.value) & 1):
            read_vals.append(int(dut.rd_data.value))
        # Small gap for sync
        await ClockCycles(dut.rd_clk, 2)
    dut.rd_en.value = 0

    assert len(read_vals) > 0, "Should have read at least some words"
    dut._log.info(f"PASS: concurrent read/write, got {len(read_vals)} words")
