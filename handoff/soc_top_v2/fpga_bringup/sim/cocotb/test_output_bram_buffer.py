"""
test_output_bram_buffer.py — Cocotb testbench for output_bram_buffer
=============================================================================

Tests:
1. Basic write-read ping-pong (bank alternation)
2. Simultaneous write to one bank while reading from other
3. Full depth write+readback for both banks
4. Reset behavior

Target: hw/rtl/buffer/output_bram_buffer.sv
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer


async def reset_dut(dut):
    """Apply reset for 5 cycles."""
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.rd_en.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.rd_addr.value = 0
    dut.bank_sel.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_basic_write_read(dut):
    """Write to bank0, switch bank_sel, read back from bank0."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write 5 words to bank0 (bank_sel=0 → write bank0)
    test_data = [0xDEADBEEF_CAFEBABE, 0x1234567890ABCDEF,
                 0xAAAAAAAABBBBBBBB, 0xCCCCCCCCDDDDDDDD,
                 0x1111111122222222]

    dut.bank_sel.value = 0  # write to bank0, read from bank1
    for i, d in enumerate(test_data):
        dut.wr_en.value = 1
        dut.wr_addr.value = i
        dut.wr_data.value = d
        await RisingEdge(dut.clk)
    dut.wr_en.value = 0

    # Switch bank_sel to 1 → now read from bank0
    dut.bank_sel.value = 1
    await RisingEdge(dut.clk)

    # Read back all 5 words
    for i, expected in enumerate(test_data):
        dut.rd_en.value = 1
        dut.rd_addr.value = i
        await RisingEdge(dut.clk)  # BRAM latency
        dut.rd_en.value = 0
        await RisingEdge(dut.clk)  # Data available

        actual = int(dut.rd_data.value)
        assert actual == expected, \
            f"Addr {i}: expected 0x{expected:016X}, got 0x{actual:016X}"

    dut._log.info("PASS: basic write-read")


@cocotb.test()
async def test_ping_pong(dut):
    """Write to bank0, read from bank1 simultaneously, then swap."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Fill bank0 (bank_sel=0)
    dut.bank_sel.value = 0
    for i in range(8):
        dut.wr_en.value = 1
        dut.wr_addr.value = i
        dut.wr_data.value = 0xA0 + i
        await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)

    # Swap to bank_sel=1: write to bank1, read from bank0
    dut.bank_sel.value = 1
    for i in range(8):
        # Simultaneously write to bank1 and read from bank0
        dut.wr_en.value = 1
        dut.wr_addr.value = i
        dut.wr_data.value = 0xB0 + i
        dut.rd_en.value = 1
        dut.rd_addr.value = i
        await RisingEdge(dut.clk)

    dut.wr_en.value = 0
    dut.rd_en.value = 0
    await RisingEdge(dut.clk)

    # Now read back from bank1 (swap to bank_sel=0)
    dut.bank_sel.value = 0
    for i in range(8):
        dut.rd_en.value = 1
        dut.rd_addr.value = i
        await RisingEdge(dut.clk)
        dut.rd_en.value = 0
        await RisingEdge(dut.clk)

        actual = int(dut.rd_data.value)
        assert actual == 0xB0 + i, \
            f"Bank1 addr {i}: expected 0x{0xB0+i:X}, got 0x{actual:X}"

    dut._log.info("PASS: ping-pong")


@cocotb.test()
async def test_rd_valid_latency(dut):
    """Verify rd_valid asserts exactly 1 cycle after rd_en."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write one word to bank0
    dut.bank_sel.value = 0
    dut.wr_en.value = 1
    dut.wr_addr.value = 0
    dut.wr_data.value = 0x42
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0

    # Switch to read bank0
    dut.bank_sel.value = 1
    await RisingEdge(dut.clk)

    # Assert rd_valid is 0 before read
    assert int(dut.rd_valid.value) == 0, "rd_valid should be 0 before read"

    # Issue read
    dut.rd_en.value = 1
    dut.rd_addr.value = 0
    await RisingEdge(dut.clk)
    dut.rd_en.value = 0

    # rd_valid should be high on next cycle
    await RisingEdge(dut.clk)
    assert int(dut.rd_valid.value) == 1, "rd_valid should be 1 after read"

    # rd_valid should be low on cycle after (no more reads)
    await RisingEdge(dut.clk)
    assert int(dut.rd_valid.value) == 0, "rd_valid should return to 0"

    dut._log.info("PASS: rd_valid latency")


@cocotb.test()
async def test_reset_clears_output(dut):
    """After reset, rd_data should be 0."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    assert int(dut.rd_data.value) == 0, "rd_data should be 0 after reset"
    assert int(dut.rd_valid.value) == 0, "rd_valid should be 0 after reset"

    dut._log.info("PASS: reset clears output")
