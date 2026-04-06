"""
test_sram.py — Cocotb testbench for sram_ctrl AXI-Lite slave
=============================================================================

Tests:
1. Write word, read back
2. Byte-enable (WSTRB): write byte 0 only, verify bytes 1-3 unchanged
3. Fill and verify a region
4. Read-after-write same address
5. Alternating reads and writes

Target: hw/rtl/memory/sram_ctrl.sv
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    """Apply reset and initialize all inputs."""
    dut.rst_n.value = 0
    dut.awvalid.value = 0
    dut.awaddr.value = 0
    dut.awsize.value = 0b010
    dut.awburst.value = 0
    dut.awid.value = 0
    dut.wvalid.value = 0
    dut.wdata.value = 0
    dut.wstrb.value = 0xF
    dut.wlast.value = 1
    dut.bready.value = 1
    dut.arvalid.value = 0
    dut.araddr.value = 0
    dut.arsize.value = 0b010
    dut.arburst.value = 0
    dut.arid.value = 0
    dut.rready.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def axi_read(dut, addr, rid=0):
    dut.arvalid.value = 1
    dut.araddr.value = addr
    dut.arid.value = rid

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.arready.value) == 1:
            break
    else:
        dut.arvalid.value = 0
        raise TimeoutError("AR handshake timed out")

    dut.arvalid.value = 0

    dut.rready.value = 1
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.rvalid.value) == 1:
            rdata = int(dut.rdata.value)
            rresp = int(dut.rresp.value)
            dut.rready.value = 0
            return rdata, rresp

    dut.rready.value = 0
    raise TimeoutError("R channel timed out")


async def axi_write(dut, addr, data, wstrb=0xF, wid=0):
    """Perform a single AXI-Lite write transaction with byte-enable."""
    dut.awvalid.value = 1
    dut.awaddr.value = addr
    dut.awid.value = wid

    dut.wvalid.value = 1
    dut.wdata.value = data
    dut.wstrb.value = wstrb
    dut.wlast.value = 1

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.awready.value) == 1:
            dut.awvalid.value = 0
            break

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.wready.value) == 1:
            dut.wvalid.value = 0
            break

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.bvalid.value) == 1:
            bresp = int(dut.bresp.value)
            return bresp

    raise TimeoutError(f"AXI write timeout at addr 0x{addr:08X}")


@cocotb.test()
async def test_write_read_back(dut):
    """Write a word, read it back, verify match."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    test_data = 0xCAFEBABE
    test_addr = 0x100

    bresp = await axi_write(dut, test_addr, test_data)
    assert bresp == 0, f"Write returned error {bresp}"

    rdata, rresp = await axi_read(dut, test_addr)
    assert rresp == 0, f"Read returned error {rresp}"
    assert rdata == test_data, \
        f"Mismatch: wrote 0x{test_data:08X}, read 0x{rdata:08X}"

    dut._log.info("PASS: write-read-back verified")


@cocotb.test()
async def test_byte_enable_wstrb(dut):
    """Write with partial WSTRB, verify only targeted bytes change."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    addr = 0x200

    # Write full word first
    await axi_write(dut, addr, 0xAABBCCDD)

    # Overwrite only byte 0 (bits [7:0]) with 0xFF
    await axi_write(dut, addr, 0x000000FF, wstrb=0x1)

    rdata, _ = await axi_read(dut, addr)
    # Bytes 1-3 should be unchanged: 0xAABBCC, byte 0 becomes 0xFF
    expected = 0xAABBCCFF
    assert rdata == expected, \
        f"WSTRB byte-enable failed: expected 0x{expected:08X}, got 0x{rdata:08X}"

    dut._log.info("PASS: byte-enable (WSTRB) works correctly")


@cocotb.test()
async def test_fill_and_verify(dut):
    """Fill 64 words, read them all back, verify content."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    num_words = 64
    base_addr = 0x400

    # Write phase
    for i in range(num_words):
        addr = base_addr + i * 4
        data = (i * 0x01010101) & 0xFFFFFFFF
        await axi_write(dut, addr, data)

    # Read-back phase
    for i in range(num_words):
        addr = base_addr + i * 4
        expected = (i * 0x01010101) & 0xFFFFFFFF
        rdata, _ = await axi_read(dut, addr)
        assert rdata == expected, \
            f"Mismatch at 0x{addr:04X}: expected 0x{expected:08X}, got 0x{rdata:08X}"

    dut._log.info(f"PASS: filled and verified {num_words} words")


@cocotb.test()
async def test_overwrite(dut):
    """Write value, overwrite with different value, verify latest data."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    addr = 0x300

    await axi_write(dut, addr, 0x11111111)
    await axi_write(dut, addr, 0x22222222)

    rdata, _ = await axi_read(dut, addr)
    assert rdata == 0x22222222, \
        f"Overwrite failed: expected 0x22222222, got 0x{rdata:08X}"

    dut._log.info("PASS: overwrite works correctly")


@cocotb.test()
async def test_alternating_rw(dut):
    """Alternating writes and reads to different addresses."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write addr A, read addr A, write addr B, read addr B, read addr A again
    await axi_write(dut, 0x000, 0xAAAAAAAA)
    rdata_a1, _ = await axi_read(dut, 0x000)
    assert rdata_a1 == 0xAAAAAAAA

    await axi_write(dut, 0x004, 0xBBBBBBBB)
    rdata_b, _ = await axi_read(dut, 0x004)
    assert rdata_b == 0xBBBBBBBB

    # Re-read addr A to verify it wasn't corrupted
    rdata_a2, _ = await axi_read(dut, 0x000)
    assert rdata_a2 == 0xAAAAAAAA, \
        f"Addr A corrupted: expected 0xAAAAAAAA, got 0x{rdata_a2:08X}"

    dut._log.info("PASS: alternating R/W pattern correct")
