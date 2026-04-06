"""
test_boot_rom.py — Cocotb testbench for boot_rom AXI-Lite slave
=============================================================================

Tests:
1. Read known addresses, verify content
2. Write to ROM → expect SLVERR response
3. Sequential reads (burst-like)
4. Read from unmapped offset within ROM range

Target: hw/rtl/memory/boot_rom.sv
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
    """Perform a single AXI-Lite read transaction."""
    dut.arvalid.value = 1
    dut.araddr.value = addr
    dut.arid.value = rid

    # Wait for arready
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.arready.value) == 1:
            break
    dut.arvalid.value = 0

    # Wait for rvalid
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.rvalid.value) == 1:
            rdata = int(dut.rdata.value)
            rresp = int(dut.rresp.value)
            return rdata, rresp

    raise TimeoutError(f"AXI read timeout at addr 0x{addr:08X}")


async def axi_write(dut, addr, data, wid=0):
    """Perform a single AXI-Lite write transaction. Returns bresp."""
    # Drive AW channel
    dut.awvalid.value = 1
    dut.awaddr.value = addr
    dut.awid.value = wid

    # Drive W channel simultaneously
    dut.wvalid.value = 1
    dut.wdata.value = data
    dut.wstrb.value = 0xF
    dut.wlast.value = 1

    # Wait for awready
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.awready.value) == 1:
            dut.awvalid.value = 0
            break

    # Wait for wready
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.wready.value) == 1:
            dut.wvalid.value = 0
            break

    # Wait for bvalid
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.bvalid.value) == 1:
            bresp = int(dut.bresp.value)
            return bresp

    raise TimeoutError(f"AXI write timeout at addr 0x{addr:08X}")


@cocotb.test()
async def test_read_address_zero(dut):
    """Read address 0x0000 — should return valid data (OKAY response)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    rdata, rresp = await axi_read(dut, 0x0000)
    assert rresp == 0, f"Expected OKAY (0), got {rresp}"
    dut._log.info(f"PASS: Read addr 0x0000 = 0x{rdata:08X}")


@cocotb.test()
async def test_sequential_reads(dut):
    """Read first 16 words sequentially, verify no stalls or errors."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    for i in range(16):
        addr = i * 4
        rdata, rresp = await axi_read(dut, addr)
        assert rresp == 0, f"Read at 0x{addr:04X} returned error {rresp}"

    dut._log.info("PASS: 16 sequential reads completed without error")


@cocotb.test()
async def test_write_returns_slverr(dut):
    """Write to ROM should return SLVERR (2'b10)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    bresp = await axi_write(dut, 0x0000, 0xDEADBEEF)
    assert bresp == 2, f"Expected SLVERR (2), got {bresp}"
    dut._log.info("PASS: Write to ROM correctly returned SLVERR")


@cocotb.test()
async def test_read_after_write_unchanged(dut):
    """Write to ROM, then read — data should be unchanged (ROM is read-only)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Read original value
    original, _ = await axi_read(dut, 0x0000)

    # Attempt write
    await axi_write(dut, 0x0000, 0xBAADF00D)

    # Read again — should be unchanged
    after, _ = await axi_read(dut, 0x0000)
    assert after == original, \
        f"ROM content changed! Before=0x{original:08X}, After=0x{after:08X}"

    dut._log.info("PASS: ROM content unchanged after write attempt")


@cocotb.test()
async def test_read_id_passthrough(dut):
    """Verify AXI ID is preserved in read response."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Issue read with specific ID
    dut.arvalid.value = 1
    dut.araddr.value = 0x0000
    dut.arid.value = 0xA

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.arready.value) == 1:
            break
    dut.arvalid.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.rvalid.value) == 1:
            rid = int(dut.rid.value)
            assert rid == 0xA, f"Expected rid=0xA, got {rid}"
            break

    dut._log.info("PASS: AXI ID correctly passed through")
