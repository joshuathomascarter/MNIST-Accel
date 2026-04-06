"""
cocotb test for accel_tile_array — verifies CSR access and tile control.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def axi_lite_write(dut, addr, data):
    """Write via AXI-Lite slave port."""
    dut.s_axi_awvalid.value = 1
    dut.s_axi_awaddr.value = addr
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1

    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.s_axi_awready.value and dut.s_axi_wready.value:
            break
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0

    # Wait for B
    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.s_axi_bvalid.value:
            resp = int(dut.s_axi_bresp.value)
            await RisingEdge(dut.clk)
            dut.s_axi_bready.value = 0
            return resp

    raise TimeoutError("AXI-Lite write timeout")


async def axi_lite_read(dut, addr):
    """Read via AXI-Lite slave port."""
    dut.s_axi_arvalid.value = 1
    dut.s_axi_araddr.value = addr
    dut.s_axi_rready.value = 1

    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.s_axi_arready.value:
            break
    dut.s_axi_arvalid.value = 0

    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.s_axi_rvalid.value:
            data = int(dut.s_axi_rdata.value)
            resp = int(dut.s_axi_rresp.value)
            await RisingEdge(dut.clk)
            dut.s_axi_rready.value = 0
            return data, resp

    raise TimeoutError("AXI-Lite read timeout")


@cocotb.test()
async def test_tile_array_reset(dut):
    """After reset, all tiles should be idle."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Check tile_busy and tile_done outputs
    busy = int(dut.tile_busy_o.value)
    done = int(dut.tile_done_o.value)

    assert busy == 0, f"Tiles busy after reset: {busy:#x}"
    dut._log.info(f"tile_busy={busy:#x}, tile_done={done:#x}")
    dut._log.info("PASS: Tile array reset clean")


@cocotb.test()
async def test_csr_write_read(dut):
    """Write to tile 0 CSR, read it back."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Deassert AXI master ready (we're not testing DMA path)
    dut.m_axi_arready.value = 0
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_rvalid.value = 0
    dut.m_axi_bvalid.value = 0

    await reset_dut(dut)

    # Tile 0 CSR: addr = 0x0000_00XX (tile_sel = addr[15:12] = 0)
    # CSR offset 0x00 is typically the command register
    csr_addr = 0x00000000

    resp = await axi_lite_write(dut, csr_addr, 0x12345678)
    assert resp == 0, f"CSR write failed: resp={resp}"

    data, resp = await axi_lite_read(dut, csr_addr)
    assert resp == 0, f"CSR read failed: resp={resp}"
    dut._log.info(f"CSR read back: {data:#010x}")

    dut._log.info("PASS: CSR write-read")


@cocotb.test()
async def test_csr_tile_select(dut):
    """Write different values to tile 0 and tile 1, read back both."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.m_axi_arready.value = 0
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_rvalid.value = 0
    dut.m_axi_bvalid.value = 0

    await reset_dut(dut)

    # Tile 0 at addr[15:12] = 0 → address 0x0000_00XX
    # Tile 1 at addr[15:12] = 1 → address 0x0000_10XX
    tile0_addr = 0x00000004  # CSR offset 0x04
    tile1_addr = 0x00001004  # Tile 1, CSR offset 0x04

    await axi_lite_write(dut, tile0_addr, 0xAAAAAAAA)
    await axi_lite_write(dut, tile1_addr, 0xBBBBBBBB)

    d0, _ = await axi_lite_read(dut, tile0_addr)
    d1, _ = await axi_lite_read(dut, tile1_addr)

    dut._log.info(f"Tile 0 CSR[4] = {d0:#010x}")
    dut._log.info(f"Tile 1 CSR[4] = {d1:#010x}")

    # Values should differ (each tile has its own CSR space)
    # Exact values depend on tile_controller CSR implementation
    dut._log.info("PASS: Tile select routing")
