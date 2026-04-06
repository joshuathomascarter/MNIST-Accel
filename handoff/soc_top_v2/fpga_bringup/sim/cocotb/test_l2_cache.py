"""
cocotb test for l2_cache_top — verifies basic read/write hits and misses.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

AXI_OKAY = 0
DRAM_BASE = 0x40000000


async def reset_dut(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def axi_write(dut, addr, data, strb=0xF):
    """Issue a single AXI write through the slave port."""
    dut.s_axi_awvalid.value = 1
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awid.value = 0
    dut.s_axi_awlen.value = 0
    dut.s_axi_awsize.value = 2
    dut.s_axi_awburst.value = 1
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wstrb.value = strb
    dut.s_axi_wlast.value = 1
    dut.s_axi_bready.value = 1

    # Wait for AW handshake
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.s_axi_awready.value:
            break
    dut.s_axi_awvalid.value = 0

    # Wait for W handshake
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.s_axi_wready.value:
            break
    dut.s_axi_wvalid.value = 0

    # Wait for B response
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.s_axi_bvalid.value:
            resp = int(dut.s_axi_bresp.value)
            dut.s_axi_bready.value = 1
            await RisingEdge(dut.clk)
            dut.s_axi_bready.value = 0
            return resp

    raise TimeoutError("AXI write did not complete")


async def axi_read(dut, addr):
    """Issue a single AXI read through the slave port."""
    dut.s_axi_arvalid.value = 1
    dut.s_axi_araddr.value = addr
    dut.s_axi_arid.value = 0
    dut.s_axi_arlen.value = 0
    dut.s_axi_arsize.value = 2
    dut.s_axi_arburst.value = 1
    dut.s_axi_rready.value = 1

    # Wait for AR handshake
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.s_axi_arready.value:
            break
    dut.s_axi_arvalid.value = 0

    # Wait for R response
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.s_axi_rvalid.value:
            data = int(dut.s_axi_rdata.value)
            resp = int(dut.s_axi_rresp.value)
            dut.s_axi_rready.value = 1
            await RisingEdge(dut.clk)
            dut.s_axi_rready.value = 0
            return data, resp

    raise TimeoutError("AXI read did not complete")


async def dram_responder(dut):
    """Simple DRAM model — responds to downstream AXI with stored data."""
    mem = {}

    while True:
        await RisingEdge(dut.clk)

        # Handle write requests
        if hasattr(dut, 'm_axi_awvalid') and dut.m_axi_awvalid.value:
            dut.m_axi_awready.value = 1
            addr = int(dut.m_axi_awaddr.value)
            await RisingEdge(dut.clk)
            dut.m_axi_awready.value = 0

            # Wait for W
            for _ in range(100):
                await RisingEdge(dut.clk)
                if dut.m_axi_wvalid.value:
                    data = int(dut.m_axi_wdata.value)
                    mem[addr] = data
                    dut.m_axi_wready.value = 1
                    await RisingEdge(dut.clk)
                    dut.m_axi_wready.value = 0

                    if dut.m_axi_wlast.value:
                        break

            # Send B response
            dut.m_axi_bvalid.value = 1
            dut.m_axi_bresp.value = AXI_OKAY
            dut.m_axi_bid.value = 0
            for _ in range(100):
                await RisingEdge(dut.clk)
                if dut.m_axi_bready.value:
                    break
            dut.m_axi_bvalid.value = 0

        # Handle read requests
        if hasattr(dut, 'm_axi_arvalid') and dut.m_axi_arvalid.value:
            dut.m_axi_arready.value = 1
            addr = int(dut.m_axi_araddr.value)
            burst_len = int(dut.m_axi_arlen.value) + 1
            await RisingEdge(dut.clk)
            dut.m_axi_arready.value = 0

            # Send R data
            for beat in range(burst_len):
                raddr = addr + beat * 4
                dut.m_axi_rvalid.value = 1
                dut.m_axi_rdata.value = mem.get(raddr, 0)
                dut.m_axi_rresp.value = AXI_OKAY
                dut.m_axi_rid.value = 0
                dut.m_axi_rlast.value = (beat == burst_len - 1)
                for _ in range(100):
                    await RisingEdge(dut.clk)
                    if dut.m_axi_rready.value:
                        break
            dut.m_axi_rvalid.value = 0


@cocotb.test()
async def test_l2_reset(dut):
    """After reset, cache should not be busy."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    busy = int(dut.cache_busy.value)
    assert busy == 0, f"Cache busy after reset: {busy}"
    dut._log.info("PASS: L2 cache reset clean")


@cocotb.test()
async def test_l2_write_read_hit(dut):
    """Write then read same address — should hit in L2."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    cocotb.start_soon(dram_responder(dut))

    await reset_dut(dut)

    addr = DRAM_BASE + 0x100
    wdata = 0xDEADBEEF

    resp = await axi_write(dut, addr, wdata)
    assert resp == AXI_OKAY, f"Write failed with resp={resp}"

    rdata, rresp = await axi_read(dut, addr)
    assert rresp == AXI_OKAY, f"Read failed with resp={rresp}"
    assert rdata == wdata, f"Read mismatch: got {rdata:#x}, expected {wdata:#x}"

    dut._log.info("PASS: L2 write-read hit")


@cocotb.test()
async def test_l2_multiple_sets(dut):
    """Write to different cache sets, read back all."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    cocotb.start_soon(dram_responder(dut))

    await reset_dut(dut)

    test_data = {}
    for i in range(4):
        addr = DRAM_BASE + i * 0x1000
        data = 0xA0000000 | i
        await axi_write(dut, addr, data)
        test_data[addr] = data

    for addr, expected in test_data.items():
        rdata, rresp = await axi_read(dut, addr)
        assert rresp == AXI_OKAY
        assert rdata == expected, f"Addr {addr:#x}: got {rdata:#x} expected {expected:#x}"

    dut._log.info("PASS: L2 multiple sets")
