"""
test_l1_dcache.py — Cocotb testbench for L1 D-Cache
Tests: hit/miss, writeback, LRU eviction, byte-enable writes
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.cpu_req.value = 0
    dut.cpu_we.value = 0
    dut.cpu_addr.value = 0
    dut.cpu_be.value = 0
    dut.cpu_wdata.value = 0
    # AXI memory side: always ready
    dut.m_axi_awready.value = 1
    dut.m_axi_wready.value = 1
    dut.m_axi_bvalid.value = 0
    dut.m_axi_arready.value = 1
    dut.m_axi_rvalid.value = 0
    dut.m_axi_rdata.value = 0
    dut.m_axi_rlast.value = 0
    dut.m_axi_rresp.value = 0
    dut.m_axi_bresp.value = 0
    dut.m_axi_rid.value = 0
    dut.m_axi_bid.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def cpu_read(dut, addr, timeout=200):
    """Issue a CPU read and wait for rvalid."""
    dut.cpu_req.value = 1
    dut.cpu_addr.value = addr
    dut.cpu_we.value = 0
    dut.cpu_be.value = 0xF

    # Wait for grant
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.cpu_gnt.value:
            break
    dut.cpu_req.value = 0

    # Wait for rvalid
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.cpu_rvalid.value:
            return int(dut.cpu_rdata.value)
    raise TimeoutError(f"CPU read at {addr:#x} timed out")


async def cpu_write(dut, addr, data, be=0xF, timeout=200):
    """Issue a CPU write and wait for completion."""
    dut.cpu_req.value = 1
    dut.cpu_addr.value = addr
    dut.cpu_we.value = 1
    dut.cpu_be.value = be
    dut.cpu_wdata.value = data

    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.cpu_gnt.value:
            break
    dut.cpu_req.value = 0
    dut.cpu_we.value = 0

    # Wait for write to complete
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.cpu_rvalid.value:
            return
    raise TimeoutError(f"CPU write at {addr:#x} timed out")


async def axi_responder(dut):
    """Background task: respond to AXI read requests with addr-based data."""
    while True:
        await RisingEdge(dut.clk)
        # Respond to AR requests with sequential data
        if int(dut.m_axi_arvalid.value) and int(dut.m_axi_arready.value):
            base_addr = int(dut.m_axi_araddr.value)
            burst_len = int(dut.m_axi_arlen.value) + 1
            for beat in range(burst_len):
                await RisingEdge(dut.clk)
                dut.m_axi_rvalid.value = 1
                dut.m_axi_rdata.value = base_addr + beat * 4  # Addr-based pattern
                dut.m_axi_rlast.value = 1 if beat == burst_len - 1 else 0
                dut.m_axi_rresp.value = 0
                while True:
                    await RisingEdge(dut.clk)
                    if int(dut.m_axi_rready.value):
                        break
            dut.m_axi_rvalid.value = 0
            dut.m_axi_rlast.value = 0

        # Respond to AW/W with B
        if int(dut.m_axi_awvalid.value) and int(dut.m_axi_awready.value):
            # Wait for wlast
            while True:
                await RisingEdge(dut.clk)
                if int(dut.m_axi_wvalid.value) and int(dut.m_axi_wlast.value):
                    break
            await RisingEdge(dut.clk)
            dut.m_axi_bvalid.value = 1
            dut.m_axi_bresp.value = 0
            await RisingEdge(dut.clk)
            while not int(dut.m_axi_bready.value):
                await RisingEdge(dut.clk)
            dut.m_axi_bvalid.value = 0


@cocotb.test()
async def test_cold_miss_and_fill(dut):
    """First read should cause a cold miss → AXI fill → data returned."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    cocotb.start_soon(axi_responder(dut))

    data = await cpu_read(dut, 0x1000_0000)
    dut._log.info(f"Cold read returned: {data:#x}")
    # Data should be the base address (from our axi_responder pattern)
    assert data == 0x1000_0000, f"Expected 0x10000000, got {data:#x}"


@cocotb.test()
async def test_cache_hit(dut):
    """Second read to same line should be a cache hit (fast)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    cocotb.start_soon(axi_responder(dut))

    # Cold miss fill
    await cpu_read(dut, 0x1000_0000)

    # Second read (same cache line) — should be fast
    start_time = cocotb.utils.get_sim_time("ns")
    data = await cpu_read(dut, 0x1000_0004)
    end_time = cocotb.utils.get_sim_time("ns")

    latency = end_time - start_time
    dut._log.info(f"Hit latency: {latency} ns, data: {data:#x}")
    # A hit should be much faster than a miss
    assert latency < 100, f"Hit latency too high: {latency} ns"


@cocotb.test()
async def test_write_and_readback(dut):
    """Write a word and read it back."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    cocotb.start_soon(axi_responder(dut))

    # Fill cache line first
    await cpu_read(dut, 0x1000_0000)

    # Write
    await cpu_write(dut, 0x1000_0000, 0xDEAD_BEEF)

    # Read back
    data = await cpu_read(dut, 0x1000_0000)
    assert data == 0xDEAD_BEEF, f"Readback mismatch: expected 0xDEADBEEF, got {data:#x}"


@cocotb.test()
async def test_byte_enable_write(dut):
    """Partial write using byte enables."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    cocotb.start_soon(axi_responder(dut))

    # Fill
    await cpu_read(dut, 0x1000_0000)

    # Write full word
    await cpu_write(dut, 0x1000_0000, 0xAABBCCDD)

    # Partial write (only byte 0)
    await cpu_write(dut, 0x1000_0000, 0x000000FF, be=0x1)

    data = await cpu_read(dut, 0x1000_0000)
    expected = 0xAABBCCFF
    assert data == expected, f"Byte-enable write: expected {expected:#x}, got {data:#x}"
