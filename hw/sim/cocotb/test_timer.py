"""
test_timer.py — Cocotb testbench for timer_ctrl AXI-Lite slave
=============================================================================

Tests:
1. mtime auto-increments
2. Set mtimecmp, verify interrupt fires when mtime >= mtimecmp
3. Clear interrupt by writing mtimecmp > mtime
4. 64-bit timer read (lo/hi)

Target: hw/rtl/periph/timer_ctrl.sv
Register map: MTIME_LO=0x00, MTIME_HI=0x04, MTIMECMP_LO=0x08, MTIMECMP_HI=0x0C
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
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


async def csr_write(dut, addr, data):
    dut.awvalid.value = 1
    dut.awaddr.value = addr
    dut.wvalid.value = 1
    dut.wdata.value = data
    dut.wstrb.value = 0xF
    dut.wlast.value = 1
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.awready.value):
            dut.awvalid.value = 0
        if int(dut.wready.value):
            dut.wvalid.value = 0
        if int(dut.awready.value) and int(dut.wready.value):
            break
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.bvalid.value):
            return


async def csr_read(dut, addr):
    dut.arvalid.value = 1
    dut.araddr.value = addr
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.arready.value):
            dut.arvalid.value = 0
            break
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.rvalid.value):
            return int(dut.rdata.value)
    return -1


@cocotb.test()
async def test_mtime_increments(dut):
    """mtime should auto-increment every clock cycle."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    val1 = await csr_read(dut, 0x00)
    await ClockCycles(dut.clk, 100)
    val2 = await csr_read(dut, 0x00)

    assert val2 > val1, f"mtime did not increment: {val1} → {val2}"
    dut._log.info(f"PASS: mtime incremented from {val1} to {val2}")


@cocotb.test()
async def test_interrupt_fires(dut):
    """Set mtimecmp close to current mtime, verify irq_timer_o asserts."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Read current mtime
    mtime_lo = await csr_read(dut, 0x00)

    # Set mtimecmp = mtime + 50 (will fire in ~50 cycles)
    target = mtime_lo + 50
    await csr_write(dut, 0x0C, 0)           # MTIMECMP_HI = 0
    await csr_write(dut, 0x08, target)       # MTIMECMP_LO = target

    # Wait for interrupt
    irq_seen = False
    for _ in range(200):
        await RisingEdge(dut.clk)
        if int(dut.irq_timer_o.value) == 1:
            irq_seen = True
            break

    assert irq_seen, "Timer interrupt did not fire"
    dut._log.info("PASS: Timer interrupt fired")


@cocotb.test()
async def test_clear_interrupt(dut):
    """Set mtimecmp far in future to clear interrupt."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Force interrupt by setting mtimecmp = 0 (mtime >= 0 always true)
    await csr_write(dut, 0x0C, 0)
    await csr_write(dut, 0x08, 0)
    await ClockCycles(dut.clk, 5)

    assert int(dut.irq_timer_o.value) == 1, "IRQ should be asserted when mtime >= mtimecmp=0"

    # Clear by setting mtimecmp very high
    await csr_write(dut, 0x0C, 0xFFFFFFFF)
    await csr_write(dut, 0x08, 0xFFFFFFFF)
    await ClockCycles(dut.clk, 5)

    assert int(dut.irq_timer_o.value) == 0, "IRQ should clear when mtimecmp >> mtime"
    dut._log.info("PASS: Timer interrupt cleared")


@cocotb.test()
async def test_write_mtime(dut):
    """Writing to MTIME_LO should set the counter."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await csr_write(dut, 0x00, 1000)
    await ClockCycles(dut.clk, 5)

    val = await csr_read(dut, 0x00)
    assert val >= 1000, f"mtime should be >= 1000 after write, got {val}"
    dut._log.info(f"PASS: mtime writable ({val})")
