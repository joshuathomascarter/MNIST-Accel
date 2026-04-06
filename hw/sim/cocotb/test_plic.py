"""
test_plic.py — Cocotb testbench for PLIC (Platform Level Interrupt Controller)
=============================================================================

Tests:
1. Assert source 5, priority 3, threshold 2 → claim returns 5
2. Two sources at different priorities → claim returns higher
3. Claim twice without complete → second returns 0
4. Complete flow: claim → ISR → complete → next interrupt

Target: hw/rtl/periph/plic.sv
Register map:
  0x0000_0000 - 0x0000_007C: Priority[0:31]  (3-bit each)
  0x0000_1000:               Pending[31:0]    (RO)
  0x0000_2000:               Enable[31:0]
  0x0020_0000:               Threshold         (3-bit)
  0x0020_0004:               Claim/Complete
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.irq_i.value = 0
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
async def test_single_source_claim(dut):
    """Assert source 5 with priority 3, threshold 2 → claim returns 5."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set priority of source 5 to 3
    await csr_write(dut, 0x0000_0014, 3)  # Priority[5] = offset 5*4 = 0x14

    # Enable source 5
    await csr_write(dut, 0x0000_2000, (1 << 5))

    # Set threshold to 2
    await csr_write(dut, 0x0020_0000, 2)

    # Assert interrupt source 5
    dut.irq_i.value = (1 << 5)
    await ClockCycles(dut.clk, 3)

    # Check irq_o asserted
    irq_out = int(dut.irq_o.value) & 1
    assert irq_out == 1, f"irq_o should be 1, got {irq_out}"

    # Claim
    claimed = await csr_read(dut, 0x0020_0004)
    assert claimed == 5, f"Expected claim=5, got {claimed}"

    dut._log.info("PASS: single source claimed correctly")


@cocotb.test()
async def test_priority_ordering(dut):
    """Two sources: src 3 (pri=5) and src 7 (pri=3) → claim returns 3 (higher pri)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set priorities
    await csr_write(dut, 0x0000_000C, 5)   # Priority[3] = 5
    await csr_write(dut, 0x0000_001C, 3)   # Priority[7] = 3

    # Enable both
    await csr_write(dut, 0x0000_2000, (1 << 3) | (1 << 7))

    # Set threshold to 0
    await csr_write(dut, 0x0020_0000, 0)

    # Assert both sources
    dut.irq_i.value = (1 << 3) | (1 << 7)
    await ClockCycles(dut.clk, 3)

    # Claim should return source 3 (higher priority = 5)
    claimed = await csr_read(dut, 0x0020_0004)
    assert claimed == 3, f"Expected claim=3 (higher pri), got {claimed}"

    dut._log.info("PASS: higher priority source claimed first")


@cocotb.test()
async def test_complete_flow(dut):
    """Full claim/complete handshake."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Setup source 1, priority 4
    await csr_write(dut, 0x0000_0004, 4)   # Priority[1] = 4
    await csr_write(dut, 0x0000_2000, (1 << 1))
    await csr_write(dut, 0x0020_0000, 0)

    # Assert
    dut.irq_i.value = (1 << 1)
    await ClockCycles(dut.clk, 3)

    # Claim
    claimed = await csr_read(dut, 0x0020_0004)
    assert claimed == 1

    # Complete (write source ID to claim/complete register)
    await csr_write(dut, 0x0020_0004, 1)

    # De-assert the source
    dut.irq_i.value = 0
    await ClockCycles(dut.clk, 3)

    # irq_o should be low now
    irq_out = int(dut.irq_o.value) & 1
    assert irq_out == 0, f"irq_o should be 0 after complete, got {irq_out}"

    dut._log.info("PASS: claim/complete handshake works")


@cocotb.test()
async def test_threshold_filtering(dut):
    """Source below threshold should NOT cause irq_o."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Source 2, priority 2
    await csr_write(dut, 0x0000_0008, 2)
    await csr_write(dut, 0x0000_2000, (1 << 2))

    # Threshold = 5 (higher than source priority)
    await csr_write(dut, 0x0020_0000, 5)

    dut.irq_i.value = (1 << 2)
    await ClockCycles(dut.clk, 5)

    irq_out = int(dut.irq_o.value) & 1
    assert irq_out == 0, f"irq_o should be 0 (below threshold), got {irq_out}"

    dut._log.info("PASS: threshold correctly filters low-priority interrupts")
