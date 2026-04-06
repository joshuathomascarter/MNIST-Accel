"""
test_gpio.py — Cocotb testbench for gpio_ctrl AXI-Lite slave
=============================================================================

Tests:
1. Set direction output, write value, verify gpio_o pin
2. Set direction input, drive pin externally, read value
3. Mixed direction — some pins in, some out

Target: hw/rtl/periph/gpio_ctrl.sv
Register map: DIR=0x00, OUT=0x04, IN=0x08
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.gpio_i.value = 0
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
async def test_output_direction(dut):
    """Set all pins to output, write value, verify gpio_o."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set direction: all output
    await csr_write(dut, 0x00, 0xFF)

    # Write output value
    await csr_write(dut, 0x04, 0xA5)

    await ClockCycles(dut.clk, 3)
    gpio_out = int(dut.gpio_o.value) & 0xFF
    assert gpio_out == 0xA5, f"Expected gpio_o=0xA5, got 0x{gpio_out:02X}"

    gpio_oe = int(dut.gpio_oe.value) & 0xFF
    assert gpio_oe == 0xFF, f"Expected gpio_oe=0xFF, got 0x{gpio_oe:02X}"

    dut._log.info("PASS: output direction and value correct")


@cocotb.test()
async def test_input_read(dut):
    """Set all pins to input, drive external value, read IN register."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set direction: all input
    await csr_write(dut, 0x00, 0x00)

    # Drive external input
    dut.gpio_i.value = 0x3C

    # Wait for 2-FF synchronizer (2 cycles + margin)
    await ClockCycles(dut.clk, 5)

    # Read IN register
    in_val = await csr_read(dut, 0x08)
    assert (in_val & 0xFF) == 0x3C, \
        f"Expected IN=0x3C, got 0x{in_val & 0xFF:02X}"

    dut._log.info("PASS: input read through synchronizer")


@cocotb.test()
async def test_mixed_direction(dut):
    """Pins 0-3 output, pins 4-7 input."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # DIR: lower 4 bits output, upper 4 bits input
    await csr_write(dut, 0x00, 0x0F)

    # Write output value (lower nibble)
    await csr_write(dut, 0x04, 0x05)

    # Drive input pins (upper nibble)
    dut.gpio_i.value = 0xB0
    await ClockCycles(dut.clk, 5)

    # Check output
    gpio_out = int(dut.gpio_o.value) & 0x0F
    assert gpio_out == 0x05, f"Output nibble wrong: 0x{gpio_out:02X}"

    # Check OE
    gpio_oe = int(dut.gpio_oe.value) & 0xFF
    assert gpio_oe == 0x0F, f"OE wrong: 0x{gpio_oe:02X}"

    # Read input
    in_val = await csr_read(dut, 0x08)
    in_upper = (in_val & 0xF0)
    assert in_upper == 0xB0, f"Input nibble wrong: 0x{in_upper:02X}"

    dut._log.info("PASS: mixed direction correct")


@cocotb.test()
async def test_dir_register_readback(dut):
    """DIR register should be readable."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await csr_write(dut, 0x00, 0x55)
    dir_val = await csr_read(dut, 0x00)
    assert (dir_val & 0xFF) == 0x55, f"DIR readback failed: 0x{dir_val:02X}"

    dut._log.info("PASS: DIR register readable")
