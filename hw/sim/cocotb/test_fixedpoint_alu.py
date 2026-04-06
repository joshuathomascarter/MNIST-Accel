"""
test_fixedpoint_alu.py — Cocotb testbench for fixedpoint_alu (Q16.16)
=====================================================================

Tests:
1. ADD — two positive Q16.16 values
2. SUB — result crosses zero
3. MUL — 1.5 × 2.0 = 3.0
4. ABS — negative → positive
5. CMP — greater-than, equal, less-than

Target: hw/rtl/hft/fixedpoint_alu.sv  (WIDTH=32, FRAC=16, REG_OUT=1)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


def q16(f):
    """Convert float to Q16.16 (signed 32-bit)."""
    val = int(f * (1 << 16))
    if val < 0:
        val = val & 0xFFFF_FFFF
    return val


def from_q16(u):
    """Convert unsigned 32-bit Q16.16 back to float."""
    if u & 0x8000_0000:
        u = u - (1 << 32)
    return u / (1 << 16)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.op.value = 0
    dut.valid_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def alu_op(dut, a, b, op):
    """Drive one operation and return (result, overflow) after REG_OUT pipeline."""
    dut.a.value = a & 0xFFFF_FFFF
    dut.b.value = b & 0xFFFF_FFFF
    dut.op.value = op
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    # Wait for registered output
    await RisingEdge(dut.clk)
    return int(dut.result.value), int(dut.overflow.value) & 1


@cocotb.test()
async def test_add(dut):
    """ADD: 1.25 + 2.75 = 4.0"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    a = q16(1.25)
    b = q16(2.75)
    expected = q16(4.0)

    result, ovf = await alu_op(dut, a, b, 0b000)
    assert result == expected, f"ADD: expected 0x{expected:08X}, got 0x{result:08X}"
    assert ovf == 0
    dut._log.info(f"PASS: 1.25 + 2.75 = {from_q16(result):.4f}")


@cocotb.test()
async def test_sub(dut):
    """SUB: 1.0 - 3.5 = -2.5"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    a = q16(1.0)
    b = q16(3.5)
    expected = q16(-2.5)

    result, ovf = await alu_op(dut, a, b, 0b001)
    assert result == expected, f"SUB: expected 0x{expected:08X}, got 0x{result:08X}"
    assert ovf == 0
    dut._log.info(f"PASS: 1.0 - 3.5 = {from_q16(result):.4f}")


@cocotb.test()
async def test_mul(dut):
    """MUL: 1.5 × 2.0 = 3.0"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    a = q16(1.5)
    b = q16(2.0)
    expected = q16(3.0)

    result, ovf = await alu_op(dut, a, b, 0b010)
    assert result == expected, f"MUL: expected 0x{expected:08X}, got 0x{result:08X}"
    assert ovf == 0
    dut._log.info(f"PASS: 1.5 × 2.0 = {from_q16(result):.4f}")


@cocotb.test()
async def test_abs(dut):
    """ABS: |-7.25| = 7.25"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    a = q16(-7.25)
    expected = q16(7.25)

    result, _ = await alu_op(dut, a, 0, 0b011)
    assert result == expected, f"ABS: expected 0x{expected:08X}, got 0x{result:08X}"
    dut._log.info(f"PASS: |-7.25| = {from_q16(result):.4f}")


@cocotb.test()
async def test_cmp(dut):
    """CMP: 5.0 > 3.0 → bit1=1, bit0=0; 3.0 == 3.0 → bit1=0, bit0=1."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # 5.0 > 3.0
    result, _ = await alu_op(dut, q16(5.0), q16(3.0), 0b100)
    gt = (result >> 1) & 1
    eq = result & 1
    assert gt == 1, f"CMP GT: expected 1, got {gt}"
    assert eq == 0, f"CMP EQ: expected 0, got {eq}"

    # 3.0 == 3.0
    result, _ = await alu_op(dut, q16(3.0), q16(3.0), 0b100)
    gt = (result >> 1) & 1
    eq = result & 1
    assert gt == 0, f"CMP GT: expected 0, got {gt}"
    assert eq == 1, f"CMP EQ: expected 1, got {eq}"

    dut._log.info("PASS: CMP greater-than and equal")
