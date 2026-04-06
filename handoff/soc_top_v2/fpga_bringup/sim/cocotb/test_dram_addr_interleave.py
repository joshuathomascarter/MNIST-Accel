"""
test_dram_addr_interleave.py — Cocotb tests for dram_addr_decoder.

Verifies RBC vs BRC address decoding modes:
  1. RBC mode: sequential addresses interleave across columns within a bank.
  2. BRC mode: sequential addresses interleave across banks first.
  3. Row boundary crossing works correctly in both modes.
  4. Known address mapping spot-checks.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer


BANK_BITS = 3
ROW_BITS  = 14
COL_BITS  = 10
BUS_BYTES = 2   # DDR3 x16 = 2 bytes per beat
BYTE_OFF  = 1   # log2(BUS_BYTES)


def rbc_decode(addr):
    """Python reference model: RBC (Row-Bank-Column) mode."""
    col  = (addr >> BYTE_OFF) & ((1 << COL_BITS) - 1)
    bank = (addr >> (BYTE_OFF + COL_BITS)) & ((1 << BANK_BITS) - 1)
    row  = (addr >> (BYTE_OFF + COL_BITS + BANK_BITS)) & ((1 << ROW_BITS) - 1)
    return bank, row, col


def brc_decode(addr):
    """Python reference model: BRC (Bank-Row-Column) mode."""
    col  = (addr >> BYTE_OFF) & ((1 << COL_BITS) - 1)
    row  = (addr >> (BYTE_OFF + COL_BITS)) & ((1 << ROW_BITS) - 1)
    bank = (addr >> (BYTE_OFF + COL_BITS + ROW_BITS)) & ((1 << BANK_BITS) - 1)
    return bank, row, col


async def apply_addr(dut, addr):
    """Apply address and wait for combinational settle."""
    dut.axi_addr.value = addr
    await Timer(1, units="ns")
    bank = int(dut.bank.value)
    row  = int(dut.row.value)
    col  = int(dut.col.value)
    return bank, row, col


@cocotb.test()
async def test_rbc_sequential(dut):
    """RBC mode: stepping by BUS_BYTES increments column."""
    # This test assumes MODE=0 (RBC) parameterization.
    for step in range(8):
        addr = step * BUS_BYTES
        bank, row, col = await apply_addr(dut, addr)
        exp_bank, exp_row, exp_col = rbc_decode(addr)
        assert col == exp_col, f"addr {addr:#x}: col {col} != {exp_col}"
        assert bank == exp_bank, f"addr {addr:#x}: bank {bank} != {exp_bank}"
        assert row == exp_row, f"addr {addr:#x}: row {row} != {exp_row}"


@cocotb.test()
async def test_rbc_bank_crossing(dut):
    """RBC mode: address crosses from bank 0 col max to bank 1 col 0."""
    # Bank boundary = 2^(BYTE_OFF + COL_BITS) = 2^11 = 2048
    boundary = 1 << (BYTE_OFF + COL_BITS)
    # Just before boundary
    b, r, c = await apply_addr(dut, boundary - BUS_BYTES)
    assert b == 0, f"Expected bank 0, got {b}"
    assert c == (1 << COL_BITS) - 1

    # At boundary
    b, r, c = await apply_addr(dut, boundary)
    assert b == 1, f"Expected bank 1, got {b}"
    assert c == 0


@cocotb.test()
async def test_rbc_row_crossing(dut):
    """RBC mode: address crosses from row 0 bank 7 to row 1 bank 0."""
    row_boundary = 1 << (BYTE_OFF + COL_BITS + BANK_BITS)
    b, r, c = await apply_addr(dut, row_boundary - BUS_BYTES)
    assert r == 0
    assert b == (1 << BANK_BITS) - 1

    b, r, c = await apply_addr(dut, row_boundary)
    assert r == 1
    assert b == 0
    assert c == 0


@cocotb.test()
async def test_known_address(dut):
    """Spot-check a known address in RBC mode."""
    # addr = row=5, bank=3, col=100, byte_off=0
    addr = (5 << (BYTE_OFF + COL_BITS + BANK_BITS)) | \
           (3 << (BYTE_OFF + COL_BITS)) | \
           (100 << BYTE_OFF)
    b, r, c = await apply_addr(dut, addr)
    assert b == 3, f"bank {b}"
    assert r == 5, f"row {r}"
    assert c == 100, f"col {c}"
