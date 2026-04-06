"""
cocotb test for page_table_walker — verifies Sv32 page walk FSM.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


PTE_V = 1 << 0      # Valid
PTE_R = 1 << 1      # Read
PTE_W = 1 << 2      # Write
PTE_X = 1 << 3      # Execute
PTE_U = 1 << 4      # User
PTE_A = 1 << 5      # Accessed
PTE_D = 1 << 6      # Dirty


async def reset_dut(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def mem_responder(dut, page_table):
    """Responds to PTW memory read requests from the page table dict."""
    while True:
        await RisingEdge(dut.clk)
        if dut.mem_req_valid.value and dut.mem_req_ready.value:
            addr = int(dut.mem_req_addr.value)
            pte = page_table.get(addr, 0)
            dut.mem_resp_valid.value = 1
            dut.mem_resp_data.value = pte
            await RisingEdge(dut.clk)
            dut.mem_resp_valid.value = 0


@cocotb.test()
async def test_ptw_reset(dut):
    """PTW should be idle after reset."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # No walk should be in progress
    dut.walk_req_valid.value = 0
    await ClockCycles(dut.clk, 5)

    assert dut.walk_done.value == 0, "PTW signaled done without request"
    assert dut.walk_fault.value == 0, "PTW signaled fault without request"

    dut._log.info("PASS: PTW idle after reset")


@cocotb.test()
async def test_ptw_simple_walk(dut):
    """Walk a valid 2-level Sv32 page table.

    VA = 0x0040_1000
    VPN[1] = 0x001, VPN[0] = 0x001
    page_table_root = PPN 0x80000 (physical page)

    Level 1 PTE at: root_ppn << 12 + VPN[1] * 4 = 0x80000000 + 0x004
    Level 0 PTE at: l1_ppn << 12 + VPN[0] * 4
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Page table root at PPN 0x80000
    satp_ppn = 0x80000

    # L1 PTE: non-leaf, points to L0 page table at PPN 0x80001
    l1_addr = (satp_ppn << 12) + (1 * 4)  # VPN[1] = 1
    l1_pte = (0x80001 << 10) | PTE_V       # Non-leaf (V but no R/W/X)

    # L0 PTE: leaf, maps to PPN 0xABCDE
    l0_addr = (0x80001 << 12) + (1 * 4)    # VPN[0] = 1
    l0_pte = (0xABCDE << 10) | PTE_V | PTE_R | PTE_W | PTE_A | PTE_D

    page_table = {l1_addr: l1_pte, l0_addr: l0_pte}

    cocotb.start_soon(mem_responder(dut, page_table))
    await reset_dut(dut)

    dut.satp_ppn.value = satp_ppn
    dut.satp_mode.value = 1  # Sv32 enabled
    dut.mem_req_ready.value = 1

    # Issue walk request
    dut.walk_req_valid.value = 1
    dut.walk_va.value = 0x00401000
    dut.walk_asid.value = 0
    dut.walk_is_store.value = 0
    dut.walk_is_exec.value = 0

    # Wait for done
    for _ in range(100):
        await RisingEdge(dut.clk)
        dut.walk_req_valid.value = 0  # Deassert after first cycle
        if dut.walk_done.value:
            break

    assert dut.walk_done.value == 1, "PTW did not complete"
    assert dut.walk_fault.value == 0, "PTW faulted on valid page"

    result_ppn = int(dut.walk_result_ppn.value)
    assert result_ppn == 0xABCDE, f"Wrong PPN: {result_ppn:#x} != 0xABCDE"

    dut._log.info("PASS: Simple Sv32 walk")


@cocotb.test()
async def test_ptw_invalid_pte(dut):
    """Walk should fault on invalid (V=0) PTE."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    satp_ppn = 0x80000

    # L1 PTE: invalid
    l1_addr = (satp_ppn << 12) + (0 * 4)
    l1_pte = 0  # V=0

    page_table = {l1_addr: l1_pte}
    cocotb.start_soon(mem_responder(dut, page_table))

    await reset_dut(dut)
    dut.satp_ppn.value = satp_ppn
    dut.satp_mode.value = 1
    dut.mem_req_ready.value = 1

    dut.walk_req_valid.value = 1
    dut.walk_va.value = 0x00000000
    dut.walk_asid.value = 0
    dut.walk_is_store.value = 0
    dut.walk_is_exec.value = 0

    for _ in range(100):
        await RisingEdge(dut.clk)
        dut.walk_req_valid.value = 0
        if dut.walk_done.value or dut.walk_fault.value:
            break

    assert dut.walk_fault.value == 1, "PTW should fault on invalid PTE"

    dut._log.info("PASS: PTW faults on invalid PTE")
