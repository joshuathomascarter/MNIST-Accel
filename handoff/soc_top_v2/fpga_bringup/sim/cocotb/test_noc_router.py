"""
cocotb test for noc_router — verifies XY routing, VC allocation, and credit flow.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer

NUM_PORTS = 5
NUM_VCS = 4
PORT_LOCAL = 0
PORT_NORTH = 1
PORT_SOUTH = 2
PORT_EAST = 3
PORT_WEST = 4


async def reset_dut(dut):
    """Assert reset for 10 cycles."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


def build_flit(flit_type, src, dst, vc_id, msg_type, payload):
    """Build a flit word.

    Flit layout (from noc_pkg):
      [63]     valid (unused here, driven by valid signal)
      [62:61]  flit_type (2b)
      [60:57]  src (4b)
      [56:53]  dst (4b)
      [52:51]  vc_id (2b)
      [50:48]  msg_type (3b)
      [47:0]   payload (48b)
    """
    flit = 0
    flit |= (flit_type & 0x3) << 61
    flit |= (src & 0xF) << 57
    flit |= (dst & 0xF) << 53
    flit |= (vc_id & 0x3) << 51
    flit |= (msg_type & 0x7) << 48
    flit |= payload & 0xFFFFFFFFFFFF
    return flit


async def inject_flit(dut, port, flit_val, vc=0):
    """Inject a flit into the given input port."""
    dut.in_flit[port].value = flit_val
    dut.in_valid[port].value = 1
    await RisingEdge(dut.clk)
    dut.in_valid[port].value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_router_reset(dut):
    """After reset, no output should be valid."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for p in range(NUM_PORTS):
        assert dut.out_valid[p].value == 0, f"Port {p} valid after reset"

    dut._log.info("PASS: Router reset clean")


@cocotb.test()
async def test_local_to_east(dut):
    """Flit from local destined east should exit east port.

    Router at (0,0), destination at (0,1): XY routing goes east.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Provide credits on all output ports
    for p in range(NUM_PORTS):
        dut.out_credit[p].value = (1 << NUM_VCS) - 1  # All VCs have credit

    # Build flit: src=0 (local node 0,0), dst=1 (node 0,1)
    # FLIT_HEAD_TAIL = 2'b11
    flit = build_flit(0x3, src=0, dst=1, vc_id=0, msg_type=0, payload=0xDEAD)

    await inject_flit(dut, PORT_LOCAL, flit, vc=0)

    # Wait for pipeline
    await ClockCycles(dut.clk, 5)

    # Check the east port got the flit
    east_valid = dut.out_valid[PORT_EAST].value
    assert east_valid == 1, f"Expected east valid=1, got {east_valid}"

    dut._log.info("PASS: Local-to-east routing works")


@cocotb.test()
async def test_no_valid_no_output(dut):
    """With no input valid, outputs should stay quiet."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # All inputs invalid
    for p in range(NUM_PORTS):
        dut.in_valid[p].value = 0

    await ClockCycles(dut.clk, 10)

    for p in range(NUM_PORTS):
        assert dut.out_valid[p].value == 0, f"Port {p} spuriously valid"

    dut._log.info("PASS: No input → no output")


@cocotb.test()
async def test_credit_return(dut):
    """After a flit is consumed, a credit should be returned on that VC."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Give credits
    for p in range(NUM_PORTS):
        dut.out_credit[p].value = (1 << NUM_VCS) - 1

    # Inject from local to local (same node)
    flit = build_flit(0x3, src=0, dst=0, vc_id=0, msg_type=0, payload=0xBEEF)
    await inject_flit(dut, PORT_LOCAL, flit)

    # Wait and check credits returned
    await ClockCycles(dut.clk, 5)

    # The input port should eventually return a credit for VC 0
    credit_out = int(dut.in_credit[PORT_LOCAL].value)
    dut._log.info(f"Credit returned: {credit_out:#06b}")
    # At least VC 0 bit should be set
    assert credit_out & 0x1, "No credit returned for VC 0"

    dut._log.info("PASS: Credit return on consumption")
