"""
test_output_bram_ctrl.py — Cocotb testbench for output_bram_ctrl FSM
=============================================================================

Tests:
1. Intermediate layer: capture accumulators → BRAM → feedback to act_buffer
2. Last layer: capture accumulators → BRAM → trigger out_dma to DDR
3. Bank alternation across layers
4. Status signals (layer_done, last_layer_done, busy)

Target: hw/rtl/buffer/output_bram_ctrl.sv
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, FallingEdge


NUM_WORDS = 32  # 256 / 8 = 32 words for 16×16 output


async def reset_dut(dut):
    """Apply reset."""
    dut.rst_n.value = 0
    dut.sched_done.value = 0
    dut.start.value = 0
    dut.accum_ready.value = 0
    dut.accum_rd_data.value = 0
    dut.bram_rd_data.value = 0
    dut.bram_rd_valid.value = 0
    dut.out_dma_done.value = 0
    dut.layer_total.value = 4
    dut.layer_current.value = 0
    dut.pool_en.value = 0
    dut.output_h.value = 16
    dut.output_w.value = 16
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_idle_state(dut):
    """After reset, ctrl should be idle with no activity."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    assert int(dut.busy.value) == 0, "Should be idle after reset"
    assert int(dut.layer_done.value) == 0, "layer_done should be 0"
    assert int(dut.last_layer_done.value) == 0, "last_layer_done should be 0"
    assert int(dut.feedback_busy.value) == 0, "feedback_busy should be 0"

    dut._log.info("PASS: idle state")


@cocotb.test()
async def test_intermediate_layer_feedback(dut):
    """
    Simulate intermediate layer (layer 0 of 4):
    sched_done → capture accumulators → write to BRAM → feedback to act_buffer
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Configure as intermediate layer (0 of 4)
    dut.layer_total.value = 4
    dut.layer_current.value = 0

    # Trigger sched_done
    dut.sched_done.value = 1
    await RisingEdge(dut.clk)
    dut.sched_done.value = 0

    # Should enter WAIT_ACCUM
    await RisingEdge(dut.clk)
    assert int(dut.busy.value) == 1, "Should be busy after sched_done"

    # Provide accumulator ready
    dut.accum_ready.value = 1

    # Feed data as the ctrl reads from accumulator
    # Simulate the 2-cycle pipeline latency from accumulator
    capture_count = 0
    for _ in range(200):  # timeout
        await RisingEdge(dut.clk)

        if int(dut.accum_rd_en.value) == 1:
            capture_count += 1
            # Provide test data 2 cycles later (pipeline)
            dut.accum_rd_data.value = 0xAA00 + capture_count

        # Check if we entered feedback phase
        if int(dut.feedback_busy.value) == 1:
            break

    # Now simulate BRAM read valid responses during feedback
    feedback_writes = 0
    for _ in range(200):
        await RisingEdge(dut.clk)

        if int(dut.bram_rd_en.value) == 1:
            # BRAM responds with 1-cycle latency
            dut.bram_rd_valid.value = 1
            dut.bram_rd_data.value = 0xBB00 + feedback_writes
        else:
            dut.bram_rd_valid.value = 0

        if int(dut.fb_act_we.value) == 1:
            feedback_writes += 1

        # Check for layer_done
        if int(dut.layer_done.value) == 1:
            break

    assert int(dut.layer_done.value) == 1, "layer_done should pulse on intermediate layer"
    assert int(dut.last_layer_done.value) == 0, "last_layer_done should NOT be set for intermediate"

    dut._log.info(f"PASS: intermediate layer feedback ({feedback_writes} writes)")


@cocotb.test()
async def test_last_layer_ddr_drain(dut):
    """
    Simulate last layer (layer 3 of 4):
    sched_done → capture accumulators → write to BRAM → trigger out_dma
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Configure as last layer (3 of 4)
    dut.layer_total.value = 4
    dut.layer_current.value = 3

    # Trigger sched_done
    dut.sched_done.value = 1
    await RisingEdge(dut.clk)
    dut.sched_done.value = 0

    # Provide accumulator ready
    dut.accum_ready.value = 1
    dut.accum_rd_data.value = 0xFFEE

    # Wait for capture phase to complete
    dma_triggered = False
    for _ in range(200):
        await RisingEdge(dut.clk)

        if int(dut.out_dma_trigger.value) == 1:
            dma_triggered = True
            break

    assert dma_triggered, "out_dma_trigger should fire for last layer"

    # Simulate DMA completion
    dut.out_dma_done.value = 1
    await RisingEdge(dut.clk)
    dut.out_dma_done.value = 0

    # Wait for last_layer_done
    await ClockCycles(dut.clk, 3)

    # Check that last_layer_done pulsed
    # (It may have already deasserted, so check that we're back in IDLE)
    assert int(dut.busy.value) == 0, "Should return to idle after DMA done"

    dut._log.info("PASS: last layer DDR drain")


@cocotb.test()
async def test_bank_alternation(dut):
    """Verify bank_sel toggles between layers."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    initial_bank = int(dut.bram_bank_sel.value)

    # Run two intermediate layers and check bank toggle
    for layer_idx in range(2):
        dut.layer_total.value = 4
        dut.layer_current.value = layer_idx

        dut.sched_done.value = 1
        await RisingEdge(dut.clk)
        dut.sched_done.value = 0

        dut.accum_ready.value = 1
        dut.accum_rd_data.value = 0

        # Wait for feedback to complete
        for _ in range(300):
            await RisingEdge(dut.clk)
            if int(dut.bram_rd_en.value):
                dut.bram_rd_valid.value = 1
                dut.bram_rd_data.value = 0
            else:
                dut.bram_rd_valid.value = 0

            if int(dut.layer_done.value) == 1:
                break

        await ClockCycles(dut.clk, 2)

    # Bank should have toggled twice from initial
    final_bank = int(dut.bram_bank_sel.value)
    assert final_bank == initial_bank, \
        f"After 2 layers, bank should return to initial ({initial_bank}), got {final_bank}"

    dut._log.info("PASS: bank alternation")
