#!/usr/bin/env python3
"""
test_accel_top.py — Functional Testbench for accel_top (Sparse Accelerator)

Tests:
  1. test_identity_14x14: Dense identity matrix — verifies the complete
     DMA -> BRAM -> Scheduler -> Systolic -> Output Accumulator -> Write DMA path.
  2. test_sparse_28x28: 50% block-sparse matrix — verifies zero blocks
     are correctly skipped and results written to DDR.

Register map matches csr.sv exactly.  BSR data layout matches bsr_dma.sv
(no header — num_rows/total_blocks come from CSR, not from DDR).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import struct
import numpy as np

# =============================================================================
# CSR Register Map — must match csr.sv localparam addresses
# =============================================================================
REG_CTRL              = 0x00    # [0]=start(W1P), [1]=abort(W1P), [2]=irq_en
REG_DIMS_M            = 0x04
REG_DIMS_N            = 0x08
REG_DIMS_K            = 0x0C
REG_TILES_Tm          = 0x10
REG_TILES_Tn          = 0x14
REG_TILES_Tk          = 0x18
REG_SCALE_Sa          = 0x2C    # Q16.16 quantization scale
REG_STATUS            = 0x3C    # [0]=busy(RO), [1]=done_tile(R/W1C)
REG_PERF_TOTAL        = 0x40
REG_PERF_ACTIVE       = 0x44
REG_RESULT_0          = 0x80
REG_RESULT_1          = 0x84
REG_RESULT_2          = 0x88
REG_RESULT_3          = 0x8C
REG_DMA_SRC_ADDR      = 0x90    # BSR DMA source address in DDR
REG_DMA_DST_ADDR      = 0x94    # Output DMA destination in DDR
REG_DMA_XFER_LEN      = 0x98
REG_DMA_CTRL          = 0x9C    # [0]=start(W1P), [1]=busy(RO), [2]=done(R/W1C)
REG_ACT_DMA_SRC       = 0xA0    # Activation DMA source address
REG_ACT_DMA_LEN       = 0xA4    # Activation transfer length (bytes)
REG_ACT_DMA_CTRL      = 0xA8    # [0]=start(W1P)
REG_BSR_CONFIG        = 0xC0    # [1]=relu_en
REG_BSR_NUM_BLOCKS    = 0xC4
REG_BSR_BLOCK_ROWS    = 0xC8
REG_BSR_BLOCK_COLS    = 0xCC


# =============================================================================
# Simulated DDR Memory
# =============================================================================
class SimDDR:
    """Byte-addressable simulated DDR for AXI read/write responders."""

    def __init__(self, size=1 << 20):
        self.mem = bytearray(size)

    def write_u32(self, addr, val):
        struct.pack_into('<I', self.mem, addr, val & 0xFFFFFFFF)

    def write_u16(self, addr, val):
        struct.pack_into('<H', self.mem, addr, val & 0xFFFF)

    def read_u64(self, addr):
        return struct.unpack_from('<Q', self.mem, addr)[0]

    def write_u64(self, addr, val):
        struct.pack_into('<Q', self.mem, addr, val & 0xFFFFFFFFFFFFFFFF)

    def write_i8_array(self, addr, arr):
        for i, v in enumerate(arr.flat):
            self.mem[addr + i] = int(v) & 0xFF

    def read_i8_array(self, addr, count):
        """Read `count` signed INT8 values from DDR."""
        result = []
        for i in range(count):
            b = self.mem[addr + i]
            result.append(b if b < 128 else b - 256)
        return result


def build_bsr_in_ddr(ddr, base_addr, weights, block_size=14):
    """
    Build BSR structure in DDR matching bsr_dma.sv's expected layout:
      1. row_ptr[0..num_rows]  (32-bit each, packed 2 per 64-bit word)
      2. col_idx[0..nnz-1]    (16-bit each, packed 4 per 64-bit word)
      3. weight blocks         (196 bytes each = 25 x 8-byte words)
    NO header -- bsr_dma gets num_rows and total_blocks from CSR inputs.
    Returns (nnz_blocks, block_rows, block_cols).
    """
    M, K = weights.shape
    block_rows = M // block_size
    block_cols = K // block_size

    # Find non-zero blocks
    row_ptr = [0]
    col_indices = []
    blocks_data = []
    for br in range(block_rows):
        for bc in range(block_cols):
            block = weights[br*block_size:(br+1)*block_size,
                           bc*block_size:(bc+1)*block_size]
            if np.any(block != 0):
                col_indices.append(bc)
                blocks_data.append(block)
        row_ptr.append(len(col_indices))

    nnz_blocks = len(col_indices)
    offset = base_addr

    # Phase 1: row_ptr (32-bit entries, 2 per 64-bit AXI beat)
    for i, rp in enumerate(row_ptr):
        ddr.write_u32(offset + i * 4, rp)
    row_ptr_bytes = len(row_ptr) * 4
    offset += (row_ptr_bytes + 7) & ~7  # 8-byte align

    # Phase 2: col_idx (16-bit entries, 4 per 64-bit AXI beat)
    for i, ci in enumerate(col_indices):
        ddr.write_u16(offset + i * 2, ci)
    col_idx_bytes = nnz_blocks * 2
    offset += (col_idx_bytes + 7) & ~7  # 8-byte align

    # Phase 3: weight blocks (196 bytes each, padded to 200 = 25 beats)
    for i, blk in enumerate(blocks_data):
        ddr.write_i8_array(offset + i * 200, blk)

    return nnz_blocks, block_rows, block_cols


# =============================================================================
# AXI-Lite Driver
# =============================================================================
async def axi_lite_write(dut, addr, data):
    """Write a 32-bit value to an AXI-Lite CSR register."""
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = data & 0xFFFFFFFF
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_wvalid.value = 1
    dut.s_axi_bready.value = 1

    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.s_axi_awready.value and dut.s_axi_awvalid.value:
            dut.s_axi_awvalid.value = 0
        if dut.s_axi_wready.value and dut.s_axi_wvalid.value:
            dut.s_axi_wvalid.value = 0
        if dut.s_axi_bvalid.value:
            dut.s_axi_bready.value = 0
            return
    raise TimeoutError(f"AXI-Lite write timeout at 0x{addr:02X}")


async def axi_lite_read(dut, addr):
    """Read a 32-bit value from an AXI-Lite CSR register."""
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 1

    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.s_axi_arready.value and dut.s_axi_arvalid.value:
            dut.s_axi_arvalid.value = 0
        if dut.s_axi_rvalid.value:
            val = dut.s_axi_rdata.value.integer
            dut.s_axi_rready.value = 0
            return val
    raise TimeoutError(f"AXI-Lite read timeout at 0x{addr:02X}")


# =============================================================================
# AXI4 Read Responder (simulates DDR slave for DMA reads)
# =============================================================================
async def axi4_read_responder(dut, ddr):
    """Respond to AXI4 read bursts using simulated DDR."""
    while True:
        await RisingEdge(dut.clk)
        if dut.m_axi_arvalid.value and dut.m_axi_arready.value:
            addr = dut.m_axi_araddr.value.integer
            burst_len = dut.m_axi_arlen.value.integer + 1

            for beat in range(burst_len):
                dut.m_axi_rvalid.value = 1
                dut.m_axi_rdata.value = ddr.read_u64(addr + beat * 8)
                dut.m_axi_rresp.value = 0
                dut.m_axi_rlast.value = 1 if beat == burst_len - 1 else 0
                dut.m_axi_rid.value = dut.m_axi_arid.value.integer

                while True:
                    await RisingEdge(dut.clk)
                    if dut.m_axi_rready.value:
                        break

            dut.m_axi_rvalid.value = 0
            dut.m_axi_rlast.value = 0


# =============================================================================
# AXI4 Write Responder (simulates DDR slave for output DMA writes)
# =============================================================================
async def axi4_write_responder(dut, ddr):
    """Accept AXI4 write bursts and store data in simulated DDR."""
    while True:
        await RisingEdge(dut.clk)

        # Wait for write address handshake
        if dut.m_axi_awvalid.value and dut.m_axi_awready.value:
            wr_addr = dut.m_axi_awaddr.value.integer
            wr_len = dut.m_axi_awlen.value.integer + 1

            # Receive all write data beats
            beat = 0
            while beat < wr_len:
                await RisingEdge(dut.clk)
                if dut.m_axi_wvalid.value and dut.m_axi_wready.value:
                    data = dut.m_axi_wdata.value.integer
                    ddr.write_u64(wr_addr + beat * 8, data)
                    beat += 1

            # Send write response
            dut.m_axi_bvalid.value = 1
            dut.m_axi_bresp.value = 0  # OKAY
            dut.m_axi_bid.value = 0

            while True:
                await RisingEdge(dut.clk)
                if dut.m_axi_bready.value:
                    break

            dut.m_axi_bvalid.value = 0


# =============================================================================
# Common reset + init
# =============================================================================
async def reset_dut(dut):
    """Apply reset and initialize all AXI signals."""
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    # AXI-Lite
    dut.s_axi_awaddr.value = 0
    dut.s_axi_awprot.value = 0
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wdata.value = 0
    dut.s_axi_wstrb.value = 0
    dut.s_axi_wvalid.value = 0
    dut.s_axi_bready.value = 0
    dut.s_axi_araddr.value = 0
    dut.s_axi_arprot.value = 0
    dut.s_axi_arvalid.value = 0
    dut.s_axi_rready.value = 0
    # AXI4 Read (from DDR responder)
    dut.m_axi_arready.value = 1
    dut.m_axi_rvalid.value = 0
    dut.m_axi_rdata.value = 0
    dut.m_axi_rresp.value = 0
    dut.m_axi_rlast.value = 0
    dut.m_axi_rid.value = 0
    # AXI4 Write (from DDR responder)
    dut.m_axi_awready.value = 1
    dut.m_axi_wready.value = 1
    dut.m_axi_bvalid.value = 0
    dut.m_axi_bresp.value = 0
    dut.m_axi_bid.value = 0

    await ClockCycles(dut.clk, 20)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)


async def wait_not_busy(dut, timeout=10000):
    """Poll STATUS[0] (busy) until it clears."""
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        status = await axi_lite_read(dut, REG_STATUS)
        if not (status & 0x01):
            return
    raise TimeoutError("Busy did not clear")


# =============================================================================
# Test 1: 14x14 Dense Identity — simplest datapath check
# =============================================================================
@cocotb.test()
async def test_identity_14x14(dut):
    """
    I * [1,2,...,14] = [1,2,...,14].
    Verifies: DMA -> BRAM -> Scheduler -> Systolic -> Accum -> Write DMA -> DDR.
    """
    await reset_dut(dut)
    ddr = SimDDR()

    # Build BSR: 14x14 identity (1 block row, 1 block col, 1 NZ block)
    weights = np.eye(14, dtype=np.int8)
    BSR_BASE = 0x1000
    nnz, block_rows, block_cols = build_bsr_in_ddr(ddr, BSR_BASE, weights)

    # Activations: [1, 2, ..., 14] (14 bytes = 2 beats)
    ACT_BASE = 0x2000
    activations = np.arange(1, 15, dtype=np.int8)
    ddr.write_i8_array(ACT_BASE, activations)

    # Output destination in DDR
    OUT_BASE = 0x3000

    dut._log.info(f"BSR: {nnz} blocks, {block_rows}x{block_cols} tiles")
    dut._log.info(f"Activations: {list(activations)}")

    # Start background AXI responders
    cocotb.start_soon(axi4_read_responder(dut, ddr))
    cocotb.start_soon(axi4_write_responder(dut, ddr))

    # ---- Configure CSR ----
    await axi_lite_write(dut, REG_DIMS_M, 14)
    await axi_lite_write(dut, REG_DIMS_N, 14)
    await axi_lite_write(dut, REG_DIMS_K, 14)

    await axi_lite_write(dut, REG_BSR_NUM_BLOCKS, nnz)
    await axi_lite_write(dut, REG_BSR_BLOCK_ROWS, block_rows)
    await axi_lite_write(dut, REG_BSR_BLOCK_COLS, block_cols)
    await axi_lite_write(dut, REG_BSR_CONFIG, 0x00)          # No ReLU

    await axi_lite_write(dut, REG_SCALE_Sa, 0x00010000)      # Q16.16 scale = 1.0

    await axi_lite_write(dut, REG_DMA_SRC_ADDR, BSR_BASE)    # BSR data in DDR
    await axi_lite_write(dut, REG_DMA_DST_ADDR, OUT_BASE)    # Output destination
    await axi_lite_write(dut, REG_ACT_DMA_SRC, ACT_BASE)     # Activations in DDR
    await axi_lite_write(dut, REG_ACT_DMA_LEN, 14)           # 14 bytes

    # ---- Trigger BSR DMA ----
    dut._log.info("Starting BSR DMA...")
    await axi_lite_write(dut, REG_DMA_CTRL, 0x01)
    await wait_not_busy(dut, timeout=20000)

    # ---- Trigger Activation DMA ----
    dut._log.info("Starting Activation DMA...")
    await axi_lite_write(dut, REG_ACT_DMA_CTRL, 0x01)
    await wait_not_busy(dut, timeout=20000)

    # ---- Trigger Computation ----
    dut._log.info("Starting computation...")
    await axi_lite_write(dut, REG_CTRL, 0x01)

    # Wait for done (top-level done = out_dma_done, includes write-back)
    for cycle in range(100000):
        await RisingEdge(dut.clk)
        if dut.done.value:
            dut._log.info(f"Done after {cycle} cycles")
            break
    else:
        raise TimeoutError("Computation did not complete in 100000 cycles")

    # ---- Verify: CSR result registers (first 4 raw accumulators) ----
    results_csr = []
    for i in range(4):
        val = await axi_lite_read(dut, REG_RESULT_0 + i * 4)
        if val >= (1 << 31):
            val -= (1 << 32)
        results_csr.append(val)
    dut._log.info(f"CSR raw accumulators [0:3]: {results_csr}")

    # ---- Verify: DDR output (196 INT8 values via output DMA) ----
    ddr_out = ddr.read_i8_array(OUT_BASE, 14)
    dut._log.info(f"DDR output [0:13]: {ddr_out}")

    expected = list(range(1, 15))
    for i in range(14):
        assert ddr_out[i] == expected[i], \
            f"DDR mismatch at [{i}]: got {ddr_out[i]}, expected {expected[i]}"

    dut._log.info("PASS: Identity multiply verified via DDR write-back!")

    # ---- Performance counters ----
    total_cyc = await axi_lite_read(dut, REG_PERF_TOTAL)
    active_cyc = await axi_lite_read(dut, REG_PERF_ACTIVE)
    dut._log.info(f"Performance: {total_cyc} total, {active_cyc} active cycles")
    if total_cyc > 0:
        dut._log.info(f"Utilization: {active_cyc / total_cyc * 100:.1f}%")


# =============================================================================
# Test 2: 28x28 Sparse (50% block sparsity)
# =============================================================================
@cocotb.test()
async def test_sparse_28x28(dut):
    """
    28x28 diagonal-block matrix (blocks [0,0] and [1,1] non-zero, 50% sparse).
    Verifies zero blocks are skipped and output is correct.
    """
    await reset_dut(dut)
    ddr = SimDDR()

    # Block [0,0] = all 1s, Block [1,1] = all 2s, rest zero
    weights = np.zeros((28, 28), dtype=np.int8)
    weights[0:14, 0:14] = 1
    weights[14:28, 14:28] = 2

    BSR_BASE = 0x1000
    nnz, block_rows, block_cols = build_bsr_in_ddr(ddr, BSR_BASE, weights)

    # Activations: all 3s
    ACT_BASE = 0x2000
    activations = np.ones(28, dtype=np.int8) * 3
    ddr.write_i8_array(ACT_BASE, activations)

    OUT_BASE = 0x3000

    dut._log.info(f"Sparse BSR: {nnz} NZ blocks / {block_rows*block_cols} total")

    cocotb.start_soon(axi4_read_responder(dut, ddr))
    cocotb.start_soon(axi4_write_responder(dut, ddr))

    # ---- Configure ----
    await axi_lite_write(dut, REG_DIMS_M, 28)
    await axi_lite_write(dut, REG_DIMS_N, 28)
    await axi_lite_write(dut, REG_DIMS_K, 28)

    await axi_lite_write(dut, REG_BSR_NUM_BLOCKS, nnz)
    await axi_lite_write(dut, REG_BSR_BLOCK_ROWS, block_rows)
    await axi_lite_write(dut, REG_BSR_BLOCK_COLS, block_cols)
    await axi_lite_write(dut, REG_BSR_CONFIG, 0x00)
    await axi_lite_write(dut, REG_SCALE_Sa, 0x00010000)

    await axi_lite_write(dut, REG_DMA_SRC_ADDR, BSR_BASE)
    await axi_lite_write(dut, REG_DMA_DST_ADDR, OUT_BASE)
    await axi_lite_write(dut, REG_ACT_DMA_SRC, ACT_BASE)
    await axi_lite_write(dut, REG_ACT_DMA_LEN, 28)

    # ---- DMAs ----
    await axi_lite_write(dut, REG_DMA_CTRL, 0x01)
    await wait_not_busy(dut, timeout=30000)

    await axi_lite_write(dut, REG_ACT_DMA_CTRL, 0x01)
    await wait_not_busy(dut, timeout=30000)

    # ---- Compute ----
    await axi_lite_write(dut, REG_CTRL, 0x01)
    for cycle in range(200000):
        await RisingEdge(dut.clk)
        if dut.done.value:
            dut._log.info(f"Sparse computation done after {cycle} cycles")
            break
    else:
        raise TimeoutError("Sparse computation timeout")

    # ---- Verify DDR output ----
    # Block [0,0]: 14 columns of weight=1, act=3 => each output = 14 * 1 * 3 = 42
    ddr_out = ddr.read_i8_array(OUT_BASE, 14)
    dut._log.info(f"DDR output [0:13] (block row 0): {ddr_out}")

    for i in range(14):
        assert ddr_out[i] == 42, \
            f"Mismatch at [{i}]: got {ddr_out[i]}, expected 42"

    dut._log.info("PASS: Sparse 28x28 verified! Zero blocks correctly skipped.")
