#!/bin/bash
# =============================================================================
# run_accel_tb.sh — Run accel_top testbench with Verilator
# =============================================================================
set -e

cd "$(dirname "$0")"

echo "=============================================="
echo "  Building accel_top testbench with Verilator"
echo "=============================================="

# Create output directory
mkdir -p sim_out

# Compile with Verilator
verilator --binary --trace -j 0 \
    -Wall -Wno-fatal -Wno-MULTIDRIVEN -Wno-BLKANDNBLK -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
    -DVERILATOR \
    -Irtl/top -Irtl/buffer -Irtl/control -Irtl/dma \
    -Irtl/mac -Irtl/systolic -Irtl/host_iface -Irtl/monitor -Irtl/meta \
    --Mdir sim_out \
    --top accel_top_tb \
    -o accel_top_tb \
    testbench/verilator/accel_top_tb.sv \
    rtl/top/accel_top.sv \
    rtl/host_iface/axi_lite_slave.sv \
    rtl/host_iface/axi_dma_bridge.sv \
    rtl/control/csr.sv \
    rtl/control/bsr_scheduler.sv \
    rtl/dma/act_dma.sv \
    rtl/dma/bsr_dma.sv \
    rtl/buffer/act_buffer.sv \
    rtl/buffer/wgt_buffer.sv \
    rtl/systolic/systolic_array_sparse.sv \
    rtl/systolic/pe.sv \
    rtl/meta/meta_decode.sv \
    rtl/monitor/perf.sv

echo ""
echo "=============================================="
echo "  Running testbench"
echo "=============================================="

./sim_out/accel_top_tb

echo ""
echo "Waveform saved to: accel_top_tb.vcd"
echo "View with: gtkwave accel_top_tb.vcd"
