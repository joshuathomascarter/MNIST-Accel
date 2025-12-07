#!/bin/bash
# =============================================================================
# run_coverage.sh — Run accel_top testbench with Verilator coverage
# =============================================================================
set -e

cd "$(dirname "$0")"

echo "=============================================="
echo "  Building Full Testbench with Coverage"
echo "=============================================="

# Create output directory
mkdir -p sim_out_cov

# Compile main testbench with Verilator and coverage
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal -Wno-MULTIDRIVEN -Wno-BLKANDNBLK -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
    -DVERILATOR \
    -Irtl/top -Irtl/buffer -Irtl/control -Irtl/dma \
    -Irtl/mac -Irtl/systolic -Irtl/host_iface -Irtl/monitor -Irtl/meta \
    --Mdir sim_out_cov \
    --top accel_top_tb_full \
    -o accel_top_tb_full \
    testbench/verilator/accel_top_tb_full.sv \
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
    rtl/monitor/perf.sv \
    rtl/mac/mac8.sv

echo ""
echo "=============================================="
echo "  Building PE Unit Test with Coverage"
echo "=============================================="

# Compile PE unit test
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal \
    -DVERILATOR \
    -Irtl/systolic -Irtl/mac \
    --Mdir sim_out_cov/pe \
    --top pe_tb \
    -o pe_tb \
    testbench/verilator/pe_tb.sv \
    rtl/systolic/pe.sv \
    rtl/mac/mac8.sv

echo ""
echo "=============================================="
echo "  Building Systolic Array Test with Coverage"
echo "=============================================="

# Compile Systolic Array unit test
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal \
    -DVERILATOR \
    -Irtl/systolic -Irtl/mac \
    --Mdir sim_out_cov/systolic \
    --top systolic_tb \
    -o systolic_tb \
    testbench/verilator/systolic_tb.sv \
    rtl/systolic/systolic_array_sparse.sv \
    rtl/systolic/pe.sv \
    rtl/mac/mac8.sv

echo ""
echo "=============================================="
echo "  Running Full Testbench"
echo "=============================================="

cd sim_out_cov
./accel_top_tb_full

echo ""
echo "=============================================="
echo "  Running PE Unit Test"
echo "=============================================="

cd pe
./pe_tb
cd ..

echo ""
echo "=============================================="
echo "  Running Systolic Array Test"
echo "=============================================="

cd systolic
./systolic_tb
cd ..

echo ""
echo "=============================================="
echo "  Building Meta Decode Test with Coverage"
echo "=============================================="

cd ..

# Compile Meta Decode unit test
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal \
    -DVERILATOR \
    -Irtl/meta \
    --Mdir sim_out_cov/meta \
    --top meta_decode_tb \
    -o meta_decode_tb \
    testbench/verilator/meta_decode_tb.sv \
    rtl/meta/meta_decode.sv

echo ""
echo "=============================================="
echo "  Running Meta Decode Test"
echo "=============================================="

cd sim_out_cov/meta
./meta_decode_tb
cd ..

echo ""
echo "=============================================="
echo "  Building Performance Monitor Test"
echo "=============================================="

cd ..

# Compile Perf Monitor unit test
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal \
    -DVERILATOR \
    -Irtl/monitor \
    --Mdir sim_out_cov/perf \
    --top perf_tb \
    -o perf_tb \
    testbench/verilator/perf_tb.sv \
    rtl/monitor/perf.sv

echo ""
echo "=============================================="
echo "  Running Performance Monitor Test"
echo "=============================================="

cd sim_out_cov/perf
./perf_tb
cd ..

echo ""
echo "=============================================="
echo "  Building BSR DMA Test"
echo "=============================================="

cd ..

# Compile BSR DMA unit test
verilator --binary --trace -j 0 \
    --coverage --coverage-line --coverage-toggle \
    -Wall -Wno-fatal \
    -DVERILATOR \
    -Irtl/dma \
    --Mdir sim_out_cov/bsr_dma \
    --top bsr_dma_tb \
    -o bsr_dma_tb \
    testbench/verilator/bsr_dma_tb.sv \
    rtl/dma/bsr_dma.sv

echo ""
echo "=============================================="
echo "  Running BSR DMA Test"
echo "=============================================="

cd sim_out_cov/bsr_dma
./bsr_dma_tb
cd ..

echo ""
echo "=============================================="
echo "  Merging Coverage Data"
echo "=============================================="

# Merge coverage from all testbenches
verilator_coverage --write merged_coverage.dat coverage.dat pe/coverage.dat systolic/coverage.dat meta/coverage.dat perf/coverage.dat bsr_dma/coverage.dat
cd ..

echo ""
echo "=============================================="
echo "  Generating Coverage Report"
echo "=============================================="

# Generate annotated coverage report from merged data
verilator_coverage --annotate coverage_annotate --annotate-min 1 sim_out_cov/merged_coverage.dat
verilator_coverage --write-info coverage.info sim_out_cov/merged_coverage.dat

echo ""
echo "Coverage data written to: sim_out_cov/merged_coverage.dat"
echo "Annotated source in: coverage_annotate/"

# Print summary
echo ""
echo "=============================================="
echo "  Coverage Summary"
echo "=============================================="
verilator_coverage --rank coverage.dat 2>/dev/null | head -40 || true
