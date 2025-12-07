#!/bin/bash
# Quick start guide for AXI-based ACCEL-v1

echo "================================"
echo "ACCEL-v1 AXI Burst DMA Guide"
echo "================================"
echo ""
echo "Speed comparison:"
echo "  UART:  14.4 KB/s  (82 sec for 1.18 MB)"
echo "  AXI:   400 MB/s   (3 ms for 1.18 MB)"
echo "  Speedup: 27,000×"
echo ""

# Test in simulator mode
echo "Running GEMM via AXI (simulator mode)..."
cd /workspaces/ACCEL-v1
python3 accel/python/host_axi/run_gemm_axi.py \
  --M 8 \
  --N 8 \
  --K 8 \
  --Tm 8 \
  --Tn 8 \
  --Tk 8 \
  --verbose \
  --simulator

echo ""
echo "Expected output: ✓ GEMM successful!"
echo ""
echo "For larger matrices (slower but still fast):"
echo "  python3 accel/python/host_axi/run_gemm_axi.py --M 32 --N 32 --K 32"
echo ""
echo "For FPGA deployment:"
echo "  1. Set up Vivado project with AXI interconnect"
echo "  2. Connect m_axi ports to AXI HP0 (DDR)"
echo "  3. Configure DDR base address in run_gemm_axi.py (default: 0x80000000)"
echo "  4. Run: python3 run_gemm_axi.py --M <size> --N <size> --K <size>"
