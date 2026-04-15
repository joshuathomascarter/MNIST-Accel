#!/usr/bin/env bash
# run_e2e_inference.sh — Build and run the end-to-end MNIST inference testbench
# Usage:  ./run_e2e_inference.sh [--build-only] [--run-only]
#
# Requires:  verilator >= 5.0 on PATH
# Outputs:   obj_dir_tb_inference/Vtb_mnist_inference
#            logs/e2e_inference.log
set -euo pipefail
cd "$(dirname "$0")"

MDIR="obj_dir_tb_inference"
LOG="logs/e2e_inference.log"
TB="hw/sim/sv/tb_mnist_inference.sv"
FLIST="hw/sim/sv/filelist.f"
FW="fw/firmware_inference.hex"
DRAM="data/dram_init.hex"

mkdir -p logs

BUILD=1
RUN=1
for arg in "$@"; do
  [[ "$arg" == "--build-only" ]] && RUN=0
  [[ "$arg" == "--run-only"   ]] && BUILD=0
done

if [[ $BUILD -eq 1 ]]; then
  echo "[$(date)] Building tb_mnist_inference with Verilator..."
  verilator --sv --binary --timing \
    -f "$FLIST" \
    "$TB" \
    --top-module tb_mnist_inference \
    --Mdir "$MDIR" \
    -CFLAGS "-O2" \
    2>&1 | tee logs/verilator_build.log

  if [[ ! -f "$MDIR/Vtb_mnist_inference" ]]; then
    echo "[ERROR] Verilator build failed — check logs/verilator_build.log"
    exit 1
  fi
  echo "[$(date)] Build complete."
fi

if [[ $RUN -eq 1 ]]; then
  echo "[$(date)] Running inference simulation..."
  echo "[$(date)] firmware : $FW"
  echo "[$(date)] dram_init: $DRAM"

  "$MDIR/Vtb_mnist_inference" \
    "+firmware=$FW" \
    "+dram_init=$DRAM" \
    2>&1 | tee "$LOG"

  echo ""
  echo "=== QUICK SUMMARY ==="
  grep -E "PASS|FAIL|Predicted|Cycles|RESULT|Performance|Throughput" "$LOG" | head -20
  echo ""
  echo "Full log: $LOG"
fi
