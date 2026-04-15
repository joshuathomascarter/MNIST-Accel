#!/usr/bin/env bash
# Waits for yosys_resynth3 to finish, then runs STA automatically.
set -e
cd "$(dirname "$0")"

echo "[$(date)] Waiting for Yosys synthesis to complete..."
wait $(pgrep -f "yosys synth_native" | head -1) 2>/dev/null || true

echo "[$(date)] Synthesis done. Running STA..."
docker run --rm \
  -v "$(pwd)":/design \
  -v openlane_pdk:/root/.volare \
  efabless/openlane:latest \
  bash -c "cd /design && /nix/store/mfkfv01n6h77wmkn12bgm57m5nb64ijv-opensta/bin/sta -no_init -exit sta_prects.tcl" \
  2>&1 | tee runs/timing_run/reports/sta_final.rpt

echo "[$(date)] STA complete. Results in runs/timing_run/reports/sta_final.rpt"
grep -E "slack|DONE|Error" runs/timing_run/reports/sta_final.rpt | head -20
