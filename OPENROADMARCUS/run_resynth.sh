#!/usr/bin/env bash
# Re-synthesize with fixed RTL (blackbox SRAM, no behavioral memory)
set -e
cd "$(dirname "$0")"
mkdir -p runs/timing_run/results/synthesis runs/timing_run/reports

echo "=== Yosys re-synthesis ===" | tee runs/timing_run/reports/yosys_resynth.log
docker run --rm \
  -v "$(pwd)":/design \
  -v openlane_pdk:/root/.volare \
  efabless/openlane:latest \
  bash -c "cd /design && yosys synth_native.ys" \
  2>&1 | tee -a runs/timing_run/reports/yosys_resynth.log

echo ""
echo "=== STA ===" | tee -a runs/timing_run/reports/yosys_resynth.log
docker run --rm \
  -v "$(pwd)":/design \
  -v openlane_pdk:/root/.volare \
  efabless/openlane:latest \
  bash -c "cd /design && openroad -no_init -exit sta_native.tcl" \
  2>&1 | tee runs/timing_run/reports/sta_native.rpt
