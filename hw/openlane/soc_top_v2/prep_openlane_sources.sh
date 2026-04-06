#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HW_DIR="$REPO_ROOT/hw"
FILELIST="$HW_DIR/sim/sv/filelist.f"
TMP_PREP="$SCRIPT_DIR/.tmp_prep"
SRC_DIR="$SCRIPT_DIR/src"
CONFIG_TCL="$SCRIPT_DIR/config.tcl"
PYTHON_BIN="${PYTHON_BIN:-python3}"

rm -rf "$TMP_PREP" "$SRC_DIR"
mkdir -p "$SRC_DIR"

pushd "$HW_DIR" >/dev/null
"$PYTHON_BIN" strip_assert.py rtl "$TMP_PREP" >/dev/null
popd >/dev/null

while IFS= read -r raw_line; do
  line="${raw_line#${raw_line%%[![:space:]]*}}"
  [[ -z "$line" || "$line" == //* ]] && continue
  if [[ "$line" == *.sv ]]; then
    rel="${line#*/hw/}"
    [[ "$rel" == "rtl/top/soc_top.sv" ]] && continue
    [[ "$rel" == "rtl/memory/coherence_demo_top.sv" ]] && continue
    [[ "$rel" == "rtl/memory/directory_controller.sv" ]] && continue
    [[ "$rel" == "rtl/memory/snoop_filter.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_bandwidth_steal.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_innet_reduce.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_qos_shaper.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_router_sva.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_traffic_gen.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_vc_allocator_sparse.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_vc_allocator_qvn.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_vc_allocator_static_prio.sv" ]] && continue
    [[ "$rel" == "rtl/noc/noc_vc_allocator_weighted_rr.sv" ]] && continue
    [[ "$rel" == "rtl/noc/reduce_engine.sv" ]] && continue
    [[ "$rel" == "rtl/noc/scatter_engine.sv" ]] && continue
    mkdir -p "$SRC_DIR/$(dirname "$rel")"
    cp "$TMP_PREP/$rel" "$SRC_DIR/$rel"
  fi
done < "$FILELIST"

rm -rf "$TMP_PREP"

cat > "$CONFIG_TCL" <<'TCL'
set ::env(DESIGN_NAME) soc_top_v2
set ::env(CLOCK_PORT) clk
set ::env(CLOCK_PERIOD) 20.0
set ::env(VERILOG_DEFINES) [list SYNTHESIS ASIC_SYNTHESIS]
set ::env(BASE_SDC_FILE) $::env(DESIGN_DIR)/constraints/soc_top_v2.sdc

set ::env(VERILOG_FILES) [concat \
  [list \
    $::env(DESIGN_DIR)/src/rtl/top/soc_pkg.sv \
    $::env(DESIGN_DIR)/src/rtl/noc/noc_pkg.sv \
    $::env(DESIGN_DIR)/src/rtl/memory/coherence_pkg.sv \
  ] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/buffer/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/cache/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/control/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/dma/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/dram/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/hft/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/host_iface/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/mac/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/memory/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/monitor/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/noc/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/periph/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/systolic/*.sv]] \
  [list \
    $::env(DESIGN_DIR)/src/rtl/top/axi_addr_decoder.sv \
    $::env(DESIGN_DIR)/src/rtl/top/axi_arbiter.sv \
    $::env(DESIGN_DIR)/src/rtl/top/axi_crossbar.sv \
    $::env(DESIGN_DIR)/src/rtl/top/obi_to_axi.sv \
    $::env(DESIGN_DIR)/src/rtl/top/simple_cpu.sv \
    $::env(DESIGN_DIR)/src/rtl/top/soc_top_v2.sv \
  ] \
]

set ::env(FP_SIZING) relative
set ::env(FP_CORE_UTIL) 35
set ::env(PL_TARGET_DENSITY) 0.45
set ::env(RUN_KLAYOUT) 0
set ::env(RUN_MAGIC) 0
TCL

echo "Prepared OpenLane source bundle in: $SRC_DIR"
echo "Generated config: $CONFIG_TCL"
