#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FPGA_DIR="$REPO_ROOT/handoff/soc_top_v2/fpga_bringup"
ASIC_DIR="$REPO_ROOT/handoff/soc_top_v2/asic_bringup"
ARCHIVE_DIR="$REPO_ROOT/archive/old_project_done"
INDEX_README="$REPO_ROOT/handoff/soc_top_v2/README.md"

OPENLANE_PREP="$REPO_ROOT/hw/openlane/soc_top_v2/prep_openlane_sources.sh"

copy_files() {
    local destination="$1"
    shift
    mkdir -p "$destination"
    cp "$@" "$destination/"
}

prune_obj_dirs() {
    local root="$1"
    find "$root" -type d -name 'obj_dir*' -prune -exec rm -rf {} +
}

write_manifest() {
    local rel_dir="$1"
    local manifest_path="$REPO_ROOT/$rel_dir/MANIFEST.txt"
    local tmp_manifest
    tmp_manifest="$(mktemp)"
    (
        cd "$REPO_ROOT"
        find "$rel_dir" -type f ! -name 'MANIFEST.txt' | LC_ALL=C sort > "$tmp_manifest"
    )
    mv "$tmp_manifest" "$manifest_path"
}

rm -rf "$FPGA_DIR" "$ASIC_DIR" "$ARCHIVE_DIR"
mkdir -p "$FPGA_DIR" "$ASIC_DIR" "$ARCHIVE_DIR"

bash "$OPENLANE_PREP"

cp -R "$REPO_ROOT/hw/rtl" "$FPGA_DIR/"
cp -R "$REPO_ROOT/hw/sim" "$FPGA_DIR/"
prune_obj_dirs "$FPGA_DIR/sim"
cp -R "$REPO_ROOT/fw" "$FPGA_DIR/"
copy_files "$FPGA_DIR/tools" \
    "$REPO_ROOT/tools/run_synthesis.sh" \
    "$REPO_ROOT/tools/synthesize_vivado.tcl" \
    "$REPO_ROOT/tools/build.sh" \
    "$REPO_ROOT/tools/test.sh" \
    "$REPO_ROOT/tools/ci_verilator.sh" \
    "$REPO_ROOT/tools/Makefile.verilator" \
    "$REPO_ROOT/tools/soc_top_v2_uart_host.py" \
    "$REPO_ROOT/tools/fpga_uart_block_demo.py" \
    "$REPO_ROOT/tools/asic_uart_block_demo.py"
copy_files "$FPGA_DIR/docs" \
    "$REPO_ROOT/docs/guides/SIMULATION_GUIDE.md" \
    "$REPO_ROOT/docs/guides/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md" \
    "$REPO_ROOT/docs/verification/TEST_RESULTS.md" \
    "$REPO_ROOT/docs/architecture/ARCHITECTURE.md" \
    "$REPO_ROOT/docs/RTL_MEMORY_AND_CPP_GUIDE.md" \
    "$REPO_ROOT/docs/zynq_ports_guide.md"
copy_files "$FPGA_DIR/hw" "$REPO_ROOT/hw/firmware.hex"
copy_files "$FPGA_DIR" "$REPO_ROOT/requirements.txt"

# Board constraints (FPGA-only)
if [[ -d "$REPO_ROOT/hw/constraints" ]]; then
    cp -R "$REPO_ROOT/hw/constraints" "$FPGA_DIR/"
fi

cp -R "$REPO_ROOT/hw/openlane/soc_top_v2" "$ASIC_DIR/openlane"
cp -R "$REPO_ROOT/hw/rtl" "$ASIC_DIR/rtl_reference"
cp -R "$REPO_ROOT/fw" "$ASIC_DIR/fw_reference"
copy_files "$ASIC_DIR/hw_reference" "$REPO_ROOT/hw/firmware.hex"
copy_files "$ASIC_DIR/tools_reference" \
    "$REPO_ROOT/hw/strip_assert.py" \
    "$REPO_ROOT/hw/yosys_run.sh" \
    "$REPO_ROOT/hw/yosys_run_soc_top_v2.sh" \
    "$REPO_ROOT/hw/yosys_synth.ys" \
    "$REPO_ROOT/tools/soc_top_v2_uart_host.py" \
    "$REPO_ROOT/tools/fpga_uart_block_demo.py" \
    "$REPO_ROOT/tools/asic_uart_block_demo.py"
copy_files "$ASIC_DIR/docs_reference" \
    "$REPO_ROOT/docs/architecture/ARCHITECTURE.md" \
    "$REPO_ROOT/docs/specs/soc_microarch_spec_v0.2.md" \
    "$REPO_ROOT/docs/critical_path_timing.md" \
    "$REPO_ROOT/docs/RTL_MEMORY_AND_CPP_GUIDE.md" \
    "$REPO_ROOT/docs/verification/TEST_RESULTS.md"
cp -R "$REPO_ROOT/hw/sim/sv" "$ASIC_DIR/sim_reference"
prune_obj_dirs "$ASIC_DIR/sim_reference"
copy_files "$ASIC_DIR" "$REPO_ROOT/requirements.txt"

mkdir -p "$ARCHIVE_DIR/sw/ml_python"
mkdir -p "$ARCHIVE_DIR/sw/cpp/include"
mkdir -p "$ARCHIVE_DIR/sw/cpp/src"
mkdir -p "$ARCHIVE_DIR/sw/cpp/apps"

cp -R "$REPO_ROOT/sw/ml_python/host" "$ARCHIVE_DIR/sw/ml_python/"
cp -R "$REPO_ROOT/sw/cpp/include/driver" "$ARCHIVE_DIR/sw/cpp/include/"
cp -R "$REPO_ROOT/sw/cpp/include/memory" "$ARCHIVE_DIR/sw/cpp/include/"
cp -R "$REPO_ROOT/sw/cpp/include/compute" "$ARCHIVE_DIR/sw/cpp/include/"
cp -R "$REPO_ROOT/sw/cpp/src/driver" "$ARCHIVE_DIR/sw/cpp/src/"
cp -R "$REPO_ROOT/sw/cpp/src/memory" "$ARCHIVE_DIR/sw/cpp/src/"
cp -R "$REPO_ROOT/sw/cpp/src/compute" "$ARCHIVE_DIR/sw/cpp/src/"
copy_files "$ARCHIVE_DIR/sw/cpp/apps" \
    "$REPO_ROOT/sw/cpp/apps/benchmark.cpp" \
    "$REPO_ROOT/sw/cpp/apps/run_mnist_inference.cpp"

cat > "$INDEX_README" <<'EOF'
# soc_top_v2 Handoff Bundles

- `fpga_bringup/` contains the staged FPGA-side source bundle for `soc_top_v2`.
- `fpga_bringup/rtl/top/pynq_z2_wrapper.sv` is the FPGA bringup top with a persistent local DRAM backing store.
- `fpga_bringup/tools/fpga_uart_block_demo.py` is the FPGA preload / compute / readback host entrypoint.
- `fpga_bringup/docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md` is the concrete board-side punch list.
- `asic_bringup/` contains the staged pre-OpenROAD ASIC-side source bundle for `soc_top_v2`.
- `asic_bringup/rtl_reference/top/soc_top_v2_asic_sim_wrapper.sv` is the ASIC pre-silicon bringup wrapper with the same backing-store model.
- `asic_bringup/tools_reference/asic_uart_block_demo.py` is the ASIC preload / compute / readback host entrypoint.

Each subdirectory includes a `README.md` with usage notes and a `MANIFEST.txt`
with the exact staged files.
EOF

cat > "$FPGA_DIR/README.md" <<'EOF'
# soc_top_v2 FPGA Bringup Bundle

This directory is the staged FPGA-side handoff for `soc_top_v2`.

## Included

- `rtl/` — live SystemVerilog RTL snapshot
- `sim/` — simulation sources and filelists with generated `obj_dir*` folders pruned
- `fw/` — bare-metal firmware sources plus the current built firmware artifacts
- `tools/` — Vivado and local verification helper scripts
- `tools/fpga_uart_block_demo.py` + `tools/soc_top_v2_uart_host.py` — UART preload / block-execute / readback flow
- `docs/` — architecture, simulation, verification, and port-reference documents
- `docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md` — board-side integration punch list
- `hw/firmware.hex` — boot ROM image mirrored next to the bundle

## Not Included

- FPGA bitstream output
- Vivado-generated implementation reports
- board-specific wrapper, block design, or pin-assignment outputs

## What Is Already Done In This Workspace

- `soc_top_v2` Verilator simulation is passing on the main SoC testbench
- tile-level Verilator regression coverage is present in `sim/`
- firmware image and source are already staged for UART-led smoke bring-up

## External Steps Still Required

1. Run `tools/run_synthesis.sh` in a Vivado environment.
2. Work through `docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md`.
3. Generate the bitstream and board reports.
4. Run `tools/fpga_uart_block_demo.py --port <tty>` for the preload / compute / readback loop.

Use `MANIFEST.txt` for the exact file list.
EOF

cat > "$ASIC_DIR/README.md" <<'EOF'
# soc_top_v2 ASIC Bringup Bundle

This directory is the staged pre-OpenROAD ASIC-side handoff for `soc_top_v2`.

## Included

- `openlane/` — regenerated source bundle, constraints, config, UPF, and planning docs
- `rtl_reference/` — raw RTL snapshot for correlation against the filtered OpenLane bundle
- `sim_reference/` — source-side simulation collateral used to derive the ASIC bundle
- `fw_reference/` and `hw_reference/firmware.hex` — bring-up software context
- `tools_reference/` — source-prep and Yosys helper scripts
- `tools_reference/asic_uart_block_demo.py` + `tools_reference/soc_top_v2_uart_host.py` — UART preload / block-execute / readback flow
- `docs_reference/` — architecture, timing, verification, and microarchitecture docs

## Not Included

- OpenROAD or OpenLane run outputs
- generated DEF/GDS/LEF/LIB artifacts
- backend signoff reports

## What Is Already Done In This Workspace

- `openlane/src/` has been regenerated from the live simulation filelist
- `openlane/config.tcl` is aligned to the current `soc_top_v2` source set
- reset, power, pad, macro, and DFT planning collateral are all staged together

## External Steps Still Required

1. Run the backend flow in the target OpenROAD/OpenLane environment.
2. Bind real foundry SRAM macros and tech libraries.
3. Complete scan/DFT insertion and package-aware pad timing.
4. Review DRC, LVS, STA, IR-drop, and congestion results.
5. Use `rtl_reference/top/soc_top_v2_asic_sim_wrapper.sv` plus `tools_reference/asic_uart_block_demo.py` for pre-silicon preload / compute / readback validation.

Use `MANIFEST.txt` for the exact file list.
EOF

cat > "$ARCHIVE_DIR/README.md" <<'EOF'
# Old Project Archive

This directory snapshots the legacy host-side FPGA software stack that is no
longer part of the active `soc_top_v2` bringup path.

## Included

- `sw/ml_python/host/`
- `sw/cpp/include/{driver,memory,compute}/`
- `sw/cpp/src/{driver,memory,compute}/`
- `sw/cpp/apps/{benchmark.cpp,run_mnist_inference.cpp}`

Use this archive as reference only. Active bringup artifacts now live under
`handoff/soc_top_v2/`.
EOF

write_manifest "handoff/soc_top_v2/fpga_bringup"
write_manifest "handoff/soc_top_v2/asic_bringup"
write_manifest "archive/old_project_done"

echo "Staged soc_top_v2 FPGA bringup bundle at: $FPGA_DIR"
echo "Staged soc_top_v2 ASIC bringup bundle at: $ASIC_DIR"
echo "Archived legacy host-side stack at: $ARCHIVE_DIR"