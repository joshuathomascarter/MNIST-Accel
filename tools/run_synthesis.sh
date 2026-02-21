#!/usr/bin/env bash
# =============================================================================
# run_synthesis.sh — One-command Vivado synthesis for ACCEL-v1
# =============================================================================
# Author: Joshua Carter  |  February 2026
#
# This wraps the TCL script so you can run it from anywhere.
#
# USAGE:
#   From ANY directory:
#     bash /path/to/ResNet-Accel-2/tools/run_synthesis.sh
#
#   Or from the project root:
#     ./tools/run_synthesis.sh
#
# PREREQUISITES:
#   - Vivado 2023.2+ installed and on PATH
#     (or source /opt/Xilinx/Vivado/<version>/settings64.sh first)
#   - Zynq-7020 part available (included in free WebPack license)
#
# OUTPUTS:
#   hw/reports/                        — All utilization/timing/power reports
#   hw/reports/synthesis_summary.json  — Machine-readable summary
#   hw/vivado_proj/                    — Full Vivado project (can open in GUI)
#   hw/accel_v1.bit                    — Bitstream (if P&R succeeds)
#   hw/vivado_synth.log                — Full Vivado console log
#
# ESTIMATED TIME:
#   Synthesis:       5-15 min
#   Implementation:  10-30 min
#   Bitstream:       2-5 min
#   Total:           ~20-50 min (depends on machine)
# =============================================================================
set -euo pipefail

# ─── Locate project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HW_DIR="$PROJECT_ROOT/hw"
TCL_SCRIPT="$PROJECT_ROOT/tools/synthesize_vivado.tcl"

echo "============================================="
echo "  ACCEL-v1 Vivado Synthesis Flow"
echo "============================================="
echo "Project root: $PROJECT_ROOT"
echo "HW directory: $HW_DIR"
echo "TCL script:   $TCL_SCRIPT"
echo ""

# ─── Check Vivado is available ────────────────────────────────────────────────
if ! command -v vivado &>/dev/null; then
    echo "ERROR: 'vivado' not found on PATH."
    echo ""
    echo "Try one of:"
    echo "  source /opt/Xilinx/Vivado/2023.2/settings64.sh"
    echo "  source /opt/Xilinx/Vivado/2024.1/settings64.sh"
    echo "  source /tools/Xilinx/Vivado/<version>/settings64.sh"
    echo ""
    echo "Or add Vivado bin/ to your PATH manually."
    
    # Try to find it automatically
    VIVADO_FOUND=$(find /opt /tools /usr/local 2>/dev/null -maxdepth 5 \
                   -name "settings64.sh" -path "*/Vivado/*" 2>/dev/null | \
                   sort -V | tail -1 || true)
    if [[ -n "$VIVADO_FOUND" ]]; then
        echo ""
        echo "Found Vivado settings at: $VIVADO_FOUND"
        echo "Run:  source $VIVADO_FOUND"
        echo "Then re-run this script."
    fi
    exit 1
fi

VIVADO_VER=$(vivado -version 2>/dev/null | head -1 || echo "unknown")
echo "Vivado version: $VIVADO_VER"

# ─── Verify RTL files exist ──────────────────────────────────────────────────
RTL_COUNT=$(find "$HW_DIR/rtl" -name "*.sv" | wc -l | tr -d ' ')
echo "RTL files found: $RTL_COUNT .sv files"

if [[ "$RTL_COUNT" -lt 15 ]]; then
    echo "WARNING: Expected ~21 .sv files, found only $RTL_COUNT"
    echo "Make sure the full RTL is present."
fi
echo ""

# ─── Run Vivado ──────────────────────────────────────────────────────────────
cd "$HW_DIR"
echo "Working directory: $(pwd)"
echo "Starting Vivado in batch mode..."
echo "Log: $HW_DIR/vivado_synth.log"
echo ""
echo "This will take 20-50 minutes. Progress:"
echo "  [1/9] Create project"
echo "  [2/9] Add RTL sources"
echo "  [3/9] Add timing constraints"
echo "  [4/9] Run synthesis          (~5-15 min)"
echo "  [5/9] Check post-synth timing"
echo "  [6/9] Run implementation     (~10-30 min)"
echo "  [7/9] Check post-route timing"
echo "  [8/9] Generate bitstream     (~2-5 min)"
echo "  [9/9] Generate reports"
echo ""

START_TIME=$(date +%s)

# Run Vivado batch mode, tee to log file
vivado -mode batch -source "$TCL_SCRIPT" -nojournal 2>&1 | tee "$HW_DIR/vivado_synth.log"

VIVADO_EXIT=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "============================================="
if [[ $VIVADO_EXIT -eq 0 ]]; then
    echo "  SYNTHESIS COMPLETE — ${MINUTES}m ${SECONDS}s"
else
    echo "  SYNTHESIS FAILED (exit code $VIVADO_EXIT) — ${MINUTES}m ${SECONDS}s"
fi
echo "============================================="
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
if [[ -f "$HW_DIR/reports/synthesis_summary.json" ]]; then
    echo "── Summary (from reports/synthesis_summary.json) ──"
    cat "$HW_DIR/reports/synthesis_summary.json"
    echo ""
fi

echo "── Generated Reports ──"
ls -lh "$HW_DIR/reports/"*.rpt 2>/dev/null || echo "  (no .rpt files found)"
echo ""

if [[ -f "$HW_DIR/accel_v1.bit" ]]; then
    BITSIZE=$(ls -lh "$HW_DIR/accel_v1.bit" | awk '{print $5}')
    echo "── Bitstream ──"
    echo "  $HW_DIR/accel_v1.bit ($BITSIZE)"
    echo ""
fi

echo "── Key Reports to Check ──"
echo "  Utilization: $HW_DIR/reports/impl_utilization.rpt"
echo "  Timing:      $HW_DIR/reports/impl_timing.rpt"
echo "  Power:       $HW_DIR/reports/impl_power.rpt"
echo ""

if [[ $VIVADO_EXIT -eq 0 ]]; then
    echo "Next: copy accel_v1.bit to PYNQ-Z2 and run hardware tests."
fi

exit $VIVADO_EXIT
