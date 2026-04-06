#!/usr/bin/env bash
# =============================================================================
# run_openlane.sh — OpenLane 2 / OpenROAD P&R for soc_top_v2
# =============================================================================
# Prerequisites (install once):
#   Option A — Docker (recommended):
#     docker pull efabless/openlane2:latest
#
#   Option B — Nix (native, faster):
#     nix-env -iA nixpkgs.openlane2
#
# Usage:
#   cd hw/openlane/soc_top_v2
#   ./run_openlane.sh              # full flow
#   ./run_openlane.sh synth        # synthesis only
#   ./run_openlane.sh floorplan    # up to floorplan
#   ./run_openlane.sh route        # up to routing
#
# Outputs land in:
#   hw/openlane/soc_top_v2/runs/soc_top_v2_<timestamp>/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESIGN_NAME="soc_top_v2"
CONFIG="$SCRIPT_DIR/config.tcl"

# ── Detect run mode ──────────────────────────────────────────────────────────
MODE="${1:-full}"

case "$MODE" in
  synth)    LAST_STEP="synthesis"           ;;
  floorplan) LAST_STEP="floorplan"          ;;
  place)    LAST_STEP="placement"           ;;
  cts)      LAST_STEP="cts"                 ;;
  route)    LAST_STEP="routing"             ;;
  full|*)   LAST_STEP=""                    ;;
esac

# ── Refresh RTL snapshot if prep script exists ───────────────────────────────
if [[ -x "$SCRIPT_DIR/prep_openlane_sources.sh" ]]; then
    echo "[1/4] Refreshing RTL snapshot..."
    bash "$SCRIPT_DIR/prep_openlane_sources.sh"
fi

# ── Choose execution backend ─────────────────────────────────────────────────
if command -v openlane &>/dev/null; then
    BACKEND="native"
elif command -v docker &>/dev/null && docker image inspect efabless/openlane2 &>/dev/null; then
    BACKEND="docker"
else
    echo "ERROR: Neither 'openlane' binary nor efabless/openlane2 Docker image found."
    echo ""
    echo "Install one of:"
    echo "  Docker:  docker pull efabless/openlane2:latest"
    echo "  Nix:     nix-env -iA nixpkgs.openlane2"
    exit 1
fi

echo "[2/4] Backend: $BACKEND"
echo "[3/4] Design:  $DESIGN_NAME"
echo "[4/4] Mode:    $MODE${LAST_STEP:+ (stop after $LAST_STEP)}"
echo ""

# ── Build openlane command ───────────────────────────────────────────────────
if [[ "$BACKEND" == "native" ]]; then
    CMD=(openlane --design-dir "$SCRIPT_DIR" "$CONFIG")
    if [[ -n "$LAST_STEP" ]]; then
        CMD+=(--last-step "$LAST_STEP")
    fi
else
    # Docker execution — mount the full hw/openlane directory
    MOUNT_DIR="$(dirname "$SCRIPT_DIR")"
    INNER_CFG="/openlane/$(basename "$SCRIPT_DIR")/config.tcl"
    CMD=(docker run --rm -it
        -v "$MOUNT_DIR":/openlane
        -v "$HOME/.sky130":/root/.sky130 2>/dev/null || true
        efabless/openlane2:latest
        openlane --design-dir "/openlane/$(basename "$SCRIPT_DIR")" "$INNER_CFG")
    if [[ -n "$LAST_STEP" ]]; then
        CMD+=(--last-step "$LAST_STEP")
    fi
fi

echo "Running: ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo ""
echo "========================================================"
echo "Done.  Results in: $SCRIPT_DIR/runs/"
echo ""
echo "Next steps:"
echo "  1. Check timing:  grep 'slack' runs/*/reports/signoff/*.rpt"
echo "  2. Check DRC:     runs/*/reports/signoff/*drc*"
echo "  3. View GDS:      klayout runs/*/results/final/gds/*.gds"
