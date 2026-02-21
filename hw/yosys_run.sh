#!/usr/bin/env bash
# =============================================================================
# yosys_run.sh — Preprocess RTL (strip assertions) then run Yosys synthesis
# =============================================================================
# Run from the hw/ directory:
#   bash yosys_run.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ACCEL-v1 Yosys Synthesis (Xilinx 7-series mapping)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Preprocess — strip SVA assertions / $fatal / $error ────────────
echo "[1/3] Preprocessing RTL (stripping SVA for Yosys)..."
rm -rf .yosys_prep
mkdir -p .yosys_prep

# Collect all .sv files (excluding deprecated)
SV_FILES=$(find rtl -name "*.sv" ! -path "*/deprecated/*" ! -name "accel_top_dual_clk.sv")

python3 - "$SV_FILES" <<'PYEOF'
import sys, re, os, glob

files = sys.argv[1].split()
for f in files:
    base = os.path.basename(f)
    with open(f) as fh:
        text = fh.read()

    # Remove property...endproperty blocks
    text = re.sub(r'(?m)^\s*property\b.*?endproperty\b[^\n]*\n?', '', text, flags=re.DOTALL)

    # Remove assert property / cover property lines
    text = re.sub(r'(?m)^\s*(assert|cover)\s+property\b[^;]*;\s*\n?', '', text)

    # Remove initial begin...end blocks containing assert/$fatal
    text = re.sub(
        r'(?m)^\s*initial\s+begin\b.*?^\s*end\b[^\n]*\n?',
        lambda m: '' if 'assert' in m.group() or '$fatal' in m.group() else m.group(),
        text, flags=re.DOTALL
    )

    # Remove always @(...) begin...end blocks that are ONLY assertions
    def strip_assert_always(m):
        body = m.group()
        # Check if every statement line is an assert or $error/$fatal/$warning
        lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('//')]
        # Keep the block only if it has non-assert logic
        non_assert = [l for l in lines
                      if not re.match(r'(always|begin|end|assert|else|\$fatal|\$error|\$warning|if\s*\(.*\)\s*assert)', l)]
        if not non_assert:
            return ''
        return body

    text = re.sub(
        r'(?m)^\s*always\s+@\(posedge\s+\w+\)\s+begin\b.*?^\s*end\b[^\n]*\n?',
        strip_assert_always, text, flags=re.DOTALL
    )

    # Remove any remaining standalone assert lines (multi-line with else)
    text = re.sub(r'(?m)^\s*assert\s*\(.*?\)\s*\n\s*else\s+\$\w+\([^)]*\);\s*\n?', '', text, flags=re.DOTALL)

    # Remove any remaining single-line assert...else
    text = re.sub(r'(?m)^\s*(if\s*\([^)]*\)\s*)?assert\s*\(.*?\)\s*else\s+\$\w+\([^)]*\);\s*\n?', '', text)

    # Remove ifdef ASSERT_ON ... endif blocks
    text = re.sub(r'(?m)^\s*`ifdef\s+ASSERT_ON\b.*?`endif[^\n]*\n?', '', text, flags=re.DOTALL)

    # Remove ifndef SYNTHESIS ... endif blocks (our SVA guards)
    text = re.sub(r'(?m)^\s*`ifndef\s+SYNTHESIS\b.*?`endif[^\n]*\n?', '', text, flags=re.DOTALL)

    with open(f'.yosys_prep/{base}', 'w') as out:
        out.write(text)

print(f'    Preprocessed {len(files)} files.')
PYEOF

echo "    Files in .yosys_prep/:"
ls .yosys_prep/*.sv | wc -l | tr -d ' '

# ── Step 2: Run Yosys ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Running Yosys synthesis (synth_xilinx -family xc7)..."
echo "    This may take 2-10 minutes..."
echo ""

mkdir -p reports

START=$(date +%s)
yosys -s yosys_synth.ys 2>&1 | tee reports/yosys_synth.log
YOSYS_EXIT=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "[3/3] Done in ${ELAPSED}s (exit code: $YOSYS_EXIT)"
echo ""

# ── Step 3: Extract key stats from log ─────────────────────────────────────
if [[ $YOSYS_EXIT -eq 0 ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  SYNTHESIS RESULTS                                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Extract the final stat block (after tech mapping)
    echo "── Resource Utilization (Xilinx 7-series mapped) ──"
    echo ""
    # Pull the stat output section
    grep -A 50 "Printing statistics" reports/yosys_synth.log | tail -n +2 | head -50
    echo ""

    # Zynq-7020 resource totals
    echo "── Zynq-7020 (xc7z020clg400-1) Available Resources ──"
    echo "  Slice LUTs:    53,200"
    echo "  Slice FFs:    106,400"
    echo "  BRAM (36Kb):      140"
    echo "  DSP48E1:          220"
    echo ""
    echo "── Reports ──"
    echo "  Full log:     reports/yosys_synth.log"
    echo "  Netlist JSON: reports/yosys_netlist.json"
    echo ""
    echo "NOTE: Yosys DSP inference for signed multiply-accumulate may"
    echo "differ from Vivado. Vivado typically infers DSP48E1 more"
    echo "aggressively for 8-bit MACs. Expect Vivado to show higher"
    echo "DSP usage and lower LUT usage than Yosys."
else
    echo "ERROR: Yosys synthesis failed. Check reports/yosys_synth.log"
fi

# Cleanup
# rm -rf .yosys_prep  # Uncomment to auto-clean

exit $YOSYS_EXIT
