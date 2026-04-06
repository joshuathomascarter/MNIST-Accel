#!/usr/bin/env python3
"""
VC Allocator Area/Resource Comparison via Yosys
================================================
Synthesizes each VC allocator variant independently and reports:
  - LUT count (combinational logic complexity)
  - FF/register count (state bits)
  - Total cell count
  - Relative overhead vs baseline

Requirements: yosys installed and on PATH.

If Yosys is not available, generates ESTIMATED area from RTL analysis
(line count, register count, mux width) as a fallback.

Usage:
    python3 tools/synth_allocator_area.py
"""

import os
import re
import subprocess
import sys
import json
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL_DIR = os.path.join(PROJECT_ROOT, "hw", "rtl", "noc")

ALLOCATORS = [
    ("Baseline RR",           "noc_vc_allocator.sv",              "noc_vc_allocator"),
    ("Static Priority",       "noc_vc_allocator_static_prio.sv",  "noc_vc_allocator_static_prio"),
    ("Weighted RR (3:1)",     "noc_vc_allocator_weighted_rr.sv",  "noc_vc_allocator_weighted_rr"),
    ("QVN (2+2 split)",       "noc_vc_allocator_qvn.sv",          "noc_vc_allocator_qvn"),
    ("Sparsity-Aware (Ours)", "noc_vc_allocator_sparse.sv",       "noc_vc_allocator_sparse"),
]


def check_yosys():
    """Check if Yosys is available."""
    try:
        result = subprocess.run(["yosys", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def synthesize_with_yosys(sv_file, top_module, pkg_file):
    """Attempt Yosys synthesis. Returns None if SV parsing fails."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as f:
        script = f"""
# Auto-generated Yosys script for {top_module}
read_verilog -sv -DSYNTHESIS {pkg_file}
read_verilog -sv -DSYNTHESIS {sv_file}
hierarchy -top {top_module}
proc; opt; fsm; opt; memory; opt
techmap; opt
abc -lut 6
opt_clean
stat
"""
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["yosys", script_path],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr

        if result.returncode != 0 or 'ERROR' in output:
            return None  # Fall back to RTL analysis

        stats = {}
        cells_match = re.search(r'Number of cells:\s+(\d+)', output)
        wires_match = re.search(r'Number of wires:\s+(\d+)', output)
        lut_match = re.search(r'\$lut\s+(\d+)', output)
        ff_match = re.search(r'\$_DFF_\w+_\s+(\d+)', output)

        if cells_match:
            stats['cells'] = int(cells_match.group(1))
        if wires_match:
            stats['wires'] = int(wires_match.group(1))
        if lut_match:
            stats['luts'] = int(lut_match.group(1))
        if ff_match:
            stats['ffs'] = int(ff_match.group(1))

        if not stats:
            return None

        return stats

    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        os.unlink(script_path)


def estimate_from_rtl(sv_path):
    """Fallback: estimate area from RTL analysis when Yosys unavailable."""
    with open(sv_path, 'r') as f:
        code = f.read()

    lines = len(code.splitlines())

    # Count registers (always_ff blocks and explicit reg/logic declarations in always_ff)
    ff_blocks = len(re.findall(r'always_ff', code))
    reg_assigns = len(re.findall(r'<=', code))

    # Count muxes (ternary operators and case/if-else in always_comb)
    muxes = len(re.findall(r'\?.*:', code))
    case_labels = len(re.findall(r'(?:case|if|else if)', code))

    # Count parameters that affect size
    num_ports = 5  # Default
    num_vcs = 4    # Default

    # Estimate: each register ≈ 1 FF, each mux/comparison ≈ 2-4 LUTs
    est_ffs = reg_assigns * num_ports  # Per-port state
    est_luts = (muxes + case_labels) * 3 * num_ports  # Combinational per port
    est_cells = est_ffs + est_luts

    return {
        'lines': lines,
        'ff_blocks': ff_blocks,
        'reg_assigns': reg_assigns,
        'muxes': muxes + case_labels,
        'est_ffs': est_ffs,
        'est_luts': est_luts,
        'est_cells': est_cells,
    }


def main():
    pkg_file = os.path.join(RTL_DIR, "noc_pkg.sv")

    print("=" * 80)
    print("  VC Allocator Area/Resource Comparison")
    print("=" * 80)

    has_yosys = check_yosys()

    if has_yosys:
        print("  Yosys detected — running real synthesis (LUT-6 mapping)\n")
    else:
        print("  Yosys NOT found — using RTL analysis estimates")
        print("  Install Yosys for accurate numbers: brew install yosys\n")

    results = {}
    use_estimate = False

    for name, sv_name, top_mod in ALLOCATORS:
        sv_path = os.path.join(RTL_DIR, sv_name)
        if not os.path.exists(sv_path):
            print(f"  [SKIP] {name}: {sv_name} not found")
            continue

        print(f"  Analyzing: {name} ({sv_name})...")

        yosys_ok = False
        if has_yosys:
            stats = synthesize_with_yosys(sv_path, top_mod, pkg_file)
            if stats is not None:
                yosys_ok = True
                results[name] = stats

        if not yosys_ok:
            if has_yosys and not use_estimate:
                print("    Yosys SV parse failed — falling back to RTL analysis")
                use_estimate = True
            results[name] = estimate_from_rtl(sv_path)

    # Determine output mode
    got_yosys_data = any('cells' in v for v in results.values())

    # Print comparison table
    print(f"\n{'='*80}")

    if got_yosys_data:
        print(f"  {'Allocator':<28} {'LUTs':>8} {'FFs':>8} {'Cells':>8} {'Overhead':>10}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

        base_cells = results.get("Baseline RR", {}).get('cells', 1)
        for name, _, _ in ALLOCATORS:
            if name not in results:
                continue
            s = results[name]
            if 'error' in s:
                print(f"  {name:<28} {'ERROR':>8} — {s['error']}")
                continue
            luts = s.get('luts', '?')
            ffs = s.get('ffs', '?')
            cells = s.get('cells', '?')
            if isinstance(cells, int) and base_cells > 0:
                overhead = f"{(cells - base_cells) / base_cells * 100:+.1f}%"
            else:
                overhead = "N/A"
            print(f"  {name:<28} {str(luts):>8} {str(ffs):>8} {str(cells):>8} {overhead:>10}")
    else:
        print(f"  {'Allocator':<28} {'Lines':>7} {'Regs':>7} {'Muxes':>7} {'Est LUTs':>9} {'Est FFs':>8} {'Overhead':>10}")
        print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*10}")

        base_cells = results.get("Baseline RR", {}).get('est_cells', 1)
        for name, _, _ in ALLOCATORS:
            if name not in results:
                continue
            s = results[name]
            overhead = f"{(s['est_cells'] - base_cells) / base_cells * 100:+.1f}%" if base_cells > 0 else "N/A"
            print(f"  {name:<28} {s['lines']:>7} {s['reg_assigns']:>7} {s['muxes']:>7} {s['est_luts']:>9} {s['est_ffs']:>8} {overhead:>10}")

    print(f"{'='*80}")
    print()
    if got_yosys_data:
        print("  Note: use the Yosys-mapped rows above as the better area signal.")
    else:
        print("  Note: the table above is based on RTL-structure estimates, not real")
        print("  synthesis area. Treat the overhead percentages as directional only.")
    print()

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "tools", "allocator_area_results.json")
    # Remove raw output before saving
    save_results = {}
    for k, v in results.items():
        save_results[k] = {mk: mv for mk, mv in v.items() if mk != 'raw_output'}
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
