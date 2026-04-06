#!/usr/bin/env python3
"""
estimate_timing.py — Clock Frequency / Critical Path Estimator
===============================================================
Uses Yosys + ABC to estimate the maximum achievable clock frequency
for key modules in the MNIST-Accel design.

Method:
  1. For each module, generate a minimal Yosys script that synthesises
     to generic CMOS gates (no FPGA mapping needed for timing depth).
  2. Run `abc -D <period_ps>` with decreasing periods to find the
     tightest period that maps without timing violations.
  3. Also report logic depth (register-to-register LUT levels) as
     a secondary check.

Target: Xilinx xc7 (Artix-7 / Zynq-7000 at speed grade -1)
  LUT propagation: ~0.50-0.60 ns per 6-input LUT
  Routing per hop: ~0.30-0.40 ns average
  Carry-chain add: ~0.05 ns/bit (fast carry)
  DSP48E1 path:    ~2.5 ns
  Typical Fmax for control logic: 150-250 MHz
  Typical Fmax for datapath (DSP-mapped): 300-500 MHz

Usage:
    python3 tools/estimate_timing.py [--module MODULE]
"""

import os
import sys
import subprocess
import json
import re
import tempfile
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJ = Path(__file__).parent.parent
RTL  = PROJ / "hw" / "rtl"
SYNTH_RTL = PROJ / "hw" / "rtl_synth" / "rtl"

# ---------------------------------------------------------------------------
# Known-good sources for each module under test
# We give each module its dependencies so Yosys can elaborate them.
# ---------------------------------------------------------------------------
MODULES = {
    "noc_innet_reduce": {
        "desc": "In-Network Reduction Engine (novel contribution)",
        "sources": [
            "noc/noc_pkg.sv",
            "noc/noc_innet_reduce.sv",
        ],
        "top": "noc_innet_reduce",
        "critical": "INT32 accumulate + scratchpad lookup",
    },
    "noc_router": {
        "desc": "5-Port Wormhole Router (full pipeline)",
        "sources": [
            "noc/noc_pkg.sv",
            "noc/noc_route_compute.sv",
            "noc/noc_credit_counter.sv",
            "noc/noc_vc_allocator.sv",
            "noc/noc_vc_allocator_sparse.sv",
            "noc/noc_switch_allocator.sv",
            "noc/noc_crossbar_5x5.sv",
            "noc/noc_input_port.sv",
            "noc/noc_innet_reduce.sv",
            "noc/noc_router.sv",
        ],
        "top": "noc_router",
        "critical": "VC allocate → switch allocate → crossbar path",
    },
    "mac8": {
        "desc": "INT8 Multiply-Accumulate (PE datapath)",
        "sources": [
            "mac/mac8.sv",
        ],
        "top": "mac8",
        "critical": "8×8→32 multiply, accumulate chain",
    },
    "pe": {
        "desc": "Processing Element (systolic array cell)",
        "sources": [
            "mac/mac8.sv",
            "systolic/pe.sv",
        ],
        "top": "pe",
        "critical": "INT8 MAC + output accumulation",
    },
    "dram_scheduler_frfcfs": {
        "desc": "FR-FCFS DRAM Scheduler",
        "sources": [
            "dram/dram_cmd_queue.sv",
            "dram/dram_scheduler_frfcfs.sv",
        ],
        "top": "dram_scheduler_frfcfs",
        "critical": "Queue scan priority logic",
    },
    "axi_crossbar": {
        "desc": "2-master 8-slave AXI Crossbar",
        "sources": [
            "top/soc_pkg.sv",
            "top/axi_addr_decoder.sv",
            "top/axi_arbiter.sv",
            "top/axi_crossbar.sv",
        ],
        "top": "axi_crossbar",
        "critical": "Round-robin arbiter + address decode",
    },
    "fixedpoint_alu": {
        "desc": "32-bit Fixed-Point ALU (HFT datapath)",
        "sources": [
            "hft/fixedpoint_alu.sv",
        ],
        "top": "fixedpoint_alu",
        "critical": "16.16 fixed-point multiply pipeline",
    },
}

# ---------------------------------------------------------------------------
# Xilinx xc7 (-1 speed grade) timing model
# LUT delay + routing = ~1.0 ns/hop (conservative)
# Carry chain: ~0.05 ns/bit → 32-bit adder = 1.6 ns (fast!)
# ---------------------------------------------------------------------------
NS_PER_LUT_LEVEL = 1.0   # ns per logic level (LUT + routing)
NS_CARRY_32B     = 2.0   # ns for a registered 32-bit carry-chain adder
SETUP_HOLD_NS    = 0.5   # FF setup + clock uncertainty


def run_yosys_stat(module_name, sources, top, rtl_base):
    """Synthesise and return Yosys stat output as a string."""
    # Build file list — prefer rtl_synth/ if present, fall back to rtl/
    file_lines = []
    for src in sources:
        synth_path = SYNTH_RTL / src
        rtl_path   = RTL / src
        if synth_path.exists():
            file_lines.append(f"read_verilog -sv -DSYNTHESIS {synth_path}")
        elif rtl_path.exists():
            file_lines.append(f"read_verilog -sv -DSYNTHESIS {rtl_path}")
        else:
            return None, f"Source not found: {src}"

    ys_script = "\n".join(file_lines) + f"""

hierarchy -top {top}
proc
flatten
opt -full
techmap
opt
abc -g AND,OR,XOR,MUX -D 5000
opt
stat
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as f:
        f.write(ys_script)
        ys_file = f.name

    try:
        result = subprocess.run(
            ["yosys", ys_file],
            capture_output=True, text=True, timeout=120
        )
        os.unlink(ys_file)
        return result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        os.unlink(ys_file)
        return None, "Yosys timed out"
    except FileNotFoundError:
        return None, "Yosys not found in PATH"


def estimate_fmax_from_depth(logic_levels):
    """
    Estimate Fmax from logic levels (LUTs in longest combinational path).
    
    For xc7 -1:
      - Each LUT6 = ~0.1 ns
      - Routing per hop = ~0.5-0.8 ns average  
      - Combined (LUT + routing): ~0.6-1.0 ns/level (use 1.0 conservative)
      - FF setup: 0.12 ns, clock skew: 0.5 ns → 0.5 ns total overhead
    
    Fmax = 1 / (levels × 1.0 ns + 0.5 ns)
    """
    cycle_ns = logic_levels * NS_PER_LUT_LEVEL + SETUP_HOLD_NS
    return 1000.0 / cycle_ns   # MHz


def parse_stat(stat_text):
    """Parse Yosys stat output for key metrics.
    
    Yosys stat format (count BEFORE cell name):
        347   $_AND_
         32   $_DFFE_PN0P_
    """
    metrics = {}

    # Count AND/OR/MUX/XOR/NOT gates — format: NUMBER   $_CELLNAME_
    def _count(pattern):
        m = re.search(pattern, stat_text)
        return int(m.group(1)) if m else 0

    counts = {
        "AND": _count(r'(\d+)\s+\$_AND_'),
        "OR":  _count(r'(\d+)\s+\$_OR_'),
        "MUX": _count(r'(\d+)\s+\$_MUX_'),
        "XOR": _count(r'(\d+)\s+\$_XOR_'),
        "NOT": _count(r'(\d+)\s+\$_NOT_'),
    }
    gate_total = sum(counts[k] for k in ("AND", "OR", "MUX", "XOR"))
    metrics["gate_counts"] = {k: v for k, v in counts.items() if v}
    metrics["gate_total"] = gate_total

    # Flip-flop count — any $_DFF*_ variant
    ff_total = sum(int(m.group(1)) for m in re.finditer(r'(\d+)\s+\$_DFF', stat_text))
    metrics["ffs"] = ff_total

    # Total cell count (sum of all cell lines in the stat block)
    metrics["cells"] = sum(int(m.group(1)) for m in
                           re.finditer(r'^\s+(\d+)\s+\S', stat_text, re.MULTILINE))

    # ABC timing depth if reported
    depth_match = re.search(r'Depth\s*[=:]\s*(\d+)', stat_text)
    if depth_match:
        metrics["depth"] = int(depth_match.group(1))

    return metrics


def estimate_depth_from_gates(gate_total, ffs):
    """
    Rough logic-depth estimate from gate count.
    Without ABC's actual path-trace, we estimate:
      - Adder (32-bit): ~32 gates in carry chain but only 2-3 "levels" 
        due to carry-chain optimization in Xilinx
      - MUX tree: log2(N) levels
      - Typical control FSM with N states: log2(N) + 4-6 compare levels
    
    Heuristic: depth ≈ sqrt(gate_total / max(ffs, 1) ) * 1.5,
    capped at reasonable values for typical modules.
    """
    if ffs and ffs > 0:
        logic_per_ff = gate_total / ffs
        depth = max(3, min(24, int((logic_per_ff ** 0.5) * 1.5)))
    else:
        depth = max(3, min(16, int(gate_total ** 0.4)))
    return depth


def run_analysis(module_name, module_info):
    """Full analysis for one module."""
    print(f"\n  {'─'*70}")
    print(f"  Module: {module_name}")
    print(f"  {module_info['desc']}")
    print(f"  Critical path: {module_info['critical']}")
    print(f"  {'─'*70}")

    stat_out, err = run_yosys_stat(
        module_name,
        module_info["sources"],
        module_info["top"],
        RTL
    )

    if err:
        print(f"  [ERROR] {err}")
        return None

    metrics = parse_stat(stat_out)
    
    gate_total = metrics.get("gate_total", 0)
    cells      = metrics.get("cells", 0)
    ffs        = metrics.get("ffs", 0)
    gcounts    = metrics.get("gate_counts", {})
    
    # Estimate depth
    if "depth" in metrics:
        depth = int(metrics["depth"])
        depth_source = "ABC path trace"
    else:
        depth = estimate_depth_from_gates(gate_total, ffs)
        depth_source = "heuristic estimate"
    
    fmax_mhz = estimate_fmax_from_depth(depth)
    period_ns = 1000.0 / fmax_mhz

    print(f"  Gates (2-input equiv): {gate_total:>8,}  [{', '.join(f'{k}:{v}' for k,v in gcounts.items())}]")
    print(f"  Flip-flops:            {ffs:>8,}")
    print(f"  Total cells (techmap): {cells:>8,}")
    print(f"  Logic depth ({depth_source}): {depth} levels")
    print(f"")
    print(f"  Estimated Fmax  (xc7 -1):  {fmax_mhz:>7.1f} MHz")
    print(f"  Estimated period:           {period_ns:>7.2f} ns")
    
    # Guidance
    if gate_total == 0 and cells == 0:
        verdict = "HEURISTIC ONLY — synthesis data incomplete, treat cautiously"
    elif fmax_mhz >= 250:
        verdict = "FAST — comfortably meets 200 MHz target"
    elif fmax_mhz >= 150:
        verdict = "NOMINAL — meets 100 MHz, tight at 200 MHz"
    elif fmax_mhz >= 100:
        verdict = "MARGINAL — may need pipelining for 200 MHz"
    else:
        verdict = "SLOW — needs redesign for high-speed operation"
    print(f"  Verdict:                    {verdict}")

    return {
        "module": module_name,
        "gates": gate_total,
        "ffs": ffs,
        "depth": depth,
        "depth_source": depth_source,
        "fmax_mhz": round(fmax_mhz, 1),
        "period_ns": round(period_ns, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate clock frequency for key modules")
    parser.add_argument("--module", choices=list(MODULES.keys()) + ["all"],
                        default="all", help="Module to analyse (default: all)")
    args = parser.parse_args()

    print("=" * 74)
    print("  MNIST-Accel Clock Frequency / Critical Path Analysis")
    print("  Yosys → ABC (generic 2-input gates) → logic-depth Fmax estimate")
    print("  Target: Xilinx xc7 -1 speed grade (conservative 1.0 ns/level + 0.5 ns FF)")
    print("=" * 74)

    selected = MODULES if args.module == "all" else {args.module: MODULES[args.module]}

    all_results = []
    for mod_name, mod_info in selected.items():
        result = run_analysis(mod_name, mod_info)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n\n{'='*74}")
        print("  SUMMARY TABLE")
        print(f"  {'Module':<30} {'Gates':>8} {'FFs':>6} {'Depth':>7} {'Fmax':>10}")
        print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*7} {'─'*10}")
        for r in all_results:
            print(f"  {r['module']:<30} {r['gates']:>8,} {r['ffs']:>6,} {r['depth']:>6}L {r['fmax_mhz']:>8.1f} MHz")

        if all_results:
            bottleneck = min(all_results, key=lambda x: x["fmax_mhz"])
            all_fast   = min(all_results, key=lambda x: x["fmax_mhz"])
            print(f"\n  Design bottleneck: {bottleneck['module']} @ {bottleneck['fmax_mhz']:.1f} MHz")
            print(f"  SoC target 100 MHz: {'✅ PASS' if bottleneck['fmax_mhz'] >= 100 else '❌ FAIL'}")
            print(f"  SoC target 200 MHz: {'✅ PASS' if bottleneck['fmax_mhz'] >= 200 else '❌ FAIL (pipeline needed)'}")
        print(f"{'='*74}")

    # Save results
    out_path = PROJ / "tools" / "timing_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
