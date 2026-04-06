#!/usr/bin/env python3
"""
run_e2e.py — End-to-End MNIST Inference Test Runner
====================================================
Orchestrates the full pipeline:
  1. Generate DRAM init hex (real weights + test image)
  2. Build inference firmware
  3. Compile RTL with Verilator
  4. Run simulation
  5. Parse UART output and verify classification

Usage:
    python3 tools/run_e2e.py              # Full run
    python3 tools/run_e2e.py --skip-gen   # Skip DRAM/FW generation (reuse existing)
    python3 tools/run_e2e.py --skip-build # Skip Verilator compilation (reuse existing)
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

# --- Paths ----------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
FW_DIR    = os.path.join(ROOT, "fw")
SIM_DIR   = os.path.join(ROOT, "hw", "sim", "sv")
DATA_DIR  = os.path.join(ROOT, "data")

DRAM_HEX          = os.path.join(DATA_DIR, "dram_init.hex")
GOLDEN_JSON       = os.path.join(DATA_DIR, "golden_reference.json")
FW_HEX_INF        = os.path.join(FW_DIR, "firmware_inference.hex")
FW_HEX_FULL_INF   = os.path.join(FW_DIR, "firmware_inference_full.hex")
FW_HEX_MT_INF     = os.path.join(FW_DIR, "firmware_inference_multitile.hex")
FW_HEX_SIM        = os.path.join(SIM_DIR, "firmware_inference.hex")
DRAM_HEX_MT       = os.path.join(DATA_DIR, "dram_init_multitile.hex")
DRAM_HEX_SIM      = os.path.join(SIM_DIR, "dram_init.hex")

FILELIST      = os.path.join(SIM_DIR, "filelist.f")
TB_FILE       = os.path.join(SIM_DIR, "tb_e2e_inference.sv")
OBJ_DIR       = os.path.join(SIM_DIR, "obj_dir_e2e")
OBJ_DIR_INR   = os.path.join(SIM_DIR, "obj_dir_e2e_inr")
OBJ_DIR_MT    = os.path.join(SIM_DIR, "obj_dir_e2e_mt")
SIM_EXEC      = os.path.join(OBJ_DIR, "Vtb_e2e_inference")
SIM_EXEC_INR  = os.path.join(OBJ_DIR_INR, "Vtb_e2e_inference")
SIM_EXEC_MT   = os.path.join(OBJ_DIR_MT, "Vtb_e2e_inference")


def get_obj_dir(inr: bool, multitile: bool = False) -> str:
    if multitile:
        return OBJ_DIR_MT
    return OBJ_DIR_INR if inr else OBJ_DIR


def get_sim_exec(inr: bool, multitile: bool = False) -> str:
    if multitile:
        return SIM_EXEC_MT
    return SIM_EXEC_INR if inr else SIM_EXEC


def run_cmd(cmd, cwd=None, desc=""):
    """Run a shell command, stream output, return (returncode, stdout)."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
    return result.returncode, result.stdout


def step_gen_dram(args):
    """Step 1: Generate DRAM init hex and inference firmware."""
    if args.skip_gen:
        print("\n[SKIP] DRAM/FW generation (--skip-gen)")
        return True

    multitile = getattr(args, 'multitile', False)

    if multitile:
        # Multi-tile: use dedicated generator
        gen_script = os.path.join(TOOLS_DIR, 'gen_dram_init_multitile.py')
        gen_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(gen_script)}"
        if args.image_path:
            gen_cmd += f" --image-path {shlex.quote(args.image_path)}"
            if args.true_label is not None:
                gen_cmd += f" --true-label {args.true_label}"
        else:
            gen_cmd += f" --image-index {args.image_index}"
        rc, _ = run_cmd(gen_cmd, cwd=ROOT,
                        desc="Step 1a: Generate multi-tile DRAM hex")
        if rc != 0:
            return False
        rc, _ = run_cmd("make clean && make multitile", cwd=FW_DIR,
                        desc="Step 1b: Build multi-tile inference firmware")
        return rc == 0

    gen_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(os.path.join(TOOLS_DIR, 'gen_dram_init.py'))}"
    if args.image_path:
        gen_cmd += f" --image-path {shlex.quote(args.image_path)}"
        if args.true_label is not None:
            gen_cmd += f" --true-label {args.true_label}"
    else:
        gen_cmd += f" --image-index {args.image_index}"
    if getattr(args, 'full', False):
        gen_cmd += " --full"

    # Generate DRAM hex
    rc, _ = run_cmd(
        gen_cmd,
        cwd=ROOT,
        desc="Step 1a: Generate DRAM init hex (real weights + test image)"
    )
    if rc != 0:
        return False

    # Build inference firmware (full BSR or FC2-only)
    fw_target = "full-inference" if getattr(args, 'full', False) else "inference"
    rc, _ = run_cmd(
        f"make clean && make {fw_target}",
        cwd=FW_DIR,
        desc=f"Step 1b: Build inference firmware ({fw_target})"
    )
    return rc == 0


def step_copy_assets(full=False, multitile=False):
    """Copy the inference firmware image and DRAM init image into sim directory."""
    import shutil
    if multitile:
        src_fw   = FW_HEX_MT_INF
        src_dram = DRAM_HEX_MT
    elif full:
        src_fw   = FW_HEX_FULL_INF
        src_dram = DRAM_HEX
    else:
        src_fw   = FW_HEX_INF
        src_dram = DRAM_HEX
    print(f"\n[COPY] {os.path.basename(src_fw)} \u2192 {FW_HEX_SIM}")
    shutil.copy2(src_fw, FW_HEX_SIM)
    print(f"[COPY] {os.path.basename(src_dram)} \u2192 {DRAM_HEX_SIM}")
    shutil.copy2(src_dram, DRAM_HEX_SIM)
    return True


def step_verilator_build(args):
    """Step 2: Compile RTL + testbench with Verilator."""
    inr       = getattr(args, 'inr', False)
    multitile = getattr(args, 'multitile', False)
    obj_dir   = get_obj_dir(inr, multitile)
    sim_exec  = get_sim_exec(inr, multitile)
    if args.skip_build and os.path.isfile(sim_exec):
        print("\n[SKIP] Verilator build (--skip-build)")
        return True

    inr_define = "-GINNET_REDUCE=1" if inr else ""
    mt_define  = "-GINNET_REDUCE=1" if multitile else ""  # multitile always uses INR-capable build
    extra_defs = mt_define if multitile else inr_define
    # Verilator compile
    verilator_cmd = (
        f"verilator --sv --binary --timing "
        f"-f {FILELIST} "
        f"{TB_FILE} "
        f"--top-module tb_e2e_inference "
        f"--Mdir {obj_dir} "
        f"{extra_defs} "
        f"-Wno-WIDTH -Wno-UNUSED -Wno-UNOPTFLAT -Wno-PINCONNECTEMPTY "
        f"-Wno-TIMESCALEMOD -Wno-SELRANGE -Wno-GENUNNAMED "
        f"-Wno-DECLFILENAME -Wno-ALWNEVER -Wno-UNDRIVEN "
        f"-Wno-CASEINCOMPLETE -Wno-SYNCASYNCNET "
        f"-Wno-BLKSEQ -Wno-UNSIGNED "
        f"-CFLAGS '-std=c++17 -O2' "
        f"-LDFLAGS '-lpthread' "
        f"-j 4"
    )

    rc, _ = run_cmd(
        verilator_cmd,
        cwd=SIM_DIR,
        desc="Step 2: Verilator compile (RTL + testbench)"
    )
    return rc == 0


def step_run_sim(inr=False, multitile=False):
    """Step 3: Run the Verilator simulation."""
    sim_exec = get_sim_exec(inr, multitile)
    t_start = time.time()
    rc, output = run_cmd(
        sim_exec,
        cwd=SIM_DIR,
        desc="Step 3: Run simulation"
    )
    elapsed = time.time() - t_start
    print(f"\nSimulation wall time: {elapsed:.1f}s")
    return rc, output, elapsed


def parse_results(output):
    """Parse simulation output for test results and performance."""
    results = {
        "all_passed": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "predicted_digit": None,
        "saw_pass": False,
        "total_cycles": None,
        "accel_cycles": None,
        "dram_reads": None,
        "dram_writes": None,
        "uart_lines": [],
    }

    for line in output.split("\n"):
        # Capture UART lines
        m = re.match(r"\[UART\]\s+(.*)", line)
        if m:
            results["uart_lines"].append(m.group(1))

        # Test results
        m = re.search(r"RESULTS:\s+(\d+)\s+passed,\s+(\d+)\s+failed", line)
        if m:
            results["tests_passed"] = int(m.group(1))
            results["tests_failed"] = int(m.group(2))

        if "ALL TESTS PASSED" in line:
            results["all_passed"] = True

        # Performance
        m = re.search(r"Total simulation cycles\s*:\s*(\d+)", line)
        if m:
            results["total_cycles"] = int(m.group(1))

        m = re.search(r"Accel busy cycles\s*:\s*(\d+)", line)
        if m:
            results["accel_cycles"] = int(m.group(1))

        m = re.search(r"DRAM read transactions\s*:\s*(\d+)", line)
        if m:
            results["dram_reads"] = int(m.group(1))

        m = re.search(r"DRAM write transactions\s*:\s*(\d+)", line)
        if m:
            results["dram_writes"] = int(m.group(1))

    # Parse UART for predicted digit
    for uart_line in results["uart_lines"]:
        m = re.match(r"Predicted:\s+(\d+)", uart_line)
        if m:
            results["predicted_digit"] = int(m.group(1))
        if "PASS: matches golden" in uart_line:
            results["saw_pass"] = True

    return results


def main():
    parser = argparse.ArgumentParser(description="E2E MNIST Inference Test Runner")
    parser.add_argument("--image-index", type=int, default=0,
                        help="MNIST test-set image index for generated input")
    parser.add_argument("--image-path", default=None,
                        help="Optional path to an external image to classify")
    parser.add_argument("--true-label", type=int, default=None,
                        help="Optional ground-truth label for --image-path")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip DRAM/firmware generation (reuse existing)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Verilator compilation (reuse existing binary)")
    parser.add_argument("--full", action="store_true",
                        help="Use BSR sparse FC1+FC2 firmware (main_inference_full.c) "
                             "instead of FC2-only (main_inference.c)")
    parser.add_argument("--inr", action="store_true",
                        help="Enable INNET_REDUCE=1 in soc_top_v2 for in-network reduction comparison")
    parser.add_argument("--multitile", action="store_true",
                        help="Run multi-tile FC1 firmware (N_PARALLEL=4, K-block parallel)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MNIST-Accel End-to-End Inference Test")
    print("=" * 60)

    # Step 1: Generate DRAM hex + firmware
    if not step_gen_dram(args):
        print("\nFAILED: DRAM/firmware generation failed")
        sys.exit(1)

    # Copy assets into sim directory
    multitile = getattr(args, 'multitile', False)
    if not step_copy_assets(full=getattr(args, 'full', False), multitile=multitile):
        print("\nFAILED: Could not copy assets")
        sys.exit(1)

    # Step 2: Verilator build
    if not step_verilator_build(args):
        print("\nFAILED: Verilator compilation failed")
        sys.exit(1)

    # Step 3: Run simulation
    rc, output, wall_time = step_run_sim(inr=getattr(args, 'inr', False),
                                         multitile=getattr(args, 'multitile', False))

    # Step 4: Parse and report
    results = parse_results(output)

    print("\n" + "=" * 60)
    print("  E2E TEST SUMMARY")
    print("=" * 60)
    print(f"  Predicted digit : {results['predicted_digit']}")
    print(f"  Firmware PASS   : {results['saw_pass']}")
    print(f"  TB tests passed : {results['tests_passed']}")
    print(f"  TB tests failed : {results['tests_failed']}")
    print(f"  Sim cycles      : {results['total_cycles']}")
    print(f"  Accel cycles    : {results['accel_cycles']}")
    print(f"  DRAM reads      : {results['dram_reads']}")
    print(f"  DRAM writes     : {results['dram_writes']}")
    print(f"  Wall time       : {wall_time:.1f}s")
    print("=" * 60)

    if results["all_passed"]:
        print("  *** E2E TEST PASSED ***")
        # Save results
        results_file = os.path.join(DATA_DIR, "e2e_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "status": "PASS",
                "predicted_digit": results["predicted_digit"],
                "total_cycles": results["total_cycles"],
                "accel_cycles": results["accel_cycles"],
                "dram_reads": results["dram_reads"],
                "dram_writes": results["dram_writes"],
                "wall_time_seconds": wall_time,
                "uart_transcript": results["uart_lines"],
            }, f, indent=2)
        print(f"  Results saved to {results_file}")
        sys.exit(0)
    else:
        print("  *** E2E TEST FAILED ***")
        print("\n  UART transcript:")
        for line in results["uart_lines"]:
            print(f"    {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
