#!/usr/bin/env python3
"""
plot_dram_experiments.py — Post-simulation analysis & plotting for DRAM controller.

Reads CSV logs produced by cocotb tests and generates:
  1. Latency histogram (read latency distribution).
  2. Bank utilisation heat-map over time.
  3. Row-hit rate bar chart (per bank).
  4. Bandwidth timeline (bytes/cycle rolling average).

Usage:
    python tools/plot_dram_experiments.py [--logdir <dir>] [--output <dir>]

Requires: matplotlib, numpy.  Install via `pip install matplotlib numpy`.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install: pip install matplotlib numpy")
    sys.exit(1)


def load_csv(path):
    """Load a CSV file and return list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_latency_histogram(records, outdir):
    """
    Input CSV columns: cycle, latency_cycles, type (read/write)
    """
    reads = [int(r["latency_cycles"]) for r in records if r["type"] == "read"]
    writes = [int(r["latency_cycles"]) for r in records if r["type"] == "write"]

    fig, ax = plt.subplots(figsize=(8, 4))
    if reads:
        ax.hist(reads, bins=30, alpha=0.7, label="Read", color="#4C72B0")
    if writes:
        ax.hist(writes, bins=30, alpha=0.7, label="Write", color="#DD8452")
    ax.set_xlabel("Latency (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("DRAM Access Latency Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latency_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  → latency_histogram.png")


def plot_bank_utilisation(records, outdir, num_banks=8, window=100):
    """
    Input CSV columns: cycle, bank, event (act/read/write/pre)
    """
    if not records:
        return
    max_cycle = max(int(r["cycle"]) for r in records)
    num_windows = max_cycle // window + 1

    heatmap = np.zeros((num_banks, num_windows), dtype=float)
    for r in records:
        b = int(r["bank"])
        w = int(r["cycle"]) // window
        heatmap[b, w] += 1

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel(f"Time Window ({window} cycles each)")
    ax.set_ylabel("Bank")
    ax.set_title("Bank Utilisation Heatmap")
    fig.colorbar(im, ax=ax, label="Commands")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bank_utilisation.png"), dpi=150)
    plt.close(fig)
    print(f"  → bank_utilisation.png")


def plot_row_hit_rate(records, outdir, num_banks=8):
    """
    Input CSV columns: bank, row_hit (0 or 1)
    """
    hits = [0] * num_banks
    totals = [0] * num_banks
    for r in records:
        b = int(r["bank"])
        totals[b] += 1
        if int(r["row_hit"]):
            hits[b] += 1

    rates = [h / t if t > 0 else 0 for h, t in zip(hits, totals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(num_banks), rates, color="#55A868")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Row-Hit Rate")
    ax.set_title("Per-Bank Row-Hit Rate")
    ax.set_xticks(range(num_banks))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "row_hit_rate.png"), dpi=150)
    plt.close(fig)
    print(f"  → row_hit_rate.png")


def plot_bandwidth_timeline(records, outdir, bus_bytes=4, window=50):
    """
    Input CSV columns: cycle, bytes_transferred
    """
    if not records:
        return
    max_cycle = max(int(r["cycle"]) for r in records)
    num_windows = max_cycle // window + 1
    bw = np.zeros(num_windows, dtype=float)
    for r in records:
        w = int(r["cycle"]) // window
        bw[w] += int(r["bytes_transferred"])
    bw /= window  # bytes per cycle

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(num_windows) * window, bw, color="#8172B2", linewidth=1)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Bytes / Cycle")
    ax.set_title(f"Bandwidth Timeline (window={window} cycles)")
    ax.axhline(y=bus_bytes, color="red", linestyle="--", alpha=0.5, label=f"Peak {bus_bytes} B/cyc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bandwidth_timeline.png"), dpi=150)
    plt.close(fig)
    print(f"  → bandwidth_timeline.png")


def main():
    parser = argparse.ArgumentParser(description="DRAM experiment plotter")
    parser.add_argument("--logdir", default="hw/sim/logs", help="Directory with CSV logs")
    parser.add_argument("--output", default="docs/figs", help="Output directory for plots")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    plotters = {
        "latency.csv":       plot_latency_histogram,
        "bank_events.csv":   plot_bank_utilisation,
        "row_hits.csv":      plot_row_hit_rate,
        "bandwidth.csv":     plot_bandwidth_timeline,
    }

    found = False
    for csv_name, fn in plotters.items():
        csv_path = logdir / csv_name
        if csv_path.exists():
            print(f"Plotting {csv_name} ...")
            records = load_csv(csv_path)
            fn(records, str(outdir))
            found = True

    if not found:
        print(f"No CSV log files found in {logdir}/")
        print("Run DRAM simulation first to generate logs.")
        sys.exit(0)


if __name__ == "__main__":
    main()
