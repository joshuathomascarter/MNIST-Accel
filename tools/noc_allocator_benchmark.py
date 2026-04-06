#!/usr/bin/env python3
"""
NoC VC Allocator Benchmark — Baseline vs Sparsity-Aware
========================================================
Cycle-accurate mesh NoC simulation using REAL BSR sparse data from the
MNIST accelerator project.

The sparsity-aware model in this script mirrors the RTL policy in
noc_vc_allocator_sparse.sv:
    - sparse traffic can use any VC
    - dense traffic avoids the reserved VC only when sparse traffic dominates
    - reservation activates adaptively after enough recent requests

Usage:
        python3 tools/noc_allocator_benchmark.py
"""

import math
import os
import sys
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

# =============================================================================
# NoC Parameters (matching noc_pkg.sv)
# =============================================================================
MESH_ROWS = 4
MESH_COLS = 4
NUM_NODES = MESH_ROWS * MESH_COLS
NUM_VCS = 4
NUM_PORTS = 5  # N, E, S, W, Local
BUF_DEPTH = 4

PORT_NORTH = 0
PORT_EAST  = 1
PORT_SOUTH = 2
PORT_WEST  = 3
PORT_LOCAL = 4


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class Packet:
    src: int
    dst: int
    is_sparse: bool
    inject_cycle: int = 0
    num_flits: int = 1
    id: int = 0


@dataclass
class Metrics:
    latencies: list = field(default_factory=list)
    total_cycles: int = 0
    total_flits_delivered: int = 0
    sparse_latencies: list = field(default_factory=list)
    dense_latencies: list = field(default_factory=list)
    blocked_cycles: int = 0


# =============================================================================
# XY Routing
# =============================================================================
def node_to_rc(node_id):
    return divmod(node_id, MESH_COLS)

def rc_to_node(row, col):
    return row * MESH_COLS + col

def xy_next_hop_port(src, dst):
    """XY routing: go X first (east/west), then Y (north/south)."""
    sr, sc = node_to_rc(src)
    dr, dc = node_to_rc(dst)
    if dc > sc:
        return PORT_EAST
    elif dc < sc:
        return PORT_WEST
    elif dr > sr:
        return PORT_SOUTH
    elif dr < sr:
        return PORT_NORTH
    else:
        return PORT_LOCAL

def hop_count(src, dst):
    sr, sc = node_to_rc(src)
    dr, dc = node_to_rc(dst)
    return abs(dr - sr) + abs(dc - sc)


# =============================================================================
# Allocator: Baseline Round-Robin
# =============================================================================
class BaselineAllocator:
    """Round-robin VC allocator — same logic as noc_vc_allocator.sv."""

    def __init__(self):
        self.rr_ptr = [0] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        # Track how many flits remain on each VC (for multi-flit packets)
        self.vc_hold_until = [[0] * NUM_VCS for _ in range(NUM_PORTS)]

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        """Try to allocate a VC. hold_cycles = how long this VC is held."""
        for offset in range(NUM_VCS):
            vc = (self.rr_ptr[output_port] + offset) % NUM_VCS
            if not self.vc_busy[output_port][vc]:
                self.vc_busy[output_port][vc] = True
                self.vc_hold_until[output_port][vc] = hold_cycles
                self.rr_ptr[output_port] = (vc + 1) % NUM_VCS
                return vc
        return None

    def tick(self):
        """Advance time — decrement hold counters and free expired VCs."""
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold_until[op][vc] -= 1
                    if self.vc_hold_until[op][vc] <= 0:
                        self.vc_busy[op][vc] = False

    def free(self, output_port, vc):
        self.vc_busy[output_port][vc] = False
        self.vc_hold_until[output_port][vc] = 0

    def busy_count(self, output_port):
        return sum(1 for v in range(NUM_VCS) if self.vc_busy[output_port][v])


# =============================================================================
# Allocator: Sparsity-Aware (matching noc_vc_allocator_sparse.sv)
# =============================================================================
class SparseAwareAllocator:
    """Sparsity-aware VC allocator matching noc_vc_allocator_sparse.sv.

    Policy:
    1. Sparse packets can use any VC.
    2. Dense packets avoid the reserved VC only when sparse traffic has been
       more than 50% of recent requests and at least 64 requests were seen.
    3. Separate RR pointers are used for sparse and dense traffic classes.
    """

    RESERVED_VC = NUM_VCS - 1
    MIN_SAMPLES = 64
    MAX_COUNT = 0xFFFF

    def __init__(self):
        self.rr_sparse = [0] * NUM_PORTS
        self.rr_dense = [0] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold_until = [[0] * NUM_VCS for _ in range(NUM_PORTS)]
        self.sparse_count = 0
        self.total_count = 0

    def _record_request(self, is_sparse):
        if self.total_count == self.MAX_COUNT:
            self.total_count >>= 1
            self.sparse_count >>= 1

        self.total_count = min(self.MAX_COUNT, self.total_count + 1)
        if is_sparse:
            self.sparse_count = min(self.MAX_COUNT, self.sparse_count + 1)

    def reservation_active(self):
        return self.total_count > self.MIN_SAMPLES and self.sparse_count > (self.total_count // 2)

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        self._record_request(is_sparse)

        if is_sparse:
            for offset in range(NUM_VCS):
                vc = (self.rr_sparse[output_port] + offset) % NUM_VCS
                if not self.vc_busy[output_port][vc]:
                    self.vc_busy[output_port][vc] = True
                    self.vc_hold_until[output_port][vc] = hold_cycles
                    self.rr_sparse[output_port] = (vc + 1) % NUM_VCS
                    return vc
            return None
        else:
            dense_vcs = NUM_VCS - 1 if self.reservation_active() else NUM_VCS
            for offset in range(dense_vcs):
                vc = (self.rr_dense[output_port] + offset) % dense_vcs
                if not self.vc_busy[output_port][vc]:
                    self.vc_busy[output_port][vc] = True
                    self.vc_hold_until[output_port][vc] = hold_cycles
                    self.rr_dense[output_port] = (vc + 1) % dense_vcs
                    return vc
            return None

    def tick(self):
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold_until[op][vc] -= 1
                    if self.vc_hold_until[op][vc] <= 0:
                        self.vc_busy[op][vc] = False

    def free(self, output_port, vc):
        self.vc_busy[output_port][vc] = False
        self.vc_hold_until[output_port][vc] = 0

    def busy_count(self, output_port):
        return sum(1 for v in range(NUM_VCS) if self.vc_busy[output_port][v])


# =============================================================================
# Cycle-Accurate NoC Simulator (multi-hop, multi-flit, per-router contention)
# =============================================================================
class NoCSimulator:
    """Cycle-accurate 4x4 mesh simulator.

    Key realism features:
    - Multi-flit packets hold VCs for multiple cycles (like real wormhole routing)
    - Per-hop allocation with contention at each intermediate router
    - Injection rate limiting (models source queuing)
    - Dense packets are longer (more flits → hold VCs longer)
    - Sparse packets are shorter (fewer flits → less blocking)
    """

    def __init__(self, allocator_class):
        self.allocator_class = allocator_class
        self.routers = [allocator_class() for _ in range(NUM_NODES)]

    def simulate(self, packets, max_cycles=100000):
        metrics = Metrics()
        pending = list(packets)
        for i, p in enumerate(pending):
            p.id = i

        # In-flight: list of (current_node, packet, vc, output_port, hops_remaining, hold_remaining)
        in_flight = []
        cycle = 0
        inject_idx = 0

        while (pending or in_flight) and cycle < max_cycles:
            cycle += 1

            # --- Tick all routers (decrement hold counters) ---
            for router in self.routers:
                router.tick()

            # --- Phase 1: Advance packets that finished their hop ---
            new_in_flight = []
            for (cur_node, pkt, vc, outp, hops_rem, hold_rem) in in_flight:
                if hold_rem <= 0:
                    # This hop completed, move to next
                    sr, sc = node_to_rc(cur_node)
                    if outp == PORT_EAST:
                        nxt = rc_to_node(sr, sc + 1)
                    elif outp == PORT_WEST:
                        nxt = rc_to_node(sr, sc - 1)
                    elif outp == PORT_SOUTH:
                        nxt = rc_to_node(sr + 1, sc)
                    elif outp == PORT_NORTH:
                        nxt = rc_to_node(sr - 1, sc)
                    else:
                        nxt = cur_node

                    if hops_rem <= 1:
                        # Delivered!
                        lat = cycle - pkt.inject_cycle
                        metrics.latencies.append(lat)
                        metrics.total_flits_delivered += 1
                        if pkt.is_sparse:
                            metrics.sparse_latencies.append(lat)
                        else:
                            metrics.dense_latencies.append(lat)
                        continue

                    # Allocate at next router
                    next_outp = xy_next_hop_port(nxt, pkt.dst)
                    hold = pkt.num_flits  # hold VC for num_flits cycles
                    vc_alloc = self.routers[nxt].allocate(next_outp, pkt.is_sparse, hold)
                    if vc_alloc is not None:
                        new_in_flight.append((nxt, pkt, vc_alloc, next_outp, hops_rem - 1, hold))
                    else:
                        # Blocked — wait at current node, retry next cycle
                        new_in_flight.append((cur_node, pkt, -1, outp, hops_rem, 0))
                        metrics.blocked_cycles += 1
                else:
                    # Still traversing this hop
                    new_in_flight.append((cur_node, pkt, vc, outp, hops_rem, hold_rem - 1))

            # Retry blocked packets (vc == -1)
            retried = []
            for entry in new_in_flight:
                cur_node, pkt, vc, outp, hops_rem, hold_rem = entry
                if vc == -1:
                    # Need to move to next node and allocate there
                    sr, sc = node_to_rc(cur_node)
                    if outp == PORT_EAST:
                        nxt = rc_to_node(sr, sc + 1)
                    elif outp == PORT_WEST:
                        nxt = rc_to_node(sr, sc - 1)
                    elif outp == PORT_SOUTH:
                        nxt = rc_to_node(sr + 1, sc)
                    elif outp == PORT_NORTH:
                        nxt = rc_to_node(sr - 1, sc)
                    else:
                        nxt = cur_node

                    next_outp = xy_next_hop_port(nxt, pkt.dst)
                    hold = pkt.num_flits
                    vc_alloc = self.routers[nxt].allocate(next_outp, pkt.is_sparse, hold)
                    if vc_alloc is not None:
                        retried.append((nxt, pkt, vc_alloc, next_outp, hops_rem - 1, hold))
                    else:
                        retried.append((cur_node, pkt, -1, outp, hops_rem, 0))
                        metrics.blocked_cycles += 1
                else:
                    retried.append(entry)
            in_flight = retried

            # --- Phase 2: Inject new packets ---
            # Two-pass injection: sparse first (priority), then dense.
            # This models the RTL's two-pass allocator — sparse requests
            # are processed first within the same clock cycle.
            injected = 0
            max_inject = NUM_NODES

            # Separate sparse and dense pending packets
            sparse_pending = []
            dense_pending = []
            scan_limit = min(inject_idx + max_inject * 2, len(pending))
            for si in range(inject_idx, scan_limit):
                pkt = pending[si]
                if pkt.is_sparse:
                    sparse_pending.append(si)
                else:
                    dense_pending.append(si)

            injected_indices = set()

            # Pass 1: Inject sparse packets (priority access to ALL VCs)
            for si in sparse_pending:
                if injected >= max_inject:
                    break
                pkt = pending[si]
                pkt.inject_cycle = cycle

                if pkt.src == pkt.dst:
                    injected_indices.add(si)
                    injected += 1
                    continue

                outp = xy_next_hop_port(pkt.src, pkt.dst)
                hops = hop_count(pkt.src, pkt.dst)
                hold = pkt.num_flits

                vc = self.routers[pkt.src].allocate(outp, pkt.is_sparse, hold)
                if vc is not None:
                    in_flight.append((pkt.src, pkt, vc, outp, hops, hold))
                    injected_indices.add(si)
                    injected += 1
                else:
                    metrics.blocked_cycles += 1

            # Pass 2: Inject dense packets (remaining VCs)
            for si in dense_pending:
                if injected >= max_inject:
                    break
                pkt = pending[si]
                pkt.inject_cycle = cycle

                if pkt.src == pkt.dst:
                    injected_indices.add(si)
                    injected += 1
                    continue

                outp = xy_next_hop_port(pkt.src, pkt.dst)
                hops = hop_count(pkt.src, pkt.dst)
                hold = pkt.num_flits

                vc = self.routers[pkt.src].allocate(outp, pkt.is_sparse, hold)
                if vc is not None:
                    in_flight.append((pkt.src, pkt, vc, outp, hops, hold))
                    injected_indices.add(si)
                    injected += 1
                else:
                    metrics.blocked_cycles += 1

            # Compact: remove injected packets from pending
            if injected_indices:
                new_pending = []
                for si in range(len(pending)):
                    if si not in injected_indices:
                        new_pending.append(pending[si])
                pending = new_pending
                inject_idx = 0
            # If nothing injected this cycle, advance scan window
            elif inject_idx < len(pending):
                pass  # retry same window next cycle

        metrics.total_cycles = cycle
        return metrics


# =============================================================================
# Traffic Generators
# =============================================================================

def generate_uniform_traffic(num_packets, sparse_fraction=0.5, seed=42):
    """Uniform random with multi-flit packets.
    Dense = 4 flits (e.g., 256-bit cache line), Sparse = 1-2 flits."""
    rng = np.random.RandomState(seed)
    packets = []
    for _ in range(num_packets):
        src = rng.randint(0, NUM_NODES)
        dst = rng.randint(0, NUM_NODES)
        while dst == src:
            dst = rng.randint(0, NUM_NODES)
        is_sparse = rng.random() < sparse_fraction
        nf = rng.randint(1, 2) if is_sparse else rng.randint(3, 5)
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))
    return packets


def generate_hotspot_traffic(num_packets, hotspot=0, frac=0.6, sparse_frac=0.3, seed=42):
    """Traffic concentrated toward one node. Dense packets are 4 flits."""
    rng = np.random.RandomState(seed)
    packets = []
    for _ in range(num_packets):
        if rng.random() < frac:
            src = rng.randint(0, NUM_NODES)
            dst = hotspot
            if src == dst:
                src = (src + 1) % NUM_NODES
        else:
            src = rng.randint(0, NUM_NODES)
            dst = rng.randint(0, NUM_NODES)
            while dst == src:
                dst = rng.randint(0, NUM_NODES)
        is_sparse = rng.random() < sparse_frac
        nf = rng.randint(1, 2) if is_sparse else rng.randint(3, 5)
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))
    return packets


def generate_scatter_reduce_traffic(num_packets, controller=0, seed=42):
    """Scatter = dense multi-flit, Reduce = sparse single-flit.
    Models real accelerator: scatter full activation tiles (dense, 4 flits),
    collect sparse partial sums (1-2 flits)."""
    packets = []
    tiles = [n for n in range(NUM_NODES) if n != controller]
    half = num_packets // 2

    for i in range(half):
        dst = tiles[i % len(tiles)]
        packets.append(Packet(src=controller, dst=dst, is_sparse=False, num_flits=4))

    for i in range(num_packets - half):
        src = tiles[i % len(tiles)]
        packets.append(Packet(src=src, dst=controller, is_sparse=True, num_flits=1))

    return packets


def generate_bsr_traffic(bsr_dir, layer_name, num_tiles=15, seed=42):
    """Generate traffic from REAL BSR sparse data.

    Key realism:
    - Non-zero blocks → 2-flit sparse packets (index + data)
    - Dense control traffic → 4-flit packets (CSR writes, activation tiles)
    - Pattern follows actual BSR row_ptr/col_idx structure
    """
    rng = np.random.RandomState(seed)

    row_ptr = np.load(os.path.join(bsr_dir, layer_name, "row_ptr.npy"))
    col_idx = np.load(os.path.join(bsr_dir, layer_name, "col_idx.npy"))

    packets = []
    controller = 0

    num_block_rows = len(row_ptr) - 1

    for br in range(num_block_rows):
        nnz_start = row_ptr[br]
        nnz_end = row_ptr[br + 1]
        nnz_cols = col_idx[nnz_start:nnz_end]

        src_tile = 1 + (br % num_tiles)

        for ci in nnz_cols:
            dst_tile = 1 + (int(ci) % num_tiles)
            if dst_tile == src_tile:
                dst_tile = (dst_tile % num_tiles) + 1

            # Non-zero block → 2-flit sparse packet (header + data)
            packets.append(Packet(src=src_tile, dst=dst_tile, is_sparse=True, num_flits=2))

        # Dense control: CSR write to configure tile (4 flits)
        if nnz_end > nnz_start:
            packets.append(Packet(src=controller, dst=src_tile, is_sparse=False, num_flits=4))

    # Scatter input activations (dense, 4 flits each)
    for t in range(1, num_tiles + 1):
        for _ in range(5):
            packets.append(Packet(src=controller, dst=t, is_sparse=False, num_flits=4))

    # Reduce partial sums (sparse, 2 flits each)
    for t in range(1, num_tiles + 1):
        for _ in range(4):
            packets.append(Packet(src=t, dst=controller, is_sparse=True, num_flits=2))

    return packets


def generate_saturated_mixed_traffic(num_packets, sparse_frac=0.2, seed=42):
    """THE KEY TEST: sustained heavy load with interleaved sparse control.

    Models: dense DMA transfers (8 flits, long VC hold) running continuously
    while sparse control messages (1 flit, urgent) try to get through.
    This is the exact scenario where VC reservation helps — without it,
    sparse control gets stuck behind dense data hogging all VCs.
    """
    rng = np.random.RandomState(seed)
    packets = []

    for _ in range(num_packets):
        src = rng.randint(0, NUM_NODES)
        dst = rng.randint(0, NUM_NODES)
        while dst == src:
            dst = rng.randint(0, NUM_NODES)

        is_sparse = rng.random() < sparse_frac
        if is_sparse:
            nf = 1  # Sparse control: single flit, needs fast delivery
        else:
            nf = rng.randint(6, 10)  # Dense bulk: LONG packets hogging VCs
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))

    return packets


def generate_bsr_with_dense_bg(bsr_dir, layer_name, num_tiles=15, dense_bg_packets=300, seed=42):
    """REAL BSR data with heavy dense background traffic.

    Models: CPU/DMA doing large memory transfers (dense, 8 flits) while
    the systolic array's sparse dataflow (BSR tiles, 2 flits) runs.
    """
    rng = np.random.RandomState(seed)

    row_ptr = np.load(os.path.join(bsr_dir, layer_name, "row_ptr.npy"))
    col_idx = np.load(os.path.join(bsr_dir, layer_name, "col_idx.npy"))

    packets = []
    controller = 0
    num_block_rows = len(row_ptr) - 1

    # Dense background: DMA controller pumping large transfers
    for i in range(dense_bg_packets):
        dst = 1 + (i % num_tiles)
        packets.append(Packet(src=controller, dst=dst, is_sparse=False, num_flits=8))

    # Interleave: BSR sparse data
    for br in range(num_block_rows):
        nnz_start = row_ptr[br]
        nnz_end = row_ptr[br + 1]
        nnz_cols = col_idx[nnz_start:nnz_end]
        src_tile = 1 + (br % num_tiles)

        for ci in nnz_cols:
            dst_tile = 1 + (int(ci) % num_tiles)
            if dst_tile == src_tile:
                dst_tile = (dst_tile % num_tiles) + 1
            packets.append(Packet(src=src_tile, dst=dst_tile, is_sparse=True, num_flits=2))

    # Shuffle to interleave dense and sparse (realistic arrival)
    rng.shuffle(packets)
    return packets


def generate_bottleneck_traffic(num_dense=500, num_sparse=300, seed=42):
    """Concentrated bottleneck: dense mega-flows along the diagonal jam
    intermediate routers while sparse 1-hop packets need VCs at those
    same routers.

    Dense: node 0→15, 3→12, 1→14, 2→13 (max hops, cross the mesh)
    Sparse: adjacent node pairs (1 hop, need fast access)

    This is the EXACT scenario for VC reservation: long dense flows
    block all VCs at intermediate routers, sparse packets need just
    1 VC to traverse a 1-hop path through those same routers.
    """
    rng = np.random.RandomState(seed)
    packets = []

    # Dense mega-flows: cross the entire mesh (6 hops each, 8-12 flits)
    dense_flows = [(0, 15), (3, 12), (1, 14), (2, 13), (4, 11), (7, 8)]
    for i in range(num_dense):
        src, dst = dense_flows[i % len(dense_flows)]
        nf = rng.randint(8, 12)
        packets.append(Packet(src=src, dst=dst, is_sparse=False, num_flits=nf))

    # Sparse latency-sensitive: 1-hop adjacent pairs
    sparse_pairs = [(5, 6), (6, 7), (9, 10), (10, 11), (5, 9), (6, 10),
                    (7, 11), (4, 5), (8, 9), (1, 5), (2, 6), (3, 7)]
    for i in range(num_sparse):
        src, dst = sparse_pairs[i % len(sparse_pairs)]
        packets.append(Packet(src=src, dst=dst, is_sparse=True, num_flits=1))

    rng.shuffle(packets)
    return packets


# =============================================================================
# Result Formatting
# =============================================================================

def format_metrics(name, metrics):
    """Format metrics into a summary dict."""
    lats = metrics.latencies
    if not lats:
        return {"name": name, "avg_lat": 0, "p50": 0, "p95": 0, "p99": 0,
                "max_lat": 0, "throughput": 0, "jain": 0, "blocked": 0,
                "sparse_avg": 0, "dense_avg": 0}

    lats_sorted = sorted(lats)
    n = len(lats_sorted)

    # Jain's fairness
    sum_x = sum(lats)
    sum_x2 = sum(x*x for x in lats)
    jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0

    sparse_avg = np.mean(metrics.sparse_latencies) if metrics.sparse_latencies else 0
    dense_avg = np.mean(metrics.dense_latencies) if metrics.dense_latencies else 0

    return {
        "name": name,
        "avg_lat": np.mean(lats),
        "p50": lats_sorted[n // 2],
        "p95": lats_sorted[int(n * 0.95)],
        "p99": lats_sorted[int(n * 0.99)],
        "max_lat": max(lats),
        "throughput": metrics.total_flits_delivered / max(metrics.total_cycles, 1),
        "jain": jain,
        "blocked": metrics.blocked_cycles,
        "sparse_avg": sparse_avg,
        "dense_avg": dense_avg,
        "total_cycles": metrics.total_cycles,
        "delivered": metrics.total_flits_delivered,
    }


def print_comparison(pattern_name, baseline_m, sparse_m):
    base = format_metrics("Baseline", baseline_m)
    spa = format_metrics("Sparse-Aware", sparse_m)

    print(f"\n{'='*72}")
    print(f"  Traffic Pattern: {pattern_name}")
    print(f"  Packets: {base['delivered']} delivered")
    print(f"{'='*72}")
    print(f"  {'Metric':<28} {'Baseline':>12} {'Sparse-VC':>12} {'Improvement':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")

    rows = [
        ("Avg Latency (cycles)", "avg_lat", True),
        ("P50 Latency", "p50", True),
        ("P95 Latency", "p95", True),
        ("P99 Latency (tail)", "p99", True),
        ("Max Latency", "max_lat", True),
        ("Throughput (flits/cyc)", "throughput", False),
        ("Jain Fairness", "jain", False),
        ("Blocked Cycles", "blocked", True),
        ("Sparse Pkt Avg Lat", "sparse_avg", True),
        ("Dense Pkt Avg Lat", "dense_avg", True),
        ("Total Cycles", "total_cycles", True),
    ]

    for label, key, lower_is_better in rows:
        b = base[key]
        s = spa[key]
        if b > 0:
            pct = ((b - s) / b) * 100 if lower_is_better else ((s - b) / b) * 100
            if pct > 0:
                delta_str = f"+{pct:.1f}%"
            else:
                delta_str = f"{pct:.1f}%"
        else:
            delta_str = "N/A"
        print(f"  {label:<28} {b:>12.2f} {s:>12.2f} {delta_str:>12}")

    print(f"{'='*72}")
    return base, spa


# =============================================================================
# Main
# =============================================================================
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bsr_dir = os.path.join(project_root, "data", "bsr_export")

    print("=" * 72)
    print("  NoC VC Allocator Benchmark: Baseline vs Sparsity-Aware")
    print("  4x4 Mesh, 4 VCs/port, XY Routing")
    print("=" * 72)

    all_results = []

    # -------------------------------------------------------------------------
    # Test 1: Uniform Random (50% sparse, multi-flit)
    # -------------------------------------------------------------------------
    print("\n[1/8] Running Uniform Random traffic (1000 packets, 50% sparse, multi-flit)...")
    packets = generate_uniform_traffic(1000, sparse_fraction=0.5)

    sim_base = NoCSimulator(BaselineAllocator)
    sim_sparse = NoCSimulator(SparseAwareAllocator)

    m_base = sim_base.simulate(list(packets))
    m_sparse = sim_sparse.simulate(list(packets))
    b, s = print_comparison("Uniform Random (50% sparse)", m_base, m_sparse)
    all_results.append(("Uniform Random", b, s))

    # -------------------------------------------------------------------------
    # Test 2: Hotspot Traffic (node 0, 60% concentration)
    # -------------------------------------------------------------------------
    print("\n[2/8] Running Hotspot traffic (1000 packets, 60% to node 0)...")
    packets = generate_hotspot_traffic(1000, hotspot=0, frac=0.6, sparse_frac=0.3)

    sim_base = NoCSimulator(BaselineAllocator)
    sim_sparse = NoCSimulator(SparseAwareAllocator)

    m_base = sim_base.simulate(list(packets))
    m_sparse = sim_sparse.simulate(list(packets))
    b, s = print_comparison("Hotspot (node 0, 30% sparse)", m_base, m_sparse)
    all_results.append(("Hotspot", b, s))

    # -------------------------------------------------------------------------
    # Test 3: Scatter-Reduce (dense scatter, sparse reduce)
    # -------------------------------------------------------------------------
    print("\n[3/8] Running Scatter-Reduce traffic (800 packets)...")
    packets = generate_scatter_reduce_traffic(800)

    sim_base = NoCSimulator(BaselineAllocator)
    sim_sparse = NoCSimulator(SparseAwareAllocator)

    m_base = sim_base.simulate(list(packets))
    m_sparse = sim_sparse.simulate(list(packets))
    b, s = print_comparison("Scatter-Reduce", m_base, m_sparse)
    all_results.append(("Scatter-Reduce", b, s))

    # -------------------------------------------------------------------------
    # Test 4: REAL BSR Data — conv2 (70% sparse)
    # -------------------------------------------------------------------------
    if os.path.exists(os.path.join(bsr_dir, "conv2", "row_ptr.npy")):
        print("\n[4/8] Running REAL BSR traffic: conv2 layer (70% block sparsity)...")
        packets = generate_bsr_traffic(bsr_dir, "conv2")
        print(f"       Generated {len(packets)} packets from conv2 BSR data")

        sim_base = NoCSimulator(BaselineAllocator)
        sim_sparse = NoCSimulator(SparseAwareAllocator)

        m_base = sim_base.simulate(list(packets))
        m_sparse = sim_sparse.simulate(list(packets))
        b, s = print_comparison("REAL BSR: conv2 (70% sparse)", m_base, m_sparse)
        all_results.append(("BSR conv2", b, s))
    else:
        print("\n[4/8] SKIP — conv2 BSR data not found")

    # -------------------------------------------------------------------------
    # Test 5: REAL BSR Data — fc1 (91% sparse)
    # -------------------------------------------------------------------------
    if os.path.exists(os.path.join(bsr_dir, "fc1", "row_ptr.npy")):
        print("\n[5/8] Running REAL BSR traffic: fc1 layer (91% block sparsity)...")
        packets = generate_bsr_traffic(bsr_dir, "fc1")
        print(f"       Generated {len(packets)} packets from fc1 BSR data")

        sim_base = NoCSimulator(BaselineAllocator)
        sim_sparse = NoCSimulator(SparseAwareAllocator)

        m_base = sim_base.simulate(list(packets))
        m_sparse = sim_sparse.simulate(list(packets))
        b, s = print_comparison("REAL BSR: fc1 (91% sparse)", m_base, m_sparse)
        all_results.append(("BSR fc1", b, s))
    else:
        print("\n[5/8] SKIP — fc1 BSR data not found")

    # -------------------------------------------------------------------------
    # Test 6: Saturated Mixed Traffic (THE KEY TEST)
    # Dense=80%, 6-10 flits; Sparse=20%, 1 flit. This is where VC
    # reservation matters — dense bulk hogs VCs and sparse control needs
    # a guaranteed lane.
    # -------------------------------------------------------------------------
    print("\n[6/8] Running Saturated Mixed traffic (3000 packets, 80% dense 6-10 flits)...")
    packets = generate_saturated_mixed_traffic(3000, sparse_frac=0.2)

    sim_base = NoCSimulator(BaselineAllocator)
    sim_sparse = NoCSimulator(SparseAwareAllocator)

    m_base = sim_base.simulate(list(packets))
    m_sparse = sim_sparse.simulate(list(packets))
    b, s = print_comparison("Saturated Mixed (80% dense, long pkts)", m_base, m_sparse)
    all_results.append(("Saturated Mixed", b, s))

    # -------------------------------------------------------------------------
    # Test 7: REAL BSR + Dense Background (BSR sparse interleaved with heavy
    # DMA-style dense transfers — the realistic SoC scenario)
    # -------------------------------------------------------------------------
    if os.path.exists(os.path.join(bsr_dir, "fc1", "row_ptr.npy")):
        print("\n[7/8] Running BSR fc1 + Dense Background (500 dense 8-flit DMA packets)...")
        packets = generate_bsr_with_dense_bg(bsr_dir, "fc1", dense_bg_packets=500)
        print(f"       Generated {len(packets)} packets (BSR sparse + dense DMA background)")

        sim_base = NoCSimulator(BaselineAllocator)
        sim_sparse = NoCSimulator(SparseAwareAllocator)

        m_base = sim_base.simulate(list(packets))
        m_sparse = sim_sparse.simulate(list(packets))
        b, s = print_comparison("BSR fc1 + Dense DMA Background", m_base, m_sparse)
        all_results.append(("BSR+DMA fc1", b, s))
    else:
        print("\n[7/8] SKIP — fc1 BSR data not found")

    # -------------------------------------------------------------------------
    # Test 8: Bottleneck — Dense mega-flows along diagonal jam intermediate
    # routers, sparse 1-hop packets need VCs at those same routers.
    # This is the EXACT use case for VC reservation.
    # -------------------------------------------------------------------------
    print("\n[8/8] Running Bottleneck traffic (500 dense diagonal + 300 sparse adjacent)...")
    packets = generate_bottleneck_traffic(num_dense=500, num_sparse=300)

    sim_base = NoCSimulator(BaselineAllocator)
    sim_sparse = NoCSimulator(SparseAwareAllocator)

    m_base = sim_base.simulate(list(packets))
    m_sparse = sim_sparse.simulate(list(packets))
    b, s = print_comparison("Bottleneck (diagonal dense + adj sparse)", m_base, m_sparse)
    all_results.append(("Bottleneck", b, s))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SUMMARY: Sparsity-Aware vs Baseline Improvement")
    print(f"{'='*72}")
    print(f"  {'Pattern':<25} {'Avg Lat':>10} {'P95 Lat':>10} {'Sparse Lat':>12} {'Throughput':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for name, b, s in all_results:
        avg_imp = ((b["avg_lat"] - s["avg_lat"]) / b["avg_lat"] * 100) if b["avg_lat"] > 0 else 0
        p95_imp = ((b["p95"] - s["p95"]) / b["p95"] * 100) if b["p95"] > 0 else 0
        sp_imp = ((b["sparse_avg"] - s["sparse_avg"]) / b["sparse_avg"] * 100) if b["sparse_avg"] > 0 else 0
        tp_imp = ((s["throughput"] - b["throughput"]) / b["throughput"] * 100) if b["throughput"] > 0 else 0

        print(f"  {name:<25} {avg_imp:>+9.1f}% {p95_imp:>+9.1f}% {sp_imp:>+11.1f}% {tp_imp:>+11.1f}%")

    print(f"{'='*72}")
    print("\n  Positive % = sparsity-aware allocator is BETTER")
    print("  Key win: sparse packet latency reduction under mixed traffic\n")


if __name__ == "__main__":
    main()
