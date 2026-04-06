#!/usr/bin/env python3
"""
NoC VC Allocator Full Comparison Benchmark
============================================
Compares FIVE allocator schemes across 16 traffic patterns including
MNIST BSR, synthetic CIFAR-10, synthetic Transformer, synthetic
nuScenes mini BEV-fusion traffic, and a trace-driven nuScenes mini
BEV frame workload.

Allocators:
  1. Baseline — standard round-robin (noc_vc_allocator.sv)
  2. Static Priority — sparse always wins (noc_vc_allocator_static_prio.sv)
  3. Weighted Round-Robin — 3:1 sparse:dense ratio (noc_vc_allocator_weighted_rr.sv)
  4. QVN-Style — 2+2 VC partition (noc_vc_allocator_qvn.sv)
  5. Sparsity-Aware — 1 reserved VC + overflow (noc_vc_allocator_sparse.sv)

Usage:
    python3 tools/noc_allocator_full_comparison.py
"""

import math
import os
import sys
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# =============================================================================
# NoC Parameters (matching noc_pkg.sv)
# =============================================================================
MESH_ROWS = 4
MESH_COLS = 4
NUM_NODES = MESH_ROWS * MESH_COLS
NUM_VCS = 4
NUM_PORTS = 5
BUF_DEPTH = 4

PORT_NORTH = 0
PORT_EAST  = 1
PORT_SOUTH = 2
PORT_WEST  = 3
PORT_LOCAL = 4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(PROJECT_ROOT, "data", "noc_traces")
NUSCENES_TRACE_PATH = os.path.join(TRACE_DIR, "nuscenes_mini_bev_frame.json")
NUSCENES_REDUCE_TRACE_PATH = os.path.join(TRACE_DIR, "nuscenes_mini_bev_reduce.json")
FRAME_TRACE_ENV = "NOC_BEV_FRAME_TRACE"
REDUCE_TRACE_ENV = "NOC_BEV_REDUCE_TRACE"


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
    release_cycle: int = 0
    reduce_group: Optional[int] = None


@dataclass
class Metrics:
    latencies: list = field(default_factory=list)
    total_cycles: int = 0
    total_flits_delivered: int = 0
    sparse_latencies: list = field(default_factory=list)
    dense_latencies: list = field(default_factory=list)
    blocked_cycles: int = 0


def clone_packet(packet: Packet) -> Packet:
    return Packet(
        src=packet.src,
        dst=packet.dst,
        is_sparse=packet.is_sparse,
        num_flits=packet.num_flits,
        release_cycle=packet.release_cycle,
        reduce_group=packet.reduce_group,
    )


# =============================================================================
# XY Routing
# =============================================================================
def node_to_rc(nid):
    return divmod(nid, MESH_COLS)

def rc_to_node(r, c):
    return r * MESH_COLS + c

def xy_next_hop_port(src, dst):
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
# Allocator 1: Baseline Round-Robin (noc_vc_allocator.sv)
# =============================================================================
class BaselineAllocator:
    NAME = "Baseline RR"

    def __init__(self):
        self.rr_ptr = [0] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold = [[0] * NUM_VCS for _ in range(NUM_PORTS)]

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        for offset in range(NUM_VCS):
            vc = (self.rr_ptr[output_port] + offset) % NUM_VCS
            if not self.vc_busy[output_port][vc]:
                self.vc_busy[output_port][vc] = True
                self.vc_hold[output_port][vc] = hold_cycles
                self.rr_ptr[output_port] = (vc + 1) % NUM_VCS
                return vc
        return None

    def tick(self):
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold[op][vc] -= 1
                    if self.vc_hold[op][vc] <= 0:
                        self.vc_busy[op][vc] = False


# =============================================================================
# Allocator 2: Static Priority (noc_vc_allocator_static_prio.sv)
#   Sparse requests always win over dense. No round-robin within class.
#   Problem: dense starves under sustained sparse load.
# =============================================================================
class StaticPriorityAllocator:
    NAME = "Static Priority"

    def __init__(self):
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold = [[0] * NUM_VCS for _ in range(NUM_PORTS)]
        # Track pending requests to implement priority between simultaneous
        self.sparse_queue = [0] * NUM_PORTS
        self.dense_queue = [0] * NUM_PORTS

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        if is_sparse:
            self.sparse_queue[output_port] += 1
        else:
            self.dense_queue[output_port] += 1

        # If sparse is requesting AND there's a sparse request pending,
        # sparse gets priority — try all VCs from index 0
        if is_sparse:
            for vc in range(NUM_VCS):
                if not self.vc_busy[output_port][vc]:
                    self.vc_busy[output_port][vc] = True
                    self.vc_hold[output_port][vc] = hold_cycles
                    self.sparse_queue[output_port] = max(0, self.sparse_queue[output_port] - 1)
                    return vc
            return None
        else:
            # Dense: only allocate if no sparse is waiting
            if self.sparse_queue[output_port] > 0:
                return None  # Yield to sparse
            for vc in range(NUM_VCS):
                if not self.vc_busy[output_port][vc]:
                    self.vc_busy[output_port][vc] = True
                    self.vc_hold[output_port][vc] = hold_cycles
                    self.dense_queue[output_port] = max(0, self.dense_queue[output_port] - 1)
                    return vc
            return None

    def tick(self):
        for op in range(NUM_PORTS):
            # Decay queues to prevent stale counts
            self.sparse_queue[op] = max(0, self.sparse_queue[op] - 1)
            self.dense_queue[op] = max(0, self.dense_queue[op] - 1)
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold[op][vc] -= 1
                    if self.vc_hold[op][vc] <= 0:
                        self.vc_busy[op][vc] = False


# =============================================================================
# Allocator 3: Weighted Round-Robin (noc_vc_allocator_weighted_rr.sv)
#   Sparse gets 3x grants, dense gets 1x. Alternates phases.
# =============================================================================
class WeightedRRAllocator:
    NAME = "Weighted RR (3:1)"

    W_SPARSE = 3
    W_DENSE = 1

    def __init__(self):
        self.rr_ptr = [0] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold = [[0] * NUM_VCS for _ in range(NUM_PORTS)]
        self.serve_sparse = [True] * NUM_PORTS  # Start in sparse phase
        self.weight_cnt = [self.W_SPARSE] * NUM_PORTS

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        op = output_port
        # Check if current phase matches request type
        if self.serve_sparse[op] and not is_sparse:
            # In sparse phase but dense request — only serve if no sparse waiting
            # (simplified: just try, but with lower priority)
            pass
        elif not self.serve_sparse[op] and is_sparse:
            # In dense phase but sparse request — still try (sparse can always attempt)
            pass

        # If in correct phase, or opposite phase has no pending, try to allocate
        for offset in range(NUM_VCS):
            vc = (self.rr_ptr[op] + offset) % NUM_VCS
            if not self.vc_busy[op][vc]:
                self.vc_busy[op][vc] = True
                self.vc_hold[op][vc] = hold_cycles
                self.rr_ptr[op] = (vc + 1) % NUM_VCS

                # Update weight counter
                if (is_sparse and self.serve_sparse[op]) or \
                   (not is_sparse and not self.serve_sparse[op]):
                    self.weight_cnt[op] -= 1
                    if self.weight_cnt[op] <= 0:
                        # Switch phase
                        self.serve_sparse[op] = not self.serve_sparse[op]
                        self.weight_cnt[op] = self.W_SPARSE if self.serve_sparse[op] else self.W_DENSE

                return vc
        return None

    def tick(self):
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold[op][vc] -= 1
                    if self.vc_hold[op][vc] <= 0:
                        self.vc_busy[op][vc] = False


# =============================================================================
# Allocator 4: QVN-Style Partitioned (noc_vc_allocator_qvn.sv)
#   VCs 0,1 = dense ONLY.  VCs 2,3 = sparse ONLY.  No overflow.
# =============================================================================
class QVNAllocator:
    NAME = "QVN (2+2 split)"

    VN_SPLIT = NUM_VCS // 2  # VCs 0..1 dense, VCs 2..3 sparse

    def __init__(self):
        self.rr_dense = [0] * NUM_PORTS
        self.rr_sparse = [self.VN_SPLIT] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold = [[0] * NUM_VCS for _ in range(NUM_PORTS)]

    def allocate(self, output_port, is_sparse=False, hold_cycles=1):
        op = output_port
        if is_sparse:
            # Sparse: ONLY VCs VN_SPLIT..NUM_VCS-1
            num_sparse_vcs = NUM_VCS - self.VN_SPLIT
            for offset in range(num_sparse_vcs):
                vc = self.VN_SPLIT + (self.rr_sparse[op] - self.VN_SPLIT + offset) % num_sparse_vcs
                if not self.vc_busy[op][vc]:
                    self.vc_busy[op][vc] = True
                    self.vc_hold[op][vc] = hold_cycles
                    self.rr_sparse[op] = self.VN_SPLIT + (vc - self.VN_SPLIT + 1) % num_sparse_vcs
                    return vc
            return None  # NO overflow — this is the QVN limitation
        else:
            # Dense: ONLY VCs 0..VN_SPLIT-1
            for offset in range(self.VN_SPLIT):
                vc = (self.rr_dense[op] + offset) % self.VN_SPLIT
                if not self.vc_busy[op][vc]:
                    self.vc_busy[op][vc] = True
                    self.vc_hold[op][vc] = hold_cycles
                    self.rr_dense[op] = (vc + 1) % self.VN_SPLIT
                    return vc
            return None

    def tick(self):
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold[op][vc] -= 1
                    if self.vc_hold[op][vc] <= 0:
                        self.vc_busy[op][vc] = False


# =============================================================================
# Allocator 5: Sparsity-Aware (noc_vc_allocator_sparse.sv) — OURS
#   Adaptive reservation: dense avoids VC-3 only when sparse traffic dominates.
# =============================================================================
class SparseAwareAllocator:
    NAME = "Sparsity-Aware (Ours)"

    RESERVED_VC = NUM_VCS - 1
    MIN_SAMPLES = 64
    MAX_COUNT = 0xFFFF

    def __init__(self):
        self.rr_sparse = [0] * NUM_PORTS
        self.rr_dense = [0] * NUM_PORTS
        self.vc_busy = [[False] * NUM_VCS for _ in range(NUM_PORTS)]
        self.vc_hold = [[0] * NUM_VCS for _ in range(NUM_PORTS)]
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
        op = output_port
        self._record_request(is_sparse)

        if is_sparse:
            for offset in range(NUM_VCS):
                vc = (self.rr_sparse[op] + offset) % NUM_VCS
                if not self.vc_busy[op][vc]:
                    self.vc_busy[op][vc] = True
                    self.vc_hold[op][vc] = hold_cycles
                    self.rr_sparse[op] = (vc + 1) % NUM_VCS
                    return vc
            return None
        else:
            dense_vcs = NUM_VCS - 1 if self.reservation_active() else NUM_VCS
            for offset in range(dense_vcs):
                vc = (self.rr_dense[op] + offset) % dense_vcs
                if not self.vc_busy[op][vc]:
                    self.vc_busy[op][vc] = True
                    self.vc_hold[op][vc] = hold_cycles
                    self.rr_dense[op] = (vc + 1) % dense_vcs
                    return vc
            return None

    def tick(self):
        for op in range(NUM_PORTS):
            for vc in range(NUM_VCS):
                if self.vc_busy[op][vc]:
                    self.vc_hold[op][vc] -= 1
                    if self.vc_hold[op][vc] <= 0:
                        self.vc_busy[op][vc] = False


# =============================================================================
# Cycle-Accurate NoC Simulator
# =============================================================================
class NoCSimulator:
    def __init__(self, allocator_class):
        self.allocator_class = allocator_class
        self.routers = [allocator_class() for _ in range(NUM_NODES)]

    def simulate(self, packets, max_cycles=100000):
        metrics = Metrics()
        pending = list(packets)
        for i, p in enumerate(pending):
            p.id = i

        in_flight = []
        cycle = 0

        while (pending or in_flight) and cycle < max_cycles:
            cycle += 1

            for router in self.routers:
                router.tick()

            # Advance packets that finished their hop
            new_in_flight = []
            for (cur_node, pkt, vc, outp, hops_rem, hold_rem) in in_flight:
                if hold_rem <= 0:
                    sr, sc = node_to_rc(cur_node)
                    if outp == PORT_EAST:     nxt = rc_to_node(sr, sc + 1)
                    elif outp == PORT_WEST:   nxt = rc_to_node(sr, sc - 1)
                    elif outp == PORT_SOUTH:  nxt = rc_to_node(sr + 1, sc)
                    elif outp == PORT_NORTH:  nxt = rc_to_node(sr - 1, sc)
                    else:                     nxt = cur_node

                    if hops_rem <= 1:
                        lat = cycle - pkt.inject_cycle
                        metrics.latencies.append(lat)
                        metrics.total_flits_delivered += 1
                        if pkt.is_sparse:
                            metrics.sparse_latencies.append(lat)
                        else:
                            metrics.dense_latencies.append(lat)
                        continue

                    next_outp = xy_next_hop_port(nxt, pkt.dst)
                    hold = pkt.num_flits
                    vc_alloc = self.routers[nxt].allocate(next_outp, pkt.is_sparse, hold)
                    if vc_alloc is not None:
                        new_in_flight.append((nxt, pkt, vc_alloc, next_outp, hops_rem - 1, hold))
                    else:
                        new_in_flight.append((cur_node, pkt, -1, outp, hops_rem, 0))
                        metrics.blocked_cycles += 1
                else:
                    new_in_flight.append((cur_node, pkt, vc, outp, hops_rem, hold_rem - 1))

            # Retry blocked packets
            retried = []
            for entry in new_in_flight:
                cur_node, pkt, vc, outp, hops_rem, hold_rem = entry
                if vc == -1:
                    sr, sc = node_to_rc(cur_node)
                    if outp == PORT_EAST:     nxt = rc_to_node(sr, sc + 1)
                    elif outp == PORT_WEST:   nxt = rc_to_node(sr, sc - 1)
                    elif outp == PORT_SOUTH:  nxt = rc_to_node(sr + 1, sc)
                    elif outp == PORT_NORTH:  nxt = rc_to_node(sr - 1, sc)
                    else:                     nxt = cur_node

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

            # Inject new packets — two-pass: sparse first, then dense
            injected = 0
            max_inject = NUM_NODES

            sparse_pending = []
            dense_pending = []
            eligible_indices = [i for i, pkt in enumerate(pending) if pkt.release_cycle <= cycle]
            scan_limit = min(max_inject * 2, len(eligible_indices))
            for si in eligible_indices[:scan_limit]:
                pkt = pending[si]
                if pkt.is_sparse:
                    sparse_pending.append(si)
                else:
                    dense_pending.append(si)

            injected_indices = set()

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

            if injected_indices:
                pending = [p for i, p in enumerate(pending) if i not in injected_indices]

        metrics.total_cycles = cycle
        return metrics


# =============================================================================
# Traffic Generators
# =============================================================================

def gen_uniform(num_packets=1000, sparse_frac=0.5, seed=42):
    rng = np.random.RandomState(seed)
    packets = []
    for _ in range(num_packets):
        src = rng.randint(0, NUM_NODES)
        dst = rng.randint(0, NUM_NODES)
        while dst == src:
            dst = rng.randint(0, NUM_NODES)
        is_sparse = rng.random() < sparse_frac
        nf = rng.randint(1, 3) if is_sparse else rng.randint(3, 6)
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))
    return packets


def gen_hotspot(num_packets=1000, hotspot=0, frac=0.6, sparse_frac=0.3, seed=42):
    rng = np.random.RandomState(seed)
    packets = []
    for _ in range(num_packets):
        if rng.random() < frac:
            src = rng.randint(0, NUM_NODES)
            dst = hotspot
            if src == dst:
                src = (src + 1) % NUM_NODES
        else:
            src, dst = rng.randint(0, NUM_NODES), rng.randint(0, NUM_NODES)
            while dst == src:
                dst = rng.randint(0, NUM_NODES)
        is_sparse = rng.random() < sparse_frac
        nf = rng.randint(1, 2) if is_sparse else rng.randint(3, 6)
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))
    return packets


def gen_scatter_reduce(num_packets=800, controller=0, seed=42):
    packets = []
    tiles = [n for n in range(NUM_NODES) if n != controller]
    half = num_packets // 2
    for i in range(half):
        dst = tiles[i % len(tiles)]
        packets.append(Packet(src=controller, dst=dst, is_sparse=False, num_flits=4))
    for i in range(num_packets - half):
        src = tiles[i % len(tiles)]
        packets.append(Packet(
            src=src,
            dst=controller,
            is_sparse=True,
            num_flits=1,
            reduce_group=i // len(tiles),
        ))
    return packets


def gen_saturated_mixed(num_packets=3000, sparse_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    packets = []
    for _ in range(num_packets):
        src, dst = rng.randint(0, NUM_NODES), rng.randint(0, NUM_NODES)
        while dst == src:
            dst = rng.randint(0, NUM_NODES)
        is_sparse = rng.random() < sparse_frac
        nf = 1 if is_sparse else rng.randint(6, 11)
        packets.append(Packet(src=src, dst=dst, is_sparse=is_sparse, num_flits=nf))
    return packets


def gen_bottleneck(num_dense=500, num_sparse=300, seed=42):
    rng = np.random.RandomState(seed)
    packets = []
    flows = [(0, 15), (3, 12), (1, 14), (2, 13), (4, 11), (7, 8)]
    for i in range(num_dense):
        src, dst = flows[i % len(flows)]
        packets.append(Packet(src=src, dst=dst, is_sparse=False, num_flits=rng.randint(8, 13)))
    pairs = [(5, 6), (6, 7), (9, 10), (10, 11), (5, 9), (6, 10),
             (7, 11), (4, 5), (8, 9), (1, 5), (2, 6), (3, 7)]
    for i in range(num_sparse):
        src, dst = pairs[i % len(pairs)]
        packets.append(Packet(src=src, dst=dst, is_sparse=True, num_flits=1))
    rng.shuffle(packets)
    return packets


def gen_bsr_traffic(bsr_dir, layer_name, num_tiles=15, seed=42):
    rng = np.random.RandomState(seed)
    row_ptr = np.load(os.path.join(bsr_dir, layer_name, "row_ptr.npy"))
    col_idx = np.load(os.path.join(bsr_dir, layer_name, "col_idx.npy"))
    packets = []
    controller = 0
    num_block_rows = len(row_ptr) - 1
    for br in range(num_block_rows):
        nnz_start, nnz_end = row_ptr[br], row_ptr[br + 1]
        nnz_cols = col_idx[nnz_start:nnz_end]
        src_tile = 1 + (br % num_tiles)
        for ci in nnz_cols:
            dst_tile = 1 + (int(ci) % num_tiles)
            if dst_tile == src_tile:
                dst_tile = (dst_tile % num_tiles) + 1
            packets.append(Packet(src=src_tile, dst=dst_tile, is_sparse=True, num_flits=2))
        if nnz_end > nnz_start:
            packets.append(Packet(src=controller, dst=src_tile, is_sparse=False, num_flits=4))
    for t in range(1, num_tiles + 1):
        for _ in range(5):
            packets.append(Packet(src=controller, dst=t, is_sparse=False, num_flits=4))
    for t in range(1, num_tiles + 1):
        for _ in range(4):
            packets.append(Packet(src=t, dst=controller, is_sparse=True, num_flits=2))
    return packets


def gen_bsr_dense_bg(bsr_dir, layer_name, num_tiles=15, dense_bg=500, seed=42):
    rng = np.random.RandomState(seed)
    row_ptr = np.load(os.path.join(bsr_dir, layer_name, "row_ptr.npy"))
    col_idx = np.load(os.path.join(bsr_dir, layer_name, "col_idx.npy"))
    packets = []
    controller = 0
    num_block_rows = len(row_ptr) - 1
    for i in range(dense_bg):
        dst = 1 + (i % num_tiles)
        packets.append(Packet(src=controller, dst=dst, is_sparse=False, num_flits=8))
    for br in range(num_block_rows):
        nnz_start, nnz_end = row_ptr[br], row_ptr[br + 1]
        src_tile = 1 + (br % num_tiles)
        for ci in col_idx[nnz_start:nnz_end]:
            dst_tile = 1 + (int(ci) % num_tiles)
            if dst_tile == src_tile:
                dst_tile = (dst_tile % num_tiles) + 1
            packets.append(Packet(src=src_tile, dst=dst_tile, is_sparse=True, num_flits=2))
    rng.shuffle(packets)
    return packets


# =============================================================================
# NEW: CIFAR-10 Synthetic Sparse Workload
# =============================================================================
def gen_cifar10_traffic(num_packets=2000, seed=42):
    """Synthetic CIFAR-10 workload modeling a sparse ResNet-like network.

    CIFAR-10 characteristics vs MNIST:
    - 3 input channels (RGB) instead of 1
    - More conv layers (3 conv + 2 FC typical)
    - Block sparsity: ~50% conv1, ~65% conv2, ~75% conv3, ~85% fc1
    - Larger activation tiles (32x32 → 16x16 → 8x8)
    - More total traffic due to deeper network

    Traffic model:
    - Dense: activation tile scatter (6-8 flits), weight loading (8 flits)
    - Sparse: BSR non-zero blocks (2 flits), partial sums (1-2 flits),
              barrier syncs (1 flit)
    """
    rng = np.random.RandomState(seed)
    packets = []
    controller = 0
    num_tiles = 15

    # Layer configs: (name, block_sparsity, activation_flits, num_blocks)
    layers = [
        ("conv1", 0.50, 8, 48),   # 3ch * 16 blocks, 50% sparse
        ("conv2", 0.65, 6, 64),   # 16ch * 4 blocks, 65% sparse
        ("conv3", 0.75, 4, 128),  # 32ch * 4 blocks, 75% sparse
        ("fc1",   0.85, 2, 256),  # 256-wide, 85% sparse
        ("fc2",   0.80, 2, 64),   # 64-wide, 80% sparse
    ]

    for layer_name, sparsity, act_flits, num_blocks in layers:
        # Dense: scatter activations to tiles
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=False,
                                  num_flits=act_flits))

        # Sparse: non-zero weight blocks (only (1-sparsity) fraction)
        nnz_blocks = int(num_blocks * (1.0 - sparsity))
        for b in range(nnz_blocks):
            src = 1 + (b % num_tiles)
            dst = 1 + ((b + 3) % num_tiles)
            if dst == src:
                dst = 1 + ((dst) % num_tiles)
            packets.append(Packet(src=src, dst=dst, is_sparse=True, num_flits=2))

        # Sparse: partial sum reduction
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=t, dst=controller, is_sparse=True, num_flits=1))

        # Sparse: barrier sync per layer
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=True, num_flits=1))

        # Dense: weight loading for next layer
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=False, num_flits=8))

    rng.shuffle(packets)
    return packets


# =============================================================================
# NEW: Transformer (Small Attention) Synthetic Sparse Workload
# =============================================================================
def gen_transformer_traffic(num_packets=2500, seed=42):
    """Synthetic Transformer (small ViT/BERT-tiny) sparse workload.

    Transformer characteristics:
    - Multi-head attention: Q, K, V projections (dense), attention scores (sparse)
    - FFN layers: two FC layers with ~60-80% sparsity after ReLU/GELU
    - LayerNorm: all-reduce across tiles (sparse control)
    - Softmax attention: mostly sparse after top-k masking

    Traffic model:
    - Dense: QKV projection broadcasts (8 flits), FFN weight loading (8 flits)
    - Sparse: attention scores after masking (1-2 flits), FFN activations
              after ReLU (2 flits), LayerNorm reduce (1 flit),
              residual additions (2 flits)

    Key difference from CNN: attention creates all-to-all traffic
    (every tile needs scores from every other tile), creating much
    higher contention than CNN's structured scatter-reduce pattern.
    """
    rng = np.random.RandomState(seed)
    packets = []
    controller = 0
    num_tiles = 15
    num_heads = 4  # Multi-head attention

    # 2 transformer blocks (tiny model)
    for block in range(2):
        # === Multi-Head Attention ===

        # Dense: broadcast Q, K, V projections to all tiles
        for proj in range(3):  # Q, K, V
            for t in range(1, num_tiles + 1):
                packets.append(Packet(src=controller, dst=t, is_sparse=False,
                                      num_flits=8))

        # Sparse: attention score exchange (ALL-TO-ALL within each head)
        # This is the high-contention pattern — every tile sends to every other
        tiles_per_head = max(1, num_tiles // num_heads)
        for head in range(num_heads):
            head_start = 1 + head * tiles_per_head
            head_end = min(head_start + tiles_per_head, num_tiles + 1)
            for src_t in range(head_start, head_end):
                for dst_t in range(head_start, head_end):
                    if src_t != dst_t:
                        # After top-k masking, ~70% of attention scores are zero
                        if rng.random() > 0.70:
                            packets.append(Packet(src=src_t, dst=dst_t,
                                                  is_sparse=True, num_flits=2))

        # Sparse: attention output reduce per head
        for head in range(num_heads):
            head_start = 1 + head * tiles_per_head
            head_end = min(head_start + tiles_per_head, num_tiles + 1)
            for t in range(head_start, head_end):
                packets.append(Packet(src=t, dst=controller, is_sparse=True,
                                      num_flits=1))

        # Sparse: LayerNorm all-reduce (broadcast mean/variance)
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=t, dst=controller, is_sparse=True, num_flits=1))
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=True, num_flits=1))

        # === FFN (Feed-Forward Network) ===

        # Dense: FFN weight loading
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=False,
                                  num_flits=8))

        # Sparse: FFN activations after GELU (~65% sparsity)
        ffn_blocks = 128
        nnz_ffn = int(ffn_blocks * 0.35)
        for b in range(nnz_ffn):
            src = 1 + (b % num_tiles)
            dst = 1 + ((b + 5) % num_tiles)
            if dst == src:
                dst = 1 + (dst % num_tiles)
            packets.append(Packet(src=src, dst=dst, is_sparse=True, num_flits=2))

        # Dense: second FFN layer weights
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=False,
                                  num_flits=8))

        # Sparse: residual connection (add input to output)
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=True, num_flits=2))

        # Sparse: LayerNorm #2
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=t, dst=controller, is_sparse=True, num_flits=1))
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=True, num_flits=1))

        # Sparse: barrier sync end of block
        for t in range(1, num_tiles + 1):
            packets.append(Packet(src=controller, dst=t, is_sparse=True, num_flits=1))

    rng.shuffle(packets)
    return packets


# =============================================================================
# NEW: nuScenes mini Multi-Camera BEV Fusion Workload
# =============================================================================
def gen_nuscenes_bev_fusion_traffic(seed=42, dense_bg=0):
    """Synthetic nuScenes mini BEV-fusion workload.

    Why this is a better autonomy fit than a small drone CNN:
    - 6 camera streams create genuine multi-branch traffic
    - view fusion creates many-to-few sparse aggregation
    - detection/planning heads add dense background traffic

    Mapping used here:
    - controller / DMA source: node 0
    - camera tiles: 1, 2, 4, 7, 13, 14
    - BEV fusion tiles: 5, 6, 9, 10
    - head / planner tiles: 8, 11, 12, 15

    Traffic model:
    - Dense: per-camera feature ingress and weight loads
    - Sparse: frustum-filtered feature tokens into shared BEV tiles
    - Sparse: BEV overlap exchange and query/control traffic
    - Optional dense background: map / weight DMA bursts
    """
    rng = np.random.RandomState(seed)
    packets = []

    controller = 0
    camera_tiles = [1, 2, 4, 7, 13, 14]
    fusion_tiles = [5, 6, 9, 10]
    head_tiles = [8, 11, 12, 15]
    planner_tile = 10

    # Initial per-camera calibration / feature buffer setup.
    for cam in camera_tiles:
        packets.append(Packet(src=controller, dst=cam, is_sparse=False, num_flits=10))

    # Three feature scales moving from camera branches into a shared BEV space.
    scales = [
        ("p3", 12, 28),
        ("p4", 10, 20),
        ("p5", 8, 14),
    ]

    for _, dense_flits, sparse_tokens in scales:
        # Camera backbone / neck weight loads and dense activation ingress.
        for cam in camera_tiles:
            packets.append(Packet(src=controller, dst=cam, is_sparse=False, num_flits=dense_flits))

        # Frustum-filtered sparse tokens into BEV fusion tiles.
        for cam_idx, cam in enumerate(camera_tiles):
            primary = fusion_tiles[cam_idx % len(fusion_tiles)]
            secondary = fusion_tiles[(cam_idx + 1) % len(fusion_tiles)]
            for token_idx in range(sparse_tokens):
                dst = secondary if (token_idx % 4 == 0) else primary
                packets.append(Packet(src=cam, dst=dst, is_sparse=True, num_flits=2))

        # Lightweight per-scale camera completion / control updates.
        for cam in camera_tiles:
            packets.append(Packet(src=cam, dst=planner_tile, is_sparse=True, num_flits=1))

    # Fusion overlap exchange between neighboring BEV tiles.
    for src in fusion_tiles:
        for dst in fusion_tiles:
            if src != dst and rng.random() < 0.55:
                packets.append(Packet(src=src, dst=dst, is_sparse=True, num_flits=2))

    # Detection / planning queries and sparse responses.
    for head in head_tiles:
        packets.append(Packet(src=planner_tile, dst=head, is_sparse=True, num_flits=2))
        packets.append(Packet(src=head, dst=planner_tile, is_sparse=True, num_flits=1))

    # Dense head weight loads.
    for head in head_tiles:
        packets.append(Packet(src=controller, dst=head, is_sparse=False, num_flits=8))

    # Optional heavy dense DMA background.
    for i in range(dense_bg):
        dst = 1 + (i % 15)
        packets.append(Packet(src=controller, dst=dst, is_sparse=False, num_flits=8))

    rng.shuffle(packets)
    return packets


def resolve_trace_path(trace_path):
    return trace_path if os.path.isabs(trace_path) else os.path.join(PROJECT_ROOT, trace_path)


def load_packet_trace(trace_path):
    """Load a compact packet trace from JSON and expand it into Packet objects."""
    resolved = resolve_trace_path(trace_path)
    with open(resolved, "r") as f:
        spec = json.load(f)

    packets = []
    for entry in spec.get("packets", []):
        sources = entry.get("srcs")
        destinations = entry.get("dsts")
        if sources is None:
            sources = [entry["src"]]
        if destinations is None:
            destinations = [entry["dst"]]

        if len(sources) > 1 and len(destinations) > 1:
            if len(sources) != len(destinations):
                raise ValueError(f"Trace entry must use matching srcs/dsts lengths: {entry}")
            pairs = list(zip(sources, destinations))
        else:
            pairs = [(src, dst) for src in sources for dst in destinations]

        count = int(entry.get("count", 1))
        release_cycle = int(entry.get("release_cycle", 0))
        release_stride = int(entry.get("release_stride", 0))
        pair_release_stride = int(entry.get("pair_release_stride", 0))
        reduce_group = entry.get("reduce_group")
        reduce_group_start = entry.get("reduce_group_start")
        reduce_group_stride = int(entry.get("reduce_group_stride", 1))

        for pair_index, (src, dst) in enumerate(pairs):
            pair_release = release_cycle + pair_index * pair_release_stride
            for idx in range(count):
                group = None
                if reduce_group_start is not None:
                    group = int(reduce_group_start) + idx * reduce_group_stride
                elif reduce_group is not None:
                    group = int(reduce_group)

                packets.append(Packet(
                    src=int(src),
                    dst=int(dst),
                    is_sparse=bool(entry["is_sparse"]),
                    num_flits=int(entry.get("num_flits", 1)),
                    release_cycle=pair_release + idx * release_stride,
                    reduce_group=group,
                ))

    return spec, packets


def append_dense_dma_background(packets, dense_bg=0, start_cycle=0, num_flits=8):
    for idx in range(dense_bg):
        packets.append(Packet(
            src=0,
            dst=1 + (idx % 15),
            is_sparse=False,
            num_flits=num_flits,
            release_cycle=start_cycle + (idx // 15),
        ))
    return packets


def gen_trace_driven_nuscenes_bev_traffic(trace_path=NUSCENES_TRACE_PATH,
                                          dense_bg=0,
                                          bg_start_cycle=180):
    """Replay a compact frame trace for multi-camera BEV fusion."""
    _, packets = load_packet_trace(trace_path)
    return append_dense_dma_background(packets, dense_bg=dense_bg,
                                       start_cycle=bg_start_cycle)


# =============================================================================
# Result Formatting
# =============================================================================
def format_metrics(name, metrics):
    lats = metrics.latencies
    if not lats:
        return {"name": name, "avg_lat": 0, "p50": 0, "p95": 0, "p99": 0,
                "max_lat": 0, "throughput": 0, "jain": 0, "blocked": 0,
                "sparse_avg": 0, "dense_avg": 0, "total_cycles": 0, "delivered": 0}

    lats_sorted = sorted(lats)
    n = len(lats_sorted)
    sum_x = sum(lats)
    sum_x2 = sum(x*x for x in lats)
    jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
    sparse_avg = float(np.mean(metrics.sparse_latencies)) if metrics.sparse_latencies else 0
    dense_avg = float(np.mean(metrics.dense_latencies)) if metrics.dense_latencies else 0

    return {
        "name": name,
        "avg_lat": float(np.mean(lats)),
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


def run_all_allocators(pattern_name, packets):
    """Run all 5 allocators on the same traffic pattern."""
    allocators = [
        BaselineAllocator,
        StaticPriorityAllocator,
        WeightedRRAllocator,
        QVNAllocator,
        SparseAwareAllocator,
    ]

    results = {}
    for alloc_cls in allocators:
        sim = NoCSimulator(alloc_cls)
        m = sim.simulate([clone_packet(p) for p in packets])
        results[alloc_cls.NAME] = format_metrics(alloc_cls.NAME, m)

    return results


def print_full_comparison(pattern_name, results):
    """Print side-by-side comparison of all allocators."""
    baseline = results["Baseline RR"]
    names = ["Baseline RR", "Static Priority", "Weighted RR (3:1)",
             "QVN (2+2 split)", "Sparsity-Aware (Ours)"]

    print(f"\n{'='*96}")
    print(f"  {pattern_name}")
    print(f"  Packets delivered: {baseline['delivered']}")
    print(f"{'='*96}")

    header = f"  {'Metric':<22}"
    for n in names:
        short = n[:14]
        header += f" {short:>14}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*14}" * 5)

    metrics_to_show = [
        ("Avg Latency", "avg_lat", True),
        ("P50 Latency", "p50", True),
        ("P95 Latency", "p95", True),
        ("Sparse Avg Lat", "sparse_avg", True),
        ("Dense Avg Lat", "dense_avg", True),
        ("Throughput", "throughput", False),
        ("Blocked Cycles", "blocked", True),
        ("Jain Fairness", "jain", False),
    ]

    for label, key, lower_better in metrics_to_show:
        row = f"  {label:<22}"
        for n in names:
            val = results[n][key]
            if n == "Baseline RR":
                row += f" {val:>14.2f}"
            else:
                bval = baseline[key]
                if bval > 0:
                    pct = ((bval - val) / bval * 100) if lower_better else ((val - bval) / bval * 100)
                    row += f" {val:>8.2f}{pct:>+5.1f}%"
                else:
                    row += f" {val:>14.2f}"
        print(row)

    print(f"{'='*96}")
    return results


# =============================================================================
# In-Network Reduction Model
# =============================================================================
# Models the router-side traffic transformation when noc_innet_reduce.sv is enabled.
#
# Important scope note:
#   The live RTL currently performs aggregation inside intermediate routers,
#   but the root-side reduce_engine is not instantiated in the active tile path.
#   That means the router fabric can collapse traffic along the path, but the
#   final combine at the reduction root is still a separate concern.
#
# This model therefore separates two views:
#   1. Post-router packets still visible at the root-facing edge(s)
#   2. Full internal tree traffic (flit-hops across all links)
#
# Only packets tagged with an explicit reduce_group are eligible for INR.
# Generic one-flit sparse traffic such as barriers or control messages is not.
# =============================================================================

def xy_path_edges(src, dst):
    edges = []
    cur = src
    while cur != dst:
      sr, sc = node_to_rc(cur)
      outp = xy_next_hop_port(cur, dst)
      if outp == PORT_EAST:
          nxt = rc_to_node(sr, sc + 1)
      elif outp == PORT_WEST:
          nxt = rc_to_node(sr, sc - 1)
      elif outp == PORT_SOUTH:
          nxt = rc_to_node(sr + 1, sc)
      elif outp == PORT_NORTH:
          nxt = rc_to_node(sr - 1, sc)
      else:
          nxt = cur
      edges.append((cur, nxt))
      cur = nxt
    return edges


def _build_reduce_tree(group_packets):
    root = group_packets[0].dst
    contributors = sorted({p.src for p in group_packets if p.src != root})
    parent = {}
    children = defaultdict(set)

    for src in contributors:
        for node, next_node in xy_path_edges(src, root):
            parent[node] = next_node
            children[next_node].add(node)

    memo = {}

    def ready_cycle(node):
        if node in memo:
            return memo[node]
        child_ready = [ready_cycle(child) + 1 for child in children.get(node, ())]
        memo[node] = max(child_ready) if child_ready else 0
        return memo[node]

    return root, parent, {k: sorted(v) for k, v in children.items()}, ready_cycle


def _build_router_inr_forward_packets(group_packets):
    if len(group_packets) <= 1:
        return [clone_packet(packet) for packet in group_packets]

    root, _, children, ready_cycle = _build_reduce_tree(group_packets)
    group_id = group_packets[0].reduce_group
    forwarded = []

    for child in children.get(root, []):
        forwarded.append(Packet(
            src=child,
            dst=root,
            is_sparse=True,
            num_flits=1,
            release_cycle=ready_cycle(child),
            reduce_group=group_id,
        ))

    return forwarded


def _build_router_inr_tree_packets(group_packets):
    if len(group_packets) <= 1:
        return [clone_packet(packet) for packet in group_packets]

    _, parent, _, ready_cycle = _build_reduce_tree(group_packets)
    group_id = group_packets[0].reduce_group
    tree_packets = []

    for node, next_node in sorted(parent.items()):
        tree_packets.append(Packet(
            src=node,
            dst=next_node,
            is_sparse=True,
            num_flits=1,
            release_cycle=ready_cycle(node),
            reduce_group=group_id,
        ))

    return tree_packets


def apply_router_innet_reduction(packets):
    """Apply router-side INR only to explicit reduction groups.

    Returns:
      1. Packets still visible after router-side aggregation (root-facing view)
      2. Full internal INR tree packets for flit-hop accounting
    """
    passthrough = [clone_packet(packet) for packet in packets if packet.reduce_group is None]
    grouped = defaultdict(list)

    for packet in packets:
        if packet.reduce_group is not None:
            grouped[(packet.dst, packet.reduce_group)].append(packet)

    forwarded = []
    tree_packets = []

    for _, group_packets in sorted(grouped.items()):
        forwarded.extend(_build_router_inr_forward_packets(group_packets))
        tree_packets.extend(_build_router_inr_tree_packets(group_packets))

    return passthrough + forwarded, passthrough + tree_packets


def calc_total_flit_hops(packets):
    """Count total flit-hops: sum of (hop_count × num_flits) for all packets."""
    return sum(hop_count(p.src, p.dst) * p.num_flits for p in packets)


def gen_fc_layer_reduce(num_tiles=15, output_neurons=128, root=0, seed=42):
    """FC-layer all-reduce workload: all tiles send partial sums to root.

    Models the final layer reduction in MNIST/CIFAR inference:
    - Each tile computed a partial sum over its weight rows
    - All tiles must reduce to the root (node 0) for final softmax
    - This is exactly the traffic noc_innet_reduce.sv is designed to handle

    Each neuron is tagged as its own reduce_group so the benchmark only
    aggregates contributions that should legally combine in hardware.
    """
    packets = []
    for tile in range(1, num_tiles + 1):
        for neuron in range(output_neurons):
            packets.append(Packet(
                src=tile,
                dst=root,
                is_sparse=True,
                num_flits=1,
                reduce_group=neuron,
            ))
    return packets


def gen_fc_layer_reduce_with_dma(num_tiles=15, output_neurons=128, root=0,
                                   dma_flits=500, seed=42):
    """FC all-reduce + simultaneous DMA traffic (weight loading)."""
    rng = np.random.RandomState(seed)
    packets = gen_fc_layer_reduce(num_tiles, output_neurons, root)
    for i in range(dma_flits):
        dst = 1 + (i % num_tiles)
        packets.append(Packet(src=root, dst=dst, is_sparse=False, num_flits=8))
    rng.shuffle(packets)
    return packets


def gen_saturated_reduce_hotspot(num_tiles=15, output_neurons=192, root=0,
                                 dense_writebacks=500, seed=42):
    """Saturated root hotspot: many reduce groups plus dense writebacks to root."""
    rng = np.random.RandomState(seed)
    packets = gen_fc_layer_reduce(num_tiles=num_tiles, output_neurons=output_neurons, root=root)
    for _ in range(dense_writebacks):
        src = 1 + rng.randint(0, num_tiles)
        packets.append(Packet(src=src, dst=root, is_sparse=False, num_flits=rng.randint(6, 9)))
    rng.shuffle(packets)
    return packets


def gen_nuscenes_bev_reduce(num_bev_cells=192, root=10):
    """Reduction-heavy autonomy workload derived from multi-camera BEV fusion.

    Each BEV cell is treated as one reduction group with contributions from
    6 camera branches placed around the mesh periphery. A competent mapping
    would keep the fusion root interior; we therefore use node 10.
    """
    camera_tiles = [1, 2, 4, 7, 13, 14]
    packets = []

    for bev_cell in range(num_bev_cells):
        for cam in camera_tiles:
            packets.append(Packet(
                src=cam,
                dst=root,
                is_sparse=True,
                num_flits=1,
                reduce_group=bev_cell,
            ))

    return packets


def gen_nuscenes_bev_reduce_with_dma(num_bev_cells=192, root=10, dense_bg=400, seed=42):
    """BEV fusion reduction plus dense map / weight DMA background."""
    rng = np.random.RandomState(seed)
    packets = gen_nuscenes_bev_reduce(num_bev_cells=num_bev_cells, root=root)

    for i in range(dense_bg):
        dst = 1 + (i % 15)
        packets.append(Packet(src=0, dst=dst, is_sparse=False, num_flits=8))

    rng.shuffle(packets)
    return packets


def gen_trace_driven_nuscenes_bev_reduce(trace_path=NUSCENES_REDUCE_TRACE_PATH,
                                         dense_bg=0,
                                         bg_start_cycle=96):
    """Replay a compact BEV reduction trace with explicit reduce_group tags."""
    _, packets = load_packet_trace(trace_path)
    return append_dense_dma_background(packets, dense_bg=dense_bg,
                                       start_cycle=bg_start_cycle)


def run_innet_reduce_benchmark(trace_reduce_path=NUSCENES_REDUCE_TRACE_PATH):
    """Benchmark comparing baseline, sparse-aware, and sparse-aware + in-network reduction."""

    scenarios = [
        ("FC All-Reduce (128 neurons, 15 tiles)",
         gen_fc_layer_reduce()),
        ("FC All-Reduce + Heavy DMA (8-flit weight loads)",
         gen_fc_layer_reduce_with_dma()),
        ("Scatter-Reduce (800 pkts)",
         gen_scatter_reduce(800)),
        ("Saturated Reduce Hotspot (root writebacks)",
         gen_saturated_reduce_hotspot()),
        ("nuScenes mini BEV Fusion (6 cameras, 192 cells)",
         gen_nuscenes_bev_reduce()),
        ("nuScenes mini BEV Fusion + Dense DMA",
         gen_nuscenes_bev_reduce_with_dma()),
        ("Trace-Driven BEV Reduce Frame",
            gen_trace_driven_nuscenes_bev_reduce(trace_path=trace_reduce_path)),
        ("Trace-Driven BEV Reduce + Dense DMA",
            gen_trace_driven_nuscenes_bev_reduce(trace_path=trace_reduce_path, dense_bg=400)),
    ]

    print(f"\n\n{'='*110}")
    print("  IN-NETWORK REDUCTION BENCHMARK")
    print("  Comparing root-visible traffic + link load: Baseline vs Sparse-Aware vs Sparse-Aware + router INR")
    print(f"  RTL alignment: noc_innet_reduce.sv aggregates in intermediate routers; root-side reduce consumption is not live in accel_tile")
    print(f"{'='*110}")

    all_inr_results = {}

    for scenario_name, pkts in scenarios:
        n_orig = len(pkts)
        n_reduce_orig = sum(1 for p in pkts if p.reduce_group is not None)
        flit_hops_orig = calc_total_flit_hops(pkts)

        # Apply router-side in-network reduction transformation.
        # pkts_inr_forward models the packets still visible after aggregation.
        # pkts_inr_tree models all internal tree edges for flit-hop accounting.
        pkts_inr_forward, pkts_inr_tree = apply_router_innet_reduction(pkts)
        n_inr = len(pkts_inr_forward)
        n_reduce_inr = sum(1 for p in pkts_inr_forward if p.reduce_group is not None)
        flit_hops_inr = calc_total_flit_hops(pkts_inr_tree)

        pkt_reduction = (n_orig - n_inr) / n_orig * 100 if n_orig > 0 else 0
        reduce_pkt_red = (n_reduce_orig - n_reduce_inr) / max(n_reduce_orig, 1) * 100
        flit_hop_red = (flit_hops_orig - flit_hops_inr) / max(flit_hops_orig, 1) * 100

        # Run latency simulations
        sim_base = NoCSimulator(BaselineAllocator)
        m_base = sim_base.simulate([clone_packet(p) for p in pkts])

        sim_sparse = NoCSimulator(SparseAwareAllocator)
        m_sparse = sim_sparse.simulate([clone_packet(p) for p in pkts])

        sim_inr = NoCSimulator(SparseAwareAllocator)
        m_inr = sim_inr.simulate([clone_packet(p) for p in pkts_inr_forward])

        # Metrics
        base_lat  = float(np.mean(m_base.latencies))  if m_base.latencies  else 0
        sparse_lat = float(np.mean(m_sparse.latencies)) if m_sparse.latencies else 0
        inr_lat   = float(np.mean(m_inr.latencies))   if m_inr.latencies   else 0

        base_sp  = float(np.mean(m_base.sparse_latencies))   if m_base.sparse_latencies   else 0
        sparse_sp = float(np.mean(m_sparse.sparse_latencies)) if m_sparse.sparse_latencies else 0
        inr_sp   = float(np.mean(m_inr.sparse_latencies))    if m_inr.sparse_latencies    else 0

        lat_imp_sparse = (base_lat - sparse_lat) / max(base_lat, 1) * 100
        lat_imp_inr    = (base_lat - inr_lat)    / max(base_lat, 1) * 100
        sp_imp_sparse  = (base_sp  - sparse_sp)  / max(base_sp,  1) * 100
        sp_imp_inr     = (base_sp  - inr_sp)     / max(base_sp,  1) * 100

        print(f"\n  Scenario: {scenario_name}")
        print(f"  {'-'*108}")
        print(f"  {'Metric':<40} {'Baseline RR':>20} {'Sparse-Aware':>20} {'Sparse + INR (router)':>22}")
        print(f"  {'-'*40} {'-'*20} {'-'*20} {'-'*22}")
        print(f"  {'Modeled packets after router INR':<40} {n_orig:>20d} {n_orig:>20d} {n_inr:>22d}")
        print(f"  {'  of which: root-visible reduce packets':<40} {n_reduce_orig:>20d} {n_reduce_orig:>20d} {n_reduce_inr:>22d}")
        print(f"  {'  reduce pkt elimination before root':<40} {'':>20} {'':>20} {f'{reduce_pkt_red:+.1f}%':>22}")
        print(f"  {'Total flit-hops (all links)':<40} {flit_hops_orig:>20d} {flit_hops_orig:>20d} {flit_hops_inr:>22d}")
        print(f"  {'  flit-hop reduction':<40} {'':>20} {'':>20} {f'{flit_hop_red:+.1f}%':>22}")
        sparse_imp_str = f"({lat_imp_sparse:+.1f}%)"
        inr_imp_str    = f"({lat_imp_inr:+.1f}%)"
        ssp_imp_str    = f"({sp_imp_sparse:+.1f}%)"
        sinr_imp_str   = f"({sp_imp_inr:+.1f}%)"

        print(f"  {'Avg latency (modeled packets)':<40} {base_lat:>20.1f} {sparse_lat:>14.1f} {sparse_imp_str:>8}  {inr_lat:>14.1f} {inr_imp_str:>8}")
        print(f"  {'Avg sparse latency (modeled)':<40} {base_sp:>20.1f} {sparse_sp:>14.1f} {ssp_imp_str:>8}  {inr_sp:>14.1f} {sinr_imp_str:>8}")
        print(f"  {'Blocked cycles':<40} {m_base.blocked_cycles:>20d} {m_sparse.blocked_cycles:>20d} {m_inr.blocked_cycles:>22d}")

        all_inr_results[scenario_name] = {
            "baseline":  format_metrics("Baseline RR",   m_base),
            "sparse":    format_metrics("Sparse-Aware",  m_sparse),
            "inr":       format_metrics("Sparse+INR (router)", m_inr),
            "pkt_reduction_pct":    pkt_reduction,
            "reduce_pkt_red_pct":   reduce_pkt_red,
            "flit_hop_reduction_pct": flit_hop_red,
            "n_orig": n_orig,
            "n_inr": n_inr,
            "flit_hops_orig": flit_hops_orig,
            "flit_hops_inr": flit_hops_inr,
        }

    print(f"\n\n  {'='*108}")
    print("  SUMMARY: Router-Side In-Network Reduction Impact")
    print(f"  {'='*108}")
    print(f"  {'Scenario':<44} {'Reduce Pkts Elim':>18} {'Flit-Hop Reduc':>16} {'Sparse Lat (INR)':>18} {'vs Baseline':>12}")
    print(f"  {'-'*44} {'-'*18} {'-'*16} {'-'*18} {'-'*12}")
    for sname, r in all_inr_results.items():
        base_sp = r["baseline"]["sparse_avg"]
        inr_sp  = r["inr"]["sparse_avg"]
        imp = (base_sp - inr_sp) / max(base_sp, 1) * 100
        short = sname[:42]
        print(f"  {short:<44} {r['reduce_pkt_red_pct']:>+17.1f}% {r['flit_hop_reduction_pct']:>+15.1f}%"
              f" {inr_sp:>18.1f} {imp:>+11.1f}%")

    print(f"\n  Key: This benchmark now models router-side INR only.")
    print(f"       Genuine reduce groups are collapsed inside the mesh, while flit-hop")
    print(f"       accounting still includes all internal tree edges. Root-side final")
    print(f"       consumption remains a separate integration step outside accel_tile.")
    print(f"{'='*110}")

    return all_inr_results


# =============================================================================
# Main
# =============================================================================
def main():
    total_patterns = 16
    bsr_dir = os.path.join(PROJECT_ROOT, "data", "bsr_export")
    frame_trace_path = os.environ.get(FRAME_TRACE_ENV, NUSCENES_TRACE_PATH)
    reduce_trace_path = os.environ.get(REDUCE_TRACE_ENV, NUSCENES_REDUCE_TRACE_PATH)

    print("=" * 96)
    print("  NoC VC Allocator FULL Comparison Benchmark")
    print(f"  5 Allocators × {total_patterns} Traffic Patterns")
    print("  4×4 Mesh, 4 VCs/port, XY Routing")
    print(f"  Trace-driven frame: {frame_trace_path}")
    print(f"  Trace-driven reduce: {reduce_trace_path}")
    print("=" * 96)

    all_results = {}

    # =========================================================================
    # Synthetic Traffic Patterns
    # =========================================================================
    tests = [
        ("1. Uniform Random (50% sparse)",
         lambda: gen_uniform(1000, 0.5)),
        ("2. Hotspot (node 0, 30% sparse)",
         lambda: gen_hotspot(1000, hotspot=0, frac=0.6, sparse_frac=0.3)),
        ("3. Scatter-Reduce (800 pkts)",
         lambda: gen_scatter_reduce(800)),
        ("4. Saturated Mixed (80% dense, long pkts)",
         lambda: gen_saturated_mixed(3000, 0.2)),
        ("5. Bottleneck (diagonal dense + adj sparse)",
         lambda: gen_bottleneck(500, 300)),
    ]

    for i, (name, gen_fn) in enumerate(tests, 1):
        print(f"\n[{i}/{total_patterns}] Running {name}...")
        pkts = gen_fn()
        print(f"         Generated {len(pkts)} packets")
        results = run_all_allocators(name, pkts)
        print_full_comparison(name, results)
        all_results[name] = results

    # =========================================================================
    # MNIST BSR Real Data
    # =========================================================================
    bsr_tests = [
        ("6. REAL BSR: MNIST conv2 (70% sparse)", "conv2", False),
        ("7. REAL BSR: MNIST fc1 (91% sparse)", "fc1", False),
        ("8. REAL BSR: MNIST fc1 + Dense DMA BG", "fc1", True),
    ]

    for name, layer, with_bg in bsr_tests:
        bsr_path = os.path.join(bsr_dir, layer, "row_ptr.npy")
        if os.path.exists(bsr_path):
            idx = int(name[0])
            print(f"\n[{idx}/{total_patterns}] Running {name}...")
            if with_bg:
                pkts = gen_bsr_dense_bg(bsr_dir, layer, dense_bg=500)
            else:
                pkts = gen_bsr_traffic(bsr_dir, layer)
            print(f"         Generated {len(pkts)} packets")
            results = run_all_allocators(name, pkts)
            print_full_comparison(name, results)
            all_results[name] = results
        else:
            print(f"\n[SKIP] {name} — BSR data not found at {bsr_path}")

    # =========================================================================
    # CIFAR-10 Synthetic Sparse Workload
    # =========================================================================
    print(f"\n[9/{total_patterns}] Running 9. CIFAR-10 Sparse ResNet (5 layers)...")
    pkts = gen_cifar10_traffic(2000)
    print(f"         Generated {len(pkts)} packets")
    results = run_all_allocators("9. CIFAR-10 Sparse ResNet", pkts)
    print_full_comparison("9. CIFAR-10 Sparse ResNet", results)
    all_results["9. CIFAR-10 Sparse ResNet"] = results

    # =========================================================================
    # Transformer Synthetic Sparse Workload
    # =========================================================================
    print(f"\n[10/{total_patterns}] Running 10. Transformer (2-block ViT-tiny)...")
    pkts = gen_transformer_traffic(2500)
    print(f"         Generated {len(pkts)} packets")
    results = run_all_allocators("10. Transformer ViT-tiny", pkts)
    print_full_comparison("10. Transformer ViT-tiny", results)
    all_results["10. Transformer ViT-tiny"] = results

    # =========================================================================
    # Stress Tests: CIFAR-10 + Dense DMA, Transformer + Dense DMA
    # =========================================================================
    print(f"\n[11/{total_patterns}] Running 11. CIFAR-10 + Heavy Dense Background...")
    cifar_pkts = gen_cifar10_traffic(2000, seed=99)
    rng = np.random.RandomState(99)
    # Add 500 heavy dense DMA packets
    for i in range(500):
        src, dst = 0, 1 + (i % 15)
        cifar_pkts.append(Packet(src=src, dst=dst, is_sparse=False, num_flits=8))
    rng.shuffle(cifar_pkts)
    print(f"         Generated {len(cifar_pkts)} packets")
    results = run_all_allocators("11. CIFAR-10 + Dense DMA", cifar_pkts)
    print_full_comparison("11. CIFAR-10 + Dense DMA", results)
    all_results["11. CIFAR-10 + Dense DMA"] = results

    print(f"\n[12/{total_patterns}] Running 12. Transformer + Heavy Dense Background...")
    xfmr_pkts = gen_transformer_traffic(2500, seed=99)
    for i in range(500):
        src, dst = 0, 1 + (i % 15)
        xfmr_pkts.append(Packet(src=src, dst=dst, is_sparse=False, num_flits=8))
    rng.shuffle(xfmr_pkts)
    print(f"         Generated {len(xfmr_pkts)} packets")
    results = run_all_allocators("12. Transformer + Dense DMA", xfmr_pkts)
    print_full_comparison("12. Transformer + Dense DMA", results)
    all_results["12. Transformer + Dense DMA"] = results

    # =========================================================================
    # Autonomous Driving: nuScenes mini Multi-Camera BEV Fusion
    # =========================================================================
    print(f"\n[13/{total_patterns}] Running 13. nuScenes mini BEV Fusion...")
    bev_pkts = gen_nuscenes_bev_fusion_traffic(seed=123)
    print(f"         Generated {len(bev_pkts)} packets")
    results = run_all_allocators("13. nuScenes mini BEV Fusion", bev_pkts)
    print_full_comparison("13. nuScenes mini BEV Fusion", results)
    all_results["13. nuScenes mini BEV Fusion"] = results

    print(f"\n[14/{total_patterns}] Running 14. nuScenes mini BEV Fusion + Dense DMA...")
    bev_bg_pkts = gen_nuscenes_bev_fusion_traffic(seed=321, dense_bg=600)
    print(f"         Generated {len(bev_bg_pkts)} packets")
    results = run_all_allocators("14. nuScenes mini BEV Fusion + Dense DMA", bev_bg_pkts)
    print_full_comparison("14. nuScenes mini BEV Fusion + Dense DMA", results)
    all_results["14. nuScenes mini BEV Fusion + Dense DMA"] = results

    # =========================================================================
    # Trace-Driven Autonomous Driving: Frame-Level Replay
    # =========================================================================
    print(f"\n[15/{total_patterns}] Running 15. Trace-Driven BEV Frame...")
    trace_bev_pkts = gen_trace_driven_nuscenes_bev_traffic(trace_path=frame_trace_path)
    print(f"         Generated {len(trace_bev_pkts)} packets")
    results = run_all_allocators("15. Trace-Driven BEV Frame", trace_bev_pkts)
    print_full_comparison("15. Trace-Driven BEV Frame", results)
    all_results["15. Trace-Driven BEV Frame"] = results

    print(f"\n[16/{total_patterns}] Running 16. Trace-Driven BEV Frame + Dense DMA...")
    trace_bev_bg_pkts = gen_trace_driven_nuscenes_bev_traffic(trace_path=frame_trace_path, dense_bg=600)
    print(f"         Generated {len(trace_bev_bg_pkts)} packets")
    results = run_all_allocators("16. Trace-Driven BEV Frame + Dense DMA", trace_bev_bg_pkts)
    print_full_comparison("16. Trace-Driven BEV Frame + Dense DMA", results)
    all_results["16. Trace-Driven BEV Frame + Dense DMA"] = results

    # =========================================================================
    # GRAND SUMMARY TABLE
    # =========================================================================
    print(f"\n\n{'='*110}")
    print("  GRAND SUMMARY: Sparse Latency Improvement vs Baseline (positive = better)")
    print(f"{'='*110}")
    hdr = f"  {'Pattern':<40} {'StaticPrio':>11} {'WtdRR 3:1':>11} {'QVN 2+2':>11} {'Ours':>11} {'Ours TP':>11}"
    print(hdr)
    print(f"  {'-'*40} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for pname, res in all_results.items():
        base_sp = res["Baseline RR"]["sparse_avg"]
        base_tp = res["Baseline RR"]["throughput"]
        if base_sp <= 0:
            continue

        sp_static = res["Static Priority"]["sparse_avg"]
        sp_wrr = res["Weighted RR (3:1)"]["sparse_avg"]
        sp_qvn = res["QVN (2+2 split)"]["sparse_avg"]
        sp_ours = res["Sparsity-Aware (Ours)"]["sparse_avg"]
        tp_ours = res["Sparsity-Aware (Ours)"]["throughput"]

        imp_static = (base_sp - sp_static) / base_sp * 100
        imp_wrr = (base_sp - sp_wrr) / base_sp * 100
        imp_qvn = (base_sp - sp_qvn) / base_sp * 100
        imp_ours = (base_sp - sp_ours) / base_sp * 100
        tp_cost = (tp_ours - base_tp) / base_tp * 100 if base_tp > 0 else 0

        short_name = pname[:38]
        print(f"  {short_name:<40} {imp_static:>+10.1f}% {imp_wrr:>+10.1f}% {imp_qvn:>+10.1f}% {imp_ours:>+10.1f}% {tp_cost:>+10.1f}%")

    print(f"{'='*110}")
    print()
    print("  Legend:")
    print("    StaticPrio  = Sparse always wins, dense starves")
    print("    WtdRR 3:1   = 3 sparse grants per 1 dense grant")
    print("    QVN 2+2     = ARM-style: VCs 0,1=dense ONLY, VCs 2,3=sparse ONLY")
    print("    Ours        = Adaptive VC reservation; dense avoids VC-3 only under sparse-heavy load")
    print("    Ours TP     = Throughput cost of our approach vs baseline")
    print()
    print("  Key insight: Our approach adapts between baseline-like sharing and sparse-")
    print("  aware reservation. It avoids QVN rigidity while reducing sparse latency in")
    print("  contention-heavy cases where sparse traffic dominates recent requests.")
    print()

    # Save JSON results
    output_path = os.path.join(PROJECT_ROOT, "tools", "allocator_comparison_results.json")
    json_results = {}
    for pname, res in all_results.items():
        json_results[pname] = {k: {mk: float(mv) if isinstance(mv, (int, float, np.floating, np.integer)) else mv
                                    for mk, mv in v.items()}
                                for k, v in res.items()}
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Results saved to: {output_path}")

    # =========================================================================
    # In-Network Reduction Benchmark
    # =========================================================================
    inr_results = run_innet_reduce_benchmark(trace_reduce_path=reduce_trace_path)

    inr_path = os.path.join(PROJECT_ROOT, "tools", "innet_reduce_results.json")
    with open(inr_path, 'w') as f:
        safe = {}
        for k, v in inr_results.items():
            safe[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    safe[k][k2] = {mk: float(mv) if isinstance(mv, (int, float, np.floating, np.integer)) else mv
                                   for mk, mv in v2.items()}
                else:
                    safe[k][k2] = float(v2) if isinstance(v2, (int, float, np.floating, np.integer)) else v2
        json.dump(safe, f, indent=2)
    print(f"\n  In-Network Reduction results saved to: {inr_path}")


if __name__ == "__main__":
    main()
