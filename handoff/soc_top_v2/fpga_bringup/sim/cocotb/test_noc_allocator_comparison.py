"""
NoC VC Allocator Comparison Testbench
=====================================
Runs identical traffic patterns through both the baseline round-robin
VC allocator and the sparsity-aware VC allocator, then compares:
  - Average packet latency
  - Throughput (flits/cycle)
  - VC utilization
  - Fairness (Jain's index)

This is the key benchmark for the novel contribution.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random
import math

NUM_VCS = 4
NUM_PORTS = 5


# ===========================================================================
# Traffic pattern generators
# ===========================================================================

def uniform_random_traffic(num_nodes, num_packets, seed=42):
    """Each node sends to a uniformly random destination."""
    rng = random.Random(seed)
    packets = []
    for _ in range(num_packets):
        src = rng.randint(0, num_nodes - 1)
        dst = rng.randint(0, num_nodes - 1)
        while dst == src:
            dst = rng.randint(0, num_nodes - 1)
        sparse = rng.random() < 0.5  # 50% sparse
        packets.append((src, dst, sparse))
    return packets


def hotspot_traffic(num_nodes, num_packets, hotspot_node=0, hotspot_frac=0.5, seed=42):
    """hotspot_frac of traffic goes to/from hotspot_node."""
    rng = random.Random(seed)
    packets = []
    for _ in range(num_packets):
        if rng.random() < hotspot_frac:
            src = rng.randint(0, num_nodes - 1)
            dst = hotspot_node
            if src == dst:
                src = (src + 1) % num_nodes
        else:
            src = rng.randint(0, num_nodes - 1)
            dst = rng.randint(0, num_nodes - 1)
            while dst == src:
                dst = rng.randint(0, num_nodes - 1)
        sparse = rng.random() < 0.3  # 30% sparse in hotspot
        packets.append((src, dst, sparse))
    return packets


def sparse_burst_traffic(num_nodes, num_packets, burst_len=8, seed=42):
    """Bursts of sparse data (like BSR tiles) between random nodes."""
    rng = random.Random(seed)
    packets = []
    i = 0
    while i < num_packets:
        src = rng.randint(0, num_nodes - 1)
        dst = rng.randint(0, num_nodes - 1)
        while dst == src:
            dst = rng.randint(0, num_nodes - 1)
        blen = min(burst_len, num_packets - i)
        for _ in range(blen):
            packets.append((src, dst, True))  # All sparse
            i += 1
    return packets


# ===========================================================================
# Metric computation
# ===========================================================================

def compute_metrics(latencies, total_cycles, num_flits):
    """Compute summary stats from latency list."""
    if not latencies:
        return {"avg_lat": 0, "max_lat": 0, "throughput": 0, "jain": 0}

    avg = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    throughput = num_flits / max(total_cycles, 1)

    # Jain's fairness index over per-source latencies
    if len(latencies) > 1:
        n = len(latencies)
        sum_x = sum(latencies)
        sum_x2 = sum(x * x for x in latencies)
        jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
    else:
        jain = 1.0

    return {
        "avg_lat": avg,
        "max_lat": max_lat,
        "throughput": throughput,
        "jain": jain,
    }


def print_comparison(name, baseline, sparse):
    """Log a comparison table."""
    lines = [
        f"\n{'='*60}",
        f"  Traffic Pattern: {name}",
        f"{'='*60}",
        f"  {'Metric':<25} {'Baseline':>12} {'Sparse-VC':>12} {'Delta':>10}",
        f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}",
    ]
    for key in ["avg_lat", "max_lat", "throughput", "jain"]:
        b = baseline[key]
        s = sparse[key]
        if b > 0:
            delta = ((s - b) / b) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"
        lines.append(f"  {key:<25} {b:>12.2f} {s:>12.2f} {delta_str:>10}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ===========================================================================
# Simulation driver (software model for comparison)
# ===========================================================================

class SimpleAllocator:
    """Software model of round-robin VC allocator."""

    def __init__(self, num_vcs):
        self.num_vcs = num_vcs
        self.priority = [0] * NUM_PORTS
        self.vc_busy = [[False] * num_vcs for _ in range(NUM_PORTS)]

    def allocate(self, requests):
        """requests: list of (input_port, output_port, vc_requested).
        Returns list of (input_port, allocated_vc) or None if blocked."""
        grants = []
        for inp, outp, req_vc in requests:
            allocated = None
            # Round-robin from priority
            for offset in range(self.num_vcs):
                vc = (self.priority[outp] + offset) % self.num_vcs
                if not self.vc_busy[outp][vc]:
                    self.vc_busy[outp][vc] = True
                    self.priority[outp] = (vc + 1) % self.num_vcs
                    allocated = vc
                    break
            grants.append((inp, allocated))
        return grants

    def free(self, output_port, vc):
        self.vc_busy[output_port][vc] = False


class SparseAwareAllocator(SimpleAllocator):
    """Software model of sparsity-aware VC allocator.

    Reserves VC 0 for sparse (high-priority, short) packets.
    Dense packets use VCs 1..N-1. Sparse packets can also overflow
    to non-reserved VCs if VC 0 is busy.
    """

    SPARSE_VC = 0

    def allocate(self, requests):
        grants = []
        for inp, outp, is_sparse in requests:
            allocated = None

            if is_sparse:
                # Try reserved VC first
                if not self.vc_busy[outp][self.SPARSE_VC]:
                    self.vc_busy[outp][self.SPARSE_VC] = True
                    allocated = self.SPARSE_VC
                else:
                    # Overflow to any free VC
                    for vc in range(self.num_vcs):
                        if not self.vc_busy[outp][vc]:
                            self.vc_busy[outp][vc] = True
                            allocated = vc
                            break
            else:
                # Dense: use VCs 1..N-1 with round-robin
                start = max(1, self.priority[outp])
                for offset in range(self.num_vcs - 1):
                    vc = 1 + (start - 1 + offset) % (self.num_vcs - 1)
                    if not self.vc_busy[outp][vc]:
                        self.vc_busy[outp][vc] = True
                        self.priority[outp] = vc + 1
                        allocated = vc
                        break

            grants.append((inp, allocated))
        return grants


def simulate_traffic(allocator, packets, num_nodes):
    """Run traffic through a software allocator model.

    Simple cycle-level simulation:
    - Each packet takes hop_count cycles to traverse
    - Blocked packets retry next cycle
    """
    cycle = 0
    latencies = []
    pending = list(packets)
    in_flight = []  # (release_cycle, output_port, vc, inject_cycle)
    total_flits = 0

    while pending or in_flight:
        cycle += 1
        if cycle > 10000:
            break  # Safety

        # Release completed packets
        completed = [p for p in in_flight if p[0] <= cycle]
        for _, outp, vc, inject_cyc in completed:
            allocator.free(outp, vc)
            latencies.append(cycle - inject_cyc)
            total_flits += 1
        in_flight = [p for p in in_flight if p[0] > cycle]

        # Inject new packets (up to num_nodes per cycle)
        inject_count = 0
        still_pending = []
        for pkt in pending:
            if inject_count >= num_nodes:
                still_pending.append(pkt)
                continue

            src, dst, sparse = pkt
            # Compute XY output port from src
            src_row, src_col = divmod(src, int(math.sqrt(num_nodes)))
            dst_row, dst_col = divmod(dst, int(math.sqrt(num_nodes)))

            if dst_col > src_col:
                outp = 3  # East
            elif dst_col < src_col:
                outp = 4  # West
            elif dst_row > src_row:
                outp = 2  # South
            elif dst_row < src_row:
                outp = 1  # North
            else:
                outp = 0  # Local

            hops = abs(dst_row - src_row) + abs(dst_col - src_col)

            grants = allocator.allocate([(src, outp, sparse)])
            if grants[0][1] is not None:
                vc = grants[0][1]
                release_cycle = cycle + max(hops, 1)
                in_flight.append((release_cycle, outp, vc, cycle))
                inject_count += 1
            else:
                still_pending.append(pkt)

        pending = still_pending

    return compute_metrics(latencies, cycle, total_flits)


# ===========================================================================
# cocotb tests
# ===========================================================================

@cocotb.test()
async def test_uniform_random_comparison(dut):
    """Compare allocators under uniform random traffic."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await ClockCycles(dut.clk, 5)

    num_nodes = 16  # 4x4
    packets = uniform_random_traffic(num_nodes, 200)

    baseline_alloc = SimpleAllocator(NUM_VCS)
    sparse_alloc = SparseAwareAllocator(NUM_VCS)

    baseline = simulate_traffic(baseline_alloc, packets, num_nodes)
    sparse = simulate_traffic(sparse_alloc, packets, num_nodes)

    report = print_comparison("Uniform Random", baseline, sparse)
    dut._log.info(report)

    dut._log.info("PASS: Uniform random comparison complete")


@cocotb.test()
async def test_hotspot_comparison(dut):
    """Compare allocators under hotspot traffic."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await ClockCycles(dut.clk, 5)

    num_nodes = 16
    packets = hotspot_traffic(num_nodes, 200)

    baseline_alloc = SimpleAllocator(NUM_VCS)
    sparse_alloc = SparseAwareAllocator(NUM_VCS)

    baseline = simulate_traffic(baseline_alloc, packets, num_nodes)
    sparse = simulate_traffic(sparse_alloc, packets, num_nodes)

    report = print_comparison("Hotspot (node 0, 50%)", baseline, sparse)
    dut._log.info(report)

    dut._log.info("PASS: Hotspot comparison complete")


@cocotb.test()
async def test_sparse_burst_comparison(dut):
    """Compare allocators under bursty sparse traffic (BSR-like)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await ClockCycles(dut.clk, 5)

    num_nodes = 16
    packets = sparse_burst_traffic(num_nodes, 200)

    baseline_alloc = SimpleAllocator(NUM_VCS)
    sparse_alloc = SparseAwareAllocator(NUM_VCS)

    baseline = simulate_traffic(baseline_alloc, packets, num_nodes)
    sparse = simulate_traffic(sparse_alloc, packets, num_nodes)

    report = print_comparison("Sparse Burst (BSR-like)", baseline, sparse)
    dut._log.info(report)

    # Sparse-aware should show lower average latency for sparse bursts
    if sparse["avg_lat"] < baseline["avg_lat"]:
        dut._log.info("RESULT: Sparse-aware allocator reduces latency for sparse traffic")
    else:
        dut._log.info("RESULT: No latency improvement (may need parameter tuning)")

    dut._log.info("PASS: Sparse burst comparison complete")


@cocotb.test()
async def test_mixed_traffic_comparison(dut):
    """Compare allocators under mixed sparse+dense traffic."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await ClockCycles(dut.clk, 5)

    num_nodes = 16
    rng = random.Random(99)

    # 60% sparse, 40% dense — realistic for sparse CNN inference
    packets = []
    for _ in range(300):
        src = rng.randint(0, num_nodes - 1)
        dst = rng.randint(0, num_nodes - 1)
        while dst == src:
            dst = rng.randint(0, num_nodes - 1)
        sparse = rng.random() < 0.6
        packets.append((src, dst, sparse))

    baseline_alloc = SimpleAllocator(NUM_VCS)
    sparse_alloc = SparseAwareAllocator(NUM_VCS)

    baseline = simulate_traffic(baseline_alloc, packets, num_nodes)
    sparse = simulate_traffic(sparse_alloc, packets, num_nodes)

    report = print_comparison("Mixed (60% sparse, 40% dense)", baseline, sparse)
    dut._log.info(report)

    dut._log.info("PASS: Mixed traffic comparison complete")
