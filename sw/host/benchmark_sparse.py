#!/usr/bin/env python3
"""
benchmark_sparse.py — Sparse Matrix Multiply Benchmarks for ACCEL-v1
=====================================================================

Benchmarks the sparse accelerator with various:
  - Matrix sizes
  - Sparsity levels
  - Block patterns

Compares against NumPy CPU baseline.

Author: ACCEL-v1 Team
Date: December 2024
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import argparse

# Import our modules
from accel import AccelDriver
from memory import BSRMatrix


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str, M: int, N: int, K: int, sparsity: float):
        self.name = name
        self.M = M
        self.N = N
        self.K = K
        self.sparsity = sparsity
        
        # Timing
        self.accel_time_ms = 0.0
        self.cpu_time_ms = 0.0
        self.speedup = 0.0
        
        # Hardware metrics
        self.total_cycles = 0
        self.active_cycles = 0
        self.utilization = 0.0
        
        # Throughput
        self.gops = 0.0  # Effective GOPS
        self.gops_peak = 0.0  # Peak GOPS
        
    def compute_metrics(self, clock_freq_mhz: float = 100.0):
        """Compute derived metrics."""
        # Effective operations (accounting for sparsity)
        dense_ops = 2 * self.M * self.N * self.K  # MACs
        effective_ops = dense_ops * (1 - self.sparsity)
        
        # GOPS
        if self.accel_time_ms > 0:
            self.gops = effective_ops / (self.accel_time_ms * 1e6)
        
        # Peak GOPS (16x16 array @ 100MHz = 25.6 GOPS)
        self.gops_peak = 16 * 16 * 2 * clock_freq_mhz / 1000.0
        
        # Speedup
        if self.accel_time_ms > 0:
            self.speedup = self.cpu_time_ms / self.accel_time_ms
            
    def __repr__(self):
        return (f"Benchmark({self.name}): {self.M}x{self.K} @ {self.K}x{self.N}, "
                f"sparsity={self.sparsity:.0%}, speedup={self.speedup:.1f}x, "
                f"util={self.utilization:.1f}%")


def generate_sparse_weights(K: int, N: int, sparsity: float, 
                           block_size: int = 16) -> Tuple[BSRMatrix, np.ndarray]:
    """
    Generate sparse weight matrix with given sparsity.
    
    Args:
        K, N: Matrix dimensions
        sparsity: Target sparsity (0.0 to 1.0)
        block_size: BSR block size
        
    Returns:
        (BSRMatrix, dense_for_reference)
    """
    # Generate dense matrix
    dense = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    
    # Apply block sparsity
    for i in range(0, K, block_size):
        for j in range(0, N, block_size):
            if np.random.random() < sparsity:
                dense[i:i+block_size, j:j+block_size] = 0
    
    # Convert to BSR
    bsr = BSRMatrix.from_dense(dense, block_size)
    
    return bsr, dense


def run_cpu_baseline(A: np.ndarray, B: np.ndarray, num_runs: int = 5) -> float:
    """
    Run CPU matrix multiply baseline.
    
    Args:
        A: Activation matrix (M, K)
        B: Weight matrix (K, N)
        num_runs: Number of runs for averaging
        
    Returns:
        Average time in milliseconds
    """
    # Warmup
    _ = A.astype(np.int32) @ B.astype(np.int32)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        C = A.astype(np.int32) @ B.astype(np.int32)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    return np.mean(times)


def run_benchmark(accel: AccelDriver, M: int, N: int, K: int, 
                  sparsity: float, num_runs: int = 3) -> BenchmarkResult:
    """
    Run a single benchmark configuration.
    
    Args:
        accel: AccelDriver instance
        M, N, K: Matrix dimensions
        sparsity: Target sparsity
        num_runs: Number of runs for averaging
        
    Returns:
        BenchmarkResult
    """
    result = BenchmarkResult(f"spmm_{M}x{N}x{K}_s{int(sparsity*100)}", 
                             M, N, K, sparsity)
    
    # Generate data
    bsr, dense_weights = generate_sparse_weights(K, N, sparsity)
    activations = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    
    # Update actual sparsity
    result.sparsity = bsr.sparsity()
    
    # CPU baseline
    result.cpu_time_ms = run_cpu_baseline(activations, dense_weights, num_runs)
    
    # Configure accelerator
    accel.reset()
    accel.configure_dimensions(M, N, K)
    
    # Load data
    accel.load_sparse_weights(bsr.row_ptr, bsr.col_idx, bsr.values)
    accel.load_activations(activations)
    
    # Run accelerator (multiple times)
    accel_times = []
    for _ in range(num_runs):
        accel.reset()
        accel.configure_dimensions(M, N, K)
        accel.load_sparse_weights(bsr.row_ptr, bsr.col_idx, bsr.values)
        accel.load_activations(activations)
        
        start = time.perf_counter()
        success, metrics = accel.run_inference()
        end = time.perf_counter()
        
        if success:
            accel_times.append((end - start) * 1000)
            result.total_cycles = metrics['cycles']
            result.active_cycles = metrics['active_cycles']
            result.utilization = metrics['utilization']
    
    if accel_times:
        result.accel_time_ms = np.mean(accel_times)
    
    result.compute_metrics()
    
    return result


def run_benchmark_suite(simulation: bool = True) -> List[BenchmarkResult]:
    """
    Run full benchmark suite.
    
    Args:
        simulation: Use simulation mode
        
    Returns:
        List of BenchmarkResult
    """
    results = []
    
    # Create accelerator
    accel = AccelDriver(simulation=simulation)
    
    # Test configurations
    configs = [
        # (M, N, K, sparsity)
        (16, 16, 16, 0.0),    # Dense baseline
        (16, 16, 16, 0.5),    # 50% sparse
        (16, 16, 16, 0.75),   # 75% sparse
        (16, 16, 16, 0.9),    # 90% sparse
        
        (32, 32, 32, 0.5),    # Larger matrix
        (64, 64, 64, 0.5),
        (128, 128, 128, 0.5),
        
        (64, 64, 64, 0.0),    # Dense larger
        (64, 64, 64, 0.9),    # Highly sparse
    ]
    
    print("\nRunning benchmarks...")
    print("-" * 80)
    
    for M, N, K, sparsity in configs:
        try:
            result = run_benchmark(accel, M, N, K, sparsity)
            results.append(result)
            
            print(f"  {result.name:30s} | CPU: {result.cpu_time_ms:8.3f}ms | "
                  f"Accel: {result.accel_time_ms:8.3f}ms | "
                  f"Speedup: {result.speedup:6.1f}x | "
                  f"Util: {result.utilization:5.1f}%")
                  
        except Exception as e:
            print(f"  {M}x{N}x{K} @ {sparsity:.0%}: FAILED - {e}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} | {'Size':>12} | {'Sparsity':>8} | "
          f"{'CPU (ms)':>10} | {'Accel (ms)':>10} | {'Speedup':>8} | {'Util':>6}")
    print("-" * 80)
    
    for r in results:
        size_str = f"{r.M}x{r.K}x{r.N}"
        print(f"{r.name:<25} | {size_str:>12} | {r.sparsity:>7.0%} | "
              f"{r.cpu_time_ms:>10.3f} | {r.accel_time_ms:>10.3f} | "
              f"{r.speedup:>7.1f}x | {r.utilization:>5.1f}%")
    
    print("-" * 80)
    
    # Averages
    avg_speedup = np.mean([r.speedup for r in results if r.speedup > 0])
    avg_util = np.mean([r.utilization for r in results])
    print(f"{'AVERAGE':<25} | {'':<12} | {'':<8} | "
          f"{'':<10} | {'':<10} | {avg_speedup:>7.1f}x | {avg_util:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="ACCEL-v1 Sparse Benchmarks")
    parser.add_argument("--real", action="store_true", 
                        help="Run on real hardware (default: simulation)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset of benchmarks")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ACCEL-v1 Sparse Matrix Multiply Benchmarks")
    print("=" * 80)
    print(f"Mode: {'Real Hardware' if args.real else 'Simulation'}")
    print(f"Array: 16x16 INT8 Systolic")
    print(f"Expected Peak: 25.6 GOPS @ 100MHz")
    
    results = run_benchmark_suite(simulation=not args.real)
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    exit(main())
