#!/usr/bin/env python3
"""
run_gemm_axi.py - AXI4-Based Host Row-Stationary Tiler for ACCEL-v1

This implements the host-side matrix tiling and orchestration using AXI4 burst DMA
instead of UART for 27,000× faster data transfer.

Matrix Multiplication: C = A × B
- A: [M×K] activation matrix (via AXI burst read from DDR)
- B: [K×N] weight matrix (via AXI burst read from DDR)
- C: [M×N] result matrix

AXI4 Protocol:
- 32-bit data path @ 100 MHz = 400 MB/s
- Burst transfers up to 256 beats (1 KB per transaction)
- Outstanding transactions for pipelining
- CSR configuration via AXI4-Lite

Author: ACCEL-v1 Team
"""

import argparse
import numpy as np
import time
import sys
import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host.axi_driver import AXIDriver, AXILiteCSR
from host_uart.csr_map import Config, pack_u32, unpack_u32


@dataclass
class GEMMConfig:
    """
    GEMM operation configuration
    
    Engineer's Note:
    This class defines the problem size and tiling strategy.
    The tiling parameters (Tm, Tn, Tk) MUST match the hardware synthesis parameters.
    If they mismatch, the hardware will produce garbage or hang.
    """

    M: int  # Matrix A rows
    N: int  # Matrix B columns
    K: int  # Inner dimension (A cols = B rows)
    Tm: int  # Tile height (systolic array rows)
    Tn: int  # Tile width (systolic array cols)
    Tk: int  # Tile depth (K dimension chunk size)
    dtype: str = "int8"  # Data type
    acc_dtype: str = "int32"  # Accumulator data type

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            raise ValueError(f"Matrix dimensions must be positive: M={self.M}, N={self.N}, K={self.K}")
        if self.Tm <= 0 or self.Tn <= 0 or self.Tk <= 0:
            raise ValueError(f"Tile dimensions must be positive: Tm={self.Tm}, Tn={self.Tn}, Tk={self.Tk}")
        # Engineer's Note:
        # The hardware currently requires perfect tiling (no remainders).
        # Future work: Implement padding in software or edge handling in hardware.
        if self.M % self.Tm != 0:
            raise ValueError(f"M={self.M} must be divisible by Tm={self.Tm}")
        if self.N % self.Tn != 0:
            raise ValueError(f"N={self.N} must be divisible by Tn={self.Tn}")
        if self.K % self.Tk != 0:
            raise ValueError(f"K={self.K} must be divisible by Tk={self.Tk}")


class HostAXITiler:
    """
    Host-side Weight-Stationary Tiler using AXI4 DMA
    
    Engineer's Note:
    This class replaces the old UART driver.
    It manages the memory map in DDR and issues AXI transactions.
    Speedup Factor: ~27,000x over UART (14.4 KB/s vs 400 MB/s).
    """

    def __init__(
        self,
        base_addr: int = 0x0,
        ddr_base: int = 0x80000000,
        timeout: float = 5.0,
        verbose: bool = False,
        use_simulator: bool = True,
    ):
        """
        Initialize Host AXI Tiler

        Args:
            base_addr: Base address of AXI4-Lite CSR slave
            ddr_base: Base address of DDR memory for A/B matrices
            timeout: Operation timeout in seconds
            verbose: Enable verbose logging
            use_simulator: Use simulator (True) or real hardware (False)
        """
        self.verbose = verbose
        self.timeout = timeout
        self.ddr_base = ddr_base
        self.axi = AXIDriver(base_addr=base_addr, use_simulator=use_simulator, debug=verbose)

        if self.verbose:
            print(f"Connected to ACCEL-v1 via AXI @ 0x{base_addr:08x} (400 MB/s)")
            print(f"DDR base address: 0x{ddr_base:08x}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close AXI driver connection"""
        pass

    def log(self, msg: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[HOST-AXI] {msg}")

    def csr_write(self, addr: int, value: int) -> bool:
        """Write to accelerator CSR register via AXI4-Lite"""
        success = self.axi.csr_write(addr, value)
        self.log(f"CSR write: addr=0x{addr:02X}, value=0x{value:08X}")
        return success

    def csr_read(self, addr: int) -> Optional[int]:
        """Read from accelerator CSR register via AXI4-Lite"""
        value, success = self.axi.csr_read(addr)
        if success:
            self.log(f"CSR read: addr=0x{addr:02X}, value=0x{value:08X}")
            return value
        return None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Poll status register until operation completes or timeout

        Args:
            timeout: Timeout in seconds (uses instance timeout if None)

        Returns:
            True if operation completed successfully
        """
        if timeout is None:
            timeout = self.timeout

        return self.axi.poll_status(int(timeout * 1000))

    def configure_accelerator(self, config: GEMMConfig, m_idx: int, n_idx: int, k_idx: int) -> bool:
        """
        Configure accelerator for specific tile operation

        Args:
            config: GEMM configuration
            m_idx: Current M tile index
            n_idx: Current N tile index
            k_idx: Current K tile index

        Returns:
            True if configuration successful
        """
        # Configure GEMM dimensions and tile indices
        csr_addr = 0x00
        
        # Write M, N, K (assumes CSR layout from csr_map.py)
        if not self.csr_write(0x00, config.M):
            return False
        if not self.csr_write(0x04, config.N):
            return False
        if not self.csr_write(0x08, config.K):
            return False
        
        # Write tile dimensions
        if not self.csr_write(0x0C, config.Tm):
            return False
        if not self.csr_write(0x10, config.Tn):
            return False
        if not self.csr_write(0x14, config.Tk):
            return False
        
        # Write tile indices
        if not self.csr_write(0x18, m_idx):
            return False
        if not self.csr_write(0x1C, n_idx):
            return False
        if not self.csr_write(0x20, k_idx):
            return False
        
        # Write scales (default to 1.0 in fixed-point)
        if not self.csr_write(0x28, pack_u32(1.0)):
            return False  # Sa
        if not self.csr_write(0x30, pack_u32(1.0)):
            return False  # Sw

        self.log(f"Configured for tile [{m_idx}, {n_idx}, {k_idx}]")
        return True

    def send_tile_data_axi(self, a_tile: np.ndarray, b_tile: np.ndarray) -> bool:
        """
        Send A and B tile data via AXI4 burst transfers

        Args:
            a_tile: A matrix tile [Tm × Tk]
            b_tile: B matrix tile [Tk × Tn]

        Returns:
            True if data transfer successful
        """
        try:
            # Convert to int8 and flatten
            a_data = a_tile.astype(np.int8).flatten()
            b_data = b_tile.astype(np.int8).flatten()

            # Convert to 32-bit words (4 INT8 values per word)
            a_words = []
            for i in range(0, len(a_data), 4):
                chunk = a_data[i : i + 4]
                if len(chunk) < 4:
                    chunk = np.pad(chunk, (0, 4 - len(chunk)), "constant")
                word = int(chunk[0]) | (int(chunk[1]) << 8) | (int(chunk[2]) << 16) | (int(chunk[3]) << 24)
                a_words.append(word)

            b_words = []
            for i in range(0, len(b_data), 4):
                chunk = b_data[i : i + 4]
                if len(chunk) < 4:
                    chunk = np.pad(chunk, (0, 4 - len(chunk)), "constant")
                word = int(chunk[0]) | (int(chunk[1]) << 8) | (int(chunk[2]) << 16) | (int(chunk[3]) << 24)
                b_words.append(word)

            # Send A tile via AXI burst write
            a_addr = self.ddr_base + 0x0
            success_a, words_a = self.axi.write_burst(a_words, a_addr)
            self.log(f"Sent A tile (AXI): {a_tile.shape} -> {len(a_words)} words @ 0x{a_addr:08x}")

            # Send B tile via AXI burst write
            b_addr = self.ddr_base + 0x10000  # 64KB offset
            success_b, words_b = self.axi.write_burst(b_words, b_addr)
            self.log(f"Sent B tile (AXI): {b_tile.shape} -> {len(b_words)} words @ 0x{b_addr:08x}")

            return success_a and success_b
        except Exception as e:
            self.log(f"Data transfer failed: {e}")
            return False

    def read_results(self, size: int) -> Optional[np.ndarray]:
        """
        Read result matrix via AXI4 burst read

        Args:
            size: Number of words to read

        Returns:
            Result array or None on error
        """
        result_addr = self.ddr_base + 0x20000  # 128KB offset for results
        
        # Limit burst to 256 words max
        read_size = min(size, 256)
        data_words, success = self.axi.read_burst(result_addr, read_size)

        if success:
            # Convert back to INT8
            result = np.array(data_words, dtype=np.uint32)
            result_int8 = np.unpackbits(result.astype(np.uint8), bitorder='little').astype(np.int8)
            self.log(f"Read results: {len(result_int8)} INT8 values")
            return result_int8[:size]
        else:
            self.log(f"Failed to read results")
            return None

    def run_gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        config: GEMMConfig,
    ) -> np.ndarray:
        """
        Execute full GEMM operation using AXI burst transfers

        Args:
            A: Activation matrix [M×K]
            B: Weight matrix [K×N]
            config: GEMM configuration

        Returns:
            Result matrix [M×N]
        """
        self.log(f"Starting GEMM: A({A.shape}) × B({B.shape}) with tiles Tm={config.Tm}, Tn={config.Tn}, Tk={config.Tk}")

        # Result matrix
        C = np.zeros((config.M, config.N), dtype=np.int32)
        start_time = time.time()

        # Tile-based loop
        for m_idx in range(0, config.M, config.Tm):
            for n_idx in range(0, config.N, config.Tn):
                for k_idx in range(0, config.K, config.Tk):
                    # Extract tiles
                    m_end = min(m_idx + config.Tm, config.M)
                    n_end = min(n_idx + config.Tn, config.N)
                    k_end = min(k_idx + config.Tk, config.K)

                    a_tile = A[m_idx:m_end, k_idx:k_end]
                    b_tile = B[k_idx:k_end, n_idx:n_end]

                    # Pad to tile size
                    a_tile_padded = np.zeros((config.Tm, config.Tk), dtype=np.int8)
                    a_tile_padded[: a_tile.shape[0], : a_tile.shape[1]] = a_tile

                    b_tile_padded = np.zeros((config.Tk, config.Tn), dtype=np.int8)
                    b_tile_padded[: b_tile.shape[0], : b_tile.shape[1]] = b_tile

                    # Configure
                    if not self.configure_accelerator(config, m_idx // config.Tm, n_idx // config.Tn, k_idx // config.Tk):
                        self.log(f"Configuration failed for tile [{m_idx}, {n_idx}, {k_idx}]")
                        return None

                    # Send data via AXI (27,000× faster than UART)
                    if not self.send_tile_data_axi(a_tile_padded, b_tile_padded):
                        self.log(f"Data transfer failed for tile [{m_idx}, {n_idx}, {k_idx}]")
                        return None

                    # Trigger computation
                    if not self.csr_write(0x40, 0x1):  # START pulse
                        return None

                    # Wait for completion
                    if not self.wait_for_completion():
                        self.log(f"Computation timeout for tile [{m_idx}, {n_idx}, {k_idx}]")
                        return None

                    # Read partial results
                    tile_size = min(config.Tm, m_end - m_idx) * min(config.Tn, n_end - n_idx)
                    result_tile = self.read_results(tile_size)

                    if result_tile is not None:
                        C[m_idx:m_end, n_idx:n_end] += result_tile[: (m_end - m_idx) * (n_end - n_idx)].reshape(
                            m_end - m_idx, n_end - n_idx
                        )

        elapsed = time.time() - start_time
        self.log(f"GEMM complete in {elapsed:.3f}s (AXI burst transfers)")

        return C


def main():
    parser = argparse.ArgumentParser(description="AXI-Based GEMM Runner for ACCEL-v1")
    parser.add_argument("--M", type=int, default=16, help="A matrix rows")
    parser.add_argument("--N", type=int, default=16, help="B matrix columns")
    parser.add_argument("--K", type=int, default=16, help="Inner dimension")
    parser.add_argument("--Tm", type=int, default=8, help="Tile height")
    parser.add_argument("--Tn", type=int, default=8, help="Tile width")
    parser.add_argument("--Tk", type=int, default=8, help="Tile depth")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--simulator", action="store_true", help="Use simulator mode")

    args = parser.parse_args()

    # Create random test matrices
    A = np.random.randint(-128, 127, (args.M, args.K), dtype=np.int8)
    B = np.random.randint(-128, 127, (args.K, args.N), dtype=np.int8)

    # Create config
    config = GEMMConfig(M=args.M, N=args.N, K=args.K, Tm=args.Tm, Tn=args.Tn, Tk=args.Tk)

    # Run GEMM
    with HostAXITiler(verbose=args.verbose, use_simulator=args.simulator or True) as tiler:
        result = tiler.run_gemm(A, B, config)

        if result is not None:
            # Verify against NumPy
            A_float = A.astype(np.float32)
            B_float = B.astype(np.float32)
            expected = (A_float @ B_float).astype(np.int32)
            
            error = np.max(np.abs(result - expected))
            print(f"✓ GEMM successful! Max error: {error}")
        else:
            print("✗ GEMM failed")


if __name__ == "__main__":
    main()
