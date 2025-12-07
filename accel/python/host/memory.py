#!/usr/bin/env python3
"""
memory.py — Memory Management Utilities for ACCEL-v1
=====================================================

Provides utilities for:
  - DMA buffer allocation with PYNQ
  - BSR sparse matrix format conversion
  - Memory layout optimization
  - Cache-aligned allocation

Author: ACCEL-v1 Team
Date: December 2024
"""

import numpy as np
from typing import Tuple, List, Optional
import struct

# Try PYNQ import
try:
    from pynq import allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False


class DMABuffer:
    """
    DMA-compatible buffer wrapper.
    
    Provides a consistent interface for DMA buffers whether running
    on real hardware (PYNQ) or in simulation.
    """
    
    def __init__(self, size: int, dtype=np.uint8, simulation: bool = False):
        """
        Allocate a DMA buffer.
        
        Args:
            size: Buffer size in elements
            dtype: NumPy data type
            simulation: Use simulation mode (pure NumPy)
        """
        self.size = size
        self.dtype = dtype
        self.simulation = simulation or not PYNQ_AVAILABLE
        
        if self.simulation:
            self._buffer = np.zeros(size, dtype=dtype)
            self._device_address = 0x40000000  # Simulated address
        else:
            self._buffer = allocate(shape=(size,), dtype=dtype)
            self._device_address = self._buffer.device_address
            
    @property
    def device_address(self) -> int:
        """Get the physical address for DMA."""
        return self._device_address
    
    @property
    def data(self) -> np.ndarray:
        """Get the buffer data as numpy array."""
        return self._buffer
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, key):
        return self._buffer[key]
    
    def __setitem__(self, key, value):
        self._buffer[key] = value
        
    def sync_to_device(self):
        """Flush CPU cache to device memory."""
        if not self.simulation:
            self._buffer.sync_to_device()
            
    def sync_from_device(self):
        """Invalidate CPU cache, read from device memory."""
        if not self.simulation:
            self._buffer.sync_from_device()
            
    def free(self):
        """Free the buffer."""
        if not self.simulation:
            self._buffer.freebuffer()
        self._buffer = None


class BSRMatrix:
    """
    Block Sparse Row (BSR) matrix format for ACCEL-v1.
    
    BSR stores sparse matrices as a collection of dense blocks.
    This is efficient for neural network weights which often have
    block-structured sparsity patterns.
    
    Memory Layout (for DMA):
        [row_ptr: (K/bs + 1) x 4 bytes]
        [col_idx: nnz_blocks x 2 bytes]
        [values:  nnz_blocks x bs x bs bytes]
    """
    
    def __init__(self, block_size: int = 16):
        """
        Initialize BSR matrix container.
        
        Args:
            block_size: Block dimension (16 for 16x16 systolic array)
        """
        self.block_size = block_size
        self.row_ptr = None
        self.col_idx = None
        self.values = None
        self.shape = (0, 0)
        self.nnz_blocks = 0
        
    @classmethod
    def from_dense(cls, dense: np.ndarray, block_size: int = 16, 
                   threshold: float = 0.0) -> 'BSRMatrix':
        """
        Convert dense matrix to BSR format.
        
        Args:
            dense: Dense matrix (K, N) INT8
            block_size: Block dimension
            threshold: Sparsity threshold (blocks with L2 norm < threshold are dropped)
            
        Returns:
            BSRMatrix instance
        """
        K, N = dense.shape
        bs = block_size
        
        # Pad to block-aligned dimensions
        K_padded = ((K + bs - 1) // bs) * bs
        N_padded = ((N + bs - 1) // bs) * bs
        
        if K_padded != K or N_padded != N:
            padded = np.zeros((K_padded, N_padded), dtype=dense.dtype)
            padded[:K, :N] = dense
            dense = padded
        
        n_block_rows = K_padded // bs
        n_block_cols = N_padded // bs
        
        # Find non-zero blocks
        row_ptr = [0]
        col_idx = []
        blocks = []
        
        for i in range(n_block_rows):
            row_start = i * bs
            row_end = row_start + bs
            
            for j in range(n_block_cols):
                col_start = j * bs
                col_end = col_start + bs
                
                block = dense[row_start:row_end, col_start:col_end]
                
                # Check if block is "non-zero" (above threshold)
                if np.linalg.norm(block) >= threshold:
                    col_idx.append(j)
                    blocks.append(block.copy())
                    
            row_ptr.append(len(col_idx))
        
        # Create BSRMatrix
        bsr = cls(block_size)
        bsr.row_ptr = np.array(row_ptr, dtype=np.uint32)
        bsr.col_idx = np.array(col_idx, dtype=np.uint16)
        bsr.values = np.stack(blocks) if blocks else np.zeros((0, bs, bs), dtype=dense.dtype)
        bsr.shape = (K_padded, N_padded)
        bsr.nnz_blocks = len(blocks)
        
        return bsr
    
    def to_dense(self) -> np.ndarray:
        """
        Convert BSR back to dense matrix.
        
        Returns:
            Dense matrix (K, N)
        """
        K, N = self.shape
        bs = self.block_size
        
        dense = np.zeros((K, N), dtype=self.values.dtype)
        
        block_idx = 0
        for row in range(len(self.row_ptr) - 1):
            row_start = row * bs
            row_end = row_start + bs
            
            for ptr in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_idx[ptr]
                col_start = col * bs
                col_end = col_start + bs
                
                dense[row_start:row_end, col_start:col_end] = self.values[block_idx]
                block_idx += 1
                
        return dense
    
    def pack_for_dma(self) -> bytes:
        """
        Pack BSR data into contiguous bytes for DMA transfer.
        
        Layout:
            [row_ptr bytes][col_idx bytes][values bytes]
            
        Returns:
            Packed bytes ready for DMA
        """
        row_ptr_bytes = self.row_ptr.tobytes()
        col_idx_bytes = self.col_idx.tobytes()
        values_bytes = self.values.astype(np.int8).tobytes()
        
        return row_ptr_bytes + col_idx_bytes + values_bytes
    
    def memory_size(self) -> int:
        """Calculate total memory size in bytes."""
        return (len(self.row_ptr) * 4 + 
                len(self.col_idx) * 2 + 
                self.nnz_blocks * self.block_size * self.block_size)
    
    def sparsity(self) -> float:
        """Calculate sparsity ratio (percentage of zero blocks)."""
        K, N = self.shape
        bs = self.block_size
        total_blocks = (K // bs) * (N // bs)
        
        if total_blocks == 0:
            return 0.0
            
        return 1.0 - (self.nnz_blocks / total_blocks)
    
    def __repr__(self) -> str:
        return (f"BSRMatrix(shape={self.shape}, block_size={self.block_size}, "
                f"nnz_blocks={self.nnz_blocks}, sparsity={self.sparsity():.1%})")


def align_size(size: int, alignment: int = 64) -> int:
    """
    Align size to given boundary.
    
    Args:
        size: Size to align
        alignment: Alignment boundary (default 64 bytes for cache line)
        
    Returns:
        Aligned size
    """
    return ((size + alignment - 1) // alignment) * alignment


def pack_activations(activations: np.ndarray, block_size: int = 16) -> bytes:
    """
    Pack dense activations for DMA transfer.
    
    Args:
        activations: Activation matrix (M, K) INT8
        block_size: Block size for padding
        
    Returns:
        Packed bytes
    """
    M, K = activations.shape
    
    # Pad to block-aligned
    M_padded = ((M + block_size - 1) // block_size) * block_size
    K_padded = ((K + block_size - 1) // block_size) * block_size
    
    if M_padded != M or K_padded != K:
        padded = np.zeros((M_padded, K_padded), dtype=np.int8)
        padded[:M, :K] = activations
        activations = padded
        
    return activations.tobytes()


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    print("Memory Utilities Test")
    print("=" * 50)
    
    # Create a sparse weight matrix
    np.random.seed(42)
    K, N = 64, 64
    dense = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    
    # Make it sparse (set some blocks to zero)
    for i in range(0, K, 16):
        for j in range(0, N, 16):
            if np.random.random() > 0.5:  # 50% sparsity
                dense[i:i+16, j:j+16] = 0
    
    # Convert to BSR
    bsr = BSRMatrix.from_dense(dense, block_size=16)
    print(f"BSR: {bsr}")
    print(f"  Memory size: {bsr.memory_size()} bytes")
    print(f"  Dense size:  {K * N} bytes")
    print(f"  Compression: {bsr.memory_size() / (K * N):.1%}")
    
    # Verify reconstruction
    reconstructed = bsr.to_dense()
    assert np.array_equal(dense, reconstructed), "BSR reconstruction failed!"
    print("  Reconstruction: PASS")
    
    # Test DMA buffer
    print("\nDMA Buffer Test:")
    buf = DMABuffer(1024, simulation=True)
    buf[0:10] = np.arange(10, dtype=np.uint8)
    print(f"  Address: 0x{buf.device_address:08x}")
    print(f"  First 10 bytes: {buf[0:10]}")
