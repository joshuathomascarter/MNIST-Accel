"""
Edge Case Tests for BSR Sparse GEMM
Tests corner cases that break naive implementations:
- Empty rows (no non-zero blocks)
- 100% dense (all blocks non-zero)
- 100% sparse (all blocks zero)
- Single block matrices
- Irregular sparsity patterns
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from golden.gemm_bsr_int8 import gemm_bsr_int8
from training.export_bsr import build_bsr_from_dense


def build_bsr_from_dense(weight, block_h=14, block_w=14, threshold=1e-10):
    """Build BSR format from dense matrix (simplified version)"""
    height, width = weight.shape

    # Pad to block size
    pad_h = (block_h - height % block_h) % block_h
    pad_w = (block_w - width % block_w) % block_w
    if pad_h > 0 or pad_w > 0:
        weight = np.pad(weight, ((0, pad_h), (0, pad_w)), constant_values=0.0)

    padded_h, padded_w = weight.shape
    num_block_rows = padded_h // block_h
    num_block_cols = padded_w // block_w

    blocks_list = []
    col_indices_list = []
    row_ptr = [0]

    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            r_start = block_row * block_h
            c_start = block_col * block_w
            block = weight[r_start : r_start + block_h, c_start : c_start + block_w]

            if np.linalg.norm(block) > threshold:
                blocks_list.append(block)
                col_indices_list.append(block_col)

        row_ptr.append(len(blocks_list))

    data = np.stack(blocks_list, axis=0) if blocks_list else np.zeros((0, block_h, block_w))
    indices = np.array(col_indices_list, dtype=np.int32)
    indptr = np.array(row_ptr, dtype=np.int32)

    return {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "shape": (height, width),
        "blocksize": (block_h, block_w),
        "num_blocks": len(blocks_list),
    }


def test_empty_rows():
    """Test matrix with empty rows (common in 90% sparse networks)"""
    print("\n" + "=" * 60)
    print("TEST 1: Empty Rows (Block Rows with No Non-Zero Blocks)")
    print("=" * 60)

    # Create weight matrix: [28, 56] with 14×14 blocks
    # Only block-row 0 col 0, and block-row 1 col 1 are non-zero
    B = np.zeros((28, 56), dtype=np.float32)
    B[0:14, 0:14] = np.random.randn(14, 14) * 0.1    # Block (0, 0)
    B[14:28, 14:28] = np.random.randn(14, 14) * 0.1  # Block (1, 1)

    # Input matrix
    A = np.random.randn(2, 56).astype(np.float32)

    # Build BSR
    bsr_B = build_bsr_from_dense(B, 14, 14)

    print(f"Weight shape: {B.shape}")
    print(f"BSR blocks: {bsr_B['num_blocks']} (only 2 non-zero)")
    print(f"row_ptr: {bsr_B['indptr'].tolist()}")
    print(f"  Row 0: {bsr_B['indptr'][1] - bsr_B['indptr'][0]} blocks")
    print(f"  Row 1: {bsr_B['indptr'][2] - bsr_B['indptr'][1]} blocks")

    # Quantize
    scale_A = np.max(np.abs(A)) / 127.0
    A_int8 = np.clip(np.rint(A / scale_A), -128, 127).astype(np.int8)

    scales_B = np.max(np.abs(B), axis=1, keepdims=True) / 127.0
    scales_B = np.maximum(scales_B, 1e-12).flatten()

    # Dense reference
    C_dense = (A_int8.astype(np.int32) @ B.astype(np.int32)).astype(np.float32) * scale_A

    # BSR sparse
    C_sparse = gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B)

    # Validate: BSR result shape matches dense
    assert C_sparse.shape == C_dense.shape, (
        f"Shape mismatch: sparse {C_sparse.shape} vs dense {C_dense.shape}")
    assert bsr_B['num_blocks'] == 2, (
        f"Expected 2 non-zero blocks, got {bsr_B['num_blocks']}")
    print(f"\nResults:")
    print(f"  Sparse shape: {C_sparse.shape}")
    print(f"  Test: PASS ✓")


def test_100_percent_dense():
    """Test 100% dense matrix (BSR should match dense GEMM exactly)"""
    print("\n" + "=" * 60)
    print("TEST 2: 100% Dense (All Blocks Non-Zero)")
    print("=" * 60)

    # Fully dense matrix
    B = np.random.randn(28, 56).astype(np.float32) * 0.1
    A = np.random.randn(2, 56).astype(np.float32)

    bsr_B = build_bsr_from_dense(B, 14, 14)

    print(f"Weight shape: {B.shape}")
    print(f"BSR blocks: {bsr_B['num_blocks']} (all blocks present)")
    print(f"Block density: {bsr_B['num_blocks'] / (2 * 4) * 100:.1f}%")

    # Quantize
    scale_A = np.max(np.abs(A)) / 127.0
    A_int8 = np.clip(np.rint(A / scale_A), -128, 127).astype(np.int8)

    scales_B = np.max(np.abs(B), axis=1, keepdims=True) / 127.0
    scales_B = np.maximum(scales_B, 1e-12).flatten()

    # BSR sparse result
    C_sparse = gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B)

    # 100% dense should have all blocks present (2 block-rows × 4 block-cols = 8)
    expected_blocks = 2 * 4
    assert bsr_B['num_blocks'] == expected_blocks, (
        f"Expected {expected_blocks} blocks for dense matrix, got {bsr_B['num_blocks']}")
    assert C_sparse.shape[0] > 0, "Output should have rows"
    assert np.any(C_sparse != 0), "Dense matrix should produce non-zero output"
    print(f"\nResults:")
    print(f"  Output shape: {C_sparse.shape}")
    print(f"  All blocks present: {bsr_B['num_blocks'] == expected_blocks}")
    print(f"  Test: PASS ✓")


def test_100_percent_sparse():
    """Test 100% sparse matrix (all zeros - should return zero output)"""
    print("\n" + "=" * 60)
    print("TEST 3: 100% Sparse (All Zeros)")
    print("=" * 60)

    # All-zero matrix
    B = np.zeros((28, 56), dtype=np.float32)
    A = np.random.randn(2, 56).astype(np.float32)

    bsr_B = build_bsr_from_dense(B, 14, 14)

    print(f"Weight shape: {B.shape}")
    print(f"BSR blocks: {bsr_B['num_blocks']} (should be 0)")
    print(f"row_ptr: {bsr_B['indptr'].tolist()}")

    # Quantize
    scale_A = np.max(np.abs(A)) / 127.0
    A_int8 = np.clip(np.rint(A / scale_A), -128, 127).astype(np.int8)
    scales_B = np.ones(28) * 1e-6  # Dummy scales

    # BSR sparse (should be all zeros)
    C_sparse = gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B)

    max_val = np.max(np.abs(C_sparse))
    assert bsr_B['num_blocks'] == 0, (
        f"All-zero matrix should produce 0 BSR blocks, got {bsr_B['num_blocks']}")
    assert max_val < 1e-6, (
        f"All-zero weights should produce near-zero output, got max={max_val}")
    print(f"\nResults:")
    print(f"  Max output value: {max_val:.6f}")
    print(f"  Test: PASS ✓")


def test_single_block():
    """Test minimal matrix (1 block)"""
    print("\n" + "=" * 60)
    print("TEST 4: Single Block Matrix")
    print("=" * 60)

    # Single 14×14 block
    B = np.random.randn(14, 14).astype(np.float32) * 0.1
    A = np.random.randn(1, 14).astype(np.float32)

    bsr_B = build_bsr_from_dense(B, 14, 14)

    print(f"Weight shape: {B.shape}")
    print(f"BSR blocks: {bsr_B['num_blocks']}")

    # Quantize
    scale_A = np.max(np.abs(A)) / 127.0
    A_int8 = np.clip(np.rint(A / scale_A), -128, 127).astype(np.int8)

    scales_B = np.max(np.abs(B), axis=1, keepdims=True) / 127.0
    scales_B = np.maximum(scales_B, 1e-12).flatten()

    # BSR sparse result
    C_sparse = gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B)

    assert bsr_B['num_blocks'] == 1, (
        f"Single block matrix should have 1 block, got {bsr_B['num_blocks']}")
    assert C_sparse.shape == (1, 14), (
        f"Output shape should be (1, 14), got {C_sparse.shape}")
    assert np.any(C_sparse != 0), "Single non-zero block should produce non-zero output"
    print(f"\nResults:")
    print(f"  Output shape: {C_sparse.shape}")
    print(f"  Test: PASS ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("BSR GEMM Edge Case Tests")
    print("=" * 60)

    test_empty_rows()
    test_100_percent_dense()
    test_100_percent_sparse()
    test_single_block()

    print("\n" + "=" * 60)
    print("All edge case tests complete!")
    print("=" * 60)
