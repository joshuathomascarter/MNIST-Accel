"""
BSR Sparse GEMM with INT8 Quantization - Golden Reference
Software reference implementation for hardware validation.

Implements: C = A @ B where B is in BSR (Block Sparse Row) format
- A: Dense INT8 matrix [M, K]
- B: Sparse INT8 matrix [K, N] in BSR format (only non-zero 8×8 blocks stored)
- C: Output FP32 matrix [M, N]

This is the GOLDEN MODEL that your Verilog hardware must match.
"""

import numpy as np


def gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B):
    """
    Sparse GEMM: C = A @ B with INT8 inputs and per-channel scaling.

    Args:
        A_int8: Dense input matrix [M, K], INT8
        bsr_B: Dictionary with BSR format:
            - 'data': Non-zero blocks [num_blocks, block_h, block_w], FP32
            - 'indices': Column index of each block [num_blocks], INT32
            - 'indptr': Row pointer [num_block_rows + 1], INT32
            - 'shape': Original matrix shape (K, N)
            - 'blocksize': (block_h, block_w)
        scale_A: Quantization scale for A (single float)
        scales_B: Per-channel quantization scales for B [N], FP32

    Returns:
        C: Output matrix [M, N], FP32
    """

    # Extract BSR components
    blocks = bsr_B["data"]  # [num_blocks, block_h, block_w]
    col_idx = bsr_B["indices"]  # [num_blocks]
    row_ptr = bsr_B["indptr"]  # [num_block_rows + 1]
    K, N = bsr_B["shape"]
    block_h, block_w = bsr_B["blocksize"]

    M = A_int8.shape[0]
    num_block_rows = len(row_ptr) - 1

    # Initialize output
    C = np.zeros((M, N), dtype=np.float32)

    # ==========================================
    # CORE SPARSE GEMM LOOP
    # This is what your hardware scheduler does
    # ==========================================

    for block_row in range(num_block_rows):
        # Get range of blocks in this row
        block_start = row_ptr[block_row]
        block_end = row_ptr[block_row + 1]
        num_blocks_in_row = block_end - block_start

        # If empty row, skip
        if num_blocks_in_row == 0:
            continue

        # Process each non-zero block in this row
        for block_idx in range(block_start, block_end):
            # Get block column
            block_col = col_idx[block_idx]

            # Get block data (FP32)
            block = blocks[block_idx]  # [block_h, block_w]

            # Quantize block to INT8 using per-output-channel scales
            # block_col indexes the N (output channel) dimension
            block_int8 = np.zeros((block_h, block_w), dtype=np.int8)
            for local_col in range(block_w):
                global_col = block_col * block_w + local_col
                if global_col < N:
                    scale = scales_B[global_col] if global_col < len(scales_B) else scales_B[0]
                    block_int8[:, local_col] = np.clip(
                        np.rint(block[:, local_col] / max(scale, 1e-12)), -128, 127
                    ).astype(np.int8)

            # Compute which rows and columns this block affects
            # block_row indexes K (reduction dimension)
            # block_col indexes N (output channel dimension)
            row_start = block_row * block_h
            row_end = min(row_start + block_h, K)
            col_start = block_col * block_w
            col_end = min(col_start + block_w, N)

            actual_h = row_end - row_start
            actual_w = col_end - col_start

            # Extract corresponding slice of A (reduction dimension)
            A_slice = A_int8[:, row_start:row_end]  # [M, actual_h]

            # INT8 matrix multiply: A_slice @ block_int8[:actual_h, :actual_w]
            C_tile_int32 = A_slice.astype(np.int32) @ block_int8[:actual_h, :actual_w].astype(np.int32)  # [M, actual_w]

            # Dequantize per output channel and accumulate (once, not per block_h)
            for local_col in range(actual_w):
                global_col = col_start + local_col
                scale = scales_B[global_col] if global_col < len(scales_B) else scales_B[0]
                C[:, global_col] += C_tile_int32[:, local_col].astype(np.float32) * scale_A * scale

    return C


def gemm_bsr_int8_simple(A_int8, bsr_B, scale_A, scales_B):
    """
    Simplified version for understanding (less optimized, clearer logic).

    This version processes one 8×8 block at a time without vectorization.
    Use this to understand the algorithm, then use gemm_bsr_int8() for accuracy.
    """

    blocks = bsr_B["data"]
    col_idx = bsr_B["indices"]
    row_ptr = bsr_B["indptr"]
    K, N = bsr_B["shape"]
    block_h, block_w = bsr_B["blocksize"]

    M = A_int8.shape[0]
    num_block_rows = len(row_ptr) - 1

    C = np.zeros((M, N), dtype=np.float32)

    # For each block row
    for block_row in range(num_block_rows):
        # How many blocks in this row?
        num_blocks = row_ptr[block_row + 1] - row_ptr[block_row]

        if num_blocks == 0:
            # Empty row - skip
            continue

        # For each block in this row
        for i in range(num_blocks):
            block_idx = row_ptr[block_row] + i
            block_col = col_idx[block_idx]
            block_data = blocks[block_idx]  # [8, 8]

            # block_row indexes K (reduction), block_col indexes N (output)
            k_start = block_row * block_h
            out_col_start = block_col * block_w

            # Extract A slice for this block (reduction dimension)
            A_slice = A_int8[:, k_start : k_start + block_h]  # [M, 8]

            # Quantize B block to INT8 per output channel
            block_int8 = np.zeros((block_h, block_w), dtype=np.int8)
            for c in range(block_w):
                global_col = out_col_start + c
                if global_col < N:
                    scale = scales_B[global_col]
                    block_int8[:, c] = np.clip(np.rint(block_data[:, c] / max(scale, 1e-12)), -128, 127)

            # INT8 multiply: C_tile = A_slice @ block_int8
            C_tile = np.zeros((M, block_w), dtype=np.int32)
            for m in range(M):
                for n in range(block_w):
                    acc = 0
                    for k in range(block_h):
                        acc += int(A_slice[m, k]) * int(block_int8[k, n])
                    C_tile[m, n] = acc

            # Dequantize and accumulate (once per output channel, not per block_h)
            for n in range(block_w):
                global_col = out_col_start + n
                if global_col < N:
                    scale = scales_B[global_col]
                    C[:, global_col] += C_tile[:, n].astype(np.float32) * scale_A * scale

    return C


if __name__ == "__main__":
    print("=" * 60)
    print("BSR INT8 GEMM - Golden Reference")
    print("=" * 60)

    # Simple test case
    M, K, N = 2, 28, 14

    # Create sparse weight matrix (50% sparse)
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    B[14:28, :] = 0  # Make second block-row zero

    # Input matrix
    A = np.random.randn(M, K).astype(np.float32)

    # Build BSR format
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training.export_bsr import build_bsr_from_dense

    bsr_B = build_bsr_from_dense(B, 14, 14)

    print(f"\nMatrix shapes:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  BSR blocks: {bsr_B['num_blocks']} (out of {2*1} total)")
    print(f"  Sparsity: {bsr_B['sparsity_pct']:.1f}%")

    # Quantize
    scale_A = np.max(np.abs(A)) / 127.0
    A_int8 = np.clip(np.rint(A / scale_A), -128, 127).astype(np.int8)

    # Per output channel (column of B) scales
    scales_B = np.max(np.abs(B), axis=0) / 127.0  # axis=0 -> per column (N dimension)
    scales_B = np.maximum(scales_B.flatten(), 1e-12)

    # Compute sparse GEMM
    C_sparse = gemm_bsr_int8(A_int8, bsr_B, scale_A, scales_B)

    # Dense reference for validation
    from golden_models.gemm_int8 import gemm_int8_per_channel

    C_dense = gemm_int8_per_channel(A_int8, B, scale_A, scales_B)

    # Compare
    max_error = np.max(np.abs(C_sparse - C_dense))
    print(f"\nValidation:")
    print(f"  Max error vs dense: {max_error:.6f}")
    print(f"  Status: {'PASS ✓' if max_error < 1e-4 else 'FAIL ✗'}")

    print("\n" + "=" * 60)
    print("Golden reference test complete!")
    print("=" * 60)
