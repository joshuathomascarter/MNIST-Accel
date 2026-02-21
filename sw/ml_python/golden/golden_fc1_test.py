#!/usr/bin/env python3
"""
golden_fc1_test.py — Run golden model on FC1 layer and save expected outputs
=============================================================================
This creates reference outputs that the C++/RTL simulation should match.
"""

import numpy as np
import json
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_bsr_layer(layer_dir):
    """Load BSR layer data from exported files."""
    row_ptr = np.load(os.path.join(layer_dir, "row_ptr.npy"))
    col_idx = np.load(os.path.join(layer_dir, "col_idx.npy"))
    
    # Load weights from .bsr binary file
    weights_path = os.path.join(layer_dir, "weights.bsr")
    with open(weights_path, "rb") as f:
        weights_flat = np.frombuffer(f.read(), dtype=np.int8)
    
    # Load metadata
    meta_path = os.path.join(layer_dir, "weights.meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    block_h, block_w = meta["blocksize"]
    num_blocks = meta["num_blocks"]
    
    # Reshape weights to blocks
    weights = weights_flat.reshape(num_blocks, block_h, block_w)
    
    return {
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "weights": weights,
        "block_h": block_h,
        "block_w": block_w,
        "num_blocks": num_blocks,
        "shape": meta["padded_shape"],
        "original_shape": meta["shape"]
    }


def gemm_bsr_int8_golden(activations, bsr_layer):
    """
    Golden BSR GEMM implementation.
    
    C = A @ B where B is in BSR format
    - A: activations [M, K] INT8
    - B: weights [K, N] in BSR format
    - C: output [M, N] INT32
    
    For FC1: A[1, 9216] @ B[9216, 128] = C[1, 128]
    (But stored transposed: B is [128, 9216] in BSR)
    """
    row_ptr = bsr_layer["row_ptr"]
    col_idx = bsr_layer["col_idx"]
    weights = bsr_layer["weights"]  # [num_blocks, block_h, block_w]
    block_h = bsr_layer["block_h"]
    block_w = bsr_layer["block_w"]
    
    M = activations.shape[0]  # Batch size
    K = activations.shape[1]  # Input features
    N = len(row_ptr) - 1      # Output features (block rows * block_h)
    N = N * block_h           # Total output neurons
    
    # Initialize output accumulator
    C = np.zeros((M, N), dtype=np.int32)
    
    num_block_rows = len(row_ptr) - 1
    
    # Process each block row
    for block_row in range(num_block_rows):
        block_start = row_ptr[block_row]
        block_end = row_ptr[block_row + 1]
        
        if block_start == block_end:
            continue  # Empty row
        
        # Process each non-zero block in this row
        for block_idx in range(block_start, block_end):
            block_col = col_idx[block_idx]
            block_data = weights[block_idx]  # [block_h, block_w] INT8
            
            # Output position
            out_row_start = block_row * block_h
            
            # Input position (column in weight matrix = row in activation)
            in_col_start = block_col * block_w
            
            # Extract activation slice: A[:, in_col_start : in_col_start + block_w]
            A_slice = activations[:, in_col_start : in_col_start + block_w]  # [M, block_w]
            
            # Matrix multiply: A_slice @ block_data.T -> [M, block_h]
            # block_data is [block_h, block_w], we want [M, block_h]
            for m in range(M):
                for h in range(block_h):
                    acc = 0
                    for w in range(min(block_w, A_slice.shape[1])):
                        acc += int(A_slice[m, w]) * int(block_data[h, w])
                    C[m, out_row_start + h] += acc
    
    return C


def main():
    print("=" * 60)
    print("FC1 Golden Model Test")
    print("=" * 60)
    
    # Load BSR layer - go up three directories from sw/ml_python/golden to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_dir = os.path.join(project_root, "data/bsr_export_14x14/fc1")
    fc1 = load_bsr_layer(data_dir)
    
    print(f"\nFC1 Layer:")
    print(f"  Original shape: {fc1['original_shape']}")
    print(f"  Padded shape:   {fc1['shape']}")
    print(f"  Block size:     {fc1['block_h']}×{fc1['block_w']}")
    print(f"  Num blocks:     {fc1['num_blocks']}")
    print(f"  Sparsity:       {100 * (1 - fc1['num_blocks'] / ((fc1['shape'][0]//fc1['block_h']) * (fc1['shape'][1]//fc1['block_w']))):.1f}%")
    
    # Create test activations (same pattern as C++ test)
    K = fc1["shape"][1]  # 9216
    activations = np.zeros(K, dtype=np.int8)
    for i in range(K):
        activations[i] = (i % 256) - 128  # Same pattern as C++ test
    activations = activations.reshape(1, K)  # [1, 9216]
    
    print(f"\nActivations:")
    print(f"  Shape: {activations.shape}")
    print(f"  First 10: {activations[0, :10]}")
    
    # Run golden model
    print("\nRunning golden GEMM...")
    C = gemm_bsr_int8_golden(activations, fc1)
    
    print(f"\nOutput:")
    print(f"  Shape: {C.shape}")
    print(f"  First 16 values: {C[0, :16]}")
    print(f"  Sum: {np.sum(C)}")
    print(f"  Min: {np.min(C)}, Max: {np.max(C)}")
    
    # Save golden output
    output_path = os.path.join(project_root, "data/bsr_export_14x14/fc1/golden_output.npy")
    np.save(output_path, C)
    print(f"\nSaved golden output to: {output_path}")
    
    # Also save as text for easy comparison
    output_txt = os.path.join(project_root, "data/bsr_export_14x14/fc1/golden_output.txt")
    with open(output_txt, "w") as f:
        f.write(f"# FC1 Golden Output (INT32)\n")
        f.write(f"# Shape: {C.shape}\n")
        for i in range(C.shape[1]):
            f.write(f"{i:3d}: {C[0, i]:12d}\n")
    print(f"Saved text version to: {output_txt}")
    
    print("\n" + "=" * 60)
    print("Golden model complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
