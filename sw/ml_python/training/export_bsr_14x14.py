#!/usr/bin/env python3
"""
export_bsr_14x14.py — 14×14 Block Sparse Row Export for Zynq Accelerator
=========================================================================

PURPOSE:
    Re-exports all model weights in BSR format with 14×14 block size to match
    the Zynq-7020 systolic array hardware. The accelerator has a 14×14 PE grid
    (196 DSP48E1s), so all tiles must align to this dimension.

WHY 14×14?
    - Zynq-7020 has 220 DSP48E1 slices
    - 14×14 = 196 PEs fits within DSP budget with margin for control logic
    - Using 16×16 (256 PEs) would exceed available DSPs
    - Smaller tiles (8×8) waste bandwidth and increase scheduling overhead

MEMORY ALIGNMENT REQUIREMENTS:
    - Block data: 14×14 = 196 INT8 values = 196 bytes per block
    - 196 bytes is NOT 64-bit aligned (196 % 8 = 4)
    - We pad each block to 200 bytes (25 × 8 bytes) for AXI burst alignment
    - Alternative: Pack 196 bytes contiguous, handle unaligned access in DMA

OUTPUT FORMAT:
    For each layer (e.g., fc1/):
        row_ptr.npy     - INT32 array, len = num_block_rows + 1
        col_idx.npy     - INT32 array, len = num_blocks
        weights.bsr     - Binary INT8, num_blocks × 196 bytes (no padding)
        weights.meta.json - Metadata with block dimensions, shapes, sparsity

USAGE:
    python export_bsr_14x14.py [--input CHECKPOINT] [--output DIR] [--quantized]

Author: ACCEL-BSR Team
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from pathlib import Path

# =============================================================================
# Constants — Hardware-Defined
# =============================================================================
BLOCK_SIZE = 14  # 14×14 PE array — DO NOT CHANGE without hardware redesign
BLOCK_H = BLOCK_SIZE
BLOCK_W = BLOCK_SIZE
BLOCK_ELEMENTS = BLOCK_H * BLOCK_W  # 196 INT8 values per block

# AXI alignment: 64-bit (8-byte) aligned for efficient DMA bursts
AXI_ALIGN = 8  # bytes


# =============================================================================
# Model Definition (must match training checkpoint)
# =============================================================================
class MNISTNet(nn.Module):
    """
    Simple MNIST CNN matching the training checkpoint.
    Architecture: conv1 → conv2 → fc1 → fc2
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)     # (1, 28, 28) → (32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)    # (32, 26, 26) → (64, 24, 24)
        self.fc1 = nn.Linear(64 * 12 * 12, 140) # After 2x2 pool: (64, 12, 12) → 140
        self.fc2 = nn.Linear(140, 10)           # 140 → 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# =============================================================================
# BSR Construction with 14×14 Blocks
# =============================================================================
def build_bsr_14x14(
    weight: np.ndarray,
    threshold: float = 1e-10,
    quantize: bool = False,
    scale: Optional[np.ndarray] = None
) -> Dict:
    """
    Convert dense weight matrix to 14×14 Block Sparse Row format.

    ALGORITHM:
        1. Pad weight matrix to multiple of 14×14
        2. Divide into 14×14 blocks
        3. Compute L2 norm of each block
        4. Keep blocks with norm > threshold (non-zero)
        5. Build CSR-like structure: row_ptr, col_idx, data

    PADDING LOGIC:
        For a matrix [M, K], we compute:
            pad_M = (14 - M % 14) % 14
            pad_K = (14 - K % 14) % 14
        Padded dimensions become multiples of 14.

    WHY PADDING MATTERS:
        - Hardware iterates over complete 14×14 tiles
        - Partial tiles at edges require padding to maintain alignment
        - Padding with zeros doesn't affect computation (0 × anything = 0)
        - Alternative: Handle partial tiles in scheduler (adds complexity)

    Args:
        weight: Dense FP32 weight matrix [out_features, in_features]
        threshold: L2 norm threshold for zero blocks
        quantize: If True, output INT8 quantized blocks
        scale: Per-channel quantization scales (required if quantize=True)

    Returns:
        Dictionary with BSR components:
            - data: [num_blocks, 14, 14] INT8 or FP32
            - indices: [num_blocks] block column indices
            - indptr: [num_block_rows + 1] row pointers
            - shape: Original (unpadded) shape
            - padded_shape: After padding to 14×14 alignment
            - blocksize: (14, 14)
            - Sparsity statistics
    """
    original_shape = weight.shape
    height, width = original_shape

    # -------------------------------------------------------------------------
    # Step 1: Pad to 14×14 alignment
    # -------------------------------------------------------------------------
    # Calculate padding needed to reach next multiple of 14
    pad_h = (BLOCK_H - height % BLOCK_H) % BLOCK_H
    pad_w = (BLOCK_W - width % BLOCK_W) % BLOCK_W

    if pad_h > 0 or pad_w > 0:
        weight = np.pad(
            weight,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0.0
        )
        # NOTE: Zero padding ensures padded regions don't contribute to output

    padded_h, padded_w = weight.shape
    num_block_rows = padded_h // BLOCK_H
    num_block_cols = padded_w // BLOCK_W

    # -------------------------------------------------------------------------
    # Step 2: Extract blocks and build sparse structure
    # -------------------------------------------------------------------------
    blocks_list = []
    col_indices_list = []
    row_ptr = [0]

    for block_row in range(num_block_rows):
        row_start = block_row * BLOCK_H
        row_end = row_start + BLOCK_H

        for block_col in range(num_block_cols):
            col_start = block_col * BLOCK_W
            col_end = col_start + BLOCK_W

            # Extract 14×14 block
            block = weight[row_start:row_end, col_start:col_end]

            # Check if block is non-zero (L2 norm above threshold)
            block_norm = np.linalg.norm(block)
            if block_norm > threshold:
                # -------------------------------------------------------------------------
                # Step 3: Optionally quantize to INT8
                # -------------------------------------------------------------------------
                if quantize:
                    if scale is None:
                        raise ValueError("scale required for quantization")

                    # Per-row quantization within the block
                    block_int8 = np.zeros_like(block, dtype=np.int8)
                    for local_row in range(BLOCK_H):
                        global_row = block_row * BLOCK_H + local_row
                        if global_row < height and global_row < len(scale):
                            s = scale[global_row]
                        elif len(scale) > 0:
                            s = scale[0]  # Fallback to first scale
                        else:
                            s = 1.0

                        # Quantize: round(x / scale), clamp to INT8 range
                        row_quantized = np.clip(
                            np.rint(block[local_row, :] / s),
                            -128, 127
                        ).astype(np.int8)
                        block_int8[local_row, :] = row_quantized

                    blocks_list.append(block_int8)
                else:
                    blocks_list.append(block.astype(np.float32))

                col_indices_list.append(block_col)

        # End of block row — record how many blocks so far
        row_ptr.append(len(blocks_list))

    # -------------------------------------------------------------------------
    # Step 4: Convert to numpy arrays
    # -------------------------------------------------------------------------
    num_blocks = len(blocks_list)
    if num_blocks > 0:
        data = np.stack(blocks_list, axis=0)  # [num_blocks, 14, 14]
    else:
        dtype = np.int8 if quantize else np.float32
        data = np.zeros((0, BLOCK_H, BLOCK_W), dtype=dtype)

    indices = np.array(col_indices_list, dtype=np.int32)
    indptr = np.array(row_ptr, dtype=np.int32)

    # Sparsity metrics
    total_blocks = num_block_rows * num_block_cols
    density = num_blocks / total_blocks if total_blocks > 0 else 0.0

    return {
        'data': data,
        'indices': indices,
        'indptr': indptr,
        'shape': original_shape,
        'padded_shape': (padded_h, padded_w),
        'blocksize': (BLOCK_H, BLOCK_W),
        'num_blocks': num_blocks,
        'num_block_rows': num_block_rows,
        'num_block_cols': num_block_cols,
        'density': density,
        'sparsity_pct': (1.0 - density) * 100.0,
    }


# =============================================================================
# Binary Export Functions
# =============================================================================
def save_bsr_binary_int8(bsr_data: Dict, filepath: str):
    """
    Save BSR blocks as contiguous INT8 binary file.

    FORMAT:
        Block 0: [14×14 INT8 values, row-major] = 196 bytes
        Block 1: [14×14 INT8 values, row-major] = 196 bytes
        ...

    MEMORY LAYOUT (for hardware DMA):
        Byte offset of block N = N × 196
        Within block: element[row][col] at offset row*14 + col

    NOTE: 196 bytes is not 8-byte aligned. Options:
        1. Contiguous (current): DMA handles unaligned, simpler format
        2. Padded: Each block at 200 bytes (25 × 8), wastes 4 bytes/block
    """
    data = bsr_data['data']  # [num_blocks, 14, 14]

    if data.dtype != np.int8:
        raise ValueError(f"Expected INT8 data, got {data.dtype}")

    with open(filepath, 'wb') as f:
        for block in data:
            # Flatten in row-major (C order) and write
            block_bytes = block.flatten('C').tobytes()
            assert len(block_bytes) == BLOCK_ELEMENTS, \
                f"Block size mismatch: {len(block_bytes)} != {BLOCK_ELEMENTS}"
            f.write(block_bytes)

    print(f"  Saved INT8 binary: {filepath} ({os.path.getsize(filepath)} bytes)")


def save_bsr_metadata(bsr_data: Dict, filepath: str, layer_name: str):
    """
    Save BSR metadata as JSON for software/hardware coordination.

    METADATA FIELDS:
        layer_name: Human-readable identifier
        shape: Original weight shape [out_features, in_features]
        padded_shape: After 14×14 alignment
        blocksize: [14, 14] — MUST match hardware
        num_blocks: Number of non-zero blocks
        num_block_rows: M / 14 (output dimension tiles)
        num_block_cols: K / 14 (input dimension tiles)
        density/sparsity_pct: Block-level sparsity
        row_ptr: CSR-style row pointer (len = num_block_rows + 1)
        col_idx: Column index for each block (len = num_blocks)
        tiles_per_row: Per-row block counts (for load balancing analysis)
    """
    metadata = {
        'layer_name': layer_name,
        'shape': list(bsr_data['shape']),
        'padded_shape': list(bsr_data['padded_shape']),
        'blocksize': list(bsr_data['blocksize']),
        'num_blocks': int(bsr_data['num_blocks']),
        'num_block_rows': int(bsr_data['num_block_rows']),
        'num_block_cols': int(bsr_data['num_block_cols']),
        'density': float(bsr_data['density']),
        'sparsity_pct': float(bsr_data['sparsity_pct']),
        # Hardware scheduling data
        'row_ptr': bsr_data['indptr'].tolist(),
        'col_idx': bsr_data['indices'].tolist(),
        'tiles_per_row': [
            int(bsr_data['indptr'][i+1] - bsr_data['indptr'][i])
            for i in range(bsr_data['num_block_rows'])
        ],
        'max_tiles_per_row': int(np.max(np.diff(bsr_data['indptr']))) if len(bsr_data['indptr']) > 1 else 0,
        # Hardware alignment info
        'bytes_per_block': BLOCK_ELEMENTS,
        'total_weight_bytes': int(bsr_data['num_blocks'] * BLOCK_ELEMENTS),
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {filepath}")


# =============================================================================
# Layer Export Pipeline
# =============================================================================
def export_layer_14x14(
    name: str,
    weight: np.ndarray,
    output_dir: str,
    scales: Optional[np.ndarray] = None
) -> Dict:
    """
    Export a single layer with 14×14 BSR format.

    Args:
        name: Layer name (e.g., 'fc1')
        weight: Weight matrix [out_features, in_features]
        output_dir: Base output directory
        scales: Per-channel quantization scales (optional)

    Returns:
        BSR data dictionary
    """
    print(f"\n{'='*60}")
    print(f"Exporting {name} with 14×14 blocks")
    print(f"{'='*60}")
    print(f"  Original shape: {weight.shape}")

    # Build BSR with quantization if scales provided
    quantize = scales is not None
    bsr_data = build_bsr_14x14(weight, quantize=quantize, scale=scales)

    print(f"  Padded shape:   {bsr_data['padded_shape']}")
    print(f"  Block layout:   {bsr_data['num_block_rows']} × {bsr_data['num_block_cols']} blocks")
    print(f"  Non-zero:       {bsr_data['num_blocks']} blocks ({bsr_data['density']*100:.1f}% dense)")
    print(f"  Sparsity:       {bsr_data['sparsity_pct']:.1f}%")

    # Create layer output directory
    layer_dir = os.path.join(output_dir, name)
    os.makedirs(layer_dir, exist_ok=True)

    # Save binary weights
    weights_path = os.path.join(layer_dir, 'weights.bsr')
    if quantize:
        save_bsr_binary_int8(bsr_data, weights_path)
    else:
        # Save FP32 for debugging (not used by hardware)
        data = bsr_data['data'].astype(np.float32)
        with open(weights_path, 'wb') as f:
            for block in data:
                f.write(block.flatten('C').tobytes())
        print(f"  Saved FP32 binary: {weights_path} ({os.path.getsize(weights_path)} bytes)")

    # Save indices as numpy arrays (for easy loading)
    np.save(os.path.join(layer_dir, 'row_ptr.npy'), bsr_data['indptr'])
    np.save(os.path.join(layer_dir, 'col_idx.npy'), bsr_data['indices'])
    print(f"  Saved row_ptr.npy ({len(bsr_data['indptr'])} entries)")
    print(f"  Saved col_idx.npy ({len(bsr_data['indices'])} entries)")

    # Save metadata
    meta_path = os.path.join(layer_dir, 'weights.meta.json')
    save_bsr_metadata(bsr_data, meta_path, name)

    return bsr_data


# =============================================================================
# Quantization Helpers
# =============================================================================
def load_quantization_scales(int8_dir: str, layer_name: str) -> Optional[np.ndarray]:
    """
    Load per-channel quantization scales from INT8 export.

    Expected file: {int8_dir}/{layer_name}_weight_scales.npy
    """
    scales_path = os.path.join(int8_dir, f'{layer_name}_weight_scales.npy')
    if os.path.exists(scales_path):
        scales = np.load(scales_path)
        print(f"  Loaded scales from {scales_path} (shape: {scales.shape})")
        return scales
    else:
        print(f"  Warning: No scales found at {scales_path}")
        return None


# =============================================================================
# Direct INT8 Export (from pre-quantized .npy files)
# =============================================================================
def build_bsr_14x14_int8_direct(
    weight_int8: np.ndarray,
    threshold: float = 1e-10
) -> Dict:
    """
    Convert pre-quantized INT8 weight matrix to 14×14 BSR format.

    This is used when weights are already quantized (from data/int8/*.npy).
    No additional quantization is performed - blocks are extracted directly.

    Args:
        weight_int8: INT8 weight matrix [out_features, in_features]
        threshold: L1 norm threshold (absolute sum) for zero blocks

    Returns:
        BSR dictionary with INT8 block data
    """
    original_shape = weight_int8.shape
    height, width = original_shape

    # Pad to 14×14 alignment
    pad_h = (BLOCK_H - height % BLOCK_H) % BLOCK_H
    pad_w = (BLOCK_W - width % BLOCK_W) % BLOCK_W

    if pad_h > 0 or pad_w > 0:
        weight_int8 = np.pad(
            weight_int8,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )

    padded_h, padded_w = weight_int8.shape
    num_block_rows = padded_h // BLOCK_H
    num_block_cols = padded_w // BLOCK_W

    blocks_list = []
    col_indices_list = []
    row_ptr = [0]

    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            r0 = block_row * BLOCK_H
            c0 = block_col * BLOCK_W
            block = weight_int8[r0:r0+BLOCK_H, c0:c0+BLOCK_W]

            # Use L1 norm (sum of absolute values) for INT8
            block_norm = np.abs(block.astype(np.int32)).sum()
            if block_norm > threshold:
                blocks_list.append(block.astype(np.int8))
                col_indices_list.append(block_col)

        row_ptr.append(len(blocks_list))

    num_blocks = len(blocks_list)
    if num_blocks > 0:
        data = np.stack(blocks_list, axis=0)
    else:
        data = np.zeros((0, BLOCK_H, BLOCK_W), dtype=np.int8)

    indices = np.array(col_indices_list, dtype=np.int32)
    indptr = np.array(row_ptr, dtype=np.int32)

    total_blocks = num_block_rows * num_block_cols
    density = num_blocks / total_blocks if total_blocks > 0 else 0.0

    return {
        'data': data,
        'indices': indices,
        'indptr': indptr,
        'shape': original_shape,
        'padded_shape': (padded_h, padded_w),
        'blocksize': (BLOCK_H, BLOCK_W),
        'num_blocks': num_blocks,
        'num_block_rows': num_block_rows,
        'num_block_cols': num_block_cols,
        'density': density,
        'sparsity_pct': (1.0 - density) * 100.0,
    }


def export_int8_layer_14x14(
    name: str,
    weight_int8: np.ndarray,
    output_dir: str
) -> Dict:
    """
    Export a single layer from pre-quantized INT8 weights.
    """
    print(f"\n{'='*60}")
    print(f"Exporting {name} (INT8) with 14×14 blocks")
    print(f"{'='*60}")
    print(f"  Original shape: {weight_int8.shape}")
    print(f"  Weight range:   [{weight_int8.min()}, {weight_int8.max()}]")

    bsr_data = build_bsr_14x14_int8_direct(weight_int8)

    print(f"  Padded shape:   {bsr_data['padded_shape']}")
    print(f"  Block layout:   {bsr_data['num_block_rows']} × {bsr_data['num_block_cols']} blocks")
    print(f"  Non-zero:       {bsr_data['num_blocks']} blocks ({bsr_data['density']*100:.1f}% dense)")
    print(f"  Sparsity:       {bsr_data['sparsity_pct']:.1f}%")

    layer_dir = os.path.join(output_dir, name)
    os.makedirs(layer_dir, exist_ok=True)

    # Save binary INT8 weights
    save_bsr_binary_int8(bsr_data, os.path.join(layer_dir, 'weights.bsr'))

    # Save indices
    np.save(os.path.join(layer_dir, 'row_ptr.npy'), bsr_data['indptr'])
    np.save(os.path.join(layer_dir, 'col_idx.npy'), bsr_data['indices'])
    print(f"  Saved row_ptr.npy ({len(bsr_data['indptr'])} entries)")
    print(f"  Saved col_idx.npy ({len(bsr_data['indices'])} entries)")

    # Save metadata
    save_bsr_metadata(bsr_data, os.path.join(layer_dir, 'weights.meta.json'), name)

    return bsr_data


def export_from_int8_dir(int8_dir: str, output_dir: str) -> Dict:
    """
    Export all layers directly from pre-quantized INT8 .npy files.

    This is the preferred path for production - uses already-quantized weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("BSR Export from INT8 Weights (14×14 blocks)")
    print("="*70)
    print(f"Source:  {int8_dir}")
    print(f"Output:  {output_dir}")

    # Discover layers
    layers = ['conv1', 'conv2', 'fc1', 'fc2']
    layer_stats = []
    total_blocks = 0
    total_nonzero = 0

    for name in layers:
        weight_path = os.path.join(int8_dir, f'{name}_weight_int8.npy')
        if not os.path.exists(weight_path):
            print(f"  Skipping {name}: {weight_path} not found")
            continue

        weight_int8 = np.load(weight_path)

        # For conv layers, the weights may be 4D - flatten to 2D
        if weight_int8.ndim == 4:
            out_c, in_c, kH, kW = weight_int8.shape
            print(f"  Reshaping conv {name}: {weight_int8.shape} → ({out_c}, {in_c*kH*kW})")
            weight_int8 = weight_int8.reshape(out_c, -1)

        bsr = export_int8_layer_14x14(name, weight_int8, output_dir)

        layer_stats.append({
            'name': name,
            'original_shape': list(bsr['shape']),
            'padded_shape': list(bsr['padded_shape']),
            'blocksize': [BLOCK_H, BLOCK_W],
            'num_blocks': bsr['num_blocks'],
            'total_blocks': bsr['num_block_rows'] * bsr['num_block_cols'],
            'density': bsr['density'],
            'sparsity_pct': bsr['sparsity_pct'],
        })

        total_blocks += bsr['num_block_rows'] * bsr['num_block_cols']
        total_nonzero += bsr['num_blocks']

    # Summary
    summary = {
        'model': 'MNIST CNN INT8 (14×14 BSR)',
        'hardware_block_size': BLOCK_SIZE,
        'source': 'int8_quantized',
        'total_blocks': total_blocks,
        'nonzero_blocks': total_nonzero,
        'overall_density': total_nonzero / total_blocks if total_blocks > 0 else 0.0,
        'overall_sparsity_pct': (1.0 - total_nonzero / total_blocks) * 100 if total_blocks > 0 else 0.0,
        'layers': layer_stats,
    }

    summary_path = os.path.join(output_dir, 'model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("Export Summary (INT8)")
    print("="*70)
    print(f"Block size:     {BLOCK_H}×{BLOCK_W}")
    print(f"Total blocks:   {total_blocks}")
    print(f"Non-zero:       {total_nonzero}")
    print(f"Sparsity:       {summary['overall_sparsity_pct']:.1f}%")
    print(f"Output:         {output_dir}")
    print("="*70)

    return summary


# =============================================================================
# Full Model Export
# =============================================================================
def export_model_14x14(
    model: nn.Module,
    output_dir: str,
    int8_dir: Optional[str] = None
) -> Dict:
    """
    Export all layers of model in 14×14 BSR format.

    Args:
        model: PyTorch model
        output_dir: Output directory for BSR files
        int8_dir: Directory with INT8 quantization data (optional)

    Returns:
        Summary dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("BSR Export for Zynq 14×14 Systolic Array")
    print("="*70)
    print(f"Block size: {BLOCK_H}×{BLOCK_W} (hardware-defined)")
    print(f"Output directory: {output_dir}")

    layer_stats = []
    total_blocks = 0
    total_nonzero = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data.cpu().numpy()

            # Load quantization scales if available
            scales = None
            if int8_dir:
                scales = load_quantization_scales(int8_dir, name)

            bsr = export_layer_14x14(name, weight, output_dir, scales)

            layer_stats.append({
                'name': name,
                'original_shape': list(bsr['shape']),
                'padded_shape': list(bsr['padded_shape']),
                'blocksize': [BLOCK_H, BLOCK_W],
                'num_blocks': bsr['num_blocks'],
                'total_blocks': bsr['num_block_rows'] * bsr['num_block_cols'],
                'density': bsr['density'],
                'sparsity_pct': bsr['sparsity_pct'],
            })

            total_blocks += bsr['num_block_rows'] * bsr['num_block_cols']
            total_nonzero += bsr['num_blocks']

        elif isinstance(module, nn.Conv2d):
            # For conv layers, reshape to 2D: [out_channels, in_channels * kH * kW]
            weight = module.weight.data.cpu().numpy()
            out_c, in_c, kH, kW = weight.shape
            weight_2d = weight.reshape(out_c, -1)

            print(f"\n  Conv {name}: {weight.shape} → {weight_2d.shape}")

            scales = None
            if int8_dir:
                scales = load_quantization_scales(int8_dir, name)

            bsr = export_layer_14x14(name, weight_2d, output_dir, scales)

            layer_stats.append({
                'name': name,
                'original_shape': list(weight.shape),
                'flattened_shape': list(weight_2d.shape),
                'padded_shape': list(bsr['padded_shape']),
                'blocksize': [BLOCK_H, BLOCK_W],
                'num_blocks': bsr['num_blocks'],
                'total_blocks': bsr['num_block_rows'] * bsr['num_block_cols'],
                'density': bsr['density'],
                'sparsity_pct': bsr['sparsity_pct'],
            })

            total_blocks += bsr['num_block_rows'] * bsr['num_block_cols']
            total_nonzero += bsr['num_blocks']

    # Save summary
    summary = {
        'model': 'MNIST CNN (14×14 BSR)',
        'hardware_block_size': BLOCK_SIZE,
        'total_blocks': total_blocks,
        'nonzero_blocks': total_nonzero,
        'overall_density': total_nonzero / total_blocks if total_blocks > 0 else 0.0,
        'overall_sparsity_pct': (1.0 - total_nonzero / total_blocks) * 100 if total_blocks > 0 else 0.0,
        'layers': layer_stats,
    }

    summary_path = os.path.join(output_dir, 'model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("Export Summary")
    print("="*70)
    print(f"Block size:     {BLOCK_H}×{BLOCK_W}")
    print(f"Total blocks:   {total_blocks}")
    print(f"Non-zero:       {total_nonzero}")
    print(f"Sparsity:       {summary['overall_sparsity_pct']:.1f}%")
    print(f"Output:         {output_dir}")
    print("="*70)

    return summary


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Export model weights in 14×14 BSR format for Zynq accelerator'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../../data/checkpoints/mnist_fp32.pt',
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/bsr_export_14x14',
        help='Output directory'
    )
    parser.add_argument(
        '--int8-dir',
        type=str,
        default='../../data/int8',
        help='Directory with pre-quantized INT8 weights'
    )
    parser.add_argument(
        '--from-int8',
        action='store_true',
        help='Export directly from INT8 .npy files (preferred for hardware)'
    )
    parser.add_argument(
        '--quantized',
        action='store_true',
        help='Export quantized INT8 weights from FP32 checkpoint'
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output)
    int8_dir = os.path.join(script_dir, args.int8_dir)

    # Mode 1: Direct INT8 export (preferred)
    if args.from_int8:
        print("\n🔧 Mode: Direct INT8 Export")
        if not os.path.exists(int8_dir):
            print(f"Error: INT8 directory not found: {int8_dir}")
            return None
        summary = export_from_int8_dir(int8_dir, output_dir)
        print("\n✅ INT8 Export complete!")
        return summary

    # Mode 2: Export from checkpoint
    checkpoint_path = os.path.join(script_dir, args.checkpoint)

    # Create model
    model = MNISTNet().eval()

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized weights for demonstration")

    # Export
    scales_dir = int8_dir if args.quantized else None
    summary = export_model_14x14(model, output_dir, scales_dir)

    print("\n✅ Export complete!")
    return summary


if __name__ == '__main__':
    main()