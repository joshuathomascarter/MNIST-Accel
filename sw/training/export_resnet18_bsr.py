"""
ResNet-18 BSR Export for ACCEL-v1 Hardware (14×14 Systolic Array)
=================================================================

Exports ResNet-18 weights in hardware-ready BSR format with metadata.
Uses 14×14 block size to match the systolic array dimensions (PYNQ-Z2).

REPLACES: sw/training/export_bsr.py (MNIST version)

Output format:
- *.bsr: Binary payload with non-zero 14×14 blocks
- *.meta.json: Metadata with row_ptr, col_idx, block dimensions

ResNet-18 Layer Structure:
- conv1: 64 output channels, 7×7 kernel, stride 2
- layer1: 2 BasicBlocks, 64 channels
- layer2: 2 BasicBlocks, 128 channels (with downsample)
- layer3: 2 BasicBlocks, 256 channels (with downsample)
- layer4: 2 BasicBlocks, 512 channels (with downsample)
- fc: 512 → 1000 (ImageNet) or custom num_classes

Author: ACCEL-v1 Team
Date: December 2024
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, List, Optional
import struct

# ----------------------------
# Configuration
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(ROOT, "data", "resnet18_bsr_export")

# Block size for 14×14 systolic array (PYNQ-Z2 DSP budget)
BLOCK_SIZE = 14
BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE  # 196 elements per block


# ----------------------------
# ResNet-18 Layer Configuration
# ----------------------------
def get_resnet18_layer_config() -> Dict[str, Dict]:
    """
    Return block size and sparsity configuration for each ResNet-18 layer.
    
    FC layers use 14×14 blocks (matches systolic array for PYNQ-Z2)
    Conv layers use 4×4 blocks (better for 3×3 kernels)
    
    Returns:
        Dictionary mapping layer names to their configurations
    """
    return {
        # Initial conv layer (7×7 kernel)
        "conv1": {"block_size": (4, 4), "min_keep": 0.50, "type": "conv"},
        
        # Layer 1 (64 channels)
        "layer1.0.conv1": {"block_size": (4, 4), "min_keep": 0.30, "type": "conv"},
        "layer1.0.conv2": {"block_size": (4, 4), "min_keep": 0.30, "type": "conv"},
        "layer1.1.conv1": {"block_size": (4, 4), "min_keep": 0.30, "type": "conv"},
        "layer1.1.conv2": {"block_size": (4, 4), "min_keep": 0.30, "type": "conv"},
        
        # Layer 2 (128 channels, has downsample)
        "layer2.0.conv1": {"block_size": (4, 4), "min_keep": 0.20, "type": "conv"},
        "layer2.0.conv2": {"block_size": (4, 4), "min_keep": 0.20, "type": "conv"},
        "layer2.0.downsample.0": {"block_size": (4, 4), "min_keep": 0.50, "type": "conv"},
        "layer2.1.conv1": {"block_size": (4, 4), "min_keep": 0.20, "type": "conv"},
        "layer2.1.conv2": {"block_size": (4, 4), "min_keep": 0.20, "type": "conv"},
        
        # Layer 3 (256 channels, has downsample)
        "layer3.0.conv1": {"block_size": (4, 4), "min_keep": 0.15, "type": "conv"},
        "layer3.0.conv2": {"block_size": (4, 4), "min_keep": 0.15, "type": "conv"},
        "layer3.0.downsample.0": {"block_size": (4, 4), "min_keep": 0.50, "type": "conv"},
        "layer3.1.conv1": {"block_size": (4, 4), "min_keep": 0.15, "type": "conv"},
        "layer3.1.conv2": {"block_size": (4, 4), "min_keep": 0.15, "type": "conv"},
        
        # Layer 4 (512 channels, has downsample)
        "layer4.0.conv1": {"block_size": (4, 4), "min_keep": 0.10, "type": "conv"},
        "layer4.0.conv2": {"block_size": (4, 4), "min_keep": 0.10, "type": "conv"},
        "layer4.0.downsample.0": {"block_size": (4, 4), "min_keep": 0.50, "type": "conv"},
        "layer4.1.conv1": {"block_size": (4, 4), "min_keep": 0.10, "type": "conv"},
        "layer4.1.conv2": {"block_size": (4, 4), "min_keep": 0.10, "type": "conv"},
        
        # Final FC layer (512 → num_classes) - Uses 14×14 blocks!
        "fc": {"block_size": (14, 14), "min_keep": 0.05, "type": "linear"},
    }


# ----------------------------
# BSR Extraction Functions
# ----------------------------
def build_bsr_from_dense(weight: np.ndarray, block_h: int, block_w: int, 
                          threshold: float = 1e-10) -> Dict:
    """
    Build BSR format directly from dense weight matrix.
    Skips zero blocks automatically.

    Args:
        weight: Dense weight matrix [out_features, in_features] or flattened conv
        block_h: Block height (14 for FC layers)
        block_w: Block width (14 for FC layers)
        threshold: Blocks with L2 norm below this are considered zero

    Returns:
        BSR format dictionary with metadata
    """
    height, width = weight.shape

    # Pad to block size
    pad_h = (block_h - height % block_h) % block_h
    pad_w = (block_w - width % block_w) % block_w
    if pad_h > 0 or pad_w > 0:
        weight = np.pad(weight, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

    padded_h, padded_w = weight.shape
    num_block_rows = padded_h // block_h
    num_block_cols = padded_w // block_w

    # Extract all blocks and compute norms
    blocks_list = []
    col_indices_list = []
    row_ptr = [0]

    num_nonzero_blocks = 0

    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            r_start = block_row * block_h
            r_end = r_start + block_h
            c_start = block_col * block_w
            c_end = c_start + block_w

            block = weight[r_start:r_end, c_start:c_end]

            # Check if block is non-zero
            block_norm = np.linalg.norm(block)
            if block_norm > threshold:
                blocks_list.append(block)
                col_indices_list.append(block_col)
                num_nonzero_blocks += 1

        row_ptr.append(num_nonzero_blocks)

    # Convert to arrays
    if len(blocks_list) > 0:
        data = np.stack(blocks_list, axis=0)
    else:
        data = np.zeros((0, block_h, block_w), dtype=weight.dtype)

    indices = np.array(col_indices_list, dtype=np.int32)
    indptr = np.array(row_ptr, dtype=np.int32)

    total_blocks = num_block_rows * num_block_cols
    density = num_nonzero_blocks / total_blocks if total_blocks > 0 else 0.0

    return {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "shape": (height, width),
        "padded_shape": (padded_h, padded_w),
        "blocksize": (block_h, block_w),
        "num_blocks": num_nonzero_blocks,
        "num_block_rows": num_block_rows,
        "num_block_cols": num_block_cols,
        "density": density,
        "sparsity_pct": 100.0 * (1.0 - density),
    }


def quantize_per_channel(weight: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-channel INT8 quantization.
    
    Args:
        weight: FP32 weight matrix
        axis: Channel axis (0 for output channels)
    
    Returns:
        Tuple of (INT8 quantized weights, per-channel scales)
    """
    axes_to_reduce = tuple(i for i in range(len(weight.shape)) if i != axis)
    maxabs = np.max(np.abs(weight), axis=axes_to_reduce, keepdims=True)
    scales = np.maximum(maxabs / 127.0, 1e-12)
    
    q = np.rint(weight / scales)
    q = np.clip(q, -128, 127).astype(np.int8)
    
    scales_flat = np.squeeze(scales, axis=axes_to_reduce).astype(np.float32)
    return q, scales_flat


def save_bsr_binary_int8(bsr: Dict, scales: np.ndarray, filepath: str):
    """
    Save BSR matrix in hardware-optimized binary format.
    
    Binary layout (matches C++ serialize_for_hardware):
    - Header: 3 × uint32 (nnz_blocks, num_block_rows, num_block_cols)
    - row_ptr: (num_block_rows + 1) × uint16
    - col_idx: nnz_blocks × uint16
    - data: nnz_blocks × BLOCK_ELEMENTS × int8
    """
    block_h, block_w = bsr["blocksize"]
    block_elements = block_h * block_w
    
    with open(filepath, "wb") as f:
        # Write header
        f.write(struct.pack("<I", bsr["num_blocks"]))
        f.write(struct.pack("<I", bsr["num_block_rows"]))
        f.write(struct.pack("<I", bsr["num_block_cols"]))
        
        # Write row_ptr as uint16
        for ptr in bsr["indptr"]:
            f.write(struct.pack("<H", ptr))
        
        # Write col_idx as uint16
        for idx in bsr["indices"]:
            f.write(struct.pack("<H", idx))
        
        # Quantize blocks and write as int8
        for i, block in enumerate(bsr["data"]):
            # Get per-row scales for this block
            block_row = 0
            for br in range(len(bsr["indptr"]) - 1):
                if bsr["indptr"][br] <= i < bsr["indptr"][br + 1]:
                    block_row = br
                    break
            
            # Quantize using per-channel scale
            start_channel = block_row * block_h
            end_channel = min(start_channel + block_h, len(scales))
            
            block_int8 = np.zeros((block_h, block_w), dtype=np.int8)
            for r in range(block_h):
                channel_idx = start_channel + r
                if channel_idx < len(scales):
                    scale = scales[channel_idx]
                    block_int8[r, :] = np.clip(np.rint(block[r, :] / scale), -128, 127).astype(np.int8)
            
            # Write flattened block (row-major)
            f.write(block_int8.tobytes())


def save_bsr_metadata(bsr: Dict, filepath: str, layer_name: str = ""):
    """Save BSR metadata as JSON for debugging and verification."""
    metadata = {
        "layer_name": layer_name,
        "original_shape": list(bsr["shape"]),
        "padded_shape": list(bsr["padded_shape"]),
        "blocksize": list(bsr["blocksize"]),
        "num_blocks": int(bsr["num_blocks"]),
        "num_block_rows": int(bsr["num_block_rows"]),
        "num_block_cols": int(bsr["num_block_cols"]),
        "sparsity_pct": float(bsr["sparsity_pct"]),
        "row_ptr": bsr["indptr"].tolist(),
        "col_idx": bsr["indices"].tolist(),
    }
    
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


# ----------------------------
# ResNet-18 Export Functions
# ----------------------------
def export_resnet18_weights(
    model: nn.Module,
    output_dir: str,
    sparsity_target: float = 0.90,
    num_classes: int = 1000,
) -> Dict[str, Dict]:
    """
    Export all ResNet-18 weights in BSR format.
    
    Args:
        model: ResNet-18 model (pretrained or trained)
        output_dir: Directory to save exported weights
        sparsity_target: Target sparsity percentage (0-1)
        num_classes: Number of output classes
    
    Returns:
        Dictionary with export statistics for each layer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    layer_configs = get_resnet18_layer_config()
    export_stats = {}
    
    print("=" * 70)
    print("Exporting ResNet-18 Weights for 14×14 Systolic Array")
    print("=" * 70)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach().cpu().numpy()
            
            # Get configuration for this layer
            config = layer_configs.get(name, {"block_size": (14, 14), "min_keep": 0.10, "type": "linear"})
            block_h, block_w = config["block_size"]
            
            # Reshape for BSR packing
            if isinstance(module, nn.Conv2d):
                # Conv: [out_ch, in_ch, kH, kW] → [out_ch, in_ch * kH * kW]
                out_ch = weight.shape[0]
                weight_2d = weight.reshape(out_ch, -1)
                layer_type = "conv"
            else:
                # Linear: already [out_features, in_features]
                weight_2d = weight
                layer_type = "linear"
            
            # Build BSR
            bsr = build_bsr_from_dense(weight_2d, block_h, block_w)
            
            # Quantize
            scales = np.max(np.abs(weight_2d), axis=1, keepdims=True) / 127.0
            scales = np.maximum(scales, 1e-12).flatten()
            
            # Create layer directory
            layer_dir = os.path.join(output_dir, name.replace(".", "_"))
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save files
            save_bsr_binary_int8(bsr, scales, os.path.join(layer_dir, "weights_int8.bsr"))
            save_bsr_metadata(bsr, os.path.join(layer_dir, "weights.meta.json"), layer_name=name)
            np.save(os.path.join(layer_dir, "scales.npy"), scales)
            np.save(os.path.join(layer_dir, "row_ptr.npy"), bsr["indptr"])
            np.save(os.path.join(layer_dir, "col_idx.npy"), bsr["indices"])
            
            # Save bias if present
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
                np.save(os.path.join(layer_dir, "bias.npy"), bias)
            
            # Print stats
            print(f"\n{name}:")
            print(f"  Shape: {weight_2d.shape} ({layer_type})")
            print(f"  Block size: {block_h}×{block_w}")
            print(f"  Blocks: {bsr['num_blocks']} / {bsr['num_block_rows'] * bsr['num_block_cols']}")
            print(f"  Sparsity: {bsr['sparsity_pct']:.1f}%")
            
            export_stats[name] = {
                "shape": list(weight_2d.shape),
                "block_size": [block_h, block_w],
                "num_blocks": bsr["num_blocks"],
                "sparsity_pct": bsr["sparsity_pct"],
                "type": layer_type,
            }
    
    # Save summary
    summary = {
        "model": "resnet18",
        "num_classes": num_classes,
        "systolic_array_size": [14, 14],
        "layers": export_stats,
    }
    
    with open(os.path.join(output_dir, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Export complete! Output: {output_dir}")
    print("=" * 70)
    
    return export_stats


def load_pretrained_resnet18(num_classes: int = 1000) -> nn.Module:
    """Load pretrained ResNet-18 from torchvision."""
    if num_classes == 1000:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
    return model


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export ResNet-18 weights in BSR format")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to trained checkpoint (uses pretrained if not specified)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for BSR files")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of output classes")
    parser.add_argument("--sparsity", type=float, default=0.90,
                        help="Target sparsity (0-1)")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading ResNet-18...")
    if args.checkpoint:
        model = models.resnet18(weights=None)
        if args.num_classes != 1000:
            model.fc = nn.Linear(512, args.num_classes)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        model = load_pretrained_resnet18(args.num_classes)
    
    model.eval()
    
    # Export weights
    export_resnet18_weights(
        model,
        args.output_dir,
        sparsity_target=args.sparsity,
        num_classes=args.num_classes,
    )
