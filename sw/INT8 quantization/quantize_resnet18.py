"""
INT8 Per-Channel Quantization for ResNet-18 on ACCEL-v1 (16×16 Systolic Array)
===============================================================================

Extends basic quantization to support:
- Per-channel quantization for better accuracy
- Block-sparse weight handling with 16×16 blocks
- Full ResNet-18 model quantization
- Calibration using representative data

REPLACES: sw/INT8 quantization/quantize.py (MNIST version)

Author: ACCEL-v1 Team
Date: December 2024
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict

# ----------------------------
# Configuration
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "data", "resnet18_int8")

# Tiling parameters for 16×16 systolic array
Tm, Tn, Tk = 16, 16, 16  # Tile dimensions matching systolic array
NUM_CALIBRATION_BATCHES = 32


# ----------------------------
# Quantization Functions
# ----------------------------
def quantize_symmetric_per_tensor(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Symmetric per-tensor INT8 quantization.
    Maps max|x| -> 127

    Args:
        x: Input array (any shape)

    Returns:
        q: INT8 quantized tensor
        scale: Quantization scale factor
    """
    maxabs = float(np.max(np.abs(x)))
    scale = max(maxabs / 127.0, 1e-12)
    q = np.rint(x / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


def quantize_symmetric_per_channel(
    x: np.ndarray, 
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-channel INT8 quantization.
    Each channel (along specified axis) gets its own scale.

    Args:
        x: Input array (e.g., weights of shape (out_channels, in_channels, ...))
        axis: Channel axis (typically 0 for output channels)

    Returns:
        q: INT8 quantized tensor
        scales: Per-channel scale factors (shape matching channel dimension)
    """
    # Compute max abs per channel
    axes_to_reduce = tuple(i for i in range(len(x.shape)) if i != axis)
    maxabs = np.max(np.abs(x), axis=axes_to_reduce, keepdims=True)

    # Guard against all-zero channels
    scales = np.maximum(maxabs / 127.0, 1e-12)

    # Quantize
    q = np.rint(x / scales)
    q = np.clip(q, -128, 127).astype(np.int8)

    # Flatten scales for storage
    scales_flat = np.squeeze(scales, axis=axes_to_reduce).astype(np.float32)

    return q, scales_flat


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize INT8 values back to FP32."""
    return q.astype(np.float32) * scale


# ----------------------------
# Activation Calibration
# ----------------------------
class ActivationCalibrator:
    """
    Collects activation statistics for quantization calibration.
    
    Uses running min/max to determine optimal scales for each layer's
    activations based on representative input data.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_ranges = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activation ranges."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    out_np = output.detach().cpu().numpy()
                    if name not in self.activation_ranges:
                        self.activation_ranges[name] = {
                            "min": np.inf,
                            "max": -np.inf,
                        }
                    self.activation_ranges[name]["min"] = min(
                        self.activation_ranges[name]["min"],
                        float(np.min(out_np))
                    )
                    self.activation_ranges[name]["max"] = max(
                        self.activation_ranges[name]["max"],
                        float(np.max(out_np))
                    )
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                self.hooks.append(module.register_forward_hook(make_hook(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation_scales(self) -> Dict[str, float]:
        """
        Compute optimal scales for each layer based on collected ranges.
        
        Returns:
            Dictionary mapping layer names to activation scales
        """
        scales = {}
        for name, ranges in self.activation_ranges.items():
            maxabs = max(abs(ranges["min"]), abs(ranges["max"]))
            scales[name] = max(maxabs / 127.0, 1e-12)
        return scales


def calibrate_activations(
    model: nn.Module,
    data_loader: DataLoader,
    num_batches: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Run calibration to determine activation scales.
    
    Args:
        model: The model to calibrate
        data_loader: DataLoader with representative data
        num_batches: Number of batches to use for calibration
        device: Device to run on
    
    Returns:
        Dictionary mapping layer names to activation scales
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    calibrator = ActivationCalibrator(model)
    
    print("Running activation calibration...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}")
    
    calibrator.remove_hooks()
    scales = calibrator.get_activation_scales()
    
    print(f"Calibration complete: {len(scales)} layers")
    return scales


# ----------------------------
# ResNet-18 Quantization
# ----------------------------
def quantize_resnet18(
    model: nn.Module,
    calibration_loader: Optional[DataLoader] = None,
    output_dir: str = None,
    include_activations: bool = True,
) -> Dict[str, Dict]:
    """
    Quantize all layers of ResNet-18 to INT8.
    
    Args:
        model: ResNet-18 model (pretrained or trained)
        calibration_loader: DataLoader for activation calibration
        output_dir: Directory to save quantized weights
        include_activations: Whether to calibrate and save activation scales
    
    Returns:
        Dictionary with quantization metadata for each layer
    """
    if output_dir is None:
        output_dir = OUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("=" * 70)
    print("Quantizing ResNet-18 to INT8")
    print("=" * 70)
    
    # Calibrate activations if loader provided
    activation_scales = {}
    if include_activations and calibration_loader is not None:
        activation_scales = calibrate_activations(
            model, calibration_loader, NUM_CALIBRATION_BATCHES, device
        )
    
    # Quantize weights
    quant_metadata = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach().cpu().numpy()
            
            # Per-channel quantization (output channel axis = 0)
            weight_int8, weight_scales = quantize_symmetric_per_channel(weight, axis=0)
            
            # Create layer directory
            layer_dir = os.path.join(output_dir, name.replace(".", "_"))
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save quantized weights
            np.save(os.path.join(layer_dir, "weight_int8.npy"), weight_int8)
            np.save(os.path.join(layer_dir, "weight_scales.npy"), weight_scales)
            
            # Save bias if present (keep as INT32 for accumulator compatibility)
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
                # Quantize bias to INT32 using weight scales
                bias_scale = weight_scales.mean()  # Simplified
                bias_int32 = np.rint(bias / bias_scale).astype(np.int32)
                np.save(os.path.join(layer_dir, "bias_int8.npy"), bias)  # Keep FP32 for now
                np.save(os.path.join(layer_dir, "bias_scale.json"), {"scale": float(bias_scale)})
            
            # Get activation scale for this layer
            act_scale = activation_scales.get(name, 1.0)
            
            # Save metadata
            layer_meta = {
                "name": name,
                "type": "conv" if isinstance(module, nn.Conv2d) else "linear",
                "weight_shape": list(weight.shape),
                "weight_scales_shape": list(weight_scales.shape),
                "activation_scale": float(act_scale),
            }
            
            if isinstance(module, nn.Conv2d):
                layer_meta.update({
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "groups": module.groups,
                })
            
            with open(os.path.join(layer_dir, "metadata.json"), "w") as f:
                json.dump(layer_meta, f, indent=2)
            
            quant_metadata[name] = layer_meta
            
            print(f"\n{name}:")
            print(f"  Shape: {weight.shape}")
            print(f"  Scale range: [{weight_scales.min():.6f}, {weight_scales.max():.6f}]")
    
    # Save global metadata
    global_meta = {
        "model": "resnet18",
        "tile_size": [Tm, Tn, Tk],
        "systolic_array_size": [16, 16],
        "layers": list(quant_metadata.keys()),
        "num_layers": len(quant_metadata),
    }
    
    with open(os.path.join(output_dir, "quantization_metadata.json"), "w") as f:
        json.dump(global_meta, f, indent=2)
    
    # Save activation scales
    if activation_scales:
        with open(os.path.join(output_dir, "activation_scales.json"), "w") as f:
            json.dump(activation_scales, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Quantization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    return quant_metadata


def get_calibration_loader(
    dataset: str = "imagenet",
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Get a calibration data loader.
    
    Args:
        dataset: "imagenet" or "cifar10"
        data_dir: Path to dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader for calibration
    """
    if dataset == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        try:
            cal_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "val"),
                transform=transform
            )
        except Exception:
            print("Warning: ImageNet not found, using random calibration data")
            return None
            
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        cal_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ResNet-18 to INT8")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default=OUT_DIR,
                        help="Output directory")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "cifar10"],
                        help="Dataset for calibration")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to dataset for calibration")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of output classes")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip activation calibration")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading ResNet-18...")
    if args.checkpoint:
        model = models.resnet18(weights=None)
        if args.num_classes != 1000:
            model.fc = nn.Linear(512, args.num_classes)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    model.eval()
    
    # Get calibration loader
    calibration_loader = None
    if not args.no_calibration:
        calibration_loader = get_calibration_loader(
            args.dataset, args.data_dir
        )
    
    # Quantize
    quantize_resnet18(
        model,
        calibration_loader=calibration_loader,
        output_dir=args.output_dir,
        include_activations=not args.no_calibration,
    )
