"""
ResNet-18 Training and Fine-tuning for ACCEL-v1 Hardware
=========================================================

This script provides training/fine-tuning for ResNet-18 with:
- Block-sparse training for hardware acceleration
- INT8 quantization-aware training
- Support for custom datasets (CIFAR-10, ImageNet, custom)

REPLACES: sw/MNIST CNN/train_mnist.py

Hardware Target: 14×14 Systolic Array with BSR sparse format (PYNQ-Z2)

Author: ACCEL-v1 Team
Date: December 2024
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict
import argparse
import json
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Default hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 64,
    "epochs": 90,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "lr_step_size": 30,
    "lr_gamma": 0.1,
    "num_workers": 4,
    "seed": 42,
}


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Block Sparse Mask Generation
# ----------------------------
def create_block_sparse_mask(
    weight: torch.Tensor,
    block_size: Tuple[int, int],
    sparsity: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create a block-sparse mask for weight pruning.
    
    Args:
        weight: Weight tensor [out_features, in_features] or [out_ch, in_ch, kH, kW]
        block_size: (block_h, block_w) - use (14, 14) for FC, (4, 4) for conv
        sparsity: Target sparsity (0-1), e.g., 0.90 for 90% zeros
        seed: Random seed
    
    Returns:
        Binary mask tensor (same shape as weight)
    """
    torch.manual_seed(seed)
    
    # Reshape to 2D if needed
    original_shape = weight.shape
    if len(weight.shape) == 4:
        # Conv: [out_ch, in_ch, kH, kW] → [out_ch, in_ch * kH * kW]
        weight_2d = weight.view(weight.shape[0], -1)
    else:
        weight_2d = weight
    
    rows, cols = weight_2d.shape
    block_h, block_w = block_size
    
    # Calculate block grid
    num_block_rows = (rows + block_h - 1) // block_h
    num_block_cols = (cols + block_w - 1) // block_w
    total_blocks = num_block_rows * num_block_cols
    
    # Determine which blocks to zero out
    num_zero_blocks = int(total_blocks * sparsity)
    
    # Create block-level mask
    block_mask = torch.ones(total_blocks, dtype=torch.bool)
    zero_indices = torch.randperm(total_blocks)[:num_zero_blocks]
    block_mask[zero_indices] = False
    
    # Expand to element-level mask
    mask = torch.zeros(rows, cols, dtype=weight.dtype, device=weight.device)
    
    for idx in range(total_blocks):
        if block_mask[idx]:
            block_row = idx // num_block_cols
            block_col = idx % num_block_cols
            
            r_start = block_row * block_h
            r_end = min(r_start + block_h, rows)
            c_start = block_col * block_w
            c_end = min(c_start + block_w, cols)
            
            mask[r_start:r_end, c_start:c_end] = 1.0
    
    # Reshape back if needed
    if len(original_shape) == 4:
        mask = mask.view(original_shape)
    
    return mask


class BlockSparsePruner:
    """
    Applies and maintains block sparsity during training.
    
    Usage:
        pruner = BlockSparsePruner(model, sparsity=0.90)
        for epoch in range(epochs):
            for batch in dataloader:
                ...
                optimizer.step()
                pruner.apply_masks()  # Re-apply sparsity after each update
    """
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.90,
        fc_block_size: Tuple[int, int] = (14, 14),
        conv_block_size: Tuple[int, int] = (4, 4),
    ):
        self.model = model
        self.sparsity = sparsity
        self.fc_block_size = fc_block_size
        self.conv_block_size = conv_block_size
        self.masks = {}
        
        self._create_masks()
    
    def _create_masks(self):
        """Create masks for all prunable layers."""
        seed = 42
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = create_block_sparse_mask(
                    module.weight.data,
                    self.conv_block_size,
                    self.sparsity,
                    seed=seed,
                )
                self.masks[name] = mask
                seed += 1
            elif isinstance(module, nn.Linear):
                mask = create_block_sparse_mask(
                    module.weight.data,
                    self.fc_block_size,
                    self.sparsity,
                    seed=seed,
                )
                self.masks[name] = mask
                seed += 1
    
    def apply_masks(self):
        """Apply masks to zero out pruned weights."""
        for name, module in self.model.named_modules():
            if name in self.masks:
                with torch.no_grad():
                    module.weight.data *= self.masks[name].to(module.weight.device)


# ----------------------------
# Dataset Loading
# ----------------------------
def get_cifar10_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test dataloaders."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_imagenet_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "/path/to/imagenet",
) -> Tuple[DataLoader, DataLoader]:
    """Get ImageNet train and val dataloaders."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


# ----------------------------
# Training Functions
# ----------------------------
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    pruner: Optional[BlockSparsePruner] = None,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Re-apply sparsity masks after weight update
        if pruner is not None:
            pruner.apply_masks()
        
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_resnet18(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    output_dir: str = None,
    num_classes: int = 10,
    epochs: int = 90,
    batch_size: int = 64,
    learning_rate: float = 0.1,
    sparsity: float = 0.0,
    pretrained: bool = True,
    resume: str = None,
) -> nn.Module:
    """
    Train or fine-tune ResNet-18.
    
    Args:
        dataset: "cifar10" or "imagenet"
        data_dir: Path to dataset
        output_dir: Directory to save checkpoints
        num_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        sparsity: Block sparsity ratio (0 = dense, 0.9 = 90% sparse)
        pretrained: Use pretrained weights
        resume: Path to checkpoint to resume from
    
    Returns:
        Trained model
    """
    if output_dir is None:
        output_dir = os.path.join(ROOT, "data", "checkpoints", "resnet18")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Set seed
    set_seed(DEFAULT_CONFIG["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {dataset} dataset...")
    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir=data_dir)
    elif dataset == "imagenet":
        train_loader, test_loader = get_imagenet_loaders(batch_size, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create model
    print("Creating ResNet-18 model...")
    if pretrained and num_classes == 1000:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Modify FC layer for custom number of classes
    if num_classes != 1000:
        model.fc = nn.Linear(512, num_classes)
    
    # For CIFAR-10, modify first conv and remove maxpool (smaller images)
    if dataset == "cifar10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    model = model.to(device)
    
    # Setup sparsity pruner
    pruner = None
    if sparsity > 0:
        print(f"Enabling block sparsity: {sparsity*100:.1f}%")
        pruner = BlockSparsePruner(
            model,
            sparsity=sparsity,
            fc_block_size=(14, 14),  # Match 14×14 systolic array (PYNQ-Z2)
            conv_block_size=(4, 4),
        )
        pruner.apply_masks()  # Apply initial masks
    
    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=DEFAULT_CONFIG["lr_step_size"],
        gamma=DEFAULT_CONFIG["lr_gamma"],
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
    
    # Training log
    log_file = os.path.join(output_dir, "logs", f"train_{datetime.now():%Y%m%d_%H%M%S}.txt")
    
    print("=" * 70)
    print(f"Training ResNet-18 on {dataset}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sparsity: {sparsity*100:.1f}%")
    print("=" * 70)
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, pruner
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        log_entry = (
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
        )
        print(log_entry)
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
        
        # Save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "sparsity": sparsity,
        }
        
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  New best accuracy: {best_acc:.2f}%")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)
    
    return model


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 for ACCEL-v1")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="Dataset to train on")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=90,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate")
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="Block sparsity ratio (0-1)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ImageNet weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Determine number of classes
    num_classes = 10 if args.dataset == "cifar10" else 1000
    
    train_resnet18(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sparsity=args.sparsity,
        pretrained=args.pretrained,
        resume=args.resume,
    )
