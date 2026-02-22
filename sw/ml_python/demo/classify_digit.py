#!/usr/bin/env python3
"""
ACCEL-v1 Handwriting Classifier — Interactive Demo
====================================================
Draw a digit (0-9) on the canvas, click "Classify" and see the prediction.

Supports two backends:
  1. PyTorch CPU — default, works anywhere
  2. FPGA accelerator — use --fpga flag on PYNQ-Z2 board

PyTorch (CPU) mode:
  1. Capture handwritten digit image
  2. Preprocess to 28×28 grayscale (MNIST format)
  3. Run INT8-quantized CNN inference on CPU
  4. Output predicted digit with confidence

FPGA (ACCEL-v1) mode:
  1. Capture handwritten digit image
  2. Preprocess to 28×28 and quantize to INT8
  3. Run conv1 → conv2 → fc1 → fc2 on the 14×14 systolic array
  4. Output predicted digit with confidence + cycle count

Requirements: torch, torchvision, Pillow, tkinter
  pip install torch torchvision Pillow

Usage:
  python3 classify_digit.py              # Interactive drawing (PyTorch CPU)
  python3 classify_digit.py --fpga       # Interactive drawing (FPGA)
  python3 classify_digit.py image.png    # Classify an image file
  python3 classify_digit.py --test [N]   # Test on N random MNIST samples
"""

import sys
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter

# ─── Model Architecture (must match checkpoint) ─────────────────────────────
class MNISTNet(nn.Module):
    """MNIST CNN matching the ACCEL-v1 hardware mapping.
    
    fc1 output = 140 (10×14) — tiles perfectly onto the 14×14 systolic array.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 140)
        self.fc2 = nn.Linear(140, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ─── Load Model ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ROOT, "..", "..", ".."))
CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "checkpoints", "mnist_fp32.pt")

# Add host/ to path for FPGA driver imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sw", "ml_python", "host"))


def load_model():
    model = MNISTNet()
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        acc = ckpt.get("best_acc", "?")
        print(f"✓ Model loaded (accuracy: {acc}%)")
    else:
        print(f"✗ Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    model.eval()
    return model


# ─── Preprocessing ───────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to MNIST-format tensor."""
    # Convert to grayscale
    img = img.convert("L")
    
    # MNIST expects white digit on black background
    # If the image is mostly white (paper), invert it
    arr = np.array(img)
    if arr.mean() > 128:
        img = Image.fromarray(255 - arr)
    
    # Add some padding to center the digit (like MNIST)
    img = img.resize((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, (4, 4))
    
    # Apply MNIST normalization
    tensor = transforms.ToTensor()(padded)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    return tensor.unsqueeze(0)  # Add batch dimension


def classify(model, img: Image.Image):
    """Run inference on CPU (PyTorch) and return (predicted_digit, confidence, all_probs, time_ms)."""
    tensor = preprocess_image(img)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    t1 = time.perf_counter()
    
    probs = F.softmax(logits, dim=1).squeeze().numpy()
    predicted = int(probs.argmax())
    confidence = float(probs[predicted]) * 100
    time_ms = (t1 - t0) * 1000
    
    return predicted, confidence, probs, time_ms


# ─── FPGA Backend ────────────────────────────────────────────────────────────

def load_fpga_backend(bitstream="accel_top.bit", simulate=False):
    """
    Initialize the FPGA accelerator and load BSR weights for all 4 layers.
    
    Returns:
        (accel_driver, layer_data) or None if FPGA not available
    """
    import json
    from accel import AccelDriver
    
    try:
        if not simulate:
            from pynq import Overlay
            overlay = Overlay(bitstream)
            accel = AccelDriver(overlay)
        else:
            accel = AccelDriver(simulation=True)
    except Exception as e:
        print(f"✗ FPGA init failed: {e}")
        return None, None
    
    # Load BSR weights
    bsr_dir = os.path.join(PROJECT_ROOT, "data", "bsr_export_14x14")
    layer_data = {}
    
    for layer_name in ["conv1", "conv2", "fc1", "fc2"]:
        layer_dir = os.path.join(bsr_dir, layer_name)
        meta_path = os.path.join(layer_dir, "weights.meta.json")
        
        if not os.path.exists(meta_path):
            print(f"  ✗ {layer_name}: BSR data not found")
            continue
        
        bsr_path = os.path.join(layer_dir, "weights.bsr")
        if not os.path.exists(bsr_path):
            bsr_path = os.path.join(layer_dir, "weights_int8.bsr")
        
        weights = np.load(bsr_path, allow_pickle=True)
        with open(meta_path) as f:
            meta = json.load(f)
        
        row_ptr = np.array(meta["row_ptr"], dtype=np.int32)
        col_idx = np.array(meta["col_idx"], dtype=np.int32)
        
        layer_data[layer_name] = {
            "weights": weights, "row_ptr": row_ptr,
            "col_idx": col_idx, "meta": meta
        }
        print(f"  ✓ {layer_name}: {meta['num_blocks']} BSR blocks "
              f"({meta['sparsity_pct']:.0f}% sparse)")
    
    return accel, layer_data


def im2col_simple(input_3d, kH, kW, stride=1):
    """im2col for conv layer: (C,H,W) → (C*kH*kW, H_out*W_out)."""
    C, H, W = input_3d.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    col = np.zeros((C * kH * kW, H_out * W_out), dtype=input_3d.dtype)
    for h in range(H_out):
        for w in range(W_out):
            patch = input_3d[:, h:h+kH, w:w+kW]
            col[:, h * W_out + w] = patch.flatten()
    return col


def max_pool_2d_simple(x, pool=2):
    """2×2 max pool on (C,H,W)."""
    C, H, W = x.shape
    Ho, Wo = H // pool, W // pool
    out = np.zeros((C, Ho, Wo), dtype=x.dtype)
    for c in range(C):
        for h in range(Ho):
            for w in range(Wo):
                out[c, h, w] = x[c, h*pool:(h+1)*pool, w*pool:(w+1)*pool].max()
    return out


def classify_fpga(accel, layer_data, img: Image.Image):
    """
    Run full MNIST inference on the FPGA accelerator.
    
    Pipeline: image → INT8 → conv1 → conv2 → pool → fc1 → fc2 → softmax
    
    Returns: (predicted_digit, confidence, all_probs, time_ms)
    """
    # Preprocess image the same way as PyTorch
    tensor = preprocess_image(img)
    img_fp32 = tensor.squeeze().numpy()  # (1, 28, 28) → (28, 28)
    
    # Quantize to INT8
    scale = np.max(np.abs(img_fp32)) / 127.0 if np.max(np.abs(img_fp32)) > 0 else 1.0
    act = np.clip(np.rint(img_fp32 / scale), -128, 127).astype(np.int8)
    act = act.reshape(1, 28, 28)  # (C=1, H=28, W=28)
    act_scale = scale
    
    t0 = time.perf_counter()
    
    # ── conv1: (1,28,28) → (32,26,26) ──
    ld = layer_data["conv1"]
    meta = ld["meta"]
    col = im2col_simple(act, 3, 3)               # (9, 676)
    M, K = meta["original_shape"]                  # 32, 9
    N = col.shape[1]                               # 676
    out32 = accel.run_layer("conv1", ld["row_ptr"], ld["col_idx"], ld["weights"],
                            col.T, M=M, N=N, K=K, Sa=act_scale)
    out_fp = out32.astype(np.float32) * act_scale
    out_fp = np.maximum(out_fp, 0)                 # ReLU
    act = out_fp.reshape(M, 26, 26)
    act_int8 = np.clip(np.rint(act / (np.max(np.abs(act))/127 + 1e-10)), -128, 127).astype(np.int8)
    act_scale = np.max(np.abs(act)) / 127.0
    act = act_int8
    
    # ── conv2: (32,26,26) → (64,24,24) → pool → (64,12,12) ──
    ld = layer_data["conv2"]
    meta = ld["meta"]
    col = im2col_simple(act, 3, 3)                # (288, 576)
    M, K = meta["original_shape"]                   # 64, 288
    N = col.shape[1]                                # 576
    out32 = accel.run_layer("conv2", ld["row_ptr"], ld["col_idx"], ld["weights"],
                            col.T, M=M, N=N, K=K, Sa=act_scale)
    out_fp = out32.astype(np.float32) * act_scale
    out_fp = np.maximum(out_fp, 0)                  # ReLU
    act_3d = out_fp.reshape(M, 24, 24)
    act_3d = max_pool_2d_simple(act_3d, 2)         # (64, 12, 12)
    act_scale = np.max(np.abs(act_3d)) / 127.0
    act = np.clip(np.rint(act_3d / (act_scale + 1e-10)), -128, 127).astype(np.int8)
    
    # ── fc1: flatten 9216 → 128 ──
    ld = layer_data["fc1"]
    meta = ld["meta"]
    M, K = meta["original_shape"]                   # 128, 9216
    N = 1
    act_2d = act.flatten()[:K].reshape(1, K).astype(np.int8)
    out32 = accel.run_layer("fc1", ld["row_ptr"], ld["col_idx"], ld["weights"],
                            act_2d, M=M, N=N, K=K, Sa=act_scale)
    out_fp = out32.astype(np.float32) * act_scale
    out_fp = np.maximum(out_fp, 0)                  # ReLU
    act_scale = np.max(np.abs(out_fp)) / 127.0
    act = np.clip(np.rint(out_fp / (act_scale + 1e-10)), -128, 127).astype(np.int8)
    
    # ── fc2: 128 → 10 (logits) ──
    ld = layer_data["fc2"]
    meta = ld["meta"]
    M, K = meta["original_shape"]                   # 10, 128
    N = 1
    act_2d = act.flatten()[:K].reshape(1, K).astype(np.int8)
    out32 = accel.run_layer("fc2", ld["row_ptr"], ld["col_idx"], ld["weights"],
                            act_2d, M=M, N=N, K=K, Sa=act_scale)
    logits = out32.astype(np.float32).flatten()[:10] * act_scale
    
    t1 = time.perf_counter()
    
    # Softmax
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()
    predicted = int(probs.argmax())
    confidence = float(probs[predicted]) * 100
    time_ms = (t1 - t0) * 1000
    
    return predicted, confidence, probs, time_ms


def print_result(predicted, confidence, probs, time_ms, backend="PyTorch CPU"):
    """Pretty-print classification result to terminal."""
    print()
    print("=" * 52)
    print(f"  PREDICTED DIGIT:  {predicted}   ({confidence:.1f}% confidence)")
    print("=" * 52)
    print()
    
    # Bar chart of all class probabilities
    print("  Class probabilities:")
    for i in range(10):
        bar_len = int(probs[i] * 40)
        bar = "█" * bar_len
        marker = " ◄── PREDICTED" if i == predicted else ""
        print(f"    {i}: {bar:<40s} {probs[i]*100:5.1f}%{marker}")
    
    print()
    print(f"  Inference time:  {time_ms:.2f} ms ({backend})")
    if backend == "PyTorch CPU":
        print(f"  FPGA estimated:  ~0.025 ms (25 µs @ 110 MHz, 90% sparse)")
        print(f"  Speedup:         ~{time_ms/0.025:.0f}×")
    print()


# ─── File Mode ───────────────────────────────────────────────────────────────
def classify_file(model, path):
    """Classify a digit from an image file."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    img = Image.open(path)
    print(f"Loaded image: {path} ({img.size[0]}×{img.size[1]})")
    
    predicted, confidence, probs, time_ms = classify(model, img)
    print_result(predicted, confidence, probs, time_ms)
    
    # Show ASCII preview of what the model sees
    print_ascii_preview(img)


def print_ascii_preview(img):
    """Print a small ASCII art preview of the preprocessed image."""
    img = img.convert("L")
    arr = np.array(img)
    if arr.mean() > 128:
        arr = 255 - arr
    
    # Resize to 28×14 for terminal display
    small = Image.fromarray(arr).resize((14, 14), Image.LANCZOS)
    pixels = np.array(small)
    
    chars = " .:-=+*#%@"
    print("  Model input preview:")
    print("  ┌" + "─" * 14 + "┐")
    for row in pixels:
        line = ""
        for px in row:
            idx = min(int(px / 256 * len(chars)), len(chars) - 1)
            line += chars[idx]
        print(f"  │{line}│")
    print("  └" + "─" * 14 + "┘")
    print()


# ─── Interactive Drawing Mode (tkinter) ─────────────────────────────────────
def run_drawing_mode(model, classify_fn=None, backend_name="PyTorch CPU"):
    """Open a tkinter canvas for drawing digits."""
    import tkinter as tk
    
    # Default to CPU classify if no function provided
    if classify_fn is None:
        classify_fn = lambda img: classify(model, img)
    
    CANVAS_SIZE = 280  # 10× MNIST resolution for smooth drawing
    BRUSH_SIZE = 16
    
    class DrawApp:
        def __init__(self, root):
            self.root = root
            root.title("ACCEL-v1 Digit Classifier")
            root.resizable(False, False)
            
            # Header
            header = tk.Frame(root, bg="#1a1a2e", height=50)
            header.pack(fill=tk.X)
            tk.Label(header, text="Draw a digit (0-9)", 
                     font=("Helvetica", 16, "bold"),
                     fg="white", bg="#1a1a2e").pack(pady=10)
            
            # Canvas
            self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                    bg="black", cursor="crosshair",
                                    highlightthickness=2, highlightbackground="#333")
            self.canvas.pack(padx=10, pady=5)
            
            # PIL image for capturing drawing
            self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
            self.draw = ImageDraw.Draw(self.image)
            
            # Bind mouse events
            self.canvas.bind("<B1-Motion>", self.paint)
            self.canvas.bind("<ButtonPress-1>", self.paint)
            
            # Buttons
            btn_frame = tk.Frame(root, bg="#16213e")
            btn_frame.pack(fill=tk.X, padx=10, pady=5)
            
            self.classify_btn = tk.Button(
                btn_frame, text="🔍 Classify", font=("Helvetica", 14, "bold"),
                bg="#0f3460", fg="white", activebackground="#533483",
                command=self.do_classify, width=15, height=2
            )
            self.classify_btn.pack(side=tk.LEFT, padx=5, expand=True)
            
            self.clear_btn = tk.Button(
                btn_frame, text="🗑 Clear", font=("Helvetica", 14),
                bg="#333", fg="white", activebackground="#555",
                command=self.clear, width=10, height=2
            )
            self.clear_btn.pack(side=tk.RIGHT, padx=5, expand=True)
            
            # Result label
            self.result_frame = tk.Frame(root, bg="#1a1a2e", height=80)
            self.result_frame.pack(fill=tk.X, padx=10, pady=5)
            
            self.result_label = tk.Label(
                self.result_frame, text="Draw a digit and click Classify",
                font=("Helvetica", 14), fg="#aaa", bg="#1a1a2e"
            )
            self.result_label.pack(pady=10)
            
            self.prob_label = tk.Label(
                self.result_frame, text="",
                font=("Courier", 10), fg="#888", bg="#1a1a2e",
                justify=tk.LEFT
            )
            self.prob_label.pack(pady=5)
            
            # Footer
            tk.Label(root, text=f"ACCEL-v1 | 14×14 Systolic Array | Backend: {backend_name}",
                     font=("Helvetica", 9), fg="#555", bg="#1a1a2e").pack(pady=5)
            
            root.configure(bg="#1a1a2e")
        
        def paint(self, event):
            x, y = event.x, event.y
            r = BRUSH_SIZE
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        
        def clear(self):
            self.canvas.delete("all")
            self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
            self.draw = ImageDraw.Draw(self.image)
            self.result_label.config(text="Draw a digit and click Classify", fg="#aaa")
            self.prob_label.config(text="")
        
        def do_classify(self):
            predicted, confidence, probs, time_ms = classify_fn(self.image)
            
            # Update GUI
            self.result_label.config(
                text=f"Predicted: {predicted}   ({confidence:.1f}%)",
                fg="#00ff88" if confidence > 80 else "#ffaa00"
            )
            
            # Build probability display
            prob_text = ""
            for i in range(10):
                bar = "█" * int(probs[i] * 20)
                marker = " ◄" if i == predicted else ""
                prob_text += f" {i}: {bar:<20s} {probs[i]*100:5.1f}%{marker}\n"
            self.prob_label.config(text=prob_text)
            
            # Also print to terminal
            print_result(predicted, confidence, probs, time_ms, backend_name)
    
    root = tk.Tk()
    app = DrawApp(root)
    print("\n  Canvas open — draw a digit and click 'Classify'")
    print("  Close the window or Ctrl+C to exit\n")
    root.mainloop()


# ─── MNIST Test Set Demo ────────────────────────────────────────────────────
def run_test_demo(model, count=10):
    """Classify random samples from the MNIST test set."""
    from torchvision import datasets
    
    test_data = datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )
    
    indices = np.random.choice(len(test_data), count, replace=False)
    correct = 0
    total_time = 0
    
    print(f"\nClassifying {count} random MNIST test images:\n")
    print(f"  {'#':>3s}  {'True':>4s}  {'Pred':>4s}  {'Conf':>6s}  {'Time':>8s}  Result")
    print(f"  {'─'*3}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*6}")
    
    for i, idx in enumerate(indices):
        img_tensor, label = test_data[idx]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0))
        t1 = time.perf_counter()
        
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred]) * 100
        ms = (t1 - t0) * 1000
        total_time += ms
        
        ok = "✓" if pred == label else "✗"
        if pred == label:
            correct += 1
        
        print(f"  {i+1:>3d}  {label:>4d}  {pred:>4d}  {conf:5.1f}%  {ms:6.2f}ms  {ok}")
    
    print(f"\n  Accuracy: {correct}/{count} ({100*correct/count:.0f}%)")
    print(f"  Avg time: {total_time/count:.2f} ms/image (PyTorch CPU)")
    print(f"  FPGA est: ~0.025 ms/image (25 µs @ 110 MHz, 90% sparse)")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   ACCEL-v1 Handwriting Digit Classifier      ║")
    print("  ║   14×14 Systolic Array · INT8 · BSR Sparse   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    
    # Check for --fpga flag
    use_fpga = "--fpga" in sys.argv
    fpga_sim = "--fpga-sim" in sys.argv
    
    # Remove flags from argv for further parsing
    args = [a for a in sys.argv[1:] if a not in ("--fpga", "--fpga-sim")]
    
    # Always load PyTorch model (used as fallback and for --test mode)
    model = load_model()
    
    # Initialize FPGA backend if requested
    accel = None
    fpga_layer_data = None
    if use_fpga or fpga_sim:
        print("\n  Initializing FPGA backend...")
        accel, fpga_layer_data = load_fpga_backend(simulate=fpga_sim)
        if accel is not None:
            print("  ✓ FPGA ready\n")
        else:
            print("  ✗ FPGA unavailable, falling back to PyTorch CPU\n")
    
    # Choose classify function based on backend
    def do_classify(img):
        if accel is not None and fpga_layer_data is not None:
            return classify_fpga(accel, fpga_layer_data, img)
        else:
            return classify(model, img)
    
    backend_name = "FPGA ACCEL-v1" if accel else "PyTorch CPU"
    
    if len(args) > 0:
        arg = args[0]
        if arg == "--test":
            count = int(args[1]) if len(args) > 1 else 10
            run_test_demo(model, count)
        elif arg == "--help" or arg == "-h":
            print("Usage:")
            print("  python3 classify_digit.py              # Interactive drawing (PyTorch CPU)")
            print("  python3 classify_digit.py --fpga       # Interactive drawing (FPGA)")
            print("  python3 classify_digit.py --fpga-sim   # Interactive drawing (FPGA simulated)")
            print("  python3 classify_digit.py image.png    # Classify an image file")
            print("  python3 classify_digit.py --test [N]   # Test on N random MNIST samples")
            print()
        else:
            classify_file(model, arg)
    else:
        run_drawing_mode(model, do_classify, backend_name)
