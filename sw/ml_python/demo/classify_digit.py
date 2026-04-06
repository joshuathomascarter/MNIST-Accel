#!/usr/bin/env python3
"""
ACCEL-v1 Handwriting Classifier — Interactive Demo
====================================================
Draw a digit (0-9) on the canvas, click "Classify" and see the prediction.

Supports three backends:
  1. PyTorch CPU   — default, works anywhere
  2. ZCU104 UART   — real hardware via USB-TTL on PMOD J160 (use --zcu104 PORT)
  3. SoC RTL e2e   -- Verilator simulation (use --soc-e2e, slow)

PyTorch (CPU) mode:
  1. Capture handwritten digit image
  2. Preprocess to 28×28 grayscale (MNIST format)
  3. Run FP32 CNN inference on CPU
  4. Output predicted digit, confidence, and latency

ZCU104 (hardware) mode:
  1. Capture handwritten digit image
  2. Run conv1 → conv2 → pool → fc1 on CPU (fast, FP32)
  3. Send first fc1 16×16 tile to ZCU104 via UART and run on systolic array
  4. Report predicted digit (CPU classification), confidence, and real UART-roundtrip latency

SoC E2E (Verilator) mode:
  1. Same pipeline but routed through full RTL simulation via run_e2e.py
  2. Slow (~10 min) but exercises the complete RTL path

Requirements: torch, torchvision, Pillow, tkinter, pyserial
  pip install torch torchvision Pillow pyserial

Usage:
  python3 classify_digit.py                       # Interactive (PyTorch CPU)
  python3 classify_digit.py --zcu104 /dev/ttyUSB0 # Interactive (ZCU104 hardware)
  python3 classify_digit.py --soc-e2e             # Interactive (full SoC RTL, slow)
  python3 classify_digit.py --soc-e2e-rebuild     # Force RTL rebuild before e2e
  python3 classify_digit.py image.png             # Classify an image file
  python3 classify_digit.py --test [N]            # Test on N random MNIST samples
"""

import sys
import os
import re
import subprocess
import tempfile
import threading
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
    
    fc1 output = 140 — padded to 144 (9×16) for systolic tiling.
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

# Add tools/ to path for ZCU104 UART host imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))


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


# ─── ZCU104 UART Backend ─────────────────────────────────────────────────────

def classify_zcu104(port: str, baud: int, model, img: Image.Image):
    """
    Run inference using the ZCU104 hardware over UART.

    Pipeline:
      CPU:    preprocess → conv1 → conv2 → pool → fc1 (FP32, ~2 ms)
      ZCU104: first fc1 16×16 tile via UART systolic array → measure real HW latency
      CPU:    full PyTorch forward pass returns the final digit + confidence
              (multi-tile fc2 accumulation on HW is future work)

    Returns: (predicted_digit, confidence, all_probs, time_ms)
      where time_ms = wall-clock UART roundtrip for the first fc1 16×16 tile.
    """
    from soc_top_v2_uart_host import SocTopV2UartHost, run_block_demo

    # ── Full CPU inference (for digit + confidence) ──────────────────────────
    tensor = preprocess_image(img)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().numpy()
    predicted = int(probs.argmax())
    confidence = float(probs[predicted]) * 100

    # ── Extract fc1 first 16×16 activation block (INT8) ─────────────────────
    # Pass the image through conv1 + conv2 + pool to get the 9216-dim fc1 input
    with torch.no_grad():
        x = tensor
        x = F.relu(model.conv1(x))
        x = F.relu(model.conv2(x))
        x = F.max_pool2d(x, 2)
        fc1_input = torch.flatten(x, 1).squeeze().numpy()  # (9216,)

    act_flat = fc1_input[:256]  # first 16×16 slice
    act_scale = float(np.max(np.abs(act_flat))) / 127.0 if np.max(np.abs(act_flat)) > 0 else 1.0
    act_block = np.clip(np.rint(act_flat / act_scale), -128, 127).astype(np.int8).reshape(16, 16)

    # First 16×16 slice of fc1 weight matrix
    wgt_np = model.fc1.weight.detach().numpy()[:16, :256]  # (16, 256) → first 16×16
    wgt_scale = float(np.max(np.abs(wgt_np))) / 127.0 if np.max(np.abs(wgt_np)) > 0 else 1.0
    wgt_block = np.clip(np.rint(wgt_np[:, :16] / wgt_scale), -128, 127).astype(np.int8)

    # ── Send one 16×16 tile to ZCU104 and time the UART roundtrip ────────────
    host = SocTopV2UartHost(port=port, baud=baud, timeout=5.0)
    version = host.ping()
    print(f"  ZCU104 ping OK — firmware version 0x{version:08x}")

    t0 = time.perf_counter()
    run_block_demo(host, act_block, wgt_block, tile=0)
    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0

    return predicted, confidence, probs, time_ms


def classify_soc_e2e(img: Image.Image, skip_build=True, full=True):
    """Run the full soc_top_v2 e2e flow on an arbitrary input image."""
    runner = os.path.join(PROJECT_ROOT, "tools", "run_e2e.py")

    with tempfile.TemporaryDirectory(prefix="mnist_accel_gui_") as tmpdir:
        image_path = os.path.join(tmpdir, "drawn_digit.png")
        img.save(image_path)

        cmd = [sys.executable, runner, "--image-path", image_path]
        if skip_build:
            cmd.append("--skip-build")
        if full:
            cmd.append("--full")

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        t1 = time.perf_counter()

    output = proc.stdout
    if proc.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-20:])
        raise RuntimeError(f"SoC e2e failed:\n{tail}")

    pred_match = re.search(r"Predicted digit\s*:\s*(\d+)", output)
    if not pred_match:
        pred_match = re.search(r"Predicted:\s*(\d+)", output)
    if not pred_match:
        raise RuntimeError("SoC e2e completed, but no predicted digit was found in the output")
    predicted = int(pred_match.group(1))

    logits_match = re.search(r"LogitsHex:\s+([0-9a-fA-F\s]+)", output)
    if logits_match:
        tokens = logits_match.group(1).split()
        logits = []
        for token in tokens[:10]:
            value = int(token, 16)
            if value & 0x80000000:
                value -= 1 << 32
            logits.append(float(value))
        logits = np.array(logits, dtype=np.float64)
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
    else:
        probs = np.zeros(10, dtype=np.float64)
        probs[predicted] = 1.0

    confidence = float(probs[predicted]) * 100.0
    time_ms = (t1 - t0) * 1000.0
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
        print(f"  ZCU104 est:      ~0.025 ms (25 µs @ 50 MHz, fc1+fc2 tiled)")
        print(f"  Potential speedup: ~{time_ms/0.025:.0f}×")
    elif "ZCU104" in backend:
        print(f"  Includes UART roundtrip overhead (first fc1 tile, 16×16)")
    print()


# ─── File Mode ───────────────────────────────────────────────────────────────
def classify_file(path, classify_fn, backend_name):
    """Classify a digit from an image file."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    img = Image.open(path)
    print(f"Loaded image: {path} ({img.size[0]}×{img.size[1]})")
    
    predicted, confidence, probs, time_ms = classify_fn(img)
    print_result(predicted, confidence, probs, time_ms, backend_name)
    
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
            tk.Label(root, text=f"ACCEL-v1 | 16×16 Systolic Array | Backend: {backend_name}",
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

        def finish_classify(self, result):
            predicted, confidence, probs, time_ms = result

            self.classify_btn.config(state=tk.NORMAL, text="🔍 Classify")
            self.clear_btn.config(state=tk.NORMAL)
            self.result_label.config(
                text=f"Predicted: {predicted}   ({confidence:.1f}%)",
                fg="#00ff88" if confidence > 80 else "#ffaa00"
            )

            prob_text = ""
            for i in range(10):
                bar = "█" * int(probs[i] * 20)
                marker = " ◄" if i == predicted else ""
                prob_text += f" {i}: {bar:<20s} {probs[i]*100:5.1f}%{marker}\n"
            self.prob_label.config(text=prob_text)

            print_result(predicted, confidence, probs, time_ms, backend_name)

        def finish_error(self, message):
            self.classify_btn.config(state=tk.NORMAL, text="🔍 Classify")
            self.clear_btn.config(state=tk.NORMAL)
            self.result_label.config(text="Classification failed", fg="#ff6666")
            self.prob_label.config(text=message)
        
        def do_classify(self):
            image_copy = self.image.copy()
            self.classify_btn.config(state=tk.DISABLED, text="Running...")
            self.clear_btn.config(state=tk.DISABLED)
            self.result_label.config(text=f"Running {backend_name}...", fg="#66ccff")
            self.prob_label.config(text="")

            def worker():
                try:
                    result = classify_fn(image_copy)
                    self.root.after(0, lambda: self.finish_classify(result))
                except Exception as exc:
                    self.root.after(0, lambda: self.finish_error(str(exc)))

            threading.Thread(target=worker, daemon=True).start()
    
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
    print(f"  ZCU104 est: ~0.025 ms/image (25 µs @ 50 MHz, tiled fc1+fc2)")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   ACCEL-v1 Handwriting Digit Classifier      ║")
    print("  ║   16×16 Systolic Array · INT8 · BSR Sparse   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    
    # Check for mode flags
    use_zcu104 = "--zcu104" in sys.argv
    zcu104_port = None
    zcu104_baud = 115200
    use_soc_e2e = "--soc-e2e" in sys.argv
    soc_e2e_rebuild = "--soc-e2e-rebuild" in sys.argv

    # Extract --zcu104 PORT [--baud BAUD] values
    _argv = sys.argv[1:]
    if use_zcu104:
        try:
            zcu104_port = _argv[_argv.index("--zcu104") + 1]
        except (IndexError, ValueError):
            print("Error: --zcu104 requires a port argument, e.g. --zcu104 /dev/ttyUSB0")
            sys.exit(1)
        if "--baud" in _argv:
            try:
                zcu104_baud = int(_argv[_argv.index("--baud") + 1])
            except (IndexError, ValueError):
                pass

    # Remove flags from argv for further parsing
    args = [
        a for a in _argv
        if a not in ("--zcu104", zcu104_port, "--baud", str(zcu104_baud),
                     "--soc-e2e", "--soc-e2e-rebuild")
    ]
    
    # Always load PyTorch model (used for CPU inference and ZCU104 conv pipeline)
    model = load_model()

    # Choose classify function based on backend
    if use_soc_e2e:
        def do_classify(img):
            return classify_soc_e2e(img, skip_build=not soc_e2e_rebuild, full=True)
        backend_name = "SoC E2E RTL (Verilator)"
    elif use_zcu104:
        def do_classify(img):
            return classify_zcu104(zcu104_port, zcu104_baud, model, img)
        backend_name = f"ZCU104 UART ({zcu104_port})"
    else:
        def do_classify(img):
            return classify(model, img)
        backend_name = "PyTorch CPU"
    
    if len(args) > 0:
        arg = args[0]
        if arg == "--test":
            count = int(args[1]) if len(args) > 1 else 10
            run_test_demo(model, count)
        elif arg == "--help" or arg == "-h":
            print("Usage:")
            print("  python3 classify_digit.py                        # Interactive (PyTorch CPU)")
            print("  python3 classify_digit.py --zcu104 /dev/ttyUSB0  # Interactive (ZCU104 hardware)")
            print("  python3 classify_digit.py --zcu104 /dev/ttyUSB0 --baud 115200")
            print("  python3 classify_digit.py --soc-e2e              # Interactive (full SoC RTL, slow)")
            print("  python3 classify_digit.py --soc-e2e-rebuild      # Force RTL rebuild before SoC e2e")
            print("  python3 classify_digit.py image.png              # Classify an image file")
            print("  python3 classify_digit.py --test [N]             # Test on N random MNIST samples")
            print()
        else:
            classify_file(arg, do_classify, backend_name)
    else:
        run_drawing_mode(model, do_classify, backend_name)
