#!/usr/bin/env python3
"""
ACCEL-v1 Handwriting Classifier — Interactive Demo
====================================================
Draw a digit (0-9) on the canvas, click "Classify" and see the prediction.

This simulates what the FPGA accelerator does:
  1. Capture handwritten digit image
  2. Preprocess to 28×28 grayscale (MNIST format)
  3. Run INT8-quantized CNN inference
  4. Output predicted digit with confidence

Requirements: torch, torchvision, Pillow, tkinter
  pip install torch torchvision Pillow

Usage:
  python3 classify_digit.py              # Interactive drawing mode
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
    """Run inference and return (predicted_digit, confidence, all_probs, time_ms)."""
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


def print_result(predicted, confidence, probs, time_ms):
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
    print(f"  Inference time:  {time_ms:.2f} ms (PyTorch CPU)")
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
def run_drawing_mode(model):
    """Open a tkinter canvas for drawing digits."""
    import tkinter as tk
    
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
            tk.Label(root, text="ACCEL-v1 | 14×14 Systolic Array | ~90 GOPS @ 90% sparse",
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
            predicted, confidence, probs, time_ms = classify(model, self.image)
            
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
            print_result(predicted, confidence, probs, time_ms)
    
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
    
    model = load_model()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--test":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            run_test_demo(model, count)
        elif arg == "--help" or arg == "-h":
            print("Usage:")
            print("  python3 classify_digit.py              # Interactive drawing canvas")
            print("  python3 classify_digit.py image.png    # Classify an image file")
            print("  python3 classify_digit.py --test [N]   # Test on N random MNIST samples")
            print()
        else:
            classify_file(model, arg)
    else:
        run_drawing_mode(model)
