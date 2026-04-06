#!/usr/bin/env python3
"""Smoke test: run every layer (conv1, conv2, pool, fc1, fc2) through PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, sys, os, struct, gzip

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1   = nn.Linear(64 * 12 * 12, 140)
        self.fc2   = nn.Linear(140, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MNISTNet()
ckpt = torch.load("data/checkpoints/mnist_fp32.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Model loaded  —  checkpoint accuracy:", ckpt.get("best_acc", "?"), "%")

def read_idx(path):
    """Read IDX file format directly (handles .gz or raw)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim  = magic & 0xFF
        dims  = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        data  = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)

RAW = "data/MNIST/raw"
# Prefer uncompressed if present, fall back to .gz
def open_idx(stem):
    plain = os.path.join(RAW, stem)
    gz    = plain + ".gz"
    if os.path.exists(plain):
        return read_idx(plain)
    return read_idx(gz)

images = open_idx("t10k-images-idx3-ubyte").astype(np.float32) / 255.0
labels = open_idx("t10k-labels-idx1-ubyte")
print("Test set size:", len(images), "samples\n")

MEAN, STD = 0.1307, 0.3081
def to_tensor(img_np):
    """Normalize and convert single 28x28 uint8 image to (1,1,28,28) tensor."""
    t = torch.from_numpy((img_np - MEAN) / STD).float()
    return t.unsqueeze(0).unsqueeze(0)

# ── Layer-by-layer trace on sample 0 ────────────────────────────────────────
img0, label0 = images[0], int(labels[0])
x = to_tensor(img0)

print("-" * 62)
print("Layer-by-layer trace  (true label =", label0, ")")
print("-" * 62)

with torch.no_grad():
    print("  Input:           ", tuple(x.shape), " range [%.3f, %.3f]" % (x.min(), x.max()))

    x1 = F.relu(model.conv1(x))
    print("  conv1 (32x3x3):  ", tuple(x1.shape), " range [%.3f, %.3f]" % (x1.min(), x1.max()),
          " nonzero=%d/%d" % (int((x1 > 0).sum()), x1.numel()))

    x2 = F.relu(model.conv2(x1))
    print("  conv2 (64x3x3):  ", tuple(x2.shape), " range [%.3f, %.3f]" % (x2.min(), x2.max()),
          " nonzero=%d/%d" % (int((x2 > 0).sum()), x2.numel()))

    x3 = F.max_pool2d(x2, 2)
    print("  max_pool2d(2):   ", tuple(x3.shape), " range [%.3f, %.3f]" % (x3.min(), x3.max()))

    x4 = torch.flatten(x3, 1)
    print("  flatten:         ", tuple(x4.shape), " (= 64x12x12 = 9216 dims)")

    x5 = F.relu(model.fc1(x4))
    nz  = int((x5 > 0).sum())
    print("  fc1 (->140 ReLU):", tuple(x5.shape), " range [%.3f, %.3f]" % (x5.min(), x5.max()),
          " nonzero=%d/140" % nz)

    logits = model.fc2(x5)
    vals   = ["% 6.2f" % v for v in logits.squeeze().tolist()]
    print("  fc2 (->10):      ", tuple(logits.shape), " logits: [%s]" % ", ".join(vals))

    probs  = F.softmax(logits, dim=1).squeeze()
    pred   = int(probs.argmax())
    conf   = float(probs[pred]) * 100
    result = "PASS" if pred == label0 else "FAIL"
    print("\n  --> Predicted: %d  True: %d  Confidence: %.1f%%  %s" % (pred, label0, conf, result))

# ── Accuracy over first 100 samples ─────────────────────────────────────────
print("\n" + "-" * 62)
print("Accuracy check — first 100 test samples")
print("-" * 62)

correct = 0
N = 100
with torch.no_grad():
    for i in range(N):
        img, label = images[i], int(labels[i])
        logits = model(to_tensor(img))
        pred   = int(logits.argmax(dim=1))
        correct += (pred == label)
        status  = "PASS" if pred == label else "FAIL (got %d)" % pred
        if i < 20 or pred != label:
            print("  [%3d] true=%d  pred=%d  %s" % (i, label, pred, status))

print("\n" + "=" * 62)
print("  Result: %d/%d correct  (%d%%)" % (correct, N, correct))
if correct < 95:
    print("  WARNING: below expected 99% — check weights or model arch")
else:
    print("  All layers working correctly.")
print("=" * 62)
