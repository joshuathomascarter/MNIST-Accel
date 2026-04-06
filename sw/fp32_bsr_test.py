#!/usr/bin/env python3
"""
FP32 BSR accuracy test: same sparsity pattern as INT8 but FP32 arithmetic.
If this hits ~98%, the remaining gap is from INT8 quantization noise.
If it's lower, the sparsity pattern itself is losing information.
"""
import sys, struct, numpy as np
from pathlib import Path

ROOT  = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import gen_dram_init as _gdi
import torch

SYST = 16
CKPT = str(sorted((ROOT / "data/checkpoints").glob("*.pt"))[-1])

with open(ROOT / "data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
    _, n, h, w = struct.unpack(">4I", f.read(16))
    imgs = np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, h, w)
with open(ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte", "rb") as f:
    struct.unpack(">2I", f.read(8))
    labels = np.frombuffer(f.read(n), dtype=np.uint8)

ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
sd    = ckpt["state_dict"]
fc1_w = sd["fc1.weight"].numpy()
fc1_b = sd["fc1.bias"].numpy()
fc2_w = sd["fc2.weight"].numpy()
fc2_b = sd["fc2.bias"].numpy()

# Get INT8 weight matrix to identify which blocks are non-zero
fc_layers = _gdi.prepare_fc_weights()
w_int8    = fc_layers["fc1"]["weights"]

M_fp32, K_fp32 = fc1_w.shape
M_pad,  K_pad  = w_int8.shape

# Apply same BSR sparsity mask: zero out FP32 blocks where INT8 block is all-zero
fc1_masked = fc1_w.copy()
zeroed = 0
total  = 0
for m in range(M_pad // SYST):
    for k in range(K_pad // SYST):
        blk_i8 = w_int8[m*SYST:(m+1)*SYST, k*SYST:(k+1)*SYST]
        rm, re = m * SYST, min((m+1)*SYST, M_fp32)
        rk, ke = k * SYST, min((k+1)*SYST, K_fp32)
        if re <= rm or ke <= rk:
            continue
        total += 1
        if not np.any(blk_i8):
            fc1_masked[rm:re, rk:ke] = 0.0
            zeroed += 1

print(f"BSR mask: {total-zeroed}/{total} blocks active  ({(total-zeroed)/total*100:.1f}% dense)")

correct = 0
for i in range(1000):
    _, _, fc1_fp32 = _gdi.run_conv_layers_golden(imgs[i].astype(np.float32) / 255.0, CKPT)
    fc1_out = fc1_masked @ fc1_fp32 + fc1_b
    relu    = np.maximum(fc1_out, 0)
    fc2_out = fc2_w @ relu + fc2_b
    if int(np.argmax(fc2_out)) == int(labels[i]):
        correct += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/1000  acc={correct/(i+1)*100:.2f}%")

print(f"\nFP32 BSR (same sparsity mask) : {correct}/1000 = {correct/10:.2f}%")
print("(Compare: INT8 4-tile = 96.50%,  FP32 dense = 98.70%)")
print()
if correct >= 985:
    print("=> Gap is from INT8 quantization noise only.")
else:
    print(f"=> Sparsity itself loses {(987-correct)/10:.1f}% accuracy; "
          "INT8 quant adds the rest.")
