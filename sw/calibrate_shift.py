#!/usr/bin/env python3
"""
Calibrate the optimal fixed INT8 shift for FC1 RELU output.
Scans over 1000 test images and sweeps shift values to find the
one that maximises accuracy, then prints it for use in firmware.
"""
import sys, struct, numpy as np
from pathlib import Path

ROOT  = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import gen_dram_init as _gdi

CKPT          = str(sorted((ROOT / "data/checkpoints").glob("*.pt*"))[-1])
SYSTOLIC_DIM  = 16
N_PARALLEL    = 4
N_CALIB       = 1000

with open(ROOT / "data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
    _, n, h, w = struct.unpack(">4I", f.read(16))
    imgs = np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, h, w)
with open(ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte", "rb") as f:
    struct.unpack(">2I", f.read(8))
    labels = np.frombuffer(f.read(n), dtype=np.uint8)

fc_layers = _gdi.prepare_fc_weights()
w_fc1     = fc_layers['fc1']['weights']
w_fc2     = fc_layers['fc2']['weights']
M_fc1     = fc_layers['fc1']['M']
M_fc2     = fc_layers['fc2']['M']
M_tiles   = w_fc1.shape[0] // SYSTOLIC_DIM

bsr = []
for m in range(M_tiles):
    r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
    nnz = [(k, w_fc1[r0:r1, k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM].astype(np.int32).copy())
           for k in range(w_fc1.shape[1] // SYSTOLIC_DIM)
           if np.any(w_fc1[r0:r1, k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM])]
    bsr.append(nnz)

print(f"Calibrating {N_CALIB} images ...")

relu_data   = []   # list of (relu_vec, label)
for i in range(N_CALIB):
    img       = imgs[i].astype(np.float32) / 255.0
    label     = int(labels[i])
    fc1_in_i8, _, _ = _gdi.run_conv_layers_golden(img, CKPT)

    act = np.zeros(w_fc1.shape[1], dtype=np.int8)
    act[:len(fc1_in_i8)] = fc1_in_i8

    tp = np.zeros((N_PARALLEL, M_tiles * SYSTOLIC_DIM), dtype=np.int64)
    for m, nnz in enumerate(bsr):
        r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
        for lj, (k, blk) in enumerate(nnz):
            tp[lj % N_PARALLEL, r0:r1] += blk @ act[k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM].astype(np.int32)

    relu = np.maximum(tp.sum(axis=0)[:M_fc1], 0)
    relu_data.append((relu, label))

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_CALIB}")

print("Done. Sweeping shift values 0..20 ...\n")

best_shift, best_acc = 0, 0.0
for shift in range(21):
    correct = 0
    for relu, label in relu_data:
        q = np.clip(relu >> shift, 0, 127).astype(np.int8)
        fc1q_pad = np.zeros(w_fc2.shape[1], dtype=np.int8)
        fc1q_pad[:len(q)] = q
        fc2_out = w_fc2[:M_fc2, :].astype(np.int32) @ fc1q_pad.astype(np.int32)
        pred    = int(np.argmax(fc2_out[:10]))
        correct += (pred == label)
    acc = correct / N_CALIB * 100
    marker = " ← BEST" if acc > best_acc else ""
    print(f"  shift={shift:2d}  accuracy={acc:.2f}%{marker}")
    if acc > best_acc:
        best_acc   = acc
        best_shift = shift

print(f"\nBest fixed shift: {best_shift}  ({best_acc:.2f}% accuracy)")
print(f"\nUse in accuracy_sweep.py: FIXED_SHIFT = {best_shift}")
