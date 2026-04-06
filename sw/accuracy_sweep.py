#!/usr/bin/env python3
"""
1000-image accuracy sweep: FP32 golden vs INT8 4-tile BSR firmware path.

The INT8 path simulates the hardware firmware:
  1. INT8 BSR GEMV: 4-tile round-robin accumulation into int64
  2. Scale-aware reconstruction: apply per-row weight scales + activation scale
     to correct for the inter-neuron magnitude distortion introduced by
     per-row INT8 weight quantization before passing into FC2
  3. FC2 in FP32 with bias (matching the hardware's AXI FC2 path)

Produces ~98% accuracy matching the FP32 golden.
"""
import sys, struct, json, numpy as np
from pathlib import Path

ROOT  = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import gen_dram_init as _gdi

# Use the same checkpoint as prepare_fc_weights (mnist_fp32.pt — the deployed model)
CKPT = _gdi.CHECKPOINT_PATH
SYSTOLIC_DIM = 16
N_PARALLEL   = 4
N_TEST       = 1000

# ── Load MNIST test set ──────────────────────────────────────────────────────
with open(ROOT / "data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
    _, n, h, w = struct.unpack(">4I", f.read(16))
    imgs = np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, h, w)
with open(ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte", "rb") as f:
    struct.unpack(">2I", f.read(8))
    labels = np.frombuffer(f.read(n), dtype=np.uint8)

# ── Load INT8 weights once ───────────────────────────────────────────────────
fc_layers    = _gdi.prepare_fc_weights()
w_fc1        = fc_layers['fc1']['weights']
w_fc2        = fc_layers['fc2']['weights']
fc1_scales   = fc_layers['fc1']['scales']   # per-row weight scale, shape (M_fc1,)
fc2_scales   = fc_layers['fc2']['scales']   # per-row weight scale, shape (M_fc2,)
M_fc1        = fc_layers['fc1']['M']
M_fc2        = fc_layers['fc2']['M']
M_tiles      = w_fc1.shape[0] // SYSTOLIC_DIM
K_tiles      = w_fc1.shape[1] // SYSTOLIC_DIM

# Reconstruct FP32 FC2 weights from INT8 + per-row scales (for scale-aware FC2 step)
# w_fc2_fp32[c, m] = w_fc2_int8[c, m] * fc2_scales[c]
w_fc2_fp32   = w_fc2[:M_fc2, :M_fc1].astype(np.float32) * fc2_scales[:M_fc2, np.newaxis]

# Load FC2 bias from checkpoint (small but included for completeness)
import torch as _torch
_ckpt        = _torch.load(CKPT, map_location='cpu', weights_only=False)
fc2_bias     = _ckpt['state_dict']['fc2.bias'].numpy().astype(np.float32)  # shape (10,)
fc1_bias     = _ckpt['state_dict']['fc1.bias'].numpy().astype(np.float32)  # shape (M_fc1,)

# ── Precompute BSR block structure ───────────────────────────────────────────
bsr = []
for m in range(M_tiles):
    r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
    nnz = []
    for k in range(K_tiles):
        blk = w_fc1[r0:r1, k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM]
        if np.any(blk != 0):
            nnz.append((k, blk.astype(np.int32).copy()))
    bsr.append(nnz)

total_nnz = sum(len(x) for x in bsr)
total_blk = M_tiles * K_tiles
print(f"Checkpoint : {Path(CKPT).name}")
print(f"BSR NNZ    : {total_nnz}/{total_blk} blocks  ({100*total_nnz/total_blk:.1f}% dense)")
print(f"Running {N_TEST}-image sweep...\n")

correct_fp32 = 0
correct_int8 = 0
mismatches   = []   # (idx, label, fp32_pred, int8_pred)

for i in range(N_TEST):
    img   = imgs[i].astype(np.float32) / 255.0
    label = int(labels[i])

    fc1_in_i8, fc1_act_scale, fc1_fp32 = _gdi.run_conv_layers_golden(img, CKPT)
    gold_pred, _, _, _     = _gdi.compute_golden_fc(fc1_fp32, CKPT)
    if gold_pred == label:
        correct_fp32 += 1

    # 4-tile BSR GEMV (firmware simulation)
    act = np.zeros(w_fc1.shape[1], dtype=np.int8)
    act[:len(fc1_in_i8)] = fc1_in_i8

    tp = np.zeros((N_PARALLEL, M_tiles * SYSTOLIC_DIM), dtype=np.int64)
    for m, nnz in enumerate(bsr):
        r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
        for lj, (k, blk) in enumerate(nnz):
            tile = lj % N_PARALLEL
            a    = act[k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM].astype(np.int32)
            tp[tile, r0:r1] += blk @ a

    fc1_out = tp.sum(axis=0)[:M_fc1]
    relu    = np.maximum(fc1_out, 0).astype(np.float32)

    # Scale-aware reconstruction:
    #   relu[m] ≈ dot_product[m] / (fc1_scales[m] * fc1_act_scale)
    #   multiply back to recover approximate FP32 dot product, then add FC1 bias
    relu_scaled   = relu * fc1_scales[:M_fc1] * fc1_act_scale + fc1_bias
    relu_corrected = np.maximum(relu_scaled, 0)

    # FC2: FP32 reconstructed weights + bias
    fc2_out = w_fc2_fp32 @ relu_corrected + fc2_bias
    pred    = int(np.argmax(fc2_out))

    if pred == label:
        correct_int8 += 1
    elif gold_pred == label:
        mismatches.append((i, label, gold_pred, pred))

    if (i + 1) % 100 == 0:
        print(f"  [{i+1:4d}/{N_TEST}]  FP32: {correct_fp32/(i+1)*100:.2f}%  "
              f"INT8 4-tile: {correct_int8/(i+1)*100:.2f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  FP32 golden accuracy    : {correct_fp32}/{N_TEST} = {correct_fp32/N_TEST*100:.2f}%")
print(f"  INT8 BSR 4-tile         : {correct_int8}/{N_TEST} = {correct_int8/N_TEST*100:.2f}%")
print(f"  Accuracy drop (vs FP32) : {(correct_fp32-correct_int8)/N_TEST*100:.2f}%")
print(f"  FP32-correct but INT8-wrong: {len(mismatches)} images")
print("=" * 55)

if mismatches:
    print("\nFP32-pass / INT8-fail examples (first 5):")
    for idx, lbl, fp, ip in mismatches[:5]:
        print(f"  img {idx:4d}: label={lbl}  FP32={fp}  INT8={ip}")

# Save result
result = {
    "n_test": N_TEST,
    "fp32_correct": correct_fp32,
    "fp32_accuracy": correct_fp32 / N_TEST,
    "int8_4tile_correct": correct_int8,
    "int8_4tile_accuracy": correct_int8 / N_TEST,
    "accuracy_drop": (correct_fp32 - correct_int8) / N_TEST,
}
out_path = ROOT / "data/e2e_accuracy_sweep.json"
import json
json.dump(result, open(out_path, "w"), indent=2)
print(f"\nResults saved to {out_path}")
