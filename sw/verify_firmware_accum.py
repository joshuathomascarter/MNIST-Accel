#!/usr/bin/env python3
"""
Verify multi-tile firmware accumulation logic in Python (<30s).
Uses the SAME gen_dram_init.py helpers that generate the DRAM image,
so the weight layout here is byte-for-byte identical to what the firmware sees.
"""
import sys, json, numpy as np, struct
from pathlib import Path

ROOT  = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import gen_dram_init as _gdi

CKPT  = str(sorted((ROOT / "data/checkpoints").glob("*.pt*"))[-1])
SYSTOLIC_DIM = 16
N_PARALLEL   = 4

# 1. Load test image 0
with open(ROOT / "data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
    magic, n, h, w = struct.unpack(">4I", f.read(16))
    imgs   = np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, h, w)
with open(ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte", "rb") as f:
    struct.unpack(">2I", f.read(8))
    labels = np.frombuffer(f.read(n), dtype=np.uint8)

img0       = imgs[0].astype(np.float32) / 255.0
true_label = int(labels[0])
print(f"Image 0: true label = {true_label}")

# 2. Conv layers -> FC1 input (same quant as DRAM gen)
fc1_in_i8, fc1_in_scale, fc1_in_fp32 = _gdi.run_conv_layers_golden(img0, CKPT)
print(f"FC1 input: scale={fc1_in_scale:.6f}, range=[{fc1_in_i8.min()}, {fc1_in_i8.max()}]")

# 3. FP32 golden prediction
gold_pred, gold_logits, fc2_in_i8, _ = _gdi.compute_golden_fc(fc1_in_fp32, CKPT)
print(f"FP32 golden prediction: {gold_pred}")

# 4. Load INT8 weights (same per-channel quant as DRAM gen)
fc_layers = _gdi.prepare_fc_weights()
fc1_info  = fc_layers['fc1']
fc2_info  = fc_layers['fc2']
w_fc1     = fc1_info['weights']    # (144, 9216) int8
w_fc2     = fc2_info['weights']    # (16, 144)  int8
print(f"FC1 padded: {w_fc1.shape}  FC2 padded: {w_fc2.shape}")

M_tiles   = w_fc1.shape[0] // SYSTOLIC_DIM   # 9
K_tiles   = w_fc1.shape[1] // SYSTOLIC_DIM   # 576

# Pad activation to K_pad
fc1_in_padded = np.zeros(w_fc1.shape[1], dtype=np.int8)
fc1_in_padded[:len(fc1_in_i8)] = fc1_in_i8

# 5. Single-tile GEMV (reference)
fc1_ref = w_fc1[:fc1_info['M'], :].astype(np.int32) @ fc1_in_padded.astype(np.int32)

# 6. 4-tile round-robin BSR accumulation (mirrors firmware exactly)
tile_partial = np.zeros((N_PARALLEL, M_tiles * SYSTOLIC_DIM), dtype=np.int64)
per_tile_nnz = [0] * N_PARALLEL

for m in range(M_tiles):
    r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
    nnz_for_m = []
    for k in range(K_tiles):
        c0, c1 = k * SYSTOLIC_DIM, (k + 1) * SYSTOLIC_DIM
        blk = w_fc1[r0:r1, c0:c1]
        if np.any(blk != 0):
            nnz_for_m.append((k, blk))
    for local_j, (k, blk) in enumerate(nnz_for_m):
        tile = local_j % N_PARALLEL
        a_blk = fc1_in_padded[k*SYSTOLIC_DIM:(k+1)*SYSTOLIC_DIM].astype(np.int32)
        tile_partial[tile, r0:r1] += blk.astype(np.int32) @ a_blk
        per_tile_nnz[tile] += 1

# CPU fold: sum partial results across all tiles
fc1_mt      = tile_partial.sum(axis=0)[:fc1_info['M']]   # (140,)

# 7. Verify: single-tile == 4-tile sum
max_diff = int(np.abs(fc1_ref.astype(np.int64) - fc1_mt).max())
print(f"\nSingle-tile vs 4-tile max |diff| = {max_diff}  {'MATCH' if max_diff==0 else 'MISMATCH'}")

# 8. ReLU + shift-quantize (firmware FC1_SHIFT logic)
fc1_relu = np.maximum(fc1_mt, 0)
max_val  = int(fc1_relu.max()) if fc1_relu.max() > 0 else 1
shift    = 0
while (max_val >> shift) > 127:
    shift += 1
fc1_q = np.clip(fc1_relu >> shift, 0, 127).astype(np.int8)
print(f"FC1 quant: shift={shift}, max_accum={max_val}, range=[{fc1_q.min()},{fc1_q.max()}]")

# Pad fc1 output to FC2 K_pad=144
fc1_q_padded = np.zeros(w_fc2.shape[1], dtype=np.int8)
fc1_q_padded[:len(fc1_q)] = fc1_q

# 9. FC2 dense GEMV
fc2_out   = w_fc2[:fc2_info['M'], :].astype(np.int32) @ fc1_q_padded.astype(np.int32)
pred_int8 = int(np.argmax(fc2_out[:10]))

# 10. Results
golden = json.load(open(ROOT / "data/golden_reference_multitile.json"))
print()
print("=" * 55)
print(f"  Multi-tile INT8 BSR prediction : {pred_int8}")
print(f"  FP32 golden prediction         : {gold_pred}")
print(f"  Golden reference (DRAM gen)    : {golden.get('golden_prediction','?')}")
print(f"  True label                     : {true_label}")
print("=" * 55)

ok = (pred_int8 == true_label) and (max_diff == 0)
print("ACCUMULATION LOGIC: PASS" if ok else
      f"ACCUMULATION LOGIC: FAIL  (pred={pred_int8}, expected={true_label}, diff={max_diff})")
print(f"\nPer-tile NNZ blocks : {per_tile_nnz}  total={sum(per_tile_nnz)}")
print(f"Load imbalance      : {max(per_tile_nnz) - min(per_tile_nnz)} blocks")
