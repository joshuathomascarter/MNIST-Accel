#!/usr/bin/env python3
"""
gen_dram_init.py — Generate DRAM initialisation hex for end-to-end inference
============================================================================

Packs real INT8 weights and a real MNIST test image into a Verilog $readmemh
file that dram_phy_simple_mem.sv loads at simulation start.  This removes
every stub from the data path: the systolic array receives genuine weights
and activations, computes real MAC results, and the firmware reads back the
actual classification.

DRAM MEMORY MAP (word-addressed, 32-bit words):
    0x0000_0000  .. 0x000F_FFFF   Weight region   (up to 1 MB)
    0x0010_0000  .. 0x0013_FFFF   Activation region (fc1 input)
    0x0014_0000  .. 0x0014_FFFF   Output region   (scratch for results)
    0x0015_0000  .. 0x0015_003F   Metadata region (layer descriptors)

These are DRAM-internal word offsets.  The CPU sees DRAM at 0x4000_0000
(cached) or 0x6000_0000 (uncached), so firmware adds the base.

Layers packed into the weight region:
    fc1: 140 × 9216 INT8 padded to 16-aligned → stored as 32-bit words
    fc2:  16 × 144  INT8 padded to 16-aligned → stored after fc1

For conv layers, the design uses im2col → GEMM.  In this e2e test the
conv outputs are pre-computed in Python (golden reference) and the fc1
input vector is written to the activation region.

OUTPUT FILES:
    data/dram_init.hex          — $readmemh hex file for Verilator
    data/golden_reference.json  — expected classification + intermediate values
    data/inference_config.hex   — firmware-readable metadata (layer shapes)

Author: auto-generated for MNIST e2e verification
"""

import os
import sys
import json
import struct
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
INT8_DIR = os.path.join(ROOT_DIR, 'data', 'int8')
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'data', 'checkpoints', 'mnist_fp32.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data')

# Hardware constants
SYSTOLIC_DIM = 16  # 16×16 PE array
WORD_BYTES = 4     # 32-bit DRAM words

# DRAM layout (word addresses)
WEIGHT_BASE   = 0x00000000
ACT_BASE      = 0x00100000
OUTPUT_BASE   = 0x00140000
META_BASE     = 0x00150000


def pad_to_multiple(n, m):
    """Pad n up to next multiple of m."""
    return ((n + m - 1) // m) * m


def pack_int8_to_words(data_int8):
    """Pack a flat INT8 array into 32-bit words (4 bytes per word, little-endian)."""
    flat = data_int8.flatten().astype(np.int8)
    # Pad to multiple of 4 bytes
    pad_len = pad_to_multiple(len(flat), WORD_BYTES) - len(flat)
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int8)])
    # View as uint32 (little-endian: byte 0 is LSB)
    return flat.view(np.uint32)


def pack_int32_to_words(data_int32):
    """Pack a flat INT32 array into 32-bit words."""
    return data_int32.flatten().astype(np.uint32)


def pack_gemv_activation_blocks(data_int8, tile_dim):
    """
    Pack a length-K INT8 vector into tiled 16x16 activation blocks.

    The current hardware compute path behaves as a streamed outer-product
    microkernel. For GEMV, each K-tile is represented as a 16x16 activation
    block whose first streamed row holds the 16 input values and whose
    remaining rows are zero-padded.
    """
    padded_len = pad_to_multiple(len(data_int8), tile_dim)
    vec_padded = np.zeros(padded_len, dtype=np.int8)
    vec_padded[:len(data_int8)] = data_int8

    blocks = []
    for tile_base in range(0, padded_len, tile_dim):
        block = np.zeros((tile_dim, tile_dim), dtype=np.int8)
        block[0, :] = vec_padded[tile_base:tile_base + tile_dim]
        blocks.append(block)

    if not blocks:
        return np.zeros(0, dtype=np.uint32)

    return pack_int8_to_words(np.stack(blocks, axis=0))


def load_mnist_test_image(index=0):
    """Load a single MNIST test image and return (image_uint8, label)."""
    try:
        import torchvision
        from torchvision import transforms
        dataset = torchvision.datasets.MNIST(
            root=os.path.join(ROOT_DIR, 'data'),
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        img_tensor, label = dataset[index]
        # img_tensor is [1, 28, 28] float in [0, 1]
        return img_tensor.numpy(), label
    except Exception as e:
        print(f"Warning: Could not load MNIST dataset: {e}")
        print("Using synthetic test image (all zeros = should classify as 0 or near)")
        return np.zeros((1, 28, 28), dtype=np.float32), 0


def load_external_image(image_path):
    """Load an arbitrary image file and convert it into an MNIST-like tensor."""
    from PIL import Image

    img = Image.open(image_path).convert('L')
    arr = np.array(img)

    if arr.mean() > 128:
        arr = 255 - arr

    ys, xs = np.where(arr > 16)
    if len(xs) > 0 and len(ys) > 0:
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    cropped = Image.fromarray(arr)
    width, height = cropped.size
    if width == 0 or height == 0:
        cropped = Image.new('L', (20, 20), 0)
    else:
        scale = 20.0 / max(width, height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        cropped = cropped.resize((resized_w, resized_h), Image.LANCZOS)

    padded = Image.new('L', (28, 28), 0)
    x_off = (28 - cropped.size[0]) // 2
    y_off = (28 - cropped.size[1]) // 2
    padded.paste(cropped, (x_off, y_off))

    image_float = np.asarray(padded, dtype=np.float32) / 255.0
    return image_float[np.newaxis, :, :]


def run_conv_layers_golden(image_float, checkpoint_path):
    """
    Run conv1 + conv2 + pool + flatten in FP32 to get the fc1 input vector.
    Then quantize the fc1 input to INT8 using a simple per-tensor scale.
    Returns: (fc1_input_int8, fc1_input_scale, conv_output_fp32)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct just the conv layers
    conv1_w = checkpoint['state_dict']['conv1.weight']
    conv1_b = checkpoint['state_dict']['conv1.bias']
    conv2_w = checkpoint['state_dict']['conv2.weight']
    conv2_b = checkpoint['state_dict']['conv2.bias']

    x = torch.from_numpy(image_float).unsqueeze(0)  # [1, 1, 28, 28]

    # Conv1: (1,28,28) → (32,26,26)
    x = F.conv2d(x, conv1_w, conv1_b)
    x = F.relu(x)

    # Conv2: (32,26,26) → (64,24,24)
    x = F.conv2d(x, conv2_w, conv2_b)
    x = F.relu(x)

    # MaxPool2d(2): (64,24,24) → (64,12,12)
    x = F.max_pool2d(x, 2)

    # Flatten: (64,12,12) → (9216,)
    x = x.flatten()

    fc1_input_fp32 = x.detach().numpy()

    # Quantize fc1 input to INT8
    abs_max = np.abs(fc1_input_fp32).max()
    if abs_max < 1e-10:
        abs_max = 1.0
    fc1_input_scale = abs_max / 127.0
    fc1_input_int8 = np.clip(
        np.round(fc1_input_fp32 / fc1_input_scale),
        -127, 127
    ).astype(np.int8)

    return fc1_input_int8, fc1_input_scale, fc1_input_fp32


def compute_golden_fc(fc1_input_fp32, checkpoint_path):
    """
    Run fc1 + fc2 in FP32 to get the golden classification.
    Also returns the INT8-quantised fc2 input (= fc1 output after ReLU).
    """
    import torch
    import torch.nn.functional as F

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    fc1_w = checkpoint['state_dict']['fc1.weight']
    fc1_b = checkpoint['state_dict']['fc1.bias']
    fc2_w = checkpoint['state_dict']['fc2.weight']
    fc2_b = checkpoint['state_dict']['fc2.bias']

    x = torch.from_numpy(fc1_input_fp32).unsqueeze(0)  # [1, 9216]

    # FC1: 9216 → M (128 in checkpoint)
    x = F.linear(x, fc1_w, fc1_b)
    x = F.relu(x)
    fc1_output_fp32 = x.detach().numpy().flatten()

    # Quantise fc1 output → INT8 (this is the fc2 input on hardware)
    abs_max = np.abs(fc1_output_fp32).max()
    if abs_max < 1e-10:
        abs_max = 1.0
    fc2_input_scale = abs_max / 127.0
    fc2_input_int8 = np.clip(
        np.round(fc1_output_fp32 / fc2_input_scale),
        -127, 127
    ).astype(np.int8)

    # FC2: M → 10
    x2 = F.linear(x, fc2_w, fc2_b)

    logits = x2.detach().numpy().flatten()
    predicted = int(np.argmax(logits))

    return predicted, logits, fc2_input_int8, fc2_input_scale


def prepare_fc_weights():
    """
    Load weights directly from the FP32 checkpoint, quantise to INT8,
    pad to 16-aligned, and return as ready-to-pack data.

    Returns dict with layer info.
    """
    import torch
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

    layers = {}
    for name in ['fc1', 'fc2']:
        w_fp32 = checkpoint['state_dict'][f'{name}.weight'].numpy()
        # Per-channel quantisation (per output row)
        abs_max = np.abs(w_fp32).max(axis=1, keepdims=True)
        abs_max = np.where(abs_max < 1e-10, 1.0, abs_max)
        scales = abs_max / 127.0
        w_int8 = np.clip(np.round(w_fp32 / scales), -127, 127).astype(np.int8)

        M_orig, K_orig = w_int8.shape
        M_pad = pad_to_multiple(M_orig, SYSTOLIC_DIM)
        K_pad = pad_to_multiple(K_orig, SYSTOLIC_DIM)

        w_padded = np.zeros((M_pad, K_pad), dtype=np.int8)
        w_padded[:M_orig, :K_orig] = w_int8

        layers[name] = {
            'weights': w_padded,
            'M': M_orig,
            'K': K_orig,
            'M_padded': M_pad,
            'K_padded': K_pad,
            'scales': scales.flatten(),
        }

    return layers


def generate_dram_hex(dram_words, filepath):
    """Write a $readmemh compatible hex file."""
    with open(filepath, 'w') as f:
        for i, word in enumerate(dram_words):
            f.write(f'{int(word) & 0xFFFFFFFF:08x}\n')
    print(f"  Written {len(dram_words)} words ({len(dram_words)*4} bytes) to {filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate DRAM init hex for MNIST e2e inference')
    parser.add_argument('--image-index', type=int, default=0,
                        help='MNIST test set image index (default: 0)')
    parser.add_argument('--image-path', default=None,
                        help='Optional path to an external digit image')
    parser.add_argument('--true-label', type=int, default=None,
                        help='Optional ground-truth label for --image-path input')
    parser.add_argument('--mem-words', type=int, default=524288,
                        help='Total DRAM words (must match RTL MEM_WORDS)')
    parser.add_argument('--output', default=os.path.join(OUTPUT_DIR, 'dram_init.hex'),
                        help='Output hex file path')
    parser.add_argument('--full', action='store_true',
                        help='Pack FC1 weights + activations for full fc1+fc2 firmware '
                             '(generates inference_config_full.h). '
                             'Without this flag only FC2 is packed (fast sim mode).')
    args = parser.parse_args()

    MEM_WORDS = args.mem_words
    print("=" * 70)
    print("  MNIST E2E DRAM Initialisation Generator")
    print("=" * 70)

    # ---- Step 1: Load MNIST test image ----
    if args.image_path:
        print(f"\n[1] Loading external image: {args.image_path}...")
        image_float = load_external_image(args.image_path)
        true_label = args.true_label if args.true_label is not None else -1
    else:
        print(f"\n[1] Loading MNIST test image #{args.image_index}...")
        image_float, true_label = load_mnist_test_image(args.image_index)

    if true_label >= 0:
        print(f"    True label: {true_label}")
    else:
        print("    True label: n/a (external image)")
    print(f"    Image shape: {image_float.shape}")

    # ---- Step 2: Run conv layers to get fc1 input ----
    print(f"\n[2] Running conv layers (golden FP32)...")
    fc1_input_int8, fc1_input_scale, fc1_input_fp32 = \
        run_conv_layers_golden(image_float, CHECKPOINT_PATH)
    print(f"    fc1 input shape: {fc1_input_fp32.shape}")
    print(f"    fc1 input scale: {fc1_input_scale:.6f}")
    print(f"    fc1 input INT8 range: [{fc1_input_int8.min()}, {fc1_input_int8.max()}]")

    # ---- Step 3: Compute golden classification ----
    print(f"\n[3] Computing golden classification (FP32)...")
    golden_class, golden_logits, fc2_input_int8, fc2_input_scale = \
        compute_golden_fc(fc1_input_fp32, CHECKPOINT_PATH)
    print(f"    Golden prediction: {golden_class}")
    print(f"    Golden logits: {golden_logits}")
    if true_label >= 0:
        print(f"    Correct: {golden_class == true_label}")
    else:
        print("    Correct: n/a (no external label provided)")
    print(f"    fc2 input (fc1 output) INT8 shape: {fc2_input_int8.shape}")
    print(f"    fc2 input scale: {fc2_input_scale:.6f}")

    # ---- Step 4: Prepare FC weights ----
    print(f"\n[4] Preparing FC weight matrices...")
    fc_layers = prepare_fc_weights()
    for name, info in fc_layers.items():
        print(f"    {name}: {info['M']}×{info['K']} → padded {info['M_padded']}×{info['K_padded']}")
        print(f"           {info['weights'].nbytes} bytes")

    # ---- Step 5: Pack everything into DRAM ----
    # For the e2e simulation test we run fc2 only (10×140 → 10 classes).
    # FC1 is 140×9216 = 5184 tile ops, too slow for simulation.
    # The fc1 output (= fc2 input) is pre-computed by Python and stored
    # as activations in DRAM.  This still exercises the full datapath:
    #   DRAM → DMA → NoC → tile scratchpad → systolic 16×16 → scratchpad
    #   → DMA → DRAM → CPU argmax → UART print
    print(f"\n[5] Packing into DRAM ({MEM_WORDS} words = {MEM_WORDS*4} bytes)...")
    dram = np.zeros(MEM_WORDS, dtype=np.uint32)
    word_ptr = 0

    # Pack fc2 weights as KxN blocks for the current hardware datapath.
    # PyTorch stores fc2 as [N x K]; the tile expects 16x16 blocks laid out as
    # [K_tile x N_tile] so each PE row corresponds to one K element.
    fc2_words = pack_int8_to_words(fc_layers['fc2']['weights'].T)
    fc2_weight_offset = word_ptr
    dram[word_ptr:word_ptr + len(fc2_words)] = fc2_words
    word_ptr += len(fc2_words)
    print(f"    fc2 weights: offset=0x{fc2_weight_offset:06x}, {len(fc2_words)} words")

    # Pack fc2 input activations as 16x16 streamed blocks.
    # Row 0 carries the 16 FC2 input values for the K tile; rows 1..15 are 0.
    act_words = pack_gemv_activation_blocks(fc2_input_int8, SYSTOLIC_DIM)
    act_offset = word_ptr
    dram[word_ptr:word_ptr + len(act_words)] = act_words
    word_ptr += len(act_words)
    print(f"    fc2 activations: offset=0x{act_offset:06x}, {len(act_words)} words")

    M_pad = fc_layers['fc2']['M_padded']
    K_pad = fc_layers['fc2']['K_padded']
    num_m_tiles = M_pad // SYSTOLIC_DIM
    num_k_tiles = K_pad // SYSTOLIC_DIM

    # Reserve one 16x16 INT32 partial-product tile per K tile.
    output_offset = word_ptr
    output_words = num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM)
    print(f"    output region: offset=0x{output_offset:06x}, {output_words} words reserved")
    word_ptr += output_words

    # ── Optional: pack FC1 weights + activations (--full mode) ──────────────
    fc1_weight_offset = None
    fc1_act_offset_full = None
    fc1_partial_offset = None
    fc1_out_offset = None
    fc2_act_offset_full = None
    fc2_output_offset_full = None

    if args.full:
        print(f"\n[5b] --full: packing FC1 weights + activations...")

        # Align to a clean boundary (round up to next 4096-word page)
        word_ptr = ((word_ptr + 4095) // 4096) * 4096

        # FC1 weights in BSR (Block Sparse Row) format — M-major, 16×16 blocks.
        # Only nonzero blocks are stored; a CSR-like indptr/indices header
        # lets the firmware skip zero blocks entirely (10× memory + speed win).
        #
        # DRAM layout (all at page-aligned base):
        #   fc1_bsr_indptr  [M_tiles + 1]  INT32 words  (row block pointers)
        #   fc1_bsr_indices [NNZ]           INT32 words  (K-block column indices)
        #   fc1_bsr_data    [NNZ × 64]      INT32 words  (16×16 INT8 blocks packed 4/word)
        fc1_info = fc_layers['fc1']
        fc1_m_pad   = fc1_info['M_padded']   # 144
        fc1_k_pad   = fc1_info['K_padded']   # 9216
        fc1_m_tiles = fc1_m_pad // SYSTOLIC_DIM   # 9
        fc1_k_tiles = fc1_k_pad // SYSTOLIC_DIM   # 576

        w_fc1 = fc1_info['weights']   # [M_pad × K_pad] INT8

        # Build BSR structure
        bsr_indptr_list  = []
        bsr_indices_list = []
        bsr_data_list    = []
        nnz_count = 0
        for m in range(fc1_m_tiles):
            bsr_indptr_list.append(nnz_count)
            r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
            for k in range(fc1_k_tiles):
                c0, c1 = k * SYSTOLIC_DIM, (k + 1) * SYSTOLIC_DIM
                blk = w_fc1[r0:r1, c0:c1]   # (16, 16) INT8
                if np.any(blk != 0):
                    bsr_indices_list.append(k)
                    bsr_data_list.append(blk.flatten())
                    nnz_count += 1
        bsr_indptr_list.append(nnz_count)   # sentinel

        bsr_indptr  = np.array(bsr_indptr_list,  dtype=np.uint32)   # 10 words
        bsr_indices = np.array(bsr_indices_list, dtype=np.uint32)   # NNZ words
        bsr_data    = np.concatenate(bsr_data_list).astype(np.int8) if bsr_data_list \
                      else np.zeros(0, dtype=np.int8)               # NNZ×256 INT8

        dense_total  = fc1_m_tiles * fc1_k_tiles
        bsr_sparsity = 100.0 * (1 - nnz_count / dense_total) if dense_total else 0.0
        print(f"    fc1 BSR: {nnz_count}/{dense_total} nonzero 16×16 blocks "
              f"({bsr_sparsity:.1f}% sparse)")

        # Pack indptr
        fc1_bsr_indptr_offset = word_ptr
        dram[word_ptr:word_ptr + len(bsr_indptr)] = bsr_indptr
        word_ptr += len(bsr_indptr)

        # Pack indices
        fc1_bsr_indices_offset = word_ptr
        dram[word_ptr:word_ptr + len(bsr_indices)] = bsr_indices
        word_ptr += len(bsr_indices)

        # Pack data blocks (NNZ × 64 words)
        fc1_bsr_data_offset = word_ptr
        fc1_bsr_data_words  = pack_int8_to_words(bsr_data)
        dram[word_ptr:word_ptr + len(fc1_bsr_data_words)] = fc1_bsr_data_words
        word_ptr += len(fc1_bsr_data_words)

        # Use fc1_weight_offset to point at indptr (firmware needs its base)
        fc1_weight_offset = fc1_bsr_indptr_offset
        bsr_words_total   = len(bsr_indptr) + len(bsr_indices) + len(fc1_bsr_data_words)
        print(f"    fc1 BSR storage: {bsr_words_total} words "
              f"(vs {fc1_m_tiles * fc1_k_tiles * SYSTOLIC_DIM * SYSTOLIC_DIM // 4} dense) "
              f"— {(fc1_m_tiles * fc1_k_tiles * SYSTOLIC_DIM * SYSTOLIC_DIM // 4) // max(1, bsr_words_total):.1f}× reduction")

        # FC1 activations: fc1_input_int8 (9216-dim) as GEMV blocks
        fc1_act_words = pack_gemv_activation_blocks(fc1_input_int8, SYSTOLIC_DIM)
        fc1_act_offset_full = word_ptr
        dram[word_ptr:word_ptr + len(fc1_act_words)] = fc1_act_words
        word_ptr += len(fc1_act_words)
        print(f"    fc1 activations: offset=0x{fc1_act_offset_full:06x}, {len(fc1_act_words)} words")

        # FC1 scratch: single 16×16 partial product (16×16 INT32 = 256 words)
        fc1_partial_offset = word_ptr
        word_ptr += SYSTOLIC_DIM * SYSTOLIC_DIM
        print(f"    fc1 partial scratch: offset=0x{fc1_partial_offset:06x}, "
              f"{SYSTOLIC_DIM*SYSTOLIC_DIM} words")

        # FC1 output: FC1_M_PADDED INT32 words (after ReLU, before quantise)
        fc1_out_offset = word_ptr
        word_ptr += fc1_m_pad
        print(f"    fc1 output: offset=0x{fc1_out_offset:06x}, {fc1_m_pad} words")

        # FC2 activation scratch (firmware writes INT8 GEMV blocks here)
        fc2_act_offset_full = word_ptr
        word_ptr += num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM // 4)  # INT8 words
        print(f"    fc2 activations scratch: offset=0x{fc2_act_offset_full:06x}, "
              f"{word_ptr - fc2_act_offset_full} words")

        # FC2 output scratch (for full firmware; separate from single-layer output)
        fc2_output_offset_full = word_ptr
        word_ptr += num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM)
        print(f"    fc2 output scratch: offset=0x{fc2_output_offset_full:06x}, "
              f"{num_k_tiles * SYSTOLIC_DIM * SYSTOLIC_DIM} words")

        total_used_full = word_ptr
        print(f"    Full DRAM used: {total_used_full}/{MEM_WORDS} words "
              f"({100*total_used_full/MEM_WORDS:.1f}%)")

    # Pack metadata at end
    meta_offset = word_ptr
    # Metadata format: single-layer descriptor for fc2
    # [magic, M_padded, K_padded, M_orig, wgt_offset, act_offset, out_offset,
    #  num_m_tiles, num_k_tiles, true_label, golden_class, sentinel]
    meta_true_label = int(true_label) if true_label >= 0 else 0xFFFFFFFF
    meta = np.array([
        0xACCE1000,                     # magic
        M_pad,                          # M padded
        K_pad,                          # K padded
        fc_layers['fc2']['M'],          # M original
        fc2_weight_offset,              # weight word offset in DRAM
        act_offset,                     # activation word offset in DRAM
        output_offset,                  # output word offset in DRAM
        num_m_tiles,                    # number of M tiles
        num_k_tiles,                    # number of K tiles
        meta_true_label,                # ground truth label or unknown sentinel
        int(golden_class),              # expected prediction
        0xDEAD_BEEF,                    # sentinel
    ], dtype=np.uint32)
    if meta_offset + len(meta) <= MEM_WORDS:
        dram[meta_offset:meta_offset + len(meta)] = meta
        print(f"    metadata: offset=0x{meta_offset:06x}, {len(meta)} words")

    total_used = word_ptr + len(meta)
    print(f"\n    Total DRAM used: {total_used}/{MEM_WORDS} words ({100*total_used/MEM_WORDS:.1f}%)")

    # ---- Step 6: Write hex file ----
    print(f"\n[6] Writing DRAM hex...")
    generate_dram_hex(dram, args.output)

    # ---- Step 7: Write golden reference ----
    golden_path = os.path.join(OUTPUT_DIR, 'golden_reference.json')
    golden = {
        'image_index': args.image_index,
        'image_path': os.path.abspath(args.image_path) if args.image_path else None,
        'true_label': int(true_label) if true_label >= 0 else None,
        'golden_prediction': int(golden_class),
        'golden_logits': [float(x) for x in golden_logits],
        'fc2_input_scale': float(fc2_input_scale),
        'fc2_input_nonzero': int(np.count_nonzero(fc2_input_int8)),
        'fc2_input_total': int(len(fc2_input_int8)),
        'dram_layout': {
            'fc2_weight_offset': int(fc2_weight_offset),
            'activation_offset': int(act_offset),
            'output_offset': int(output_offset),
            'metadata_offset': int(meta_offset),
        },
        'fc2_shape': {
            'M': fc_layers['fc2']['M'], 'K': fc_layers['fc2']['K'],
            'M_padded': fc_layers['fc2']['M_padded'], 'K_padded': fc_layers['fc2']['K_padded'],
            'num_m_tiles': int(num_m_tiles), 'num_k_tiles': int(num_k_tiles),
        },
        'mem_words': MEM_WORDS,
    }
    with open(golden_path, 'w') as f:
        json.dump(golden, f, indent=2)
    print(f"  Written golden reference to {golden_path}")

    # ---- Step 8: Write firmware config header ----
    config_path = os.path.join(ROOT_DIR, 'fw', 'inference_config.h')
    with open(config_path, 'w') as f:
        f.write("/* Auto-generated by gen_dram_init.py — DO NOT EDIT */\n")
        f.write("#ifndef INFERENCE_CONFIG_H\n")
        f.write("#define INFERENCE_CONFIG_H\n\n")
        f.write(f"#define DRAM_BASE_UC        0x60000000u\n")
        f.write(f"#define DRAM_BASE_CACHED    0x40000000u\n\n")
        f.write(f"/* FC2 layer: {fc_layers['fc2']['M']}x{fc_layers['fc2']['K']} */\n")
        f.write(f"#define FC2_WEIGHT_OFFSET   0x{fc2_weight_offset:08x}u  /* word offset */\n")
        f.write(f"#define FC2_M_ORIG          {fc_layers['fc2']['M']}u\n")
        f.write(f"#define FC2_K_ORIG          {fc_layers['fc2']['K']}u\n")
        f.write(f"#define FC2_M_PADDED        {fc_layers['fc2']['M_padded']}u\n")
        f.write(f"#define FC2_K_PADDED        {fc_layers['fc2']['K_padded']}u\n")
        f.write(f"#define FC2_NUM_M_TILES     {num_m_tiles}u\n")
        f.write(f"#define FC2_NUM_K_TILES     {num_k_tiles}u\n\n")
        f.write(f"#define ACT_OFFSET          0x{act_offset:08x}u  /* word offset */\n")
        f.write(f"#define OUTPUT_OFFSET       0x{output_offset:08x}u  /* word offset */\n\n")
        f.write(f"#define SYSTOLIC_DIM        {SYSTOLIC_DIM}u\n")
        f.write(f"#define BLOCK_WORDS         ({SYSTOLIC_DIM}u * {SYSTOLIC_DIM}u / 4u)  /* 64 words per 16x16 INT8 block */\n")
        f.write(f"#define ACT_BLOCK_WORDS     BLOCK_WORDS  /* 64 words per 16x16 activation block */\n\n")
        if true_label >= 0:
            f.write(f"#define TRUE_LABEL          {int(true_label)}u\n")
        else:
            f.write("#define TRUE_LABEL          0xFFFFFFFFu\n")
        f.write(f"#define GOLDEN_PREDICTION   {int(golden_class)}u\n\n")
        f.write("#endif /* INFERENCE_CONFIG_H */\n")
    print(f"  Written firmware config to {config_path}")

    # ---- Step 8b: Write full firmware config header (--full only) ----------
    if args.full:
        full_config_path = os.path.join(ROOT_DIR, 'fw', 'inference_config_full.h')
        fc1_info = fc_layers['fc1']
        fc1_m_pad   = fc1_info['M_padded']
        fc1_k_pad   = fc1_info['K_padded']
        fc1_m_tiles = fc1_m_pad // SYSTOLIC_DIM
        fc1_k_tiles = fc1_k_pad // SYSTOLIC_DIM
        with open(full_config_path, 'w') as f:
            f.write("/* Auto-generated by gen_dram_init.py --full — DO NOT EDIT */\n")
            f.write("#ifndef INFERENCE_CONFIG_FULL_H\n")
            f.write("#define INFERENCE_CONFIG_FULL_H\n\n")
            f.write(f"#define DRAM_BASE_UC        0x60000000u\n")
            f.write(f"#define DRAM_BASE_CACHED    0x40000000u\n\n")
            f.write(f"/* FC1 BSR layer: {fc1_info['M']}x{fc1_info['K']} — {nnz_count} nonzero 16x16 blocks */\n")
            f.write(f"#define FC1_BSR_INDPTR_OFFSET  0x{fc1_bsr_indptr_offset:08x}u  /* {fc1_m_tiles+1} words */\n")
            f.write(f"#define FC1_BSR_INDICES_OFFSET 0x{fc1_bsr_indices_offset:08x}u  /* {nnz_count} words */\n")
            f.write(f"#define FC1_BSR_DATA_OFFSET    0x{fc1_bsr_data_offset:08x}u  /* NNZ*64 words */\n")
            f.write(f"#define FC1_BSR_NNZ            {nnz_count}u\n")
            f.write(f"#define FC1_M_ORIG          {fc1_info['M']}u\n")
            f.write(f"#define FC1_K_ORIG          {fc1_info['K']}u\n")
            f.write(f"#define FC1_M_PADDED        {fc1_m_pad}u\n")
            f.write(f"#define FC1_K_PADDED        {fc1_k_pad}u\n")
            f.write(f"#define FC1_NUM_M_TILES     {fc1_m_tiles}u\n")
            f.write(f"#define FC1_NUM_K_TILES     {fc1_k_tiles}u\n")
            f.write(f"#define FC1_ACT_OFFSET      0x{fc1_act_offset_full:08x}u\n")
            f.write(f"#define FC1_PARTIAL_OFFSET  0x{fc1_partial_offset:08x}u\n")
            f.write(f"#define FC1_OUT_OFFSET      0x{fc1_out_offset:08x}u\n\n")
            f.write(f"/* FC2 layer: {fc_layers['fc2']['M']}x{fc_layers['fc2']['K']} */\n")
            f.write(f"#define FC2_WEIGHT_OFFSET   0x{fc2_weight_offset:08x}u\n")
            f.write(f"#define FC2_M_ORIG          {fc_layers['fc2']['M']}u\n")
            f.write(f"#define FC2_K_ORIG          {fc_layers['fc2']['K']}u\n")
            f.write(f"#define FC2_M_PADDED        {fc_layers['fc2']['M_padded']}u\n")
            f.write(f"#define FC2_K_PADDED        {fc_layers['fc2']['K_padded']}u\n")
            f.write(f"#define FC2_NUM_M_TILES     {num_m_tiles}u\n")
            f.write(f"#define FC2_NUM_K_TILES     {num_k_tiles}u\n")
            f.write(f"#define FC2_ACT_OFFSET      0x{fc2_act_offset_full:08x}u\n")
            f.write(f"#define FC2_OUTPUT_OFFSET   0x{fc2_output_offset_full:08x}u\n\n")
            f.write(f"#define SYSTOLIC_DIM        {SYSTOLIC_DIM}u\n")
            f.write(f"#define BLOCK_WORDS         ({SYSTOLIC_DIM}u * {SYSTOLIC_DIM}u / 4u)\n\n")
            if true_label >= 0:
                f.write(f"#define TRUE_LABEL          {int(true_label)}u\n")
            else:
                f.write("#define TRUE_LABEL          0xFFFFFFFFu\n")
            f.write(f"#define GOLDEN_PREDICTION   {int(golden_class)}u\n\n")
            f.write("#endif /* INFERENCE_CONFIG_FULL_H */\n")
        print(f"  Written full firmware config to {full_config_path}")

    print(f"\n{'='*70}")
    if args.image_path:
        print(f"  DONE — External image: golden={golden_class}")
    else:
        print(f"  DONE — Image #{args.image_index}: true={true_label}, golden={golden_class}")
    print(f"  Ready for simulation: verilator ... +DRAM_INIT=data/dram_init.hex")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
