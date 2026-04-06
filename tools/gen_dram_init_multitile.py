#!/usr/bin/env python3
"""
gen_dram_init_multitile.py — Generate DRAM init hex for multi-tile FC1+FC2 inference
======================================================================================

Partitions the FC1 BSR (block sparse row) weight matrix across N_PARALLEL=4
worker tiles.  For each M-tile (output-row block), K-blocks are assigned
round-robin to the N_PARALLEL workers:

    worker w gets K-blocks j where  j % N_PARALLEL == w

Each worker tile has its own:
  - fc1_mt_indptr[w]   : row-pointers into its local index/data arrays (9+1 words)
  - fc1_mt_indices[w]  : K-column indices for non-zero blocks assigned to tile w
  - fc1_mt_data[w]     : 16×16 INT8 blocks (packed 4/word, 64 words each)

The shared FC1 activation vector (unchanged) and per-tile partial scratch slots
are also packed.

DRAM MEMORY MAP (word offsets, 32-bit words):
  [FC2 region identical to single-tile gen_dram_init.py --full]
  [FC1 multi-tile region]
    per_tile[0] : indptr, indices, data (tile 0)
    per_tile[1] : indptr, indices, data (tile 1)
    per_tile[2] : indptr, indices, data (tile 2)
    per_tile[3] : indptr, indices, data (tile 3)
    fc1_activation : shared activation blocks
    fc1_partial[0..N_PARALLEL-1] : per-tile partial-sum scratch (SYSTOLIC_DIM words each)
    fc1_output  : FC1_M_PADDED INT32 words (accumulated after all M-tiles)
    [FC2 activation & output scratch — same as single-tile]

Outputs:
  data/dram_init_multitile.hex   — $readmemh hex
  data/golden_reference.json     — updated with multitile layout info
  fw/inference_config_multitile.h — firmware constants

Usage:
    python3 tools/gen_dram_init_multitile.py [--image-index N]
"""

import os
import sys
import json
import struct
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'data', 'checkpoints', 'mnist_fp32.pt')
OUTPUT_DIR      = os.path.join(ROOT_DIR, 'data')

# Re-use helper functions from gen_dram_init.py by importing it
sys.path.insert(0, SCRIPT_DIR)
import gen_dram_init as _gdi

# --------------------------------------------------------------------------
# Multi-tile constants
# --------------------------------------------------------------------------
N_PARALLEL   = 4   # number of worker tiles for FC1
ROOT_TILE    = 0   # tile that owns IN-network reduce sink
SYSTOLIC_DIM = 16
WORD_BYTES   = 4
BLOCK_WORDS  = SYSTOLIC_DIM * SYSTOLIC_DIM // 4  # 64 words per 16×16 INT8 block


# --------------------------------------------------------------------------
# Helpers (re-use from gen_dram_init)
# --------------------------------------------------------------------------
def pad_to_multiple(n, m):
    return ((n + m - 1) // m) * m


def pack_int8_to_words(data_int8):
    return _gdi.pack_int8_to_words(data_int8)


def pack_gemv_activation_blocks(data_int8, tile_dim):
    return _gdi.pack_gemv_activation_blocks(data_int8, tile_dim)


def generate_dram_hex(dram_words, filepath):
    _gdi.generate_dram_hex(dram_words, filepath)


# --------------------------------------------------------------------------
# Multi-tile BSR partition
# --------------------------------------------------------------------------
def partition_bsr(fc1_info, n_parallel=N_PARALLEL):
    """
    Partition the FC1 BSR weight matrix across n_parallel worker tiles.

    For each M-tile m, non-zero K-blocks are assigned round-robin:
        block at position j within M-tile m → worker j % n_parallel

    Returns:
      tile_indptr  [n_parallel][M_tiles+1]  — per-tile row-block pointers
      tile_indices [n_parallel][NNZ_tile]   — K-column indices per tile
      tile_data    [n_parallel][NNZ_tile×256]  — INT8 blocks per tile
    """
    w_fc1    = fc1_info['weights']   # [M_pad × K_pad] INT8
    M_padded = fc1_info['M_padded']
    K_padded = fc1_info['K_padded']
    M_tiles  = M_padded // SYSTOLIC_DIM
    K_tiles  = K_padded // SYSTOLIC_DIM

    tile_indptr  = [[] for _ in range(n_parallel)]
    tile_indices = [[] for _ in range(n_parallel)]
    tile_data    = [[] for _ in range(n_parallel)]
    tile_nnz     = [0]  * n_parallel

    for m in range(M_tiles):
        r0, r1 = m * SYSTOLIC_DIM, (m + 1) * SYSTOLIC_DIM
        # collect ALL nonzero K-blocks for this M-tile
        nnz_for_m = []
        for k in range(K_tiles):
            c0, c1 = k * SYSTOLIC_DIM, (k + 1) * SYSTOLIC_DIM
            blk = w_fc1[r0:r1, c0:c1]
            if np.any(blk != 0):
                nnz_for_m.append((k, blk.flatten()))

        # update per-tile indptr (cumulative count up to M-tile m)
        for w in range(n_parallel):
            tile_indptr[w].append(tile_nnz[w])

        # assign blocks round-robin to worker tiles
        for local_j, (k, blk) in enumerate(nnz_for_m):
            w = local_j % n_parallel
            tile_indices[w].append(k)
            tile_data[w].append(blk)
            tile_nnz[w] += 1

    # sentinel indptr entry
    for w in range(n_parallel):
        tile_indptr[w].append(tile_nnz[w])

    # convert to numpy
    out_indptr  = [np.array(tile_indptr[w],  dtype=np.uint32) for w in range(n_parallel)]
    out_indices = [np.array(tile_indices[w], dtype=np.uint32) if tile_indices[w]
                   else np.zeros(0, dtype=np.uint32)
                   for w in range(n_parallel)]
    out_data    = [np.concatenate(tile_data[w]).astype(np.int8) if tile_data[w]
                   else np.zeros(0, dtype=np.int8)
                   for w in range(n_parallel)]

    stats = {
        'M_tiles': M_tiles,
        'K_tiles': K_tiles,
        'global_nnz': sum(tile_nnz),
        'per_tile_nnz': tile_nnz,
    }
    return out_indptr, out_indices, out_data, stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate multi-tile DRAM init hex')
    parser.add_argument('--image-index', type=int, default=0)
    parser.add_argument('--image-path',  default=None)
    parser.add_argument('--true-label',  type=int, default=None)
    parser.add_argument('--mem-words',   type=int, default=524288)
    parser.add_argument('--output', default=os.path.join(OUTPUT_DIR,'dram_init_multitile.hex'))
    args = parser.parse_args()

    MEM_WORDS = args.mem_words
    print("=" * 70)
    print("  MNIST Multi-Tile DRAM Init Generator  (N_PARALLEL=%d)" % N_PARALLEL)
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load MNIST test image                                            #
    # ------------------------------------------------------------------ #
    if args.image_path:
        print(f"\n[1] Loading external image: {args.image_path}...")
        image_float = _gdi.load_external_image(args.image_path)
        true_label  = args.true_label if args.true_label is not None else -1
    else:
        print(f"\n[1] Loading MNIST test image #{args.image_index}...")
        image_float, true_label = _gdi.load_mnist_test_image(args.image_index)
    print(f"    True label : {true_label if true_label >= 0 else 'n/a'}")

    # ------------------------------------------------------------------ #
    # 2. Run conv layers → fc1 input                                      #
    # ------------------------------------------------------------------ #
    print("\n[2] Running conv layers (golden FP32)...")
    fc1_input_int8, fc1_input_scale, fc1_input_fp32 = \
        _gdi.run_conv_layers_golden(image_float, CHECKPOINT_PATH)
    print(f"    fc1 input shape: {fc1_input_fp32.shape}")

    # ------------------------------------------------------------------ #
    # 3. Golden classification                                            #
    # ------------------------------------------------------------------ #
    print("\n[3] Computing golden classification...")
    golden_class, golden_logits, fc2_input_int8, fc2_input_scale = \
        _gdi.compute_golden_fc(fc1_input_fp32, CHECKPOINT_PATH)
    print(f"    Golden prediction: {golden_class}")

    # ------------------------------------------------------------------ #
    # 4. Prepare weights                                                  #
    # ------------------------------------------------------------------ #
    print("\n[4] Preparing FC weight matrices...")
    fc_layers = _gdi.prepare_fc_weights()
    for name, info in fc_layers.items():
        print(f"    {name}: {info['M']}×{info['K']} → padded {info['M_padded']}×{info['K_padded']}")

    # ------------------------------------------------------------------ #
    # 5. Pack FC2 region (identical to single-tile gen)                   #
    # ------------------------------------------------------------------ #
    print("\n[5] Packing FC2 region...")
    dram = np.zeros(MEM_WORDS, dtype=np.uint32)
    word_ptr = 0

    fc2_info   = fc_layers['fc2']
    M_pad      = fc2_info['M_padded']
    K_pad      = fc2_info['K_padded']
    num_m_tiles = M_pad // SYSTOLIC_DIM
    num_k_tiles = K_pad // SYSTOLIC_DIM

    fc2_words = pack_int8_to_words(fc2_info['weights'].T)
    fc2_weight_offset = word_ptr
    dram[word_ptr:word_ptr + len(fc2_words)] = fc2_words
    word_ptr += len(fc2_words)
    print(f"    fc2 weights : offset=0x{fc2_weight_offset:06x}, {len(fc2_words)} words")

    act_words  = pack_gemv_activation_blocks(fc2_input_int8, SYSTOLIC_DIM)
    act_offset = word_ptr
    dram[word_ptr:word_ptr + len(act_words)] = act_words
    word_ptr += len(act_words)
    print(f"    fc2 act     : offset=0x{act_offset:06x}, {len(act_words)} words")

    output_offset = word_ptr
    output_words  = num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM)
    word_ptr += output_words
    print(f"    fc2 output  : offset=0x{output_offset:06x}, {output_words} words")

    # ------------------------------------------------------------------ #
    # 6. Pack FC1 multi-tile BSR region                                   #
    # ------------------------------------------------------------------ #
    print("\n[6] Partitioning FC1 BSR across %d tiles..." % N_PARALLEL)
    fc1_info = fc_layers['fc1']
    tile_indptr, tile_indices, tile_data, stats = partition_bsr(fc1_info)

    M_tiles = stats['M_tiles']
    K_tiles = stats['K_tiles']
    total_nnz = stats['global_nnz']
    print(f"    FC1 {fc1_info['M']}×{fc1_info['K']} : {total_nnz} NNZ blocks, "
          f"{stats['global_nnz']}/{M_tiles*K_tiles} non-zeros, "
          f"per-tile NNZ: {stats['per_tile_nnz']}")

    # Align FC1 region to 4096-word page boundary
    word_ptr = ((word_ptr + 4095) // 4096) * 4096
    fc1_region_base = word_ptr

    indptr_offsets  = []
    indices_offsets = []
    data_offsets    = []

    for w in range(N_PARALLEL):
        # write indptr
        ip_off = word_ptr
        dram[word_ptr:word_ptr + len(tile_indptr[w])] = tile_indptr[w]
        word_ptr += len(tile_indptr[w])
        indptr_offsets.append(ip_off)

        # write indices
        idx_off = word_ptr
        if len(tile_indices[w]) > 0:
            dram[word_ptr:word_ptr + len(tile_indices[w])] = tile_indices[w]
        word_ptr += len(tile_indices[w])
        indices_offsets.append(idx_off)

        # write data blocks (packed INT8 → 32-bit words)
        dat_off = word_ptr
        if len(tile_data[w]) > 0:
            data_words = pack_int8_to_words(tile_data[w])
            dram[word_ptr:word_ptr + len(data_words)] = data_words
            word_ptr += len(data_words)
        data_offsets.append(dat_off)

        nnz_w = stats['per_tile_nnz'][w]
        print(f"    tile {w}: indptr@0x{ip_off:06x}  indices@0x{idx_off:06x}  "
              f"data@0x{dat_off:06x}  ({nnz_w} NNZ blocks)")

    # Shared FC1 activation
    fc1_act_words  = pack_gemv_activation_blocks(fc1_input_int8, SYSTOLIC_DIM)
    fc1_act_offset = word_ptr
    dram[word_ptr:word_ptr + len(fc1_act_words)] = fc1_act_words
    word_ptr += len(fc1_act_words)
    print(f"    fc1 act       : offset=0x{fc1_act_offset:06x}, {len(fc1_act_words)} words")

    # Per-tile partial-sum scratch (SYSTOLIC_DIM INT32 words for accumulator
    # + SYSTOLIC_DIM*SYSTOLIC_DIM INT32 words for per-block output staging)
    PARTIAL_SLOT = SYSTOLIC_DIM + SYSTOLIC_DIM * SYSTOLIC_DIM  # 16 + 256 = 272 words
    partial_offsets = []
    for w in range(N_PARALLEL):
        partial_offsets.append(word_ptr)
        word_ptr += PARTIAL_SLOT
    print(f"    fc1 partial   : tile 0 offset=0x{partial_offsets[0]:06x}, "
          f"stride=0x{PARTIAL_SLOT:03x} ({PARTIAL_SLOT} words ea.)")

    # FC1 output (FC1_M_PADDED INT32 words = accumulated, post-ReLU)
    fc1_m_padded   = fc1_info['M_padded']
    fc1_out_offset = word_ptr
    word_ptr      += fc1_m_padded
    print(f"    fc1 output    : offset=0x{fc1_out_offset:06x}, {fc1_m_padded} words")

    # FC2 activation scratch (INT8 GEMV blocks, rewritten from FC1 output)
    fc2_act_offset_full = word_ptr
    word_ptr += num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM // 4)
    print(f"    fc2 act full  : offset=0x{fc2_act_offset_full:06x}")

    # FC2 output scratch
    fc2_output_offset_full = word_ptr
    word_ptr += num_k_tiles * (SYSTOLIC_DIM * SYSTOLIC_DIM)
    print(f"    fc2 out full  : offset=0x{fc2_output_offset_full:06x}")

    print(f"\n    Total DRAM used: {word_ptr}/{MEM_WORDS} words "
          f"({100.0*word_ptr/MEM_WORDS:.1f}%)")

    # ------------------------------------------------------------------ #
    # 7. Write DRAM hex                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[7] Writing DRAM hex to {args.output}...")
    generate_dram_hex(dram, args.output)

    # ------------------------------------------------------------------ #
    # 8. Write firmware config header                                     #
    # ------------------------------------------------------------------ #
    config_path = os.path.join(ROOT_DIR, 'fw', 'inference_config_multitile.h')
    print(f"\n[8] Writing firmware config to {config_path}...")
    with open(config_path, 'w') as f:
        f.write("/* Auto-generated by gen_dram_init_multitile.py — DO NOT EDIT */\n")
        f.write("#ifndef INFERENCE_CONFIG_MULTITILE_H\n")
        f.write("#define INFERENCE_CONFIG_MULTITILE_H\n\n")

        f.write("#include <stdint.h>\n\n")

        f.write("/* ---- DRAM base addresses ---- */\n")
        f.write("#define DRAM_BASE_UC        0x60000000u\n")
        f.write("#define DRAM_BASE_CACHED    0x40000000u\n\n")

        f.write("/* ---- Multi-tile geometry ---- */\n")
        f.write(f"#define N_PARALLEL          {N_PARALLEL}u\n")
        f.write(f"#define ROOT_TILE           {ROOT_TILE}u\n\n")

        f.write("/* ---- FC1 dimensions ---- */\n")
        f.write(f"#define FC1_M_ORIG          {fc1_info['M']}u\n")
        f.write(f"#define FC1_K_ORIG          {fc1_info['K']}u\n")
        f.write(f"#define FC1_M_PADDED        {fc1_info['M_padded']}u\n")
        f.write(f"#define FC1_K_PADDED        {fc1_info['K_padded']}u\n")
        f.write(f"#define FC1_NUM_M_TILES     {M_tiles}u\n")
        f.write(f"#define FC1_NUM_K_TILES     {K_tiles}u\n\n")

        f.write("/* ---- FC2 dimensions ---- */\n")
        f.write(f"#define FC2_M_ORIG          {fc2_info['M']}u\n")
        f.write(f"#define FC2_K_ORIG          {fc2_info['K']}u\n")
        f.write(f"#define FC2_M_PADDED        {fc2_info['M_padded']}u\n")
        f.write(f"#define FC2_K_PADDED        {fc2_info['K_padded']}u\n")
        f.write(f"#define FC2_NUM_M_TILES     {num_m_tiles}u\n")
        f.write(f"#define FC2_NUM_K_TILES     {num_k_tiles}u\n\n")

        f.write("/* ---- Scratchpad layout (words) ---- */\n")
        f.write("#define SYSTOLIC_DIM        16u\n")
        f.write(f"#define BLOCK_WORDS         {BLOCK_WORDS}u  /* 64 words per 16×16 INT8 block */\n")
        f.write("#define SP_RESULT_WORDS     256u   /* 16×16 INT32 = 256 words */\n")
        f.write("#define SP_WGT_BASE         0u\n")
        f.write("#define SP_ACT_BASE         64u\n")
        f.write("#define SP_OUT_BASE         128u\n\n")

        f.write("/* ---- Per-tile FC1 BSR offsets (word offsets in DRAM) ---- */\n")
        for w in range(N_PARALLEL):
            f.write(f"#define FC1_MT_INDPTR_OFFSET_W{w}    0x{indptr_offsets[w]:08x}u\n")
        f.write("\n")
        for w in range(N_PARALLEL):
            f.write(f"#define FC1_MT_INDICES_OFFSET_W{w}   0x{indices_offsets[w]:08x}u\n")
        f.write("\n")
        for w in range(N_PARALLEL):
            f.write(f"#define FC1_MT_DATA_OFFSET_W{w}      0x{data_offsets[w]:08x}u\n")
        f.write("\n")

        f.write("/* ---- Shared FC1 activation ---- */\n")
        f.write(f"#define FC1_ACT_OFFSET          0x{fc1_act_offset:08x}u\n\n")

        f.write("/* ---- Per-tile FC1 partial scratch ---- */\n")
        f.write("/* Layout: [0..15] = INT32 accumulator, [16..271] = per-block output
        f.write("/* Layout: [0..15] = INT32 accumulator, [16..271] = per-block output */\n")
        for w in range(N_PARALLEL):
            f.write(f"#define FC1_PARTIAL_OFFSET_W{w}      0x{partial_offsets[w]:08x}u\n")
        f.write("\n")

        f.write("/* ---- FC1 accumulated output ---- */\n")
        f.write(f"#define FC1_OUT_OFFSET          0x{fc1_out_offset:08x}u\n\n")

        f.write("/* ---- FC2 regions (full pipeline with FC1 prefix) ---- */\n")
        f.write(f"#define FC2_WEIGHT_OFFSET       0x{fc2_weight_offset:08x}u\n")
        f.write(f"#define FC2_ACT_OFFSET          0x{fc2_act_offset_full:08x}u\n")
        f.write(f"#define FC2_OUTPUT_OFFSET       0x{fc2_output_offset_full:08x}u\n\n")

        f.write("/* ---- Convenience arrays (initialise in C) ---- */\n")
        f.write("/* Use  MT_INDPTR_OFFSETS[w]  etc. to index by tile id. */\n")
        f.write("#define MT_INDPTR_OFFSETS  { \\\n")
        f.write(",\\\n".join(
            f"    0x{indptr_offsets[w]:08x}u" for w in range(N_PARALLEL)
        ) + " \\\n}\n\n")
        f.write("#define MT_INDICES_OFFSETS  { \\\n")
        f.write(",\\\n".join(
            f"    0x{indices_offsets[w]:08x}u" for w in range(N_PARALLEL)
        ) + " \\\n}\n\n")
        f.write("#define MT_DATA_OFFSETS  { \\\n")
        f.write(",\\\n".join(
            f"    0x{data_offsets[w]:08x}u" for w in range(N_PARALLEL)
        ) + " \\\n}\n\n")
        f.write("#define MT_PARTIAL_OFFSETS  { \\\n")
        f.write(",\\\n".join(
            f"    0x{partial_offsets[w]:08x}u" for w in range(N_PARALLEL)
        ) + " \\\n}\n\n")

        if true_label >= 0:
            f.write(f"#define TRUE_LABEL          {int(true_label)}u\n")
        else:
            f.write("#define TRUE_LABEL          0xFFFFFFFFu\n")
        f.write(f"#define GOLDEN_PREDICTION   {int(golden_class)}u\n\n")

        f.write("#endif /* INFERENCE_CONFIG_MULTITILE_H */\n")

    print(f"    Written {config_path}")

    # ------------------------------------------------------------------ #
    # 9. Write golden reference                                           #
    # ------------------------------------------------------------------ #
    golden_path = os.path.join(OUTPUT_DIR, 'golden_reference_multitile.json')
    golden = {
        'image_index'     : args.image_index,
        'true_label'      : int(true_label) if true_label >= 0 else None,
        'golden_prediction': int(golden_class),
        'golden_logits'   : [float(x) for x in golden_logits],
        'n_parallel'      : N_PARALLEL,
        'fc1_nnz_per_tile': stats['per_tile_nnz'],
        'dram_layout': {
            'fc2_weight_offset'    : int(fc2_weight_offset),
            'fc2_act_offset'       : int(act_offset),
            'fc2_output_offset'    : int(output_offset),
            'fc1_indptr_offsets'   : [int(x) for x in indptr_offsets],
            'fc1_indices_offsets'  : [int(x) for x in indices_offsets],
            'fc1_data_offsets'     : [int(x) for x in data_offsets],
            'fc1_act_offset'       : int(fc1_act_offset),
            'fc1_partial_offsets'  : [int(x) for x in partial_offsets],
            'fc1_out_offset'       : int(fc1_out_offset),
            'fc2_act_offset_full'  : int(fc2_act_offset_full),
            'fc2_output_offset_full': int(fc2_output_offset_full),
        },
    }
    with open(golden_path, 'w') as gf:
        json.dump(golden, gf, indent=2)
    print(f"    Written {golden_path}")

    print("\nDone. Summary:")
    print(f"  FC1 NNZ total  : {total_nnz} blocks")
    print(f"  Per-tile NNZ   : {stats['per_tile_nnz']}")
    print(f"  Speedup target : ~{N_PARALLEL}×  (ideal K-parallel)")
    print(f"  DRAM hex       : {args.output}")
    print(f"  Firmware cfg   : {config_path}")


if __name__ == '__main__':
    main()
