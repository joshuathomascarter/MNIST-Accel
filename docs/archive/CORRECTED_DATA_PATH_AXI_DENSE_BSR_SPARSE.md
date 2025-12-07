# ACCEL-v1 Complete Architecture: Dense Activations + Sparse BSR Weights via AXI

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     HOST (Python)                                    │
├──────────────────────────────────────────────────────────────────────┤
│  run_gemm_axi_bsr.py (NEW)                                           │
│  ├─ Load A.npy (dense INT8 activations)                              │
│  ├─ Load B_bsr/ (sparse BSR weights: metadata + blocks)              │
│  └─ Convert both to AXI burst transactions                           │
└──────────────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    AXI4 Master (400 MB/s)                            │
├──────────────────────────────────────────────────────────────────────┤
│  axi_driver.py → axi_master_sim.py → axi_dma_master.sv              │
│  Burst reads from DDR:                                               │
│  ├─ Activation blocks (dense, 0x80000000)                            │
│  └─ BSR metadata + weight blocks (sparse, 0x80010000)                │
└──────────────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   FPGA Data Buffers                                  │
├──────────────────────────────────────────────────────────────────────┤
│  act_buffer.sv        │ wgt_buffer.sv      │ meta_decode.sv          │
│  (Dense activations)  │ (Dense blocks)     │ (Sparse metadata cache) │
│                       │                    │                         │
│  row_ptr_bram         │ col_idx_bram       │ block_reorder_buffer.sv │
│  (cumulative counts)  │ (column indices)   │ (reorder logic)         │
└──────────────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────────┐
│              Control: Scheduler Layer                                │
├──────────────────────────────────────────────────────────────────────┤
│  scheduler.sv → bsr_scheduler.sv (SPARSE-AWARE)                      │
│  ├─ Read row_ptr from meta_decode (which blocks to use)              │
│  ├─ Read col_idx from meta_decode (block positions)                  │
│  └─ Skip zero blocks, only process non-zero                          │
└──────────────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────────┐
│           Computation: Systolic Array (Sparse-Aware)                 │
├──────────────────────────────────────────────────────────────────────┤
│  systolic_array_sparse.sv (2x2 PE array example)                     │
│  ├─ Row-Stationary dataflow                                          │
│  ├─ Activation rows flow down                                        │
│  ├─ Weight blocks flow right (only non-zero)                         │
│  └─ Skip computation for zero blocks                                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐          │
│  │ PE[0,0]              PE[0,1]                           │          │
│  │ ├─ a_in[7:0] (act)   ├─ a_in[7:0] (act)              │          │
│  │ ├─ b_in[7:0] (wgt)   ├─ b_in[7:0] (wgt) ONLY IF      │          │
│  │ └─ mac8.sv           │   NON-ZERO (from meta_decode) │          │
│  │                                                        │          │
│  │ PE[1,0]              PE[1,1]                           │          │
│  │ ├─ a_in[7:0] (act)   ├─ a_in[7:0] (act)              │          │
│  │ ├─ b_in[7:0] (wgt)   ├─ b_in[7:0] (wgt) ONLY IF      │          │
│  │ └─ mac8.sv           │   NON-ZERO                     │          │
│  └────────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    Output Buffer                                     │
├──────────────────────────────────────────────────────────────────────┤
│  Accumulated results (32-bit partial sums)                           │
│  Read back via AXI burst                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ACTIVATIONS: Training → MAC (Dense Path)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: TRAINING & GOLDEN INPUT CAPTURE                             │
└─────────────────────────────────────────────────────────────────────┘

train_mnist.py
├─ Line 52-64: Load MNIST dataset
│  └─ 28×28 grayscale images, FP32
│
├─ Line 73-74: Define CNN model (same as before)
│  └─ Conv1(1,32,3) → Conv2(32,64,3) → FC1(9216,128) → FC2(128,10)
│
├─ Line 80-120: Train for 8 epochs
│  └─ Forward pass generates FP32 activations at each layer
│
└─ Line 165: Save test batch
   └─ np.save("python/golden/mnist_inputs.npy", golden_inputs)
      Shape: (batch, 1, 28, 28) FP32


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: QUANTIZE ACTIVATIONS TO INT8 (DENSE)                        │
└─────────────────────────────────────────────────────────────────────┘

quantize.py
├─ Line 226: Load test images (FP32)
│  └─ imgs = np.load("python/golden/mnist_inputs.npy")
│
├─ Line 230: Load trained FP32 model
│  └─ model = torch.load("data/checkpoints/mnist_fp32.pt")
│
├─ Line 237-251: Forward pass → extract activations (FP32)
│  ├─ conv1_out = Conv1(imgs) + ReLU → (batch, 32, 26, 26) FP32
│  ├─ conv2_out = Conv2(conv1) + ReLU → (batch, 64, 24, 24) FP32
│  ├─ fc1_in = Flatten(conv2_pooled) → (batch, 9216) FP32
│  └─ fc1_out = FC1(fc1_in) + ReLU → (batch, 128) FP32
│
├─ Line 255-262: Quantize activations to INT8 (per-tensor)
│  └─ for name, act in activations.items():
│       min_val = act.min()
│       max_val = act.max()
│       scale = (max_val - min_val) / 255
│       q_act = ((act - min_val) / scale).astype(np.int8)
│       quantized_acts[name] = {"data": q_act, "scale": scale}
│
└─ Line 327: Save as A.npy (dense, no sparsity)
   └─ np.save("data/quant_tiles/A.npy", A_int8)
      Shape: (M, K) = (16, 16) INT8
      Status: **ALL values stored** (100%, dense)


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: HOST LOADS DENSE ACTIVATIONS VIA AXI BURST                  │
└─────────────────────────────────────────────────────────────────────┘

run_gemm_axi_bsr.py (NEW)
├─ Line 270: Load A matrix (dense)
│  └─ A = np.load("data/quant_tiles/A.npy")
│     Shape: (16, 16) INT8, ALL values present
│
├─ Line 280-330: Pack INT8 to 32-bit words (4 INT8 per word)
│  └─ a_words = []
│     for i in range(0, len(a_data), 4):
│       chunk = a_data[i:i+4]
│       word = chunk[0]|(chunk[1]<<8)|(chunk[2]<<16)|(chunk[3]<<24)
│       a_words.append(word)
│     Result: 64 INT8 values → 16 × 32-bit words
│
├─ Line 335: Send via AXI burst write
│  └─ axi.write_burst(a_words, 0x80000000)
│     Address: 0x80000000 (DDR base)
│     Burst: 16 words (64 bytes)
│     Speed: 400 MB/s → completes in <1ms
│
└─ Result in DDR: Dense INT8 activation matrix ready


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: HARDWARE RECEIVES ACTIVATIONS (AXI → BUFFER)                │
└─────────────────────────────────────────────────────────────────────┘

accel_top.sv
├─ m_axi_* ports receive burst read requests
│  └─ axi_dma_master.sv reads from 0x80000000
│
├─ axi_dma_master.sv
│  ├─ m_axi_araddr = 0x80000000
│  ├─ m_axi_arlen = 15 (16 beats)
│  ├─ m_axi_rdata[31:0] → 4 INT8 values per cycle
│  └─ Writes to act_buffer via buf_wdata/buf_wen
│
└─ act_buffer.sv
   ├─ act_we ← dma_buf_wen (write enable)
   ├─ act_waddr ← dma_buf_waddr[ADDR_WIDTH-1:0] (address)
   ├─ act_wdata[63:0] ← dma_buf_wdata[31:0] (8 INT8 values)
   └─ Stores all 16×16 activation matrix


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: SYSTOLIC READS ACTIVATIONS (ROW BY ROW)                     │
└─────────────────────────────────────────────────────────────────────┘

scheduler.sv (or bsr_scheduler.sv)
├─ For each tile (m_idx, n_idx, k_idx):
│  ├─ act_rd_en ← 1'b1 (enable activation read)
│  └─ act_k_idx ← k_idx (which K dimension chunk)
│
└─ systolic_array_sparse.sv
   ├─ a_vec[63:0] = act_buffer[act_waddr] (read row from buffer)
   │  └─ 8 INT8 values: [a0, a1, a2, a3, a4, a5, a6, a7]
   │
   └─ Broadcast to all PE rows
      ├─ PE[0,0] receives a_vec[7:0]
      ├─ PE[0,1] receives a_vec[15:8]
      ├─ PE[1,0] receives a_vec[7:0]
      └─ PE[1,1] receives a_vec[15:8]


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: PE RECEIVES ACTIVATION + WEIGHT (SPARSE-AWARE)              │
└─────────────────────────────────────────────────────────────────────┘

pe.sv (Example: PE[0,0])
├─ a_in[7:0] ← activation from systolic_array
│  └─ Example: 0x05 = 5 (INT8)
│
├─ b_in[7:0] ← weight from wgt_buffer
│  └─ ONLY if block is non-zero (checked by bsr_scheduler)
│     Example: 0x03 = 3 (INT8)
│
└─ If block is zero: b_in ← 0x00 (skip computation)


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: MAC UNIT (8×8 Multiply-Accumulate)                          │
└─────────────────────────────────────────────────────────────────────┘

mac8.sv
├─ Input: a[7:0] = 0x05 (activation, INT8)
├─ Input: b[7:0] = 0x03 (weight, INT8)
├─ Multiply: 5 × 3 = 15 (INT16)
├─ Accumulate: acc[31:0] += 15
│  └─ Result: 32-bit partial sum stays in PE
│
└─ If weight is zero (sparse):
   └─ Multiply: 5 × 0 = 0 (no-op, can be skipped)
```

---

## WEIGHTS: Training → MAC (Sparse BSR Path)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: TRAIN WEIGHTS (FP32)                                        │
└─────────────────────────────────────────────────────────────────────┘

train_mnist.py
├─ Line 73-74: Define model with weight layers
│  └─ Conv1.weight: (32, 1, 3, 3) FP32
│     Conv2.weight: (64, 32, 3, 3) FP32
│     FC1.weight: (128, 9216) FP32
│     FC2.weight: (10, 128) FP32
│
├─ Line 80-120: Training loop (8 epochs)
│  └─ optimizer.step() updates all weights via backprop
│
└─ Line 151: Save trained model
   └─ torch.save({"state_dict": model.state_dict(), ...}, "mnist_fp32.pt")
      All weights saved as FP32


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: PRUNE WEIGHTS TO ~30% DENSITY (BSR BLOCKS)                  │
└─────────────────────────────────────────────────────────────────────┘

blocksparse_train.py
├─ Line 70: Load trained model (FP32)
│  └─ model = load_dense_model()
│     Loads all weights as FP32
│
├─ Line 120-150: Compute block-wise L2 norms (8×8 blocks)
│  ├─ For each layer, reshape weights into (num_blocks, 64) 
│  │  └─ FC1: (128, 9216) → (128/8) × (9216/8) = 16 × 144 blocks
│  │
│  └─ For each block: norm = ||block_8x8||_2
│     Example: 
│       Block 0: norm = 2.5
│       Block 1: norm = 0.8  ← LOW (will be pruned)
│       Block 2: norm = 3.1
│       Block 3: norm = 0.2  ← LOW (will be pruned)
│       ...
│       Block 143: norm = 1.5
│
├─ Line 160-180: Threshold-based pruning (~70% → zero, keep 30%)
│  ├─ Sort blocks by norm
│  ├─ Keep top 30% (threshold: median norm × 0.5)
│  ├─ Zero out bottom 70%
│  │
│  └─ Result:
│       Kept blocks (30%): 43 blocks (out of 144 for FC1)
│       Zero blocks (70%): 101 blocks (not stored, skipped in compute)
│
└─ Line 200: Create sparse weight dictionary
   └─ sparse_weights["fc1"] = {
        "kept_blocks": [0, 2, 4, 7, 15, ...],  # 43 block indices
        "block_data": [...],                    # 43×64 INT8 values
        "row_ptr": [0, 5, 12, 18, ...],         # Cumulative counts
        "col_idx": [0, 2, 4, 7, 15, ...],       # Column indices
      }


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: QUANTIZE SPARSE WEIGHTS TO INT8 (PER-CHANNEL)               │
└─────────────────────────────────────────────────────────────────────┘

quantize.py
├─ Line 170-180: Quantize weights (per-output-channel)
│  ├─ For each layer:
│  │  ├─ Load FP32 weights from checkpoint
│  │  ├─ For output_channel in num_outputs:
│  │  │  ├─ min_val = weights[output_channel].min()
│  │  │  ├─ max_val = weights[output_channel].max()
│  │  │  ├─ scale = (max_val - min_val) / 255
│  │  │  └─ w_int8[output_channel] = ((w - min_val) / scale)
│  │  │
│  │  └─ Result: INT8 weights with per-channel scales
│  │
│  └─ Example for FC1:
│       Input shape: (128, 9216) FP32
│       Output: (128, 9216) INT8 + 128 scales (one per output channel)
│
└─ Line 320: Save quantized weights
   └─ np.save("data/quant_tiles/B_quantized.npy", W_int8)
      Shape: (K, N) = (9216, 128) INT8


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: EXPORT TO BSR FORMAT (METADATA + BLOCKS)                    │
└─────────────────────────────────────────────────────────────────────┘

export_bsr.py
├─ Convert quantized sparse weights to BSR format
│
├─ For FC1 (128 output channels, 9216 input features):
│  ├─ Reshape to blocks: (16 block_rows) × (144 block_cols)
│  │  └─ Block size: 8×8
│  │
│  ├─ Create row_ptr array:
│  │  └─ Cumulative count of non-zero blocks per row
│  │     Example:
│  │       row_ptr[0] = 0     (no blocks in row 0)
│  │       row_ptr[1] = 5     (5 blocks in row 1, indices 0-4)
│  │       row_ptr[2] = 12    (7 blocks in row 2, indices 5-11)
│  │       row_ptr[3] = 18    (6 blocks in row 3, indices 12-17)
│  │       ...
│  │       row_ptr[16] = 43   (total 43 blocks)
│  │
│  ├─ Create col_idx array:
│  │  └─ Column index for each block
│  │     Example:
│  │       col_idx[0] = 5   (block at row 1, col 5)
│  │       col_idx[1] = 12  (block at row 1, col 12)
│  │       col_idx[2] = 45
│  │       col_idx[3] = 78
│  │       col_idx[4] = 130
│  │       col_idx[5] = 2   (block at row 2, col 2)
│  │       ...
│  │
│  └─ Create blocks array:
│     └─ Actual 8×8 INT8 block data (only 43 blocks, not 144)
│        blocks shape: (43, 8, 8) = 3,456 INT8 values (vs 147,456 if dense)
│        **Compression: 2.3% of original (97.7% reduction!)**
│
└─ Save BSR files:
   ├─ data/bsr_export/row_ptr.npy  (shape: 17, dtype: int32)
   ├─ data/bsr_export/col_idx.npy  (shape: 43, dtype: int32)
   └─ data/bsr_export/blocks.npy   (shape: 43×8×8, dtype: int8)


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: HOST LOADS BSR METADATA + BLOCKS VIA AXI BURST              │
└─────────────────────────────────────────────────────────────────────┘

run_gemm_axi_bsr.py (NEW)
├─ Line 275-290: Load BSR sparse weights
│  ├─ row_ptr = np.load("data/bsr_export/row_ptr.npy")
│  │  └─ Shape: (17,) INT32, values: [0, 5, 12, 18, ...]
│  │
│  ├─ col_idx = np.load("data/bsr_export/col_idx.npy")
│  │  └─ Shape: (43,) INT32, values: [5, 12, 45, 78, ...]
│  │
│  └─ blocks = np.load("data/bsr_export/blocks.npy")
│     └─ Shape: (43, 8, 8) INT8, **only 43 blocks** (not 144!)
│
├─ Line 320-350: Pack BSR data to AXI bursts
│  ├─ row_ptr as 32-bit words
│  ├─ col_idx as 32-bit words
│  └─ blocks flattened to 32-bit words (4 INT8 per word)
│
├─ Line 360-380: Send via AXI burst (TWO TRANSFERS)
│  ├─ Metadata (row_ptr + col_idx):
│  │  └─ axi.write_burst(meta_words, 0x80010000)
│  │     Address: 0x80010000 (metadata region)
│  │     Size: ~200 bytes
│  │
│  └─ Blocks:
│     └─ axi.write_burst(block_words, 0x80010100)
│        Address: 0x80010100 (block data region)
│        Size: 43×64 = 2,752 bytes (vs 9,408 if dense)
│        **10.7× compression for bandwidth!**
│
└─ Result in DDR:
   ├─ row_ptr: [0, 5, 12, 18, ...]
   ├─ col_idx: [5, 12, 45, ...]
   └─ blocks: 43 × 8×8 non-zero blocks only


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: HARDWARE RECEIVES METADATA (AXI → META_DECODE)              │
└─────────────────────────────────────────────────────────────────────┘

accel_top.sv
├─ axi_dma_master.sv reads metadata from 0x80010000
│  └─ m_axi_rdata carries row_ptr and col_idx words
│
├─ dma_meta_wen ← 1'b1 (write to metadata cache)
├─ dma_meta_data ← m_axi_rdata[31:0] (metadata word)
│
└─ Metadata written to cache during DMA transfer


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: META_DECODE CACHES METADATA (PARALLEL ACCESS)               │
└─────────────────────────────────────────────────────────────────────┘

meta_decode.sv (Metadata Cache)
├─ DMA input interface (write side):
│  ├─ dma_meta_wen (pulse to write metadata)
│  ├─ dma_meta_data[31:0] (row_ptr or col_idx word)
│  └─ dma_meta_type (0=row_ptr, 1=col_idx)
│
├─ Caches both:
│  ├─ row_ptr cache (e.g., [0, 5, 12, 18, ...])
│  └─ col_idx cache (e.g., [5, 12, 45, 78, ...])
│
└─ Scheduler output interface (read side):
   ├─ sched_meta_raddr[7:0] (which metadata entry)
   ├─ sched_meta_ren (read enable)
   ├─ sched_meta_rdata[31:0] (returned data)
   ├─ sched_meta_rvalid (data valid)
   └─ perf_cache_hits/misses (for profiling)
      
   **BENEFIT: Scheduler can read metadata in parallel with DMA loading blocks!**


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8: HARDWARE RECEIVES WEIGHT BLOCKS (AXI → WGT_BUFFER)          │
└─────────────────────────────────────────────────────────────────────┘

accel_top.sv
├─ axi_dma_master.sv reads blocks from 0x80010100
│  └─ m_axi_rdata carries 4 INT8 block values per cycle
│
├─ wgt_buffer.sv receives all 43 blocks
│  ├─ wgt_we ← dma_buf_wen (write enable)
│  ├─ wgt_waddr ← block_index (0-42, only 43 blocks!)
│  └─ wgt_wdata ← block data
│
└─ **ONLY 43 blocks stored** (not 144!)
   Space saved: 1.6× smaller BRAM


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 9: SCHEDULER DETERMINES WHICH BLOCKS TO USE (BSR_SCHEDULER)    │
└─────────────────────────────────────────────────────────────────────┘

bsr_scheduler.sv (Sparse-Aware Scheduling)
├─ For each output channel i:
│  ├─ block_start ← row_ptr[i] (first block for this output)
│  ├─ block_end ← row_ptr[i+1] (one past last block)
│  ├─ num_blocks ← block_end - block_start (usually 2-5 blocks)
│  │
│  └─ For each block j in [block_start, block_end):
│     ├─ block_col ← col_idx[j] (which input dimension)
│     ├─ block_idx ← j (which block to fetch from wgt_buffer)
│     │
│     └─ Send to systolic_array_sparse:
│        ├─ Valid this block? Yes! (non-zero)
│        ├─ Skip computation for zero blocks
│        └─ Process only these ~30% of blocks
│
└─ **RESULT: Skip 70% of computation** (zero blocks are no-ops)
   Performance: 3.3× speedup vs dense computation!


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 10: SYSTOLIC READS WEIGHTS (ONLY NON-ZERO BLOCKS)              │
└─────────────────────────────────────────────────────────────────────┘

systolic_array_sparse.sv
├─ Receive block_valid from bsr_scheduler
│  └─ If block_valid = 1: read weight block
│     If block_valid = 0: inject zeros (skip computation)
│
├─ b_vec[63:0] = wgt_buffer[block_idx] IF valid
│  └─ 8 INT8 values: [b0, b1, b2, b3, b4, b5, b6, b7]
│
├─ Broadcast to all PE columns
│  ├─ PE[0,0] receives b_vec[7:0]
│  ├─ PE[0,1] receives b_vec[15:8]
│  ├─ PE[1,0] receives b_vec[7:0]
│  └─ PE[1,1] receives b_vec[15:8]
│
└─ If block_valid = 0:
   └─ b_vec ← 0 (inject zeros)
      Computation becomes: a × 0 = 0 (no-op, hardware can skip)


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 11: PE RECEIVES ACTIVATION + WEIGHT (SPARSE-AWARE)             │
└─────────────────────────────────────────────────────────────────────┘

pe.sv (Example: PE[0,0])
├─ a_in[7:0] ← activation (always valid, from act_buffer)
│  └─ Example: 0x05 = 5 (INT8)
│
├─ b_in[7:0] ← weight from wgt_buffer
│  ├─ If bsr_scheduler says block is non-zero:
│  │  └─ b_in ← 0x03 = 3 (INT8)
│  │
│  └─ If bsr_scheduler says block is zero:
│     └─ b_in ← 0x00 (injected zero)
│
└─ Decision made by bsr_scheduler based on metadata:
   ├─ Non-zero block: Process normally
   └─ Zero block: a × 0 = 0 (can be optimized as no-op)


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 12: MAC UNIT (SAME FOR BOTH PATHS)                             │
└─────────────────────────────────────────────────────────────────────┘

mac8.sv
├─ Input: a[7:0] (INT8 activation from act_buffer)
│  └─ Example: 0x05 = 5
│
├─ Input: b[7:0] (INT8 weight from wgt_buffer, may be zero if sparse)
│  ├─ If non-zero block: Example: 0x03 = 3
│  └─ If zero block: 0x00 (injected zero)
│
├─ Multiply: product = a × b
│  ├─ Non-zero case: 5 × 3 = 15
│  └─ Zero case: 5 × 0 = 0 (no-op)
│
├─ Accumulate: acc[31:0] += sign_extend(product[15:0])
│  ├─ Non-zero: acc += 15
│  └─ Zero: acc += 0 (no change)
│
└─ Output: acc_out[31:0] (partial sum)
   └─ Result stays in PE, flows to next PE or output BRAM
```

---

## Summary: Complete File Flow (Training → MAC)

### **Activation Path (DENSE):**
```
1. train_mnist.py          → FP32 images (28×28)
2. quantize.py             → INT8 activations (all values)
3. run_gemm_axi_bsr.py     → Pack A.npy to 32-bit words
4. axi_driver.py           → Format AXI burst
5. axi_dma_master.sv       → Read from DDR @ 0x80000000
6. act_buffer.sv           → Store all INT8 activations
7. systolic_array_sparse.sv → Broadcast activation rows
8. pe.sv                    → Receive a_in[7:0]
9. mac8.sv                 → **a[7:0] × b[7:0]**
```

### **Weight Path (SPARSE BSR - 30% density, 70% skipped):**
```
1. train_mnist.py          → FP32 weights (128×9216)
2. blocksparse_train.py    → Prune to ~30% (43 blocks of 144)
3. quantize.py             → INT8 weights (per-channel)
4. export_bsr.py           → Convert to BSR format:
                              - row_ptr[17]
                              - col_idx[43]
                              - blocks[43×8×8]
5. run_gemm_axi_bsr.py     → Pack BSR to 32-bit words
6. axi_driver.py           → Format AXI bursts (metadata + blocks)
7. axi_dma_master.sv       → Read metadata @ 0x80010000
8. meta_decode.sv          → Cache row_ptr + col_idx
9. axi_dma_master.sv       → Read blocks @ 0x80010100
10. wgt_buffer.sv          → Store only 43 blocks (not 144!)
11. bsr_scheduler.sv       → Determine which blocks to use:
                             - Read row_ptr/col_idx from meta_decode
                             - Skip 70% of blocks (zero, not stored)
12. systolic_array_sparse.sv → Broadcast weight blocks (only valid ones)
13. pe.sv                   → Receive b_in[7:0] (or 0x00 if skipped)
14. mac8.sv                 → **a[7:0] × b[7:0]** (or a × 0 if sparse)
```

---

## Performance Gains

| Metric | Dense | Sparse BSR | Gain |
|--------|-------|-----------|------|
| Weight storage | 147,456 bytes | 3,456 bytes | **42.7× compression** |
| DMA transfer | 147 KB | 3.5 KB | **42.7× faster** |
| Computation | 16×16×16 MACs | ~4.8×16×16 MACs | **3.3× fewer ops** |
| Metadata overhead | 0 | ~300 bytes | Negligible |
| Power | Full usage | ~30% active | **3.3× less power** |

---

## Key Innovation: Parallel Metadata Access

**Without meta_decode.sv (sequential):**
```
Time: [Load metadata] → wait → [Load blocks] → wait → [Scheduler reads] → Compute
      (slow, metadata not available during block load)
```

**With meta_decode.sv (parallel):**
```
Time: [Load metadata to cache] (in parallel with)
      [Load blocks] (in parallel with)
      [Scheduler reads from meta_decode cache]
      Compute
      (faster, no waiting)
```

**Benefit:** Scheduler can start planning work while DMA is still loading blocks!

