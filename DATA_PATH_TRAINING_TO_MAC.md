# Complete Data Path: Training → MAC for ACCEL-v1

## ACTIVATIONS: Training → MAC

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TRAINING & DATASET GENERATION                              │
└─────────────────────────────────────────────────────────────────────┘

train_mnist.py
├─ Line 52-64: Load MNIST dataset (torchvision.datasets.MNIST)
│  └─ Downloads 60,000 training images + 10,000 test images (28×28 grayscale)
│
├─ Line 65: Create DataLoader with batch_size=64
│  └─ Creates train_loader (batches of 64 images)
│
├─ Line 73-74: Define CNN model
│  └─ class Net(nn.Module):
│     - Conv2d(1, 32, 3, 1) → outputs (batch, 32, 26, 26)
│     - Conv2d(32, 64, 3, 1) → outputs (batch, 64, 24, 24)
│     - MaxPool2d(2) → outputs (batch, 64, 12, 12)
│     - Flatten() → (batch, 9216)
│     - Linear(9216, 128) → (batch, 128)
│     - Linear(128, 10) → (batch, 10)
│
├─ Line 80-120: Training loop (8 epochs)
│  └─ For each batch: forward() → generates intermediate activations
│     - conv1_out = Conv1(image) → (batch, 32, 26, 26)
│     - relu1 = ReLU(conv1_out)
│     - conv2_out = Conv2(relu1) → (batch, 64, 24, 24)
│     - relu2 = ReLU(conv2_out)
│     - pool = MaxPool2d(relu2) → (batch, 64, 12, 12)
│     - flat = Flatten(pool) → (batch, 9216)
│     - fc1_out = FC1(flat) → (batch, 128)
│     - relu_fc = ReLU(fc1_out)
│     - logits = FC2(relu_fc) → (batch, 10)
│
├─ Line 165: Save first test batch
│  └─ np.save("python/golden/mnist_inputs.npy", golden_inputs)
│     Saved as: float32 (28×28) images, shape (batch_size, 1, 28, 28)
│
└─ Line 151: Save trained model checkpoint
   └─ torch.save({"state_dict": model.state_dict(), ...}, "data/checkpoints/mnist_fp32.pt")
      Saved as: FP32 weights for all layers


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: GOLDEN ACTIVATION EXTRACTION (Quantization)                │
└─────────────────────────────────────────────────────────────────────┘

quantize.py
├─ Line 23: Define path to golden inputs
│  └─ GOLDEN_INPUTS_PATH = "python/golden/mnist_inputs.npy"
│
├─ Line 226: Load saved test images
│  └─ imgs = np.load(GOLDEN_INPUTS_PATH)[:num_samples]
│     Shape: (num_samples, 1, 28, 28) as float32
│
├─ Line 230: Load trained model checkpoint
│  └─ checkpoint = torch.load("data/checkpoints/mnist_fp32.pt")
│     model.load_state_dict(checkpoint["state_dict"])
│     model.eval() → FP32 weights loaded
│
├─ Line 237-251: Forward pass to capture intermediate activations
│  │
│  ├─ For each test batch:
│  │  ├─ x = Conv1(imgs) + ReLU
│  │  │  └─ activations["conv1_out"] = output[0].cpu().numpy() 
│  │  │     Shape: (batch, 32, 26, 26) as FP32
│  │  │
│  │  ├─ x = Conv2(x) + ReLU
│  │  │  └─ activations["conv2_out"] = output[0].cpu().numpy()
│  │  │     Shape: (batch, 64, 24, 24) as FP32
│  │  │
│  │  ├─ x = Flatten(x)
│  │  │  └─ activations["fc1_in"] = a_flat.cpu().numpy()
│  │  │     Shape: (batch, 9216) as FP32
│  │  │
│  │  └─ x = FC1(x) + ReLU
│  │     └─ activations["fc1_out"] = output[0].cpu().numpy()
│  │        Shape: (batch, 128) as FP32
│  │
│  └─ Result: activations = {
│       "conv1_out": float32 array,
│       "conv2_out": float32 array,
│       "fc1_in": float32 array,
│       "fc1_out": float32 array
│     }
│
├─ Line 255-262: Quantize activations (per-tensor)
│  └─ for name, act in activations.items():
│       q_act, q_scale = quantize_symmetric_per_tensor(act)
│       quantized_acts[name] = {
│         "data": q_act,           # INT8 (-128 to 127)
│         "scale": q_scale,        # FP32 scaling factor
│       }
│
└─ Line 327: Save as hardware-ready tiles
   └─ np.save("data/quant_tiles/A.npy", A_int8)
      Saved as: INT8 dense matrix, shape (M, K)


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: HOST LOADS ACTIVATIONS VIA AXI                             │
└─────────────────────────────────────────────────────────────────────┘

run_gemm_axi.py
├─ Line 260-270: Load A matrix (activations)
│  └─ A = np.load("data/quant_tiles/A.npy")
│     Shape: (M, K) INT8
│     Example: (16, 16) INT8 activation matrix
│
├─ Line 280-330: Convert to 32-bit words (pack 4 INT8 per word)
│  └─ for i in range(0, len(a_data), 4):
│       chunk = a_data[i:i+4]
│       word = chunk[0] | (chunk[1]<<8) | (chunk[2]<<16) | (chunk[3]<<24)
│       a_words.append(word)
│     Example: 4 INT8 values → 1 x 32-bit word
│
├─ Line 335: Send via AXI burst write
│  └─ success, words_a = self.axi.write_burst(a_words, a_addr)
│     a_addr = 0x80000000 (DDR base)
│     Burst: Up to 256 words (1 KB) per transaction
│
└─ Result in DDR: INT8 activation matrix ready for hardware


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: HARDWARE RECEIVES ACTIVATIONS (AXI → Buffer)               │
└─────────────────────────────────────────────────────────────────────┘

accel_top.sv
├─ m_axi_* ports (AXI4 Master interface)
│  └─ m_axi_rdata[31:0] carries 4×INT8 activation bytes
│
├─ axi_dma_master.sv
│  ├─ m_axi_arvalid, m_axi_araddr, m_axi_arlen, m_axi_rdata
│  └─ Bursts read activation data from DDR
│     - Start address: 0x80000000
│     - Burst length: Up to 256 beats
│     - Data width: 32 bits (4 INT8 values)
│
├─ Buffer write path mux
│  ├─ dma_target_act = (dma_buf_waddr[31:30] == 2'b00)
│  └─ act_we ← dma_buf_wen & dma_target_act
│
└─ act_buffer.sv (Activation Storage BRAM)
   ├─ act_we (write enable)
   ├─ act_waddr (address, selects row)
   ├─ act_wdata[TM*8-1:0] (data bus, TM=8 → 64 bits = 8 INT8 values)
   └─ Stores activation matrix row-by-row
      Each row: [int8_0, int8_1, ..., int8_7]


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: SYSTOLIC ARRAY BROADCAST (Row-Stationary Dataflow)         │
└─────────────────────────────────────────────────────────────────────┘

systolic_array.sv
├─ act_rd_en (read enable from scheduler)
├─ a_vec[TN*8-1:0] (read activation row from buffer)
│  └─ TN=8 → reads 8×INT8 values per cycle (one row)
│
├─ Row-Stationary Flow:
│  ├─ Cycle 0: Load row 0 of A into PE[0,*]
│  ├─ Cycle 1: Load row 1 of A into PE[1,*]
│  └─ Continue for M/TM tiles
│
└─ PE array (N_ROWS × N_COLS = 2×2 example)
   ├─ PE[0,0] receives a_vec[7:0] (first INT8 from A)
   ├─ PE[0,1] receives a_vec[15:8] (second INT8 from A)
   ├─ PE[1,0] receives a_vec[7:0] (first INT8 from A)
   └─ PE[1,1] receives a_vec[15:8] (second INT8 from A)
      Broadcast to all PEs in the same column


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: PROCESSING ELEMENT COMPUTATION                             │
└─────────────────────────────────────────────────────────────────────┘

pe.sv (Example: PE[0,0])
├─ Input: a_in[7:0] (INT8 activation from buffer)
│  └─ Value range: -128 to 127
│
├─ Input: b_in[7:0] (INT8 weight from weight buffer)
│  └─ Value range: -128 to 127
│
├─ mac8.sv instantiation
│  ├─ a[7:0] ← a_in[7:0] (activation)
│  ├─ b[7:0] ← b_in[7:0] (weight)
│  └─ Product ← a × b (INT16 intermediate result)
│
└─ Accumulator update
   └─ acc[31:0] ← acc[31:0] + (a × b)
      32-bit accumulation to prevent overflow


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 7: MAC UNIT (8×8 Integer Multiply-Accumulate)                 │
└─────────────────────────────────────────────────────────────────────┘

mac8.sv
├─ Input: a[7:0] (INT8 activation)
│  └─ Example: 0xFE = -2 (two's complement)
│
├─ Input: b[7:0] (INT8 weight)
│  └─ Example: 0x03 = 3
│
├─ Multiply: product = a × b
│  └─ INT8 × INT8 → INT16 temporary
│     -2 × 3 = -6 (0xFFFA as INT16)
│
├─ Accumulate: accumulator += product
│  └─ accumulator[31:0] += sign_extend(product[15:0])
│     Example: 1000 + (-6) = 994
│
└─ Output: acc_out[31:0] (partial sum)
   └─ Result stays in PE until row complete


┌─────────────────────────────────────────────────────────────────────┐
│ FINAL: OUTPUT TO RESULT BUFFER                                       │
└─────────────────────────────────────────────────────────────────────┘

After all K tiles processed:
├─ c_out_flat[(N_ROWS*N_COLS*32)-1:0] (accumulated results from PE array)
│  └─ 2×2 PE array → 4 × 32-bit results
│
├─ Stored in output BRAM (result buffer)
│  └─ out_we, out_waddr, out_wdata
│
└─ Read back via AXI burst read
   └─ DDR address: 0x80000000 + 0x20000
```

---

## WEIGHTS: Training → MAC (Parallel Path)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: WEIGHT TRAINING                                            │
└─────────────────────────────────────────────────────────────────────┘

train_mnist.py
├─ Line 73-74: Define model with weight layers
│  └─ model = Net()
│     - Conv1: weight shape (32, 1, 3, 3) → FP32
│     - Conv2: weight shape (64, 32, 3, 3) → FP32
│     - FC1: weight shape (128, 9216) → FP32
│     - FC2: weight shape (10, 128) → FP32
│
├─ Line 80-120: Training (8 epochs)
│  └─ optimizer.step() updates weights via backprop
│     - Gradient descent: w_new = w_old - lr × dL/dw
│
└─ Line 151: Save trained model
   └─ torch.save({
        "state_dict": model.state_dict(),  # FP32 weights
        "best_acc": accuracy,
      }, "data/checkpoints/mnist_fp32.pt")


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: WEIGHT QUANTIZATION & SPARSIFICATION                       │
└─────────────────────────────────────────────────────────────────────┘

blocksparse_train.py
├─ Line 70: Load trained model checkpoint
│  └─ model = load_dense_model()
│     Loads FP32 weights from mnist_fp32.pt
│
├─ Line 120-150: Compute block norms (L2 norm of 8×8 blocks)
│  ├─ For Conv1: weight (32, 1, 3, 3) → compute block norms
│  ├─ For Conv2: weight (64, 32, 3, 3) → compute block norms
│  ├─ For FC1: weight (128, 9216) → group into (128/8)×(9216/8) blocks
│  └─ For FC2: weight (10, 128) → group into blocks
│
├─ Line 160-180: Block-wise pruning (keep only high-norm blocks)
│  ├─ Sort blocks by norm (L2 norm of each 8×8 block)
│  ├─ Keep top ~30% of blocks (threshold-based)
│  ├─ Zero out low-norm blocks
│  └─ Result: Sparse weights with BSR metadata
│
└─ Result: Sparse weight matrices saved as BSR format


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: BSR EXPORT & SERIALIZATION                                 │
└─────────────────────────────────────────────────────────────────────┘

export_bsr.py (from blocksparse_train output)
├─ Converts sparse weight matrices to BSR format:
│  └─ Block Sparse Row format (COO-like but for blocks)
│     - row_ptr[]: cumulative block counts per row
│     - col_idx[]: column index of each block
│     - blocks[]: actual 8×8 blocks
│
├─ Example for Conv2 (64×32 output features, 32×3×3 filters):
│  ├─ Total 8×8 blocks: (64/8) × (32/8) = 64 blocks max
│  ├─ If 30% sparse: ~45 blocks stored (not 64)
│  ├─ Metadata:
│  │  - row_ptr[8] = cumulative counts [0, 6, 12, 18, ...]
│  │  - col_idx[45] = column positions for each block
│  │  - blocks[45][64] = actual 8×8 INT8 blocks
│  │
│  └─ File saved: "data/bsr_export/conv2_weights.bsr"
│
└─ All layers exported as separate BSR files


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: HOST LOADS WEIGHTS VIA AXI (Alternative to BSR)             │
└─────────────────────────────────────────────────────────────────────┘

quantize.py
├─ Line 170-180: Quantize weights (per-channel)
│  ├─ For each layer (Conv1, Conv2, FC1, FC2):
│  │  ├─ weight_int8, weight_scales = quantize_symmetric_per_channel(w)
│  │  └─ Converts FP32 weights → INT8 per output channel
│  │
│  └─ Result: quantized_model = {
│       "conv1.weight": {
│         "data": INT8 array,
│         "scales": FP32 per-channel scales,
│         "shape": original_shape
│       },
│       ...
│     }
│
├─ Line 320: Save as dense tiles (for AXI path)
│  └─ np.save("data/quant_tiles/B.npy", B_int8)
│     Saved as: INT8 dense matrix, shape (K, N)
│     Example: FC1 → (9216, 128) INT8
│
└─ B matrix ready for AXI transfer


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: HOST LOADS WEIGHTS VIA AXI                                 │
└─────────────────────────────────────────────────────────────────────┘

run_gemm_axi.py
├─ Line 273-280: Load B matrix (weights)
│  └─ B = np.load("data/quant_tiles/B.npy")
│     Shape: (K, N) INT8
│     Example: (16, 16) INT8 weight matrix
│
├─ Line 280-330: Convert to 32-bit words (pack 4 INT8 per word)
│  └─ for i in range(0, len(b_data), 4):
│       chunk = b_data[i:i+4]
│       word = chunk[0] | (chunk[1]<<8) | (chunk[2]<<16) | (chunk[3]<<24)
│       b_words.append(word)
│
├─ Line 345: Send via AXI burst write
│  └─ success, words_b = self.axi.write_burst(b_words, b_addr)
│     b_addr = 0x80000000 + 0x10000 (weights offset)
│
└─ Result in DDR: INT8 weight matrix ready for hardware


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: HARDWARE RECEIVES WEIGHTS (AXI → Buffer)                   │
└─────────────────────────────────────────────────────────────────────┘

accel_top.sv
├─ m_axi_* ports (AXI4 Master interface)
│  └─ m_axi_rdata[31:0] carries 4×INT8 weight bytes
│
├─ axi_dma_master.sv
│  ├─ Read from address: 0x80000000 + 0x10000
│  └─ Burst read 256 words (1 KB) of weight data
│
├─ Buffer write path mux
│  ├─ dma_target_wgt = (dma_buf_waddr[31:30] == 2'b01)
│  └─ wgt_we ← dma_buf_wen & dma_target_wgt
│
└─ wgt_buffer.sv (Weight Storage BRAM)
   ├─ wgt_we (write enable)
   ├─ wgt_waddr (address, selects weight row)
   ├─ wgt_wdata[TN*8-1:0] (data bus, TN=8 → 64 bits = 8 INT8 values)
   └─ Stores weight matrix row-by-row
      Each row: [int8_0, int8_1, ..., int8_7]


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 7: SYSTOLIC ARRAY BROADCAST (Column-wise, Row-Stationary)     │
└─────────────────────────────────────────────────────────────────────┘

systolic_array.sv
├─ wgt_rd_en (read enable from scheduler)
├─ b_vec[TM*8-1:0] (read weight column from buffer)
│  └─ TM=8 → reads 8×INT8 values per cycle (one column)
│
├─ Row-Stationary Flow:
│  ├─ Weights flow horizontally (columns move right through PE rows)
│  ├─ Cycle 0: Load column 0 of B into PE[*,0]
│  ├─ Cycle 1: Load column 1 of B into PE[*,1]
│  └─ Continue for N/TN tiles
│
└─ PE array (N_ROWS × N_COLS = 2×2 example)
   ├─ PE[0,0] receives b_vec[7:0] (first INT8 from B)
   ├─ PE[0,1] receives b_vec[15:8] (second INT8 from B)
   ├─ PE[1,0] receives b_vec[7:0] (first INT8 from B)
   └─ PE[1,1] receives b_vec[15:8] (second INT8 from B)
      Broadcast to all PEs in the same row


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 8: PROCESSING ELEMENT COMPUTATION (SAME MAC UNIT)             │
└─────────────────────────────────────────────────────────────────────┘

pe.sv (Example: PE[0,0])
├─ Input: a_in[7:0] (INT8 activation from activation buffer)
├─ Input: b_in[7:0] (INT8 weight from weight buffer)
│
└─ mac8.sv
   ├─ Multiply: product = a[7:0] × b[7:0]
   ├─ Accumulate: acc += sign_extend(product)
   └─ Output: partial sum


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 9: MAC UNIT (8×8 Integer Multiply-Accumulate)                 │
└─────────────────────────────────────────────────────────────────────┘

mac8.sv (Same as activations)
├─ Input: a[7:0] (INT8 activation)
├─ Input: b[7:0] (INT8 weight)
├─ Multiply: product = a × b (INT16)
├─ Accumulate: acc[31:0] += sign_extend(product[15:0])
└─ Output: acc_out[31:0]
```

---

## Complete File Sequence (Training → MAC)

### **For Activations:**
1. `train_mnist.py` - Train model, save images
2. `quantize.py` - Extract + quantize activations  
3. `run_gemm_axi.py` - Load A.npy, convert to 32-bit words
4. `axi_driver.py` - Format AXI burst transactions
5. `axi_master_sim.py` - Simulate AXI protocol
6. `accel_top.sv` - Top-level orchestration
7. `axi_dma_master.sv` - Burst read from DDR
8. `act_buffer.sv` - Store INT8 activations
9. `systolic_array.sv` - Broadcast rows horizontally
10. `pe.sv` - Processing element
11. **`mac8.sv`** - **8×8 Multiply-Accumulate**

### **For Weights:**
1. `train_mnist.py` - Train model, save weights
2. `blocksparse_train.py` - Prune to ~30% density
3. `export_bsr.py` - Convert to BSR format (metadata)
4. **OR** `quantize.py` - Quantize to INT8 (dense)
5. `run_gemm_axi.py` - Load B.npy, convert to 32-bit words
6. `axi_driver.py` - Format AXI burst transactions
7. `axi_master_sim.py` - Simulate AXI protocol
8. `accel_top.sv` - Top-level orchestration
9. `axi_dma_master.sv` - Burst read from DDR
10. `wgt_buffer.sv` - Store INT8 weights
11. `systolic_array.sv` - Broadcast columns vertically
12. `pe.sv` - Processing element
13. **`mac8.sv`** - **8×8 Multiply-Accumulate**

---

## Summary

**Activations (A matrix):**
```
train_mnist.py (FP32 images)
  → quantize.py (INT8 activations)
  → run_gemm_axi.py (pack to 32-bit words)
  → axi_driver.py (AXI burst format)
  → axi_dma_master.sv (DDR → FPGA)
  → act_buffer.sv (storage)
  → systolic_array.sv (broadcast rows)
  → pe.sv (receive a_in[7:0])
  → mac8.sv (multiply with b_in[7:0])
```

**Weights (B matrix):**
```
train_mnist.py (FP32 weights)
  → blocksparse_train.py (sparse 30%)
  → export_bsr.py or quantize.py (INT8 weights)
  → run_gemm_axi.py (pack to 32-bit words)
  → axi_driver.py (AXI burst format)
  → axi_dma_master.sv (DDR → FPGA)
  → wgt_buffer.sv (storage)
  → systolic_array.sv (broadcast columns)
  → pe.sv (receive b_in[7:0])
  → mac8.sv (multiply with a_in[7:0])
```

**MAC Unit:**
```
pe.sv inputs: a_in[7:0] (INT8 activation), b_in[7:0] (INT8 weight)
  ↓
mac8.sv: product = a[7:0] × b[7:0] → INT16
  ↓
Accumulator: acc[31:0] += sign_extend(product[15:0])
  ↓
Output: partial sum → next PE or result buffer
```

