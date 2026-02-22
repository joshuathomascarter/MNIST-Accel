# ACCEL-v1 Architecture Deep Dive

> Technical specification for the 14×14 weight-stationary systolic array accelerator (PYNQ-Z2)

---

## Table of Contents

1. [Systolic Array Operation](#systolic-array-operation)
2. [Weight-Stationary Dataflow](#weight-stationary-dataflow)
3. [BSR Sparse Scheduling](#bsr-sparse-scheduling)
4. [Timing Analysis](#timing-analysis)
5. [MNIST CNN Layer Breakdown](#mnist-cnn-layer-breakdown)
6. [Memory Bandwidth Analysis](#memory-bandwidth-analysis)
7. [Power Estimation](#power-estimation)

---

## Systolic Array Operation

### Matrix Multiplication Mapping

For C = A × B where:
- A: Activations [M × K]
- B: Weights [K × N]
- C: Output [M × N]

The 14×14 array computes a 14×14 output tile per pass:

```
                    B (weights)
                    [K × 14]
                    ↓ ↓ ↓ ↓
              ┌─────────────────┐
   A          │                 │
[14 × K] ───▶ │  14×14 Systolic │ ───▶ C [14 × 14]
              │     Array       │
              │                 │
              └─────────────────┘
```

### Tiling for Large Matrices

For M=512, N=512, K=512:

```
Total tiles = ceil(512/14) × ceil(512/14) × ceil(512/14)
            = 37 × 37 × 37
            = 50,653 tile operations

Each tile: 14 × 14 × 14 = 2,744 MACs
Total MACs: 50,653 × 2,744 = ~139M (matches M×N×K)
```

### Tile Loop Structure

```python
# Pseudocode for tiled GEMM
for m_tile in range(0, M, 14):      # Output row tiles
    for n_tile in range(0, N, 14):  # Output col tiles
        # Initialize accumulator tile to 0
        acc[14][14] = 0
        
        for k_tile in range(0, K, 14):  # Reduction tiles
            # Load 14×14 weight block
            load_weights(B[k_tile:k_tile+14, n_tile:n_tile+14])
            
            # Stream 14×14 activation block
            stream_activations(A[m_tile:m_tile+14, k_tile:k_tile+14])
            
            # Accumulate partial products
            acc += systolic_compute()
        
        # Store output tile
        store_output(C[m_tile:m_tile+14, n_tile:n_tile+14], acc)
```

---

## Weight-Stationary Dataflow

### Why Weight-Stationary?

| Dataflow | Weight Reuse | Activation Reuse | Best For |
|----------|--------------|------------------|----------|
| Weight-Stationary | ★★★★★ | ★★☆☆☆ | Large batch, sparse weights |
| Output-Stationary | ★★☆☆☆ | ★★★★☆ | Small batch |
| Input-Stationary | ★★★☆☆ | ★★★★☆ | Balanced workloads |

We use **weight-stationary** because:
1. Weights are loaded once per K-tile, reused across all M
2. BSR sparsity means we only load non-zero weight blocks
3. Activation streaming is memory-bound anyway

### Detailed Dataflow Timing

```
Cycle:  0    1    2    3    4    5    6    7   ...   K+14  K+15
        │    │    │    │    │    │    │    │         │     │
PE[0,0]:│ w₀ │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │ a₅ │ a₆ │...│aₖ₋₁│drain│
PE[0,1]:│ w₁ │    │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │ a₅ │...│aₖ₋₂│aₖ₋₁│
PE[0,2]:│ w₂ │    │    │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │...│aₖ₋₃│aₖ₋₂│
   :    │    │    │    │    │    │    │    │    │   │     │
PE[0,13]:│w₁₅│    │    │    │    │    │    │    │...│aₖ₋₁₆│aₖ₋₁₅│
        │    │    │    │    │    │    │    │    │   │     │
        │◄──load──►│◄──────── K cycles compute ─────────►│◄drain►│
        │  (14 cyc) │                                    │(14 cyc)│
```

### PE State Machine

```
        ┌───────────┐
        │   IDLE    │
        └─────┬─────┘
              │ start
              ▼
        ┌───────────┐
        │LOAD_WEIGHT│ ← weight_in valid
        └─────┬─────┘
              │ weight_loaded
              ▼
        ┌───────────┐
        │  COMPUTE  │ ← accumulate: acc += w × a
        └─────┬─────┘
              │ k_done
              ▼
        ┌───────────┐
        │   DRAIN   │ → psum_out valid
        └─────┬─────┘
              │ drained
              ▼
        ┌───────────┐
        │   IDLE    │
        └───────────┘
```

---

## BSR Sparse Scheduling

### Block Skip Logic

The BSR scheduler reads `row_ptr` and `col_idx` to determine which weight blocks to load:

```systemverilog
// Simplified BSR scheduler logic
always_ff @(posedge clk) begin
    case (state)
        IDLE: begin
            if (start) begin
                block_row <= 0;
                block_idx <= row_ptr[0];
                state <= CHECK_ROW;
            end
        end
        
        CHECK_ROW: begin
            if (block_idx < row_ptr[block_row + 1]) begin
                // Non-zero block exists in this row
                current_col <= col_idx[block_idx];
                state <= LOAD_BLOCK;
            end else begin
                // Skip empty row
                block_row <= block_row + 1;
                if (block_row + 1 >= num_block_rows)
                    state <= DONE;
                else
                    block_idx <= row_ptr[block_row + 1];
            end
        end
        
        LOAD_BLOCK: begin
            // DMA 256 bytes from data[block_idx * 256]
            // ... load into weight buffer ...
            block_idx <= block_idx + 1;
            state <= CHECK_ROW;
        end
    endcase
end
```

### Sparsity Speedup Model

```
Dense cycles = M_tiles × N_tiles × K_tiles × (K + 15)
Sparse cycles = M_tiles × N_tiles × nnz_blocks × (K + 15)

Speedup = Dense_cycles / Sparse_cycles
        = total_blocks / nnz_blocks
        = 1 / (1 - sparsity)

Example @ 70% sparsity:
  Speedup = 1 / 0.3 = 3.33×
```

---

## Timing Analysis

### Critical Path

```
Weight BRAM → PE weight reg → Multiplier → Adder → Accumulator reg
   ↓              ↓              ↓           ↓           ↓
  1.2ns        0.3ns          2.5ns       1.5ns       0.5ns = 6.0ns total

Target: 200 MHz (5ns period)
Slack: -1.0ns 

Solution: Pipeline the multiplier
  BRAM → reg → MUL stage 1 → MUL stage 2 → ADD → ACC
  1.2    0.3      1.3            1.2        1.5   0.5  = 3.0ns per stage ✓
```

### Latency Breakdown

| Operation | Cycles | Notes |
|-----------|--------|-------|
| CSR config | 10 | AXI-Lite writes |
| DMA BSR header | 50 | 12 bytes @ 64-bit AXI |
| DMA row_ptr | 20 | Per block row |
| DMA col_idx | 5 | Per non-zero block |
| DMA block data | 32 | 256 bytes @ 64-bit |
| Weight load | 16 | Into PE array |
| Compute | K | Stream activations |
| Drain | 16 | Collect partial sums |
| Output write | 32 | 256 bytes @ 64-bit |

### Pipeline Diagram

```
Block 0:  [DMA]──[Load]──[Compute K cycles]──[Drain]
Block 1:         [DMA]──[Load]──[Compute K cycles]──[Drain]
Block 2:                [DMA]──[Load]──[Compute K cycles]──[Drain]
          │      │      │      │
          ◄──────┼──────┼──────► Overlapped: DMA while computing
                 │      │
                 ◄──────► Not overlapped: must wait for load
```

---

## MNIST CNN Layer Breakdown

### Layer Dimensions

| Layer | Type | Input Shape | Weight Shape | Output Shape | Params | MACs |
|-------|------|-------------|--------------|--------------|--------|------|
| conv1 | Conv 3×3 | 1×28×28 | 32×1×3×3 | 32×26×26 | 288 | 195K |
| conv2 | Conv 3×3 | 32×26×26 | 64×32×3×3 | 64×24×24 | 18,432 | 10.6M |
| pool | MaxPool 2×2, s2 | 64×24×24 | - | 64×12×12 | - | - |
| fc1 | Linear | 9216 | 128×9216 | 128 | 1,179,648 | 1.18M |
| fc2 | Linear | 128 | 10×128 | 10 | 1,280 | 1.3K |
| **TOTAL** | | | | | **1.2M** | **12.0M** |

> **Note:** fc1 dominates (98.3% of parameters). Block sparsity pruning has the most impact on this layer.

### im2col Transformation for Conv

Convolutions are mapped to GEMM via im2col:

```
Conv2D: Input [C_in × H × W] * Kernel [C_out × C_in × kH × kW]
      ↓ im2col
GEMM:   A [H_out×W_out × C_in×kH×kW] × B [C_in×kH×kW × C_out]
      = C [H_out×W_out × C_out]

Example: conv2
  Input: 32×26×26, Kernel: 64×32×3×3
  im2col A: [576 × 288]  (576 = 24×24, 288 = 32×3×3)
  Weight B: [288 × 64]
  Output C: [576 × 64]
  
  Tiles: ceil(576/14) × ceil(64/14) × ceil(288/14)
       = 42 × 5 × 21 = 4,410 tile ops
```

### Cycle Estimates per Layer

Assuming 200 MHz, 70% block sparsity:

| Layer | GEMM Shape | Tile-ops (Dense) | Tile-ops (Sparse) | Cycles | Time (µs) |
|-------|-----------|------------------|-------------------|--------|----------|
| conv1 | [676×9] × [9×32] | 147 | 44 | 750 | 3.8 |
| conv2 | [576×288] × [288×64] | 4,410 | 1,323 | 22.5K | 112 |
| fc1 | [1×9216] × [9216×128] | 6,590 | 1,977 | 33.6K | 168 |
| fc2 | [1×128] × [128×10] | 10 | 3 | 51 | 0.3 |
| **TOTAL** | | **11,157** | **3,347** | **~57K** | **~284 µs** |

**Throughput: ~3,500 images/second @ 70% sparsity (200 MHz)**

---

## Memory Bandwidth Analysis

### DDR Requirements

```
Per image (MNIST CNN):
  Weights: 4.8 MB (FP32) → 1.2 MB (INT8) → ~360 KB (70% sparse BSR)
  Activations: ~37 KB peak (conv2 output: 64×24×24)
  Outputs: 10 bytes (final logits)
  
Bandwidth @ 3500 img/s:
  Weights: 360 KB × 1 = 360 KB (fits in BRAM; loaded once)
  Activations: 37 KB × 3500 = 130 MB/s
  Outputs: negligible
  
Total: ~130 MB/s << Z7020's 4.2 GB/s DDR bandwidth ✓
```

### On-Chip Buffer Sizing

| Buffer | Size | Purpose |
|--------|------|---------|
| Weight BRAM | 12 KB | Current 14×14 block + next (double buffer) |
| Activation BRAM | 7 KB | 14 rows × 256 cols × 2 banks |
| Output BRAM | 3 KB | 14×14 × INT32 × 2 banks |
| BSR Metadata | 2 KB | row_ptr, col_idx for current layer |
| **Total** | **24 KB** | Fits in Z7020's 560 KB BRAM ✓ |

---

## Power Estimation

### Component Breakdown (@ 200 MHz, Z7020)

| Component | Count | Power/Unit | Total |
|-----------|-------|------------|-------|
| DSP48 (MAC) | 196 | 5 mW | 980 mW |
| BRAM | 30 | 10 mW | 300 mW |
| Logic (LUTs) | 18K | 0.01 mW | 180 mW |
| Registers | 12K | 0.005 mW | 60 mW |
| Clocking | - | - | 100 mW |
| I/O (AXI) | - | - | 50 mW |
| **Static** | - | - | 200 mW |
| **Total** | | | **1.87 W** |

### Energy Efficiency

```
Throughput: 137 img/s
Power: 1.87 W
Energy: 1.87 / 137 = 13.6 mJ/image

Comparison:
  CPU (i7): ~1000 mJ/image
  GPU (RTX 3090): ~50 mJ/image
  ACCEL-v1 (Z7020): ~14 mJ/image ← 3.5× better than GPU
```

### Power Optimization Techniques (Implemented)

1. **Clock gating**: Disable PE clocks when idle
2. **Block skipping**: Don't clock zero-weight blocks
3. **Activation gating**: Gate datapath for zero activations
4. **BRAM power-down**: Disable unused banks

---

## Summary: Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Array size | 14×14 | 196 PEs (fits Z7020's 220 DSPs) |
| Block size | 14×14 | Matches array |
| Data type | INT8 weights, INT8 activations | INT32 accumulators |
| Clock | 200 MHz target | 5ns period |
| Throughput | 39.2 GOPS (dense) | 196 MACs × 200 MHz |
| Throughput | 131 GOPS (70% sparse) | 3.3× speedup |
| Latency | ~0.28 ms/image | MNIST CNN @ 70% sparse |
| Power | ~1.7 W | Z7020 @ 200 MHz |
| Efficiency | 77 GOPS/W | At 70% sparsity |
