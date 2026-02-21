# ACCEL-BSR Sparsity Format Specification

## Overview

This document specifies the Block Sparse Row (BSR) format used by the ACCEL hardware accelerator for storing and processing sparse neural network weights.

## BSR Format Structure

### Memory Layout

BSR format stores only **non-zero blocks** in a compressed format with three components:

1. **Data Array**: Sequential storage of non-zero blocks
2. **Column Indices**: Block column position for each stored block
3. **Row Pointers**: Start index of each block-row in the data array

### Block Dimensions

- **Standard block size**: 14×14 elements (configurable)
- **Convolution blocks**: 4×4 elements for conv layers
- **Data type**: INT8 for weights, FP32 for scales

### BSR Components

```
data:    [num_blocks, block_h, block_w]  # Non-zero blocks only
indices: [num_blocks]                     # Column index of each block
indptr:  [num_block_rows + 1]            # Row pointer (CSR-like)
```

#### Row Pointer Indexing

For block row `i`:
- Number of blocks in row `i`: `indptr[i+1] - indptr[i]`
- Block indices for row `i`: `range(indptr[i], indptr[i+1])`
- Empty rows: `indptr[i+1] == indptr[i]`

#### Column Indices

For block at position `j` in data array:
- Block column: `indices[j]`
- Global column start: `indices[j] * block_w`

## Tile Ordering

### Hardware Tiling Parameters

```
Tm = 2   # Output tile rows
Tn = 2   # Output tile columns
Tk = 64  # Accumulation dimension
```

### Tile Traversal Order

For matrix `C = A @ B`:
1. Iterate over output tiles: `(tm, tn)` in row-major order
2. For each output tile, iterate accumulation tiles: `tk` from 0 to K/Tk
3. Load A tile: `[tm*Tm : (tm+1)*Tm, tk*Tk : (tk+1)*Tk]`
4. Load B tile: `[tk*Tk : (tk+1)*Tk, tn*Tn : (tn+1)*Tn]`
5. Compute: `C_tile += A_tile @ B_tile`

### Block Alignment Rules

1. **Matrix dimensions must be multiples of block size**
   - Pad with zeros if necessary
   - Padding added at right/bottom edges

2. **Block boundaries align with tile boundaries**
   - `Tk` should be multiple of `block_h` (typically 8)
   - Example: Tk=64, block_h=8 → 8 blocks per tile

3. **Zero threshold**: Blocks with L2 norm < 1e-10 are considered zero

## Binary Format

### `.bsr` File (Weight Data)

```
[Block 0: block_h × block_w elements]
[Block 1: block_h × block_w elements]
...
[Block N-1: block_h × block_w elements]
```

- **Element type**: INT8 (1 byte per element)
- **Block size**: 196 bytes for 14×14 blocks
- **Total size**: `num_blocks * block_h * block_w` bytes

### `.meta.json` File (Metadata)

```json
{
  "row_ptr": [0, 5, 12, ...],
  "col_idx": [0, 3, 5, ...],
  "shape": [128, 9216],
  "blocksize": [8, 8],
  "num_blocks": 147,
  "sparsity_pct": 90.0,
  "dtype": "int8"
}
```

### Per-Channel Scales

Stored separately as NumPy arrays:
- **Filename**: `<layer_name>_scales.npy`
- **Shape**: `[K]` for weight matrix `[K, N]`
- **Type**: FP32
- **Usage**: `output = INT32_accumulator * scale_A * scale_B[channel]`

## Hardware Scheduler Interface

### FSM States

```
IDLE → READ_ROW_PTR → CHECK_EMPTY → READ_COL_IDX → LOAD_BLOCK → COMPUTE → NEXT_BLOCK
```

### Memory Access Pattern

1. **Read `row_ptr[block_row]` and `row_ptr[block_row+1]`**
   - Compute `num_blocks = row_ptr[block_row+1] - row_ptr[block_row]`
   - If `num_blocks == 0`, skip to next row

2. **For each block in row:**
   - Read `col_idx[block_idx]` to get column position
   - Read 196-byte block from `data` array at offset `block_idx * 64`
   - Read scale from `scales[block_row * 8 : (block_row+1) * 8]`

3. **Systolic array operation:**
   - Load A slice: `[M, 8]` from input activations
   - Load B block: `[8, 8]` from BSR data
   - Compute: INT8 MAC → INT32 accumulator
   - Scale: Multiply by `scale_A * scale_B[channel]`
   - Accumulate into C at position `[row_start, col_start]`

## Example: FC1 Layer (MNIST)

### Layer Properties

```
Input:  9216 features
Output: 128 neurons
Sparsity: 90%
Block size: 14×14
```

### BSR Dimensions

```
Weight shape: [128, 9216]  (output_features, input_features)
Block rows: 128 / 8 = 16
Block cols: 9216 / 8 = 1152
Total possible blocks: 16 × 1152 = 18,432
Non-zero blocks (10%): ~1,843 blocks
```

### Memory Requirements

```
Data array:    1,843 blocks × 196 bytes = 117,952 bytes (~115 KB)
Column indices: 1,843 × 4 bytes = 7,372 bytes (~7 KB)
Row pointers:   17 × 4 bytes = 68 bytes
Scales:         128 × 4 bytes = 512 bytes
Total:          ~126 KB (vs 1.15 MB dense → 11× reduction)
```

### Hardware Scheduler Loop

```python
for block_row in range(16):
    num_blocks = row_ptr[block_row + 1] - row_ptr[block_row]
    
    if num_blocks == 0:
        continue  # Empty row
    
    for i in range(num_blocks):
        block_idx = row_ptr[block_row] + i
        block_col = col_idx[block_idx]
        
        # Load 14×14 block from BRAM
        block_data = memory[block_idx * 64 : (block_idx + 1) * 64]
        
        # Compute position in output
        out_row = block_row * 8
        out_col = block_col * 8
        
        # Feed to systolic array
        systolic_compute(A[:, out_row:out_row+8], 
                        block_data,
                        C[:, out_col:out_col+8])
```

## Validation Requirements

### Golden Model Matching

Hardware output must match software golden model within tolerance:
- **INT8 operations**: Bit-exact match for integer operations
- **Quantization**: Bit-exact match after rounding
- **Floating-point scales**: Max error < 1e-4 (FP32 precision)

### Edge Cases to Test

1. **Empty rows**: `row_ptr[i+1] == row_ptr[i]`
2. **100% sparse**: All blocks zero → `num_blocks == 0`
3. **100% dense**: All blocks present → `num_blocks == total_blocks`
4. **Single block**: Minimal case
5. **Irregular sparsity**: Random block positions

## References

- BSR Format: [scipy.sparse.bsr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html)
- INT8 Quantization: `docs/QUANTIZATION.md`
- Hardware Architecture: `docs/ARCHITECTURE.md`
