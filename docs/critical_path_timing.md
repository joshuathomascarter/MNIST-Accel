# Critical Path Timing Analysis

## Target

| Parameter | Value |
|-----------|-------|
| FPGA | Xilinx XC7Z020-1CLG400C (Zynq-7020) |
| Datapath clock | 200 MHz (5.0 ns period) |
| Control clock | 50 MHz (20.0 ns period) |
| Array size | 14x14 (196 PEs) |
| Data width | INT8 operands, INT32 accumulators |

## Critical Paths

### Path 1: MAC Datapath (Systolic Array)

The MAC unit (`mac8.sv`) performs signed 8x8 multiply + 32-bit accumulate. On Zynq-7020, this maps to one DSP48E1 slice.

```
activation_in → REG → DSP48E1 (multiply + accumulate) → acc_reg
```

- **DSP48E1 latency**: 1 cycle (pipelined multiply + add)
- **Estimated delay**: ~3.5 ns (within 5.0 ns budget)
- **Margin**: ~1.5 ns

### Path 2: Activation Propagation (Systolic Chain)

Activations propagate through 14 rows of PEs, each with a 1-cycle register.

```
act_buffer → PE[0,c] → PE[1,c] → ... → PE[13,c]
```

- **Pipeline depth**: 14 stages (one per row)
- **First valid output at PE[13,c]**: cycle 13
- **Total compute phase**: 14 + 13 = 27 cycles per tile (14 data + 13 pipeline drain)

### Path 3: Weight Load Propagation

Weights load systolically through columns. `load_weight` propagates with 1-cycle delay per PE.

```
load_weight → PE[r,0] → PE[r,1] → ... → PE[r,13]
```

- **Propagation**: 14 cycles for the signal to reach last column
- **Weight capture at PE[r,13]**: cycle 13 after load begins
- **Total load phase**: 14 cycles data + 14 cycles propagation = 28 cycles worst-case

### Path 4: Buffer Read (BRAM)

Both `act_buffer.sv` and `wgt_buffer.sv` use registered BRAM with an output register.

```
addr → BRAM (1 cycle) → output_reg (1 cycle) → data_out
```

- **Read latency**: 2 cycles
- **Impact**: Scheduler must account for 2-cycle prefetch, not 1

### Path 5: Clock Domain Crossing

Control-to-datapath CDC uses toggle-based `pulse_sync.sv`:

```
control_clk (50 MHz) → toggle → sync_2ff → detect_edge → datapath_clk (200 MHz)
```

- **Latency**: 2-3 datapath clock cycles (10-15 ns)
- **Throughput**: 1 pulse per control clock period (20 ns)

## Tile Execution Timeline

For a single 14x14 tile:

```
Phase           | Cycles | Notes
----------------|--------|--------------------------------
FETCH_ROW       |   4    | 2 BRAM reads (row_ptr[r], row_ptr[r+1])
FETCH_COL       |   4    | 2 BRAM reads (col_idx)
LOAD_WGT        |  14    | Stream 14 weight rows into array
WAIT_WGT        |  14    | Weight propagation to last column
STREAM_ACT      |  27    | 14 activations + 13 pipeline drain
DRAIN           |  14    | Read partial sums from array edge
                |--------|
Total           |  77    | Per non-zero block
```

## Performance Estimates

### Peak Throughput

- 196 MACs per cycle at 200 MHz = **39.2 GOPS** (dense)
- Compute efficiency per tile: 14 * 14 / 77 = 196/77 = ~2.5 MACs/cycle effective (amortized over overhead)
- For large tiles (K >> 14): overhead amortizes, approaching peak

### Sparse Speedup

At 70% block sparsity, only 30% of blocks are non-zero:
- 3.3x fewer LOAD_WGT + STREAM_ACT phases
- Metadata fetch overhead remains per-block
- Estimated effective speedup: 2.5-3.0x

## Placement Constraints

The 14x14 PE array should be placed as a contiguous block to minimize routing delay:

```tcl
# Vivado constraint example
create_pblock systolic_pblock
add_cells_to_pblock [get_pblocks systolic_pblock] [get_cells -hier -filter {NAME =~ */systolic_array_i/*}]
resize_pblock [get_pblocks systolic_pblock] -add {SLICE_X0Y0:SLICE_X25Y25}
```

## Known Timing Risks

1. **64-bit to 112-bit zero-extension** in `accel_top.sv` — only 8 of 14 INT8 values loaded per DMA write (see AUDIT.md R34)
2. **Combinational clock gating** in `csr.sv` — glitch-prone without latch (see AUDIT.md R14)
3. **Multi-bit CDC** on `blocks_processed` counter — needs gray coding (see AUDIT.md R23)