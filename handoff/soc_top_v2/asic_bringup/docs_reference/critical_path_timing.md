# Critical Path Timing Analysis

## Target

| Parameter | Value |
|-----------|-------|
| FPGA | Xilinx XC7Z020-1CLG400C (Zynq-7020) |
| Datapath clock | 200 MHz (5.0 ns period) |
| Control clock | 50 MHz (20.0 ns period) |
| ASIC bring-up clock | 50 MHz (20.0 ns period) |
| Array size | 16x16 (256 PEs) |
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

Activations propagate through 16 rows of PEs, each with a 1-cycle register.

```
act_buffer → PE[0,c] → PE[1,c] → ... → PE[15,c]
```

- **Pipeline depth**: 16 stages (one per row)
- **First valid output at PE[15,c]**: cycle 15
- **Total compute phase**: 16 + 15 = 31 cycles per tile (16 data + 15 pipeline drain)

### Path 3: Weight Load Propagation

Weights load systolically through columns. `load_weight` propagates with 1-cycle delay per PE.

```
load_weight → PE[r,0] → PE[r,1] → ... → PE[r,15]
```

- **Propagation**: 16 cycles for the signal to reach last column
- **Weight capture at PE[r,15]**: cycle 15 after load begins
- **Total load phase**: 16 cycles data + 16 cycles propagation = 32 cycles worst-case

### Path 4: Buffer Read (BRAM)

Both `act_buffer.sv` and `wgt_buffer.sv` use registered BRAM with an output register.

```
addr → BRAM (1 cycle) → output_reg (1 cycle) → data_out
```

- **Read latency**: 2 cycles
- **Impact**: Scheduler must account for 2-cycle prefetch, not 1

### Path 5: Current ASIC Bring-Up Clocking

The active ASIC bring-up path for the current repo is single-clock on `clk`.
The older `pulse_sync.sv` CDC note should not be treated as an active blocker
for the present `soc_top_v2` or `accel_top` handoff.

- **Active ASIC assumption**: one 50 MHz top-level clock
- **Reset policy**: asynchronous assert, synchronized deassert at the top level
- **CDC focus**: revisit only if a separate PHY or accelerator clock is
    reintroduced in the taped-out scope

## Tile Execution Timeline

For a single 16x16 tile:

```
Phase           | Cycles | Notes
----------------|--------|--------------------------------
FETCH_ROW       |   4    | 2 BRAM reads (row_ptr[r], row_ptr[r+1])
FETCH_COL       |   4    | 2 BRAM reads (col_idx)
LOAD_WGT        |  16    | Stream 16 weight rows into array
WAIT_WGT        |  16    | Weight propagation to last column
STREAM_ACT      |  31    | 16 activations + 15 pipeline drain
DRAIN           |  16    | Read partial sums from array edge
Overhead        |   6    | FSM state transitions + sync
                |--------|
Total           |  93    | Per non-zero block
```

## Performance Estimates

### Peak Throughput

- 256 MACs per cycle at 200 MHz = **51.2 GOPS** (dense)
- Compute efficiency per tile: 16 * 16 / 93 = 256/93 ≈ 2.75 MACs/cycle effective (amortized over overhead)
- For large tiles (K >> 16): overhead amortizes, approaching peak

### Sparse Speedup

At 70% block sparsity, only 30% of blocks are non-zero:
- 3.3x fewer LOAD_WGT + STREAM_ACT phases
- Metadata fetch overhead remains per-block
- Estimated effective speedup: 2.5-3.0x

## Placement Constraints

The 16x16 PE array should be placed as a contiguous block to minimize routing delay:

```tcl
# Vivado constraint example
create_pblock systolic_pblock
add_cells_to_pblock [get_pblocks systolic_pblock] [get_cells -hier -filter {NAME =~ */systolic_array_i/*}]
resize_pblock [get_pblocks systolic_pblock] -add {SLICE_X0Y0:SLICE_X25Y25}
```

## ASIC Bring-Up Timing Risks

1. **Memory timing is still provisional**. The largest buffers, scratchpads,
     metadata stores, cache arrays, and SRAM/ROM structures are still inferred in
     RTL, so generic-memory WNS is not signoff-quality.
2. **Full-chip timing risk is now dominated by `soc_top_v2` fabric**, not the
     16x16 MAC datapath. CPU, TLB/PTW, crossbar, cache hierarchy, and DRAM control
     are the largest closure risk if the target stays full-chip.
3. **Off-chip IO timing is still a starter model**. The current SDC now has
     side-grouped IO and placeholder delays, but DRAM/package timing is not yet
     package-aware signoff collateral.
4. **Clock gating is intentionally deferred**. The ASIC path now bypasses the
     manual CSR clock gate; add library ICG cells only after CTS, scan, and power
     architecture are frozen.

## Retired Risks

- The earlier `64-bit to 128-bit zero-extension` note is stale for the current
    `accel_top` RTL because the active design already includes dedicated 64 to 128
    DMA packing logic.
- The earlier `blocks_processed` multi-bit CDC note is stale for the current
    single-clock `perf.sv` integration.
- The earlier `pulse_sync.sv` note is stale for the current active ASIC
    handoff path.