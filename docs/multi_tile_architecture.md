# Multi-Tile Architecture

> Design reference for scaling ACCEL-v1 from a single accelerator pipeline to a 4x4 tile array.
> Focus: partitioning strategy, inter-tile data movement, control flow, and memory hierarchy for Month 5.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Goal](#design-goal)
3. [Core Principle: Tile-Local Datapath Stays the Same](#core-principle-tile-local-datapath-stays-the-same)
4. [System-Level Partitioning Modes](#system-level-partitioning-modes)
5. [Layer Descriptor Model](#layer-descriptor-model)
6. [Tile Controller Flow](#tile-controller-flow)
7. [MNIST Mapping Across 16 Tiles](#mnist-mapping-across-16-tiles)
8. [Inter-Tile Communication Requirements](#inter-tile-communication-requirements)
9. [Required RTL Changes](#required-rtl-changes)
10. [New Modules for Month 5](#new-modules-for-month-5)
11. [Memory Hierarchy Direction](#memory-hierarchy-direction)
12. [Tradeoffs and Risks](#tradeoffs-and-risks)
13. [Implementation Guidance](#implementation-guidance)

---

## Overview

ACCEL-v1 currently behaves as a single accelerator pipeline:

- load activations and weights from memory
- run sparse systolic compute
- accumulate and requantize outputs
- either feed outputs back on-chip or drain them to DDR

The multi-tile architecture keeps that tile-local compute pipeline intact and scales performance by instantiating multiple copies of the accelerator tile behind a shared interconnect.

The main architectural change is not inside the PE array. The main change is the addition of:

- per-tile orchestration
- inter-tile communication
- layer-dependent work partitioning
- global synchronization between tiles

---

## Design Goal

The 16-tile design should support different parallelization modes depending on layer shape:

- convolution layers should primarily use output-channel partitioning
- large fully connected layers should primarily use K-dimension splitting with reduction
- very small layers should run on a single tile when parallelism is not worth the communication cost

This avoids forcing one scheduling policy onto all layer types.

---

## Core Principle: Tile-Local Datapath Stays the Same

The internal datapath of each tile should remain weight-stationary.

Inside each tile, the following blocks stay conceptually the same:

- `bsr_scheduler`
- `act_buffer`
- `wgt_buffer`
- `systolic_array_sparse`
- `output_accumulator`
- `output_bram_ctrl`
- `output_bram_buffer`

That means:

- weights are still loaded into the systolic array and held stationary
- activations still stream through the array
- partial sums are still accumulated locally
- outputs are still staged in the output BRAM path before reuse or drain

The multi-tile system changes what each tile is assigned to compute and where outputs go after local completion.

---

## System-Level Partitioning Modes

### 1. Output-Partitioned Mode

Best fit:
- convolution layers
- layers where each tile can compute a disjoint set of output channels

Behavior:
- all tiles receive the same input activation tensor
- each tile receives a different subset of filter weights
- each tile computes a different subset of output channels
- no inter-tile reduction is required because each tile's outputs are final

Advantages:
- simple tile-local compute
- no partial sum merge
- matches convolution naturally

Cost:
- after the layer completes, the next layer may need every tile to see the full activation tensor
- this creates an all-to-all scatter phase between layers

### 2. K-Split Mode

Best fit:
- large fully connected layers
- layers where the reduction dimension is too large for one tile to process efficiently

Behavior:
- the K dimension is split across tiles
- each tile receives only its activation slice and matching weight slice
- each tile computes partial sums for the same output vector
- partial sums are reduced across tiles using the NoC

Advantages:
- parallelizes the largest dimension cleanly
- balances work across tiles

Cost:
- requires explicit partial-sum reduction
- introduces barrier and reduction latency
- demands deterministic reduce routing

### 3. Single-Tile Mode

Best fit:
- very small layers
- final classifier layers such as `FC2`

Behavior:
- one tile performs the full layer
- remaining tiles idle at a barrier

Advantages:
- simplest control path
- avoids paying communication overhead for tiny workloads

Cost:
- low hardware utilization for that layer, but often still the best latency tradeoff

---

## Layer Descriptor Model

Software should pre-compute a descriptor for each tile and each layer before inference begins.

Suggested descriptor fields:

```c
struct layer_desc {
    uint8_t  mode;             // 0=output_partition, 1=k_split, 2=single_tile, 3=idle
    uint16_t my_filter_start;  // first output filter/channel assigned to this tile
    uint16_t my_filter_count;  // number of output filters/channels assigned to this tile
    uint16_t my_k_start;       // first K index assigned to this tile
    uint16_t my_k_count;       // number of K elements assigned to this tile
    uint8_t  reduce_peer;      // peer tile used during staged reduction
    uint8_t  reduce_root;      // set for the final reduction destination
    uint8_t  bcast_src;        // source tile or controller for activation broadcast
    uint32_t weight_base;      // base address of this tile's weight slice
    uint32_t act_base;         // base address of this tile's activation slice
    uint8_t  is_last_layer;    // controls final drain behavior
    uint8_t  pool_en;          // optional post-layer pooling enable
};
```

Software responsibilities:

- choose partition mode per layer
- assign tile ownership
- compute all base addresses and slice sizes
- pre-program tile descriptors before global start

Hardware responsibilities:

- read the descriptor
- execute the assigned mode
- synchronize with peer tiles at the end of the phase

---

## Tile Controller Flow

Each tile should be wrapped by a tile controller FSM that interprets the layer descriptor and drives the existing accelerator pipeline.

### High-Level FSM

```text
LAYER_START
  -> READ_DESC
  -> MODE_SELECT

OUTPUT_PART:
  -> FETCH_MY_WEIGHTS
  -> FETCH_ACTIVATIONS
  -> COMPUTE
  -> SCATTER_OUTPUT
  -> BARRIER

K_SPLIT:
  -> FETCH_MY_K_SLICE
  -> COMPUTE
  -> REDUCE_SEND / REDUCE_RECV
  -> BARRIER

SINGLE_TILE:
  -> FETCH_ALL
  -> COMPUTE
  -> FEEDBACK_OR_DRAIN
  -> BARRIER

IDLE:
  -> BARRIER

BARRIER
  -> NEXT_LAYER
  -> LAYER_START
```

### Tile Controller Responsibilities

- issue local DMA/fetch commands
- start the existing local compute pipeline
- select whether outputs are reused locally, sent to NoC, or drained
- wait for scatter/reduce completion
- participate in layer barrier synchronization

---

## MNIST Mapping Across 16 Tiles

This section gives a concrete reference mapping for the 4-layer MNIST network.

### Conv1

Assume:
- 32 output filters
- 16 tiles

Mapping:
- 2 filters per tile
- all tiles receive the same 28x28 input activation image
- each tile loads weights for its 2 assigned filters
- each tile produces 2 output channels of size 26x26

Post-compute:
- each tile scatters its 2 channels to all tiles
- after scatter, every tile has the full `32 x 26 x 26` activation tensor

Reason:
- the next convolution layer expects the full set of input channels

### Conv2

Assume:
- 64 output filters
- 16 tiles

Mapping:
- 4 filters per tile
- activations are already present locally after conv1 scatter
- each tile loads weights for its 4 assigned output channels
- each tile computes 4 channels of size 24x24
- optional pooling reduces these to 12x12

Post-compute:
- each tile scatters its 4 output channels
- after scatter, every tile has the full `64 x 12 x 12` tensor

### FC1

Assume:
- matrix shape approximately `128 x 9216`

Mapping:
- split `K=9216` across 16 tiles
- each tile handles 576 K elements
- every tile computes a 128-element partial-sum vector

Post-compute:
- reduce the 16 partial vectors through the NoC
- a tree reduction takes 4 stages for 16 tiles
- the reduction root produces the final 128-element output
- final activations are then broadcast to tiles that need them

### FC2

Assume:
- matrix shape `10 x 128`

Mapping:
- run entirely on tile 0
- other tiles remain idle at the barrier

Reason:
- communication overhead would dominate any parallel speedup
- a single tile can complete the layer cheaply

---

## Inter-Tile Communication Requirements

### Broadcast

Used when:
- all tiles need the same activation tensor
- a reduction root needs to distribute final results

Examples:
- input image to all conv1 tiles
- FC1 reduced output broadcast before FC2 or later processing

### Scatter

Used when:
- tiles produce disjoint output-channel subsets
- the next layer needs the full activation tensor on each tile

Examples:
- conv1 output scatter
- conv2 output scatter

### Reduction

Used when:
- all tiles compute partial sums for the same outputs

Examples:
- FC1 K-split reduction

### Barrier

Used when:
- all tiles must complete their current phase before the next layer begins

Examples:
- after scatter
- after reduction
- before descriptor advance

---

## Required RTL Changes

### `hw/rtl/buffer/output_bram_ctrl.sv`

Current role:
- choose between local feedback and DDR drain

Multi-tile impact:
- `S_DECIDE` needs more destinations
- outputs may need to go to:
  - local feedback path
  - DDR drain path
  - NoC scatter path
  - NoC reduction-send path
  - NoC reduction-receive/accumulate path

Likely new states:
- `S_NOC_SCATTER`
- `S_NOC_REDUCE_SEND`
- `S_NOC_REDUCE_RECV`

### `hw/rtl/top/accel_top.sv`

Current role:
- single-pipeline integration

Multi-tile impact:
- becomes the tile-local datapath wrapper or is embedded under a tile wrapper
- should expose clean control/status hooks to the tile controller
- should expose output routing control for local reuse vs NoC vs DDR

### `sw/cpp/src/driver/accelerator.cpp`

Current role:
- configure one accelerator instance and run a layer sequence

Multi-tile impact:
- software must build and write descriptors for all tiles
- software must understand per-layer mode selection
- software must launch a global multi-tile inference run instead of a single tile job

---

## New Modules for Month 5

### `tile_controller.sv`

Purpose:
- per-tile sequencing FSM
- owns descriptor execution
- controls local fetch, compute, and post-compute flow

### `accel_tile.sv`

Purpose:
- package the existing ACCEL-v1 pipeline as one reusable tile
- expose a cleaner interface to the tile controller and NoC

### `accel_tile_array.sv`

Purpose:
- instantiate 16 tiles
- wire each tile into the NoC and global synchronization logic

### `noc_interface.sv`

Purpose:
- provide packet injection and packet receive logic
- convert local buffer traffic into NoC flits or packet streams

### `scatter_engine.sv`

Purpose:
- send output-channel slices from one tile to all peers
- place received slices into the correct local activation storage

### `reduce_engine.sv`

Purpose:
- merge partial sums during K-split execution
- support staged tree reduction across the tile array

### `barrier_sync.sv`

Purpose:
- detect when all participating tiles are complete
- release tiles into the next layer only when the system is synchronized

### `accel_scratchpad.sv`

Purpose:
- define a cleaner per-tile local memory abstraction
- hold activations, weights, and staged outputs for the tile

### `accel_dma_engine.sv`

Purpose:
- tile-facing DMA or memory-fetch interface
- move slices between shared memory and local scratchpads

---

## Memory Hierarchy Direction

The long-term ASIC-oriented memory hierarchy should look like this:

```text
DRAM controller
    |
    v
Shared L2 SRAM / weight staging SRAM
    |
    v
NoC / shared fabric
    |
    v
Per-tile L1 scratchpads
    |
    v
Tile-local systolic array and accumulators
```

### L1: Per-Tile Scratchpads

These are the local memories closest to each tile:

- activation storage
- weight storage
- output staging storage

In the current FPGA design, these correspond to structures like:

- `act_buffer`
- `wgt_buffer`
- `output_bram_buffer`

In an ASIC version, these would likely become SRAM macros.

### L2: Shared On-Chip SRAM

Primary use:
- stage weights once from DRAM
- multicast or distribute slices to tiles
- reduce repeated DRAM traffic

This is especially useful because:
- conv and FC weights can be reused heavily
- weight delivery becomes a system bandwidth bottleneck

### DRAM

Primary use:
- backing store for model weights and larger activations
- refill source when data does not fit on-chip

### Prefetching

BSR metadata makes prefetching more deterministic than speculative.

Because `row_ptr` and `col_idx` explicitly describe the non-zero block stream, software or hardware can often predict:

- which weight blocks are needed next
- which activation windows will be consumed next
- how to stage tiles before compute begins

That makes sparse scheduling more controllable than a generic cache-based design.

---

## Tradeoffs and Risks

### Output-Partitioned Convolution

Benefit:
- no reduction needed

Cost:
- scatter traffic between layers can become large
- inter-layer data movement may dominate compute

### K-Split Fully Connected

Benefit:
- distributes the largest reduction dimension cleanly

Cost:
- requires deterministic and efficient reduction
- reduction latency can erase compute gains if not pipelined well

### Single-Tile Small Layers

Benefit:
- lowest control complexity for tiny workloads

Cost:
- under-utilizes the array for those layers

### System Bottlenecks

Likely bottlenecks in a multi-tile design are:

- activation scatter bandwidth
- weight multicast bandwidth
- barrier latency
- NoC congestion during reduction

The system is unlikely to be compute-bound first. It is more likely to be movement-bound.

---

## Implementation Guidance

### Recommended Order for Month 5

1. Define `accel_tile.sv` as a clean wrapper around the existing single-tile accelerator path.
2. Add `tile_controller.sv` with descriptor-driven mode select.
3. Add barrier synchronization first, before full scatter/reduce.
4. Implement output-partitioned conv flow first.
5. Add scatter path for conv-to-conv transitions.
6. Add K-split reduction for FC1.
7. Keep FC2 single-tile until the rest of the system is stable.

### Keep the First Version Simple

For the first multi-tile version:

- support only one active mode per layer
- use static software-generated descriptors
- use deterministic reduction trees
- avoid dynamic work stealing or adaptive remapping
- keep tile-local compute unchanged as much as possible

### Success Criteria

A good Month 5 milestone would be:

- 16 tiles instantiated
- conv layers run in output-partitioned mode
- FC1 runs in K-split mode with correct reduction
- FC2 runs on one tile
- all tiles advance layer-by-layer through a barrier
- outputs match the single-tile golden model
