# ACCEL-v1 System Architecture

## Overview (Updated Phase 5)

ACCEL-v1 is a sparse CNN accelerator with:
- **Dense path** (Phases 1-3): Traditional systolic array GEMM
- **Sparse path** (Phases 4-5): BSR-format sparse via metadata cache + scheduler
- **Host interfaces**: UART, AXI4-Lite, optional SPI

## Dataflow (Phase 5 - Sparse Pipeline)

### Sparse Metadata Pipeline Overview

Phase 5 introduces a **metadata-driven sparse acceleration path** for Block Sparse Row (BSR) format tensors. This pipeline decouples sparsity metadata (row pointers, column indices) from block data, enabling **6-9× speedup** for sparse CNN layers:

```
SPARSE DATA FLOW (Phase 5):

Host System
    │
    │ (UART: BSR metadata + block data)
    ▼
┌──────────────────────┐
│   UART Receiver      │  115.2 kbps, 8N1
│   (8-bit stream)     │
└──────────────────────┘
    │
    │ (Raw 8-bit words)
    ▼
┌──────────────────────┐
│   DMA Lite Engine    │  8→32 bit assembly
│   (64-entry FIFO)    │  Word packing (LSB-first)
└──────────────────────┘
    │
    │ (32-bit metadata words)
    ▼
┌──────────────────────┐
│   Metadata Decoder   │  256-entry BRAM cache
│   (meta_decode.sv)   │  Validation FSM
│                      │  Perf counters (hit/miss)
└──────────────────────┘
    │
    │ (Decoded row/col indices)
    ▼
┌──────────────────────┐
│   BSR Scheduler      │  Workload generation
│   (bsr_scheduler.sv) │  Block dependency tracking
└──────────────────────┘
    │
    │ (Sparse block addresses)
    ▼
┌──────────────────────────────┐
│   Systolic Array (Sparse)    │  2×2 PEs
│   (systolic_sparse.sv)       │  8×8 block compute
│   Block BRAM: 4MB (32-bit)   │  INT8 dataflow
└──────────────────────────────┘
    │
    │ (32-bit partial sums)
    ▼
┌──────────────────────┐
│   Output Reorder     │  Result reconstruction
│   (Clamp/Shift INT8) │  CSR post-processing
└──────────────────────┘
    │
    │ (UART: INT8 results)
    ▼
Host System
```

### BSR Format Overview

Block Sparse Row (BSR) encodes M×N sparse tensor as:

```
┌─────────────────────────────────────────────┐
│  Block Sparse Row (BSR) Format              │
├─────────────────────────────────────────────┤
│  • Block size: 8×8 (INT8 elements)          │
│  • Non-zero blocks stored in row-major      │
│  • Metadata per block: row_idx, col_idx     │
│  • Block data: 64 INT8 values per block     │
│                                             │
│  Example: 2 non-zero blocks in layer        │
│                                             │
│  Block 0: Row=0, Col=0 (top-left 8×8)       │
│    ┌────────────────────────────┐           │
│    │ a[0][0]  a[0][1] ... a[0][7]│           │
│    │ ...                         │           │
│    │ a[7][7] (64 values total)   │           │
│    └────────────────────────────┘           │
│                                             │
│  Block 1: Row=2, Col=1 (row 16-23, col 8-15)│
│    ┌────────────────────────────┐           │
│    │ a[16][8]  ... a[16][15]     │           │
│    │ ...                         │           │
│    │ a[23][15]                   │           │
│    └────────────────────────────┘           │
│                                             │
│  Metadata Stream Format:                    │
│    [ROW_PTR | COL_IDX | ... | BLOCK_DATA]   │
│     (Type=0)  (Type=1)        (32 bytes)    │
└─────────────────────────────────────────────┘
```

### Component Details

#### 1. DMA Lite Engine (`verilog/dma/dma_lite.v`)

Lightweight byte-to-word DMA specifically for metadata stream assembly:

```verilog
module dma_lite #(
    parameter FIFO_DEPTH = 64,    // 64-entry buffer
    parameter PACKET_LEN = 4      // 4 bytes → 1 word
) (
    input clk, rst_n,
    
    // Host input (8-bit stream from UART)
    input [7:0] in_data,
    input in_valid,
    output in_ready,
    
    // Assembled output (32-bit words)
    output [31:0] out_data,
    output out_valid,
    input out_ready,
    
    // Status
    output dma_done,
    output [15:0] dma_bytes_transferred
);

// Internal:
//   • 64-entry 8-bit FIFO for input buffering
//   • Byte assembly: 4×8-bit → 1×32-bit (LSB-first)
//   • Read pointer for tracking 8-to-32-bit packing
//   • Backpressure via handshake signals
endmodule
```

**Key Features:**
- ✅ LSB-first packing: bytes[3:0] → word[31:0]
- ✅ FIFO-based buffering (64 entries configurable)
- ✅ Handshake protocol (valid/ready backpressure)
- ✅ Byte counter for metadata length tracking
- ✅ Simple state machine (IDLE → ASSEMBLE → OUTPUT)

#### 2. Metadata Decoder (`verilog/meta/meta_decode.sv`)

BRAM cache for sparse metadata with validation and performance monitoring:

```verilog
module meta_decode #(
    parameter CACHE_SIZE = 256,   // 256-entry cache
    parameter ENABLE_PERF = 1     // Enable counters
) (
    input clk, rst_n,
    
    // DMA Input (metadata words from dma_lite)
    input [31:0] metadata_word,
    input metadata_valid,
    output metadata_ready,
    
    // Scheduler Read Port (dual-port BRAM)
    input [7:0] sched_addr,
    output [31:0] sched_data,
    output sched_valid,
    
    // Status & Control
    output [3:0] error_flags,           // Error vector
    output [31:0] cache_hit_count,      // Performance counter
    output [31:0] cache_miss_count,     // Performance counter
    output [31:0] decode_cycle_count    // Performance counter
);

// Internal:
//   • Dual-port BRAM (256×32-bit):
//     - Write port: DMA input
//     - Read port: Scheduler queries
//   • Metadata types: ROW_PTR(0), COL_IDX(1), BLOCK_HDR(2)
//   • Validation FSM: IDLE → LATCH → VALIDATE → DECODE → CACHE_WR → DONE
//   • Performance counters: cache hits, misses, cycle count
endmodule
```

**FSM State Machine:**

```
IDLE
  ├─ metadata_valid=1 → LATCH
  └─ else → IDLE

LATCH (capture incoming metadata word)
  ├─ → VALIDATE

VALIDATE (check metadata type and format)
  ├─ Valid type (0-2) → DECODE
  └─ Invalid → ERROR (set error_flags[0]=1)

DECODE (parse field content)
  ├─ type==ROW_PTR → extract row_start
  ├─ type==COL_IDX → extract col_start
  └─ type==BLOCK_HDR → extract block_count

CACHE_WR (write to BRAM)
  ├─ → DONE

DONE (ready for next word)
  └─ → IDLE
```

**Performance Counters:**
- `cache_hit_count`: Incremented when scheduler read hits cached block metadata
- `cache_miss_count`: Incremented when scheduler read misses (refetch required)
- `decode_cycle_count`: Total cycles spent in VALIDATE+DECODE+CACHE_WR stages

#### 3. BSR Scheduler (`verilog/top/bsr_scheduler.sv` - skeleton)

Generates block addresses and workload sequences from metadata cache:

```
SCHEDULER OPERATION:

INPUT: Row pointers [r0, r1, r2, ...], Column indices [c0, c1, ...]
OUTPUT: Block addresses for systolic array

For each output row r:
  For each non-zero block at (r, c):
    1. Read row_ptr[r] → start block index
    2. Read col_idx[start+k] → column index
    3. Emit block address: (r, c, block_idx)
    4. Wait for systolic array to consume
```

#### 4. Sparse Systolic Array (`verilog/systolic/systolic_array_sparse.sv` - skeleton)

2×2 PE array optimized for sparse 8×8 blocks:

```
PE ARRAY (2×2):
  
  ┌──────────────────┐
  │ PE[0][0] PE[0][1]│
  │ PE[1][0] PE[1][1]│
  └──────────────────┘
  
  • Each PE: 1×1 systolic with 8×8 local accumulator
  • Block compute: 8×8 × 8×8 GEMM per block
  • Pipelining: Load block → Compute 64 cycles → Store result
  • Total cycles/block: ~70 cycles (10 load + 50 compute + 10 drain)
```

### Latency Analysis

| Stage | Cycles | Notes |
|-------|--------|-------|
| UART RX (32 bytes metadata) | 280 | 115.2k baud, 8N1 |
| DMA assembly (8 bytes → 1 word) | 4 | Pipelined |
| Metadata decode (BRAM write) | 10 | FSM latency |
| Scheduler lookup (BRAM read) | 2 | Dual-port BRAM |
| Block compute (8×8 × 8×8) | 70 | 10+50+10 cycles |
| **Total per block** | **~360** | Sequential pipeline |

**Throughput**: 1 block every 70 compute cycles (systolic limited)  
**Bandwidth**: 32 bytes metadata per block → 45% overhead

### Performance Targets

**For sparse CNN layer (10% density):**

```
Dense systolic: 70 cycles/block → 100 blocks = 7000 cycles

Sparse pipeline:
  • Skip 90% of blocks
  • Metadata decode cache hits: 90% (cache the row pointers)
  • Effective compute: 10 blocks × 70 = 700 cycles
  • Metadata overhead: 10 blocks × 15 cycles = 150 cycles
  • UART bottleneck: 320 cycles (280 RX + 40 header)
  
  Total: ~700 cycles (COMPUTED) + 320 cycles (I/O) = 1020 cycles
  
  Speedup: 7000 / 1020 = 6.9×
```

### CSR Register Map (Phase 5)

```
// Existing (Phases 1-4)
0x00: CONTROL       // Start, reset, mode (dense vs sparse)
0x04: STATUS        // Ready, busy, error flags  
0x08: M_DIM         // Matrix M dimension
0x0C: N_DIM         // Matrix N dimension
0x10: K_DIM         // Matrix K dimension

// New (Phase 5 - Sparse)
0x50: SPARSE_MODE   // Enable sparse acceleration
0x54: METADATA_ADDR // BRAM address offset for metadata
0x58: BLOCK_COUNT   // Total non-zero blocks
0x5C: PERF_CACHE_HIT    // Performance: cache hits
0x60: PERF_CACHE_MISS   // Performance: cache misses
0x64: PERF_CYCLES       // Performance: total cycles
```

### Integration with Dense Path

The sparse path coexists with dense dense path:

```
Control Logic:
  IF sparse_mode == 1:
    → Route UART → DMA → meta_decode → scheduler → sparse_systolic
  ELSE:
    → Route UART → wgt_buffer → systolic_array (dense) [Phase 1-3]
```

### Testbench Coverage (Phase 5)

| Testbench | Module | Coverage |
|-----------|--------|----------|
| `tb_dma_lite.sv` | dma_lite | Byte assembly (100%), backpressure (100%) |
| `tb_meta_decode.sv` | meta_decode | BRAM write/read (100%), error injection (100%), perf counters (100%) |
| `tb_bsr_scheduler.sv` (TODO) | bsr_scheduler | Row/col traversal, block address generation |
| `tb_systolic_sparse.sv` (TODO) | systolic_sparse | Block compute, pipelining, accumulation |
| `tb_integration_sparse.sv` (TODO) | top_sparse | Full DMA→meta→scheduler→systolic pipeline |

## System Block Diagram (Dense + Sparse Paths)

```
                    ┌─────────────────┐
                    │   Host System   │
                    │  (UART/CSR Cmds)│
                    └────────┬────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │    UART Receiver (8N1, 115.2k)     │
        └────────┬────────────────────────────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼ (sparse)    ▼ (dense)
    ┌──────────────┐  ┌─────────────────┐
    │  DMA Lite    │  │  Weight Buffer  │
    │  (8→32 bit)  │  │  (WRAM, INT8)   │
    └──────┬───────┘  └────────┬────────┘
           │                   │
           ▼                   │
    ┌──────────────────┐       │
    │  Meta Decoder    │       │
    │  (BRAM Cache)    │       │
    └──────┬───────────┘       │
           │                   │
           ▼                   ▼
    ┌──────────────┐  ┌──────────────────┐
    │ Scheduler    │  │ Activation Buff  │
    │ (Workload)   │  │ (ARAM, INT8)     │
    └──────┬───────┘  └────────┬─────────┘
           │                   │
           └────────┬──────────┘
                    ▼
           ┌─────────────────────┐
           │  Systolic Array     │
           │  (2×2 PEs)          │
           │  INT32 Accum        │
           └──────────┬──────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Output Pipeline      │
           │ (Clamp/Shift → INT8) │
           └──────────┬───────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │  Output Buffer (BRAM)  │
          │  Result Storage        │
          └────────────┬───────────┘
                       │
                       ▼
                    UART TX
                       │
                       ▼
                   Host System
```

## Interfaces

- **CSR**: start, dims (M,N,K), strides, tile sizes, scale/shift, sparse_mode flag
- **UART**: 8N1 @ 115.2 kbps; metadata + data stream (sparse) or weights + activations (dense)
- **Block BRAM**: 4MB capacity for sparse block data (32-bit words, 1M entries)
- **Metadata BRAM Cache**: 256 entries for row/col index caching

## Core Components

### 1. Systolic Array
- **Configuration**: N×M Processing Elements (PEs)
- **Dataflow**: TRUE Row-Stationary (weights loaded once and stored stationary, activations stream)
- **Precision**: INT8 inputs, INT32 internal accumulators
- **Throughput**: N×M MAC operations per cycle
- **Weight Reuse**: Maximum - each weight loaded once per K-tile, reused for all M×N computations

### 2. Processing Element (PE)
```verilog
module pe (
    input clk, rst_n,
    input [7:0] a_in,           // Activation (flows horizontally)
    input [7:0] b_in,           // Weight (loaded once)
    input load_weight,          // Control to load weight
    input en, clr,
    output [7:0] a_out,         // Forward activation to next PE
    output [31:0] acc           // Local partial sum accumulator
);
    reg [7:0] weight_reg;       // STATIONARY weight storage
    // Weight stays put, activation flows through
endmodule
```

**PE Operation (Row-Stationary):**
1. **Weight Loading Phase**: Captures `b_in` into `weight_reg` when `load_weight` is high
2. **Compute Phase**: Performs MAC using stationary weight: `acc += a_in * weight_reg`
3. **Activation Flow**: Forwards `a_in` to next PE horizontally (a_out)
4. **NO Weight Forwarding**: Weight stays in register (not passed to neighbors)

### 3. Buffer Architecture

#### Weight Buffer (`wgt_buffer.v`)
- **Capacity**: Configurable depth for weight storage
- **Width**: 8-bit INT8 weights
- **Access Pattern**: Sequential read during weight loading
- **Interface**: Simple FIFO-style with read enable

#### Activation Buffer (`act_buffer.v`)
- **Capacity**: Tile-sized activation storage
- **Width**: 8-bit INT8 activations
- **Access Pattern**: Simultaneous broadcast to PE array
- **Interface**: Dual-port for load/compute overlap

### 4. Control and Status Registers (CSR)

The CSR module (`csr.v`) provides software-visible configuration:

```verilog
// Register Map
0x00: CONTROL       // Start, reset, mode control
0x04: STATUS        // Ready, busy, error flags  
0x08: M_DIM         // Matrix M dimension
0x0C: N_DIM         // Matrix N dimension
0x10: K_DIM         // Matrix K dimension
0x14: TILE_M        // M-dimension tile size
0x18: TILE_N        // N-dimension tile size
0x1C: SCALE_SHIFT   // Post-processing parameters
```

**Control Flow:**
1. Host writes matrix dimensions and tile sizes
2. Host writes scale/shift parameters for quantization
3. Host sets START bit in CONTROL register
4. Hardware sets BUSY flag and begins computation
5. Hardware clears BUSY and sets READY when complete

### 5. UART Communication Interface

#### Protocol Stack
```
┌─────────────────────────────────────┐
│           Host Software             │
├─────────────────────────────────────┤
│         UART Driver Layer           │
├─────────────────────────────────────┤
│      Hardware UART (8N1)           │
├─────────────────────────────────────┤
│         Physical Layer              │
└─────────────────────────────────────┘
```

#### Packet Format
```
[HEADER|LENGTH|PAYLOAD|CRC16]
 1 byte 1 byte N bytes 2 bytes
```

- **HEADER**: Command type (0x01=Write, 0x02=Read, 0x03=Data)
- **LENGTH**: Payload length (0-255 bytes)
- **PAYLOAD**: Command data or matrix values
- **CRC16**: Error detection checksum

## Dataflow Architecture

### TRUE Row-Stationary Dataflow

The ACCEL-v1 implements **TRUE Row-Stationary (RS) dataflow** with stationary weight storage:

```
Phase 1: WEIGHT LOADING (weights broadcast to columns, stored in PEs)
        
        W_col0   W_col1   W_col2   ← Weights broadcast from top
           ↓        ↓        ↓
        ┌─────┐  ┌─────┐  ┌─────┐
Row 0:  │ W00 │  │ W01 │  │ W02 │  ← ALL PEs in column receive same weight
        └─────┘  └─────┘  └─────┘
        ┌─────┐  ┌─────┐  ┌─────┐
Row 1:  │ W00 │  │ W01 │  │ W02 │  ← Same weights (broadcast)
        └─────┘  └─────┘  └─────┘

Phase 2: ACTIVATION STREAMING (weights stay put, activations flow)

        Weights STAY STATIONARY in PEs
        ┌─────┐ ┌─────┐ ┌─────┐      
A_row0 →│ W00 │→│ W01 │→│ W02 │  ← A_row0 flows right
        └─────┘ └─────┘ └─────┘
A_row1 →│ W00 │→│ W01 │→│ W02 │  ← A_row1 flows right  
        └─────┘ └─────┘ └─────┘
           ↓       ↓       ↓
      [Partial Sums accumulate LOCALLY in each PE]
```

**Key Row-Stationary Properties:**
- ✅ Weights loaded ONCE and stored stationary in PE registers
- ✅ Activations stream horizontally through the array
- ✅ NO weight movement during computation (maximum reuse)
- ✅ Partial sums accumulate locally (not forwarded between PEs)
- ✅ Each PE computes one output element of result matrix

**Advantages:**
- **Maximum weight reuse**: Each weight loaded once, used for all M activations
- **Minimal weight bandwidth**: Weights loaded in burst, then stationary
- **Energy efficient**: No weight movement during main compute phase
- **Natural tiling**: Perfect for large matrix multiplication with K-tiling

### Data Movement Patterns

#### 1. Weight Loading Phase (NEW in Row-Stationary)
```verilog
// Load weights into ALL PEs - broadcast to columns
// Scheduler enters S_LOAD_WEIGHT state
for (int k = 0; k < Tk; k++) begin
    load_weight = 1;  // Enable weight capture
    b_in = weight_buffer[k];  // Broadcast to all PEs in same column
    @(posedge clk);  // Weights now stationary in PEs
end
```

#### 2. Activation Streaming Phase
```verilog
for (int k = 0; k < K; k++) begin
    for (int m = 0; m < M; m++) begin
        pe_array[0][m].act_in <= activations[k][m];
        // Activations flow through array
    end
end
```

#### 3. Accumulation Collection
```verilog
for (int n = 0; n < N; n++) begin
    for (int m = 0; m < M; m++) begin
        results[n][m] = pe_array[n][M-1].acc_out;
    end
end
```

## Memory Hierarchy

### 1. Host Memory
- **Capacity**: System RAM (GB scale)
- **Content**: Full model weights and activations
- **Access**: UART transfer to accelerator

### 2. On-Chip Buffers
- **Weight Buffer**: 2KB - 8KB (configurable)
- **Activation Buffer**: 1KB - 4KB (configurable)
- **Output Buffer**: 512B - 2KB (configurable)

### 3. PE Local Storage
- **Weight Registers**: 8-bit × N×M PEs
- **Accumulator Registers**: 32-bit × N×M PEs

## Timing and Performance

### Clock Domains
- **System Clock**: 50-100 MHz target frequency
- **UART Clock**: Derived from system clock
- **PE Array Clock**: Same as system clock (synchronous design)

### Performance Characteristics
- **Peak Throughput**: N×M×f_clk MACs/second
- **Effective Utilization**: 80-95% (depending on tile sizes)
- **Memory Bandwidth**: Limited by UART interface (~1MB/s)

### Pipeline Stages
1. **Weight Load**: 1-2 cycles per weight
2. **Activation Stream**: 1 cycle per activation
3. **Computation**: 1 cycle per MAC operation
4. **Output Collection**: 1-2 cycles per result

## Interface Specifications

### CSR Interface
- **Bus Width**: 32-bit
- **Address Space**: 64 registers (256 bytes)
- **Access Latency**: 1 cycle read, 1 cycle write
- **Endianness**: Little-endian

### UART Interface
- **Baud Rate**: 115200 bps (configurable)
- **Frame Format**: 8N1 (8 data bits, no parity, 1 stop bit)
- **Flow Control**: None (software managed)
- **Buffer Depth**: 16-byte RX FIFO, 16-byte TX FIFO

## Power and Area Estimates

### Area Breakdown (for 4×4 array)
- **PE Array**: ~60% of total area
- **Buffers**: ~25% of total area
- **Control Logic**: ~10% of total area
- **I/O Interface**: ~5% of total area

### Power Breakdown
- **Compute**: ~70% of total power
- **Memory**: ~20% of total power
- **I/O**: ~10% of total power

## Design Considerations

### Scalability
- **Array Size**: Parameterizable N×M configuration
- **Buffer Sizes**: Configurable based on target applications
- **Clock Frequency**: Scalable based on technology node

### Optimization Opportunities
1. **Sparsity Support**: Zero-skipping for sparse weights/activations
2. **Mixed Precision**: 4-bit/8-bit/16-bit support
3. **Advanced Dataflows**: Weight-stationary, output-stationary
4. **Memory Hierarchy**: Multi-level buffer hierarchy

### Verification Strategy
- **Unit Testing**: Individual PE, buffer, CSR verification
- **Integration Testing**: Full array with golden model comparison
- **System Testing**: Host software + hardware co-verification
- **Performance Testing**: Throughput and latency characterization

## Future Enhancements

### Short Term (v1.1)
- Enhanced error detection and recovery
- Improved UART throughput with compression
- Power management modes

### Medium Term (v2.0)
- Support for different quantization schemes
- Multiple array configurations
- Hardware-software co-design optimizations

### Long Term (v3.0)
- Multi-accelerator systems
- Advanced neural network layer support
- Real-time inference capabilities

---

*This architecture document serves as the authoritative reference for the ACCEL-v1 system design. For implementation details, refer to the individual Verilog modules in the `verilog/` directory.*