# ACCEL-v1 System Architecture```markdown

# Architecture

## Overview

## Dataflow (v1)

The ACCEL-v1 is an INT8 CNN accelerator built around a systolic array architecture optimized for matrix multiplication operations. This document provides a comprehensive view of the system design, component interactions, and dataflow implementation.- **Weights**: pre-quantized INT8, streamed into `wgt_buffer`.

- **Activations**: im2col tiles as INT8 into `act_buffer`.

## System Block Diagram- **Compute**: systolic array (N x M PEs), int32 accumulators.

- **Post**: clamp/shift → INT8, return via UART.

```

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐## Interfaces

│   Host System   │◄──►│   UART Interface │◄──►│   CSR Control   │- **CSR**: start, dims (M,N,K), strides, tile sizes, scale/shift.

└─────────────────┘    └──────────────────┘    └─────────────────┘- **UART**: 8N1 @ configurable baud; simple framing: [HDR|PAYLOAD|CRC].

                                                         │

                                                         ▼*(Diagram lives in README; final fig export to `docs/figs/` later.)*
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Weight Buffer  │──►│  Systolic Array  │◄──►│ Activation Buf  │
│   (INT8 Wgts)   │    │   (N×M PEs)      │    │  (INT8 Acts)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Output Pipeline │
                       │ (Clamp/Shift)   │
                       └─────────────────┘
```

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