# ACCEL-v1 System Architecture Diagram

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
  - [Host Computer (Python Software Stack)](#host-computer-python-software-stack)
  - [Communication Interface](#communication-interface)
  - [FPGA Hardware (Cyclone V)](#fpga-hardware-cyclone-v)
- [Complete System Block Diagram](#complete-system-block-diagram)
- [Data Flow Summary](#data-flow-summary)
- [Performance Characteristics](#performance-characteristics)
- [Key Design Features](#key-design-features)

## Overview

The ACCEL-v1 accelerator follows a host-FPGA co-design architecture with the following key components:

## System Components

### Host Computer (Python Software Stack)
- **Application Layer**: Neural network training/inference and GEMM operations
- **ML Framework Integration**: PyTorch/TensorFlow compatibility for QAT/PTQ workflows
- **HostRSTiler Class**: Matrix tiling, GEMM execution, quantization, and error handling
- **UART Driver**: Packet framing, stream parsing, CRC error detection, and flow control

### Communication Interface
- **USB-to-UART Converter**: FT232 or similar for 115.2k bps serial communication
- **Protocol**: Custom packet format with SYNC+LEN+CMD+PAYLOAD+CRC structure

### FPGA Hardware (Cyclone V)
- **UART Interface**: Serial conversion, packet parsing, and command dispatch
- **Control & Status Registers (CSR)**: Configuration and monitoring interface
- **Memory Subsystem**: Double-buffered activation/weight buffers and result storage
- **Systolic Array Core**: Row-stationary dataflow with INT8×INT8→INT32 MAC units
- **Control Unit**: Scheduler, address generation, state machine, and interrupt handling

---

## Complete System Block Diagram

```text
ACCEL-v1 Complete System Block Diagram
======================================

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    HOST COMPUTER                                        │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                              Python Software Stack                                  │ │
│ │                                                                                     │ │
│ │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │ │
│ │  │   Application   │  │  ML Framework   │  │    Training     │  │   Inference     │ │ │
│ │  │   (MNIST CNN)   │  │  (PyTorch/TF)   │  │   (QAT/PTQ)     │  │  (Production)   │ │ │
│ │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘ │ │
│ │           │                     │                     │                     │       │ │
│ │           └─────────────────────┼─────────────────────┼─────────────────────┘       │ │
│ │                                 ▼                     ▼                             │ │
│ │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │ │
│ │  │                        HostRSTiler Class                                        │ │ │
│ │  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │ │ │
│ │  │ │  Matrix Tiling  │ │  GEMM Executor  │ │ Quantization    │ │ Error Handling  │ │ │ │
│ │  │ │  • Tile Size    │ │  • Triple Loop  │ │ • INT8 Scales   │ │ • Timeout       │ │ │ │
│ │  │ │  • Memory Mgmt  │ │  • Accumulation │ │ • Range Clamp   │ │ • CRC Checks    │ │ │ │
│ │  │ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ │ │ │
│ │  └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│ │                                         │                                           │ │
│ │                                         ▼                                           │ │
│ │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │ │
│ │  │                         UART Driver                                             │ │ │
│ │  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │ │ │
│ │  │ │ Packet Framing  │ │ Stream Parser   │ │    CRC-8        │ │  Flow Control   │ │ │ │
│ │  │ │ • SYNC Pattern  │ │ • State Machine │ │ • Polynomial    │ │ • Backpressure  │ │ │ │
│ │  │ │ • Length Field  │ │ • Buffer Mgmt   │ │ • Error Detect  │ │ • Timeout       │ │ │ │
│ │  │ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ │ │ │
│ │  └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼ USB/UART 115.2k bps
                                    ┌─────────────────┐
                                    │   USB-to-UART   │
                                    │    Converter     │
                                    │   (FT232, etc)   │
                                    └─────────────────┘
                                              │
                                              ▼ Serial RX/TX
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  FPGA HARDWARE                                          │
│                                   (Cyclone V)                                           │
│                                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                              UART Interface                                         │ │
│ │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │ │
│ │ │    UART RX      │ │    UART TX      │ │ Command Decoder │ │ Response Gen    │     │ │
│ │ │ • Bit Sampling  │ │ • Bit Timing    │ │ • Addr Decode   │ │ • Status Read   │     │ │
│ │ │ • Frame Detect  │ │ • Stop Bits     │ │ • Data Route    │ │ • Result Mux    │     │ │
│ │ │ • Parity Check  │ │ • Flow Control  │ │ • CRC Verify    │ │ • Error Report  │     │ │
│ │ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘     │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                         │
│                                              ▼                                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                        Control & Status Registers (CSR)                             │ │
│ │                                                                                     │ │
│ │  Address Map:          │  Register Function:         │   Bit Fields:               │ │
│ │  0x00: CTRL           │  • Start/Stop Control       │   [0]: START                │ │
│ │  0x04: DIMS_M         │  • Matrix Dimensions        │   [1]: ABORT                │ │
│ │  0x08: DIMS_N         │  • Tile Configuration       │   [2]: IRQ_EN               │ │
│ │  0x0C: DIMS_K         │  • Scale Factors            │   STATUS[0]: READY          │ │
│ │  0x10: TILES_Tm       │  • Buffer Control           │   STATUS[1]: BUSY           │ │
│ │  0x14: TILES_Tn       │  • Status Monitoring        │   STATUS[2]: DONE           │ │
│ │  0x18: TILES_Tk       │                             │   STATUS[3]: ERROR          │ │
│ │  0x1C: INDEX_m        │                             │                             │ │
│ │  0x20: INDEX_n        │                             │                             │ │
│ │  0x24: INDEX_k        │                             │                             │ │
│ │  0x28: BUFF           │                             │                             │ │
│ │  0x2C: SCALE_Sa       │                             │                             │ │
│ │  0x30: SCALE_Sw       │                             │                             │ │
│ │  0x3C: STATUS         │                             │                             │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                         │
│                                              ▼                                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                            Memory Subsystem                                         │ │
│ │                                                                                     │ │
│ │ ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────────────┐ │ │
│ │ │   Activation Buffer │ │   Weight Buffer     │ │      Result Buffer              │ │ │
│ │ │                     │ │                     │ │                                 │ │ │
│ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────────────────┐ │ │ │
│ │ │ │  Bank 0 (Ping)  │ │ │ │  Bank 0 (Ping)  │ │ │ │       C Matrix              │ │ │ │
│ │ │ │  TM×TK×INT8     │ │ │ │  TK×TN×INT8     │ │ │ │       TM×TN×INT32           │ │ │ │
│ │ │ │  128×128×1B     │ │ │ │  128×128×1B     │ │ │ │       128×128×4B            │ │ │ │
│ │ │ │  = 16KB         │ │ │ │  = 16KB         │ │ │ │       = 64KB                │ │ │ │
│ │ │ └─────────────────┘ │ │ └─────────────────┘ │ │ └─────────────────────────────┘ │ │ │
│ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │                                 │ │ │
│ │ │ │  Bank 1 (Pong)  │ │ │ │  Bank 1 (Pong)  │ │ │  Address Mapping:              │ │ │
│ │ │ │  TM×TK×INT8     │ │ │ │  TK×TN×INT8     │ │ │  0x1000-0x1FFF: Act Buffer     │ │ │
│ │ │ │  128×128×1B     │ │ │ │  128×128×1B     │ │ │  0x2000-0x2FFF: Wgt Buffer     │ │ │
│ │ │ │  = 16KB         │ │ │ │  = 16KB         │ │ │  0x3000-0x3FFF: Result Buffer  │ │ │
│ │ │ └─────────────────┘ │ │ └─────────────────┘ │ │                                 │ │ │
│ │ └─────────────────────┘ └─────────────────────┘ └─────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                     │                        │                        │               │
│                     ▼                        ▼                        ▲               │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                          Systolic Array Core                                        │ │
│ │                                                                                     │ │
│ │    A Stream →                                               → Result Accumulation  │ │
│ │                 ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                │ │
│ │    A[0,k] ────▶ │ PE  │─▶│ PE  │─▶│ PE  │─▶│ PE  │ ────▶ (discard)                │ │
│ │                 │ 0,0 │  │ 0,1 │  │ 0,2 │  │ 0,3 │                                │ │
│ │                 └─────┘  └─────┘  └─────┘  └─────┘                                │ │
│ │                    │        │        │        │                                   │ │
│ │                    ▼        ▼        ▼        ▼    B Stream                       │ │
│ │                 ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐     ↓                         │ │
│ │    A[1,k] ────▶ │ PE  │─▶│ PE  │─▶│ PE  │─▶│ PE  │ ────▶ (discard)                │ │
│ │                 │ 1,0 │  │ 1,1 │  │ 1,2 │  │ 1,3 │                                │ │
│ │                 └─────┘  └─────┘  └─────┘  └─────┘                                │ │
│ │                    │        │        │        │                                   │ │
│ │                    ▼        ▼        ▼        ▼                                   │ │
│ │                 ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                │ │
│ │    A[2,k] ────▶ │ PE  │─▶│ PE  │─▶│ PE  │─▶│ PE  │ ────▶ (discard)                │ │
│ │                 │ 2,0 │  │ 2,1 │  │ 2,2 │  │ 2,3 │                                │ │
│ │                 └─────┘  └─────┘  └─────┘  └─────┘                                │ │
│ │                    │        │        │        │                                   │ │
│ │                    ▼        ▼        ▼        ▼                                   │ │
│ │                 ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                │ │
│ │    A[3,k] ────▶ │ PE  │─▶│ PE  │─▶│ PE  │─▶│ PE  │ ────▶ (discard)                │ │
│ │                 │ 3,0 │  │ 3,1 │  │ 3,2 │  │ 3,3 │                                │ │
│ │                 └─────┘  └─────┘  └─────┘  └─────┘                                │ │
│ │                    │        │        │        │                                   │ │
│ │                    ▼        ▼        ▼        ▼                                   │ │
│ │                (discard) (discard)(discard)(discard)                              │ │
│ │                                                                                   │ │
│ │                B[k,0]   B[k,1]   B[k,2]   B[k,3]                                 │ │
│ │                  ▲         ▲         ▲         ▲                                  │ │
│ │                                                                                   │ │
│ │  Each PE: acc[i,j] += A[i,k] × B[k,j]  (INT8 × INT8 → INT32)                   │ │
│ │  After K cycles: acc[i,j] = Σ(k=0..K-1) A[i,k] × B[k,j] = C[i,j]               │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                         │
│                                              ▼                                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                            Control Unit                                             │ │
│ │                                                                                     │ │
│ │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │ │
│ │ │   Scheduler     │ │  Address Gen    │ │  State Machine  │ │ Interrupt Ctrl  │     │ │
│ │ │ • Tile Loops    │ │ • Buffer Index  │ │ • IDLE/CONFIG   │ │ • IRQ Generate  │     │ │
│ │ │ • Timing Ctrl   │ │ • Increment     │ │ • LOAD/COMPUTE  │ │ • Status Update │     │ │
│ │ │ • Sync Logic    │ │ • Wrap Around   │ │ • STORE/DONE    │ │ • Error Handle  │     │ │
│ │ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘     │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘


```

## Data Flow Summary

### 1. Configuration Phase
**Host → UART → CSR**: Matrix dimensions, tile sizes, and quantization scales

### 2. Data Upload Phase
**Host → UART → Buffers**: Weight and activation matrices (INT8 format)

### 3. Computation Phase
**Buffers → Systolic Array → Result Buffer**: GEMM execution with INT32 accumulation

### 4. Result Download Phase
**Result Buffer → UART → Host**: Output matrix (INT32 or requantized INT8)

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Clock Frequency** | 100 MHz | 10ns period target |
| **Array Size** | 4×4 = 16 PEs | Configurable architecture |
| **Peak Throughput** | 1.6 GOPS | 16 PEs × 100 MHz |
| **Memory Bandwidth** | 12.8 GB/s | 128-bit × 100 MHz |
| **UART Bandwidth** | 115.2 kbps | Bottleneck for small tiles |
| **Tile Processing** | Overlapped | Compute with data transfer |
| **Power Consumption** | <1W | Estimated for Cyclone V |

## Key Design Features

**Row-Stationary Dataflow** - Optimized weight reuse pattern  
**Double-Buffered Memory** - Ping-pong operation for continuous processing  
**INT8 Quantization** - Memory and compute efficiency  
**Tiling Support** - Large matrix capability beyond on-chip memory  
**UART Control Interface** - Easy host integration and debugging  
**Error Detection/Recovery** - CRC validation and timeout handling  
**Scalable Architecture** - Configurable array size for different FPGA targets

## Architecture Benefits

### Memory Efficiency
- **Double Buffering**: Eliminates memory access bottlenecks
- **Quantization**: 4× memory reduction vs FP32
- **Tiling**: Supports matrices larger than on-chip capacity

### Compute Optimization
- **Row-Stationary**: Maximizes weight reuse, minimizes data movement
- **Pipelined PEs**: High-frequency operation with minimal latency
- **INT8 MAC Units**: Optimized for CNN inference workloads

### System Integration
- **UART Interface**: Simple, reliable, and debuggable communication
- **CSR Programming**: Flexible configuration without firmware updates
- **Error Handling**: Robust operation with comprehensive status reporting
```