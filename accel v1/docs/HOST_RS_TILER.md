# Host RS Tiler Implementation Guide

## Overview

The Host RS Tiler is a conceptual Python-based tiling system for partitioning matrix multiplication operations into tiles suitable for hardware accelerators. This document describes the intended implementation and protocol.

## System Architecture

The Host RS Tiler consists of several components:

### 1. Tiling Engine (`python/utils/tile_counts.py`)
- Calculates tile dimensions based on matrix sizes and hardware constraints
- Supports configurable tile sizes (TM, TN, TK)
- Implements ceiling division for tile count calculation

### 2. UART Communication Interface (`python/host_uart/uart_driver.py`)
- Conceptual UART communication driver
- Packet-based protocol design for command and data transfer
- Configurable baud rates

### 3. CSR Management (`python/host_uart/csr_map.py`)
- Maps software configuration to hardware control registers
- API for setting matrix dimensions, tile sizes, and control flags
- Address mapping and data formatting

### 4. Matrix Operation Engine (`python/host_uart/run_gemm.py`)
- Orchestrates GEMM operations using the tiled approach
- Manages data upload, computation scheduling, and result retrieval

## Protocol Specification

### UART Packet Format

All communication between the host and ACCEL-v1 uses a structured packet format:

```
| CMD (1 byte) | DATA (4 bytes) |
```

#### Command Types

- `0x00-0x0F`: CSR register access
- `0x10-0x1F`: Activation buffer write (address in lower 4 bits)
- `0x20-0x2F`: Weight buffer write (address in lower 4 bits)
- `0x30-0x3F`: Control commands (start, stop, reset)
- `0x40-0x4F`: Status queries

#### Data Format

- All data is transmitted in little-endian format
- 32-bit values are sent as 4 sequential bytes
- Matrix data is quantized to INT8 format before transmission

### Register Map

| Address | Name | Description |
|---------|------|-------------|
| 0x00 | CTRL | Control register (start/stop/reset) |
| 0x04 | STATUS | Status register (busy/done/error flags) |
| 0x08 | M_DIM | Matrix M dimension |
| 0x0C | N_DIM | Matrix N dimension |
| 0x10 | K_DIM | Matrix K dimension |
| 0x14 | TM_SIZE | Tile M size |
| 0x18 | TN_SIZE | Tile N size |
| 0x1C | TK_SIZE | Tile K size |

## Protocol Specification

### UART Packet Format

Communication between host and ACCEL-v1 uses a 7-byte packet format:

```
| CMD (1 byte) | ADDR_L (1 byte) | ADDR_H (1 byte) | DATA_0-3 (4 bytes) |
```

#### Command Types

- `0x0X`: CSR register access
- `0x1X`: CSR read
- `0x2X`: Activation buffer write
- `0x3X`: Weight buffer write
- `0x5X`: Start computation
- `0x6X`: Abort computation
- `0x7X`: Status query

#### Data Format

- Data transmitted in little-endian format
- 32-bit values sent as 4 sequential bytes
- Matrix data quantized to INT8 format before transmission

### Register Map

| Address | Name | Description |
|---------|------|-------------|
| 0x00 | CTRL | Control register |
| 0x04 | STATUS | Status register |
| 0x08 | M_DIM | Matrix M dimension |
| 0x0C | N_DIM | Matrix N dimension |
| 0x10 | K_DIM | Matrix K dimension |
| 0x14 | TM_SIZE | Tile M size |
| 0x18 | TN_SIZE | Tile N size |
| 0x1C | TK_SIZE | Tile K size |

## Performance Considerations

### Throughput Factors

Performance depends on:

1. Tile sizing relative to array dimensions
2. UART baud rate and protocol overhead
3. Memory bandwidth constraints
4. Computation vs communication overlap

### Theoretical Analysis

For the 2×2 systolic array implementation:
- 4 INT8 MACs per cycle
- Clock-dependent throughput
- Memory bandwidth limits sustained performance

## Usage Examples

### Basic GEMM Operation (Conceptual)

```python
from python.host_uart.run_gemm import run_gemm_operation
from python.host_uart.uart_driver import UARTDriver

# Initialize UART connection
uart = UARTDriver(port='/dev/ttyUSB0', baud=115200)

# Run matrix multiplication C = A × B
result = run_gemm_operation(
    A=input_matrix_A,  # M×K matrix
    B=input_matrix_B,  # K×N matrix
    M=64, N=64, K=64,
    TM=8, TN=8, TK=8,
    uart_driver=uart
)
```

## Error Handling

The system design includes error detection:

### UART Layer
- Framing errors detected by hardware UART receiver
- Optional parity bit checking
- Timeout handling for response packets

### Protocol Layer
- Command validation
- Address range checking
- Optional CRC for data integrity

## Implementation Notes

### Hardware Requirements
- ACCEL-v1 hardware with UART interface
- On-chip memory for buffer storage
- Systolic array dimensions must match software configuration

### Software Requirements
- Python 3.7+ with NumPy
- PySerial for UART communication

### Configuration
- Tile sizes configured based on available memory
- UART baud rate set to 115200
- Consider memory constraints when sizing tiles

## Future Work

Potential enhancements:
- DMA support for higher bandwidth
- Advanced scheduling for multi-tile operations
- Hardware quantization acceleration
- Additional data type support

## References

- [ACCEL-v1 Architecture Overview](ARCHITECTURE.md)
- [Quantization Implementation](QUANTIZATION.md)
- [Verification Results](VERIFICATION.md)