# Documentation Figures

This folder contains comprehensive illustrative figures used by the ACCEL-v1 documentation.

## Available Diagrams

### `system_architecture.svg`
Complete system architecture showing the host-FPGA co-design:
- Host software stack (Python application, HostRSTiler, UART driver, CSR mapping)
- FPGA hardware components (UART interface, control subsystem, memory buffers, systolic array)
- Data and control flow between components
- Performance and verification infrastructure

### `systolic_array_dataflow.svg`
Detailed 2×2 systolic array dataflow diagram with:
- Row-stationary operation showing PE internals
- Timing example with K=4 cycles of accumulation
- Data movement patterns (activations horizontal, weights vertical)
- Matrix multiplication context and advantages

### `uart_timing_diagram.svg`
Comprehensive UART protocol documentation including:
- Byte-level packet structure (SYNC, LENGTH, COMMAND, PAYLOAD, CRC)
- Bit-level timing for 8N1 framing at 115200 baud
- Example CSR write command breakdown
- Receiver state machine operation

## Usage

Reference these diagrams in documentation with:
```markdown
![System Architecture](figs/system_architecture.svg)
![Systolic Array Dataflow](figs/systolic_array_dataflow.svg)
![UART Protocol](figs/uart_timing_diagram.svg)
```

These SVG files are optimized for both web viewing and print documentation, with scalable vector graphics and clear typography.
