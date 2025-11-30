# ACCEL-v1 File Dependency Flow Chart

## Overview
This document maps all file-to-file connections across Verilog RTL and Python software layers.

---

## VERILOG RTL FILES (26 total)

### 1. TOP-LEVEL HIERARCHY

```
accel_top.sv (Main top-level module with CSR, DMA, UART, and systolic array)
├── accel_top_dual_clk.sv (Dual-clock variant with clock domain crossing)
└── top_sparse.sv (Sparse computation variant)
```

---

## VERILOG DATA FLOW

### Layer 1: Top-Level Orchestration (3 files)

| File | Purpose |
|------|---------|
| `rtl/top/accel_top.sv` | Main accelerator top module integrating all subsystems. |
| `rtl/top/accel_top_dual_clk.sv` | Dual-clock version with clock domain crossing for FPGA. |
| `rtl/top/top_sparse.sv` | Sparse tensor variant supporting block-sparse reorder format. |

---

### Layer 2: Systolic Array (Compute Core) (3 files)

```
accel_top.sv
└── systolic_array.sv (Dense computation)
    └── pe.sv (Processing element with MAC unit)
    
accel_top.sv  
└── systolic_array_sparse.sv (Sparse computation with skip logic)
    └── pe.sv (Same PE used in both)
```

| File | Purpose |
|------|---------|
| `rtl/systolic/systolic_array.sv` | Dense 2D array of processing elements. |
| `rtl/systolic/systolic_array_sparse.sv` | Sparse variant with metadata skipping. |
| `rtl/systolic/pe.sv` | Single processing element with multiply-accumulate. |

---

### Layer 3: Computation & MAC (1 file)

| File | Purpose |
|------|---------|
| `rtl/mac/mac8.sv` | 8-bit integer multiply-accumulate unit. |

---

### Layer 4: Data Path Buffers (2 files)

```
accel_top.sv
├── act_buffer.sv (Activation/input buffer)
└── wgt_buffer.sv (Weight buffer)
```

| File | Purpose |
|------|---------|
| `rtl/buffer/act_buffer.sv` | Stores activation data for row-stationary dataflow. |
| `rtl/buffer/wgt_buffer.sv` | Stores weight data, provides parallel reads to PE rows. |

---

### Layer 5: DMA & Data Movement (4 files)

```
accel_top.sv
├── bsr_dma.sv (Block-sparse reorder DMA engine)
│   ├── axi_dma_master.sv (AXI master interface)
│   └── dma_lite.sv (Lightweight DMA for UART path)
│
└── axi_dma_bridge.sv (Bridges UART/CSR commands to AXI DMA)
```

| File | Purpose |
|------|---------|
| `rtl/dma/bsr_dma.sv` | Block-sparse reorder DMA with metadata unpacking. |
| `rtl/dma/axi_dma_master.sv` | AXI4 master interface for high-speed data transfers. |
| `rtl/dma/dma_lite.sv` | Lightweight DMA for low-bandwidth UART mode. |
| `rtl/host_iface/axi_dma_bridge.sv` | Bridges CSR commands to AXI DMA transactions. |

---

### Layer 6: Control & Sequencing (7 files)

```
accel_top.sv
└── Control Subsystem
    ├── csr.sv (Control/status registers)
    ├── scheduler.sv (Layer scheduler)
    ├── bsr_scheduler.sv (Block-sparse specific scheduler)
    ├── block_reorder_buffer.sv (BSR metadata management)
    ├── multi_layer_buffer.sv (Multi-layer workload tracking)
    ├── pulse_sync.sv (Pulse synchronization across clock domains)
    └── sync_2ff.sv (2-FF synchronizer for CDC)
```

| File | Purpose |
|------|---------|
| `rtl/control/csr.sv` | Memory-mapped CSR for configuration and status. |
| `rtl/control/scheduler.sv` | Base layer scheduler controlling compute sequence. |
| `rtl/control/bsr_scheduler.sv` | Sparse-aware scheduler for block-sparse tensors. |
| `rtl/control/block_reorder_buffer.sv` | Stores BSR block metadata for reorder operations. |
| `rtl/control/multi_layer_buffer.sv` | Manages multi-layer workload queuing. |
| `rtl/control/pulse_sync.sv` | Clock domain crossing for pulse signals. |
| `rtl/control/sync_2ff.sv` | CDC synchronizer using 2-stage flip-flops. |

---

### Layer 7: Host Interface (3 files)

```
accel_top.sv
└── axi_lite_slave_v2.sv (Primary AXI4-Lite slave)
    ├── axi_lite_slave.sv (Legacy version)
    └── Used by csr.sv for register access
```

| File | Purpose |
|------|---------|
| `rtl/host_iface/axi_lite_slave_v2.sv` | AXI4-Lite slave endpoint for CSR reads/writes. |
| `rtl/host_iface/axi_lite_slave.sv` | Legacy AXI4-Lite slave (deprecated). |

---

### Layer 8: Serial Communication (2 files)

```
accel_top.sv (when USE_AXI_DMA=0)
├── uart_rx.sv (UART receiver)
└── uart_tx.sv (UART transmitter)
```

| File | Purpose |
|------|---------|
| `rtl/uart/uart_rx.sv` | UART receiver for low-bandwidth commands. |
| `rtl/uart/uart_tx.sv` | UART transmitter for status/result output. |

---

### Layer 9: Metadata & Monitoring (2 files)

```
accel_top.sv
├── meta_decode.sv (Block-sparse metadata decoder)
└── perf.sv (Performance counter monitor)
```

| File | Purpose |
|------|---------|
| `rtl/meta/meta_decode.sv` | Decodes block-sparse reorder format metadata. |
| `rtl/monitor/perf.sv` | Performance counter tracking (cycles, throughput). |

---

## COMPLETE VERILOG DEPENDENCY TREE

```
accel_top.sv ◄─ TOP LEVEL ENTRY POINT
├─── systolic_array.sv
│    └─── pe.sv ◄─ References mac8.sv
├─── systolic_array_sparse.sv
│    └─── pe.sv ◄─ References mac8.sv
├─── act_buffer.sv
├─── wgt_buffer.sv
├─── bsr_dma.sv
│    ├─── axi_dma_master.sv
│    └─── dma_lite.sv
├─── axi_dma_bridge.sv
├─── csr.sv
│    └─── axi_lite_slave_v2.sv
├─── scheduler.sv
├─── bsr_scheduler.sv
├─── block_reorder_buffer.sv
├─── multi_layer_buffer.sv
├─── meta_decode.sv
├─── pulse_sync.sv
├─── sync_2ff.sv
├─── perf.sv
├─── uart_rx.sv (legacy)
└─── uart_tx.sv (legacy)

Top Variants:
├─── accel_top_dual_clk.sv (uses same modules with CDC)
└─── top_sparse.sv (uses sparse array variant)
```

---

---

## PYTHON FILES (26 total)

### Layer 1: Host Communication & Drivers (5 files)

| File | Purpose |
|------|---------|
| `host/axi_driver.py` | AXI4-Lite CSR reader/writer and DMA transaction controller. |
| `host/axi_master_sim.py` | AXI4 master simulator for software testing. |
| `host_uart/uart_driver.py` | UART serial communication driver. |
| `host_uart/csr_map.py` | CSR address mapping for accelerator registers. |
| `host_uart/run_gemm.py` | Example GEMM runner using UART or AXI interface. |

---

### Layer 2: Model Exporters (4 files)

```
exporters/__init__.py
├─── export_conv.py (CNN convolution export)
├─── export_mlp.py (MLP dense layer export)
└─── export_transformer.py (Transformer export)
```

| File | Purpose |
|------|---------|
| `exporters/__init__.py` | Exporter module initialization. |
| `exporters/export_conv.py` | Converts Conv2D layers to BSR weight format. |
| `exporters/export_mlp.py` | Converts dense MLP layers to BSR format. |
| `exporters/export_transformer.py` | Exports transformer attention/FFN weights as sparse. |

---

### Layer 3: Golden Reference Models (2 files)

| File | Purpose |
|------|---------|
| `golden_models/golden_mac8.py` | Golden reference for 8-bit MAC computation. |
| `golden_models/gemm_int8.py` | Reference GEMM implementation in INT8 precision. |

---

### Layer 4: Sparse Computation (1 file)

| File | Purpose |
|------|---------|
| `golden/gemm_bsr_int8.py` | Golden GEMM with block-sparse reorder format. |

---

### Layer 5: Quantization & Training (3 files)

```
INT8 quantization/
├─── quantize.py (Main INT8 post-training quantization)
└─── quantize old.py (Legacy quantization version)

training/
└─── blocksparse_train.py (Training with sparse weights)
```

| File | Purpose |
|------|---------|
| `INT8 quantization/quantize.py` | Post-training INT8 quantization pipeline. |
| `INT8 quantization/quantize old.py` | Legacy quantization (deprecated). |
| `training/blocksparse_train.py` | Trains models with block-sparse weight sparsity. |

---

### Layer 6: Model Training & Export (2 files)

| File | Purpose |
|------|---------|
| `MNIST CNN/train_mnist.py` | Trains MNIST CNN model with quantization. |
| `training/export_bsr.py` | Exports trained weights to BSR file format. |

---

### Layer 7: Utilities (2 files)

| File | Purpose |
|------|---------|
| `utils/tile_counts.py` | Calculates optimal tiling parameters. |
| `utils/golden_c_tile.py` | Golden C-tile computation reference. |

---

### Layer 8: Testing (7 files)

```
tests/
├─── test_csr_pack.py (CSR packing validation)
├─── test_mac.py (MAC unit functional tests)
├─── test_exporters.py (Weight exporter tests)
├─── test_edges.py (Edge case testing)
├─── test_integration.py (Full integration tests)
├─── test_golden_models.py (Golden model validation)
└─── post_training_quant_tests.py (Quantization tests)
```

| File | Purpose |
|------|---------|
| `tests/test_csr_pack.py` | Validates CSR command packing/unpacking. |
| `tests/test_mac.py` | Unit tests for 8-bit MAC operation. |
| `tests/test_exporters.py` | Tests weight exporter correctness. |
| `tests/test_edges.py` | Edge case and boundary condition testing. |
| `tests/test_integration.py` | End-to-end integration validation. |
| `tests/test_golden_models.py` | Compares golden models to actual compute. |
| `tests/post_training_quant_tests.py` | Validates quantization accuracy. |

---

## COMPLETE PYTHON DEPENDENCY TREE

```
train_mnist.py ◄─ ENTRY POINT (Training)
├─── quantize.py (Post-training quantization)
│    └─── gemm_int8.py (Reference computation)
├─── export_conv.py (Extract Conv layers)
│    └─── blocksparse_train.py (Sparse weights)
│         └─── export_bsr.py (BSR export)
│
run_gemm.py ◄─ ENTRY POINT (Inference)
├─── uart_driver.py (Serial comms)
│    └─── csr_map.py (Register addresses)
│
axi_driver.py (Alternative interface)
├─── axi_master_sim.py (Sim mode)
└─── Used by run_gemm.py
│
TESTING PIPELINE:
│
test_exporters.py
├─── export_conv.py
├─── export_mlp.py
└─── export_transformer.py
│
test_golden_models.py
├─── golden_mac8.py
├─── gemm_int8.py
└─── gemm_bsr_int8.py
│
test_integration.py
├─── test_mac.py
├─── test_csr_pack.py
└─── run_gemm.py
│
UTILITIES:
├─── tile_counts.py (Used by exporters)
└─── golden_c_tile.py (Reference tiling)
```

---

## FILE-TO-FILE FLOW MATRIX

### Verilog Data Flow

| Source | Destination | Connection | Purpose |
|--------|-------------|-----------|---------|
| accel_top.sv | systolic_array.sv | Module instantiation | Compute engine |
| systolic_array.sv | pe.sv | Module instantiation | Processing elements |
| pe.sv | mac8.sv | Module instantiation | Multiply-accumulate |
| accel_top.sv | act_buffer.sv | Module instantiation | Input buffering |
| accel_top.sv | wgt_buffer.sv | Module instantiation | Weight buffering |
| act_buffer.sv | systolic_array.sv | Data bus | Feed activations |
| wgt_buffer.sv | systolic_array.sv | Data bus | Feed weights |
| bsr_dma.sv | act_buffer.sv | DMA write | Load activations |
| bsr_dma.sv | wgt_buffer.sv | DMA write | Load weights |
| csr.sv | bsr_dma.sv | Control signals | Start DMA |
| csr.sv | scheduler.sv | Control signals | Control compute |
| scheduler.sv | systolic_array.sv | Control bus | Sequencing |
| meta_decode.sv | bsr_dma.sv | Metadata parse | BSR format |
| axi_lite_slave_v2.sv | csr.sv | Write/read | CSR access |
| perf.sv | accel_top.sv | Performance data | Monitoring |

### Python Data Flow

| Source | Destination | Connection | Purpose |
|--------|-------------|-----------|---------|
| train_mnist.py | quantize.py | Function call | Quantize model |
| quantize.py | gemm_int8.py | Function call | Verify computation |
| train_mnist.py | export_conv.py | Function call | Export weights |
| export_conv.py | export_bsr.py | Function call | BSR serialization |
| run_gemm.py | uart_driver.py | Function call | Send UART commands |
| uart_driver.py | csr_map.py | Constant import | Register addresses |
| run_gemm.py | axi_driver.py | Function call | AXI interface mode |
| test_exporters.py | export_conv.py | Function call | Test export |
| test_integration.py | run_gemm.py | Function call | E2E validation |
| test_golden_models.py | gemm_int8.py | Function call | Compare results |
| tile_counts.py | export_conv.py | Function call | Optimize tiling |

---

## SUMMARY TABLE

| Layer | Verilog Count | Python Count | Purpose |
|-------|---------------|--------------|---------|
| Top-level | 3 | - | Entry points |
| Compute | 4 | - | Systolic array & MAC |
| Memory | 2 | - | Buffers |
| Data Movement | 4 | 5 | DMA & communication |
| Control | 7 | - | Sequencing & CSR |
| Host Interface | 3 | - | AXI/UART endpoints |
| Serial | 2 | - | UART transceivers |
| Monitoring | 2 | - | Performance tracking |
| **Verilog Subtotal** | **26** | - | - |
| Training & Export | - | 6 | Model preparation |
| Model Reference | - | 3 | Golden models |
| Host Drivers | - | 5 | SW communication |
| Utilities | - | 2 | Helper functions |
| Testing | - | 7 | Validation suite |
| **Python Subtotal** | - | **26** | - |
| **TOTAL** | **26** | **26** | Complete system |

---

## CROSS-LAYER INTEGRATION POINTS

### 1. Weight/Activation Loading Path
```
export_bsr.py (Python) → BSR file
→ run_gemm.py → uart_driver.py → UART → uart_rx.sv (Verilog)
→ csr.sv → bsr_dma.sv → wgt_buffer.sv/act_buffer.sv
```

### 2. Computation Path
```
Weights/Activations in Buffers
→ systolic_array.sv → pe.sv → mac8.sv (8-bit INT8 MAC)
```

### 3. Result Readback Path
```
perf.sv (Performance counters) → axi_lite_slave_v2.sv
→ csr.sv → axi_driver.py (Python) → Results
```

### 4. Quantization to Hardware Path
```
train_mnist.py (Float) → quantize.py (INT8)
→ export_conv.py → export_bsr.py (BSR format)
→ csr_map.py (Python) → uart_driver.py → accel_top.sv
```

---

## ACTIVATION DATA FLOW (Complete Path - Corrected)

### WHERE ACTIVATIONS COME FROM

**Activations are NOT block-sparse** (no BSR format). They flow from **test inputs**, not training outputs.

### ACTUAL Complete Data Path (You're Correct!)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TRAINING & GOLDEN INPUT CAPTURE                           │
└─────────────────────────────────────────────────────────────────────┘

train_mnist.py (Trains MNIST model on GPU/CPU)
    ↓
    Loads MNIST dataset (28×28 grayscale images)
    Trains CNN: Conv1→Conv2→FC1→FC2
    
    At end of training:
    ↓
    Saves FIRST BATCH of test images → python/golden/mnist_inputs.npy
    
    File: accel/python/MNIST CNN/train_mnist.py, line 165:
    np.save("python/golden/mnist_inputs.npy", golden_inputs)


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: QUANTIZATION & ACTIVATION EXTRACTION                       │
└─────────────────────────────────────────────────────────────────────┘

quantize.py reads mnist_inputs.npy:
    File: accel/python/INT8 quantization/quantize.py, line 226:
    imgs = np.load(GOLDEN_INPUTS_PATH)[:num_samples]
    
    ↓
    Runs test images through trained model to get intermediate values
    
    File: quantize.py, function quantize_activations_from_golden():
    • Loads model checkpoint
    • Forward pass on golden inputs
    • Captures layer outputs: conv1_out, conv2_out, fc1_in, fc1_out
    
    ↓
    Quantizes activations to INT8 (per-tensor, NOT per-channel)
    Saves tiles for hardware testing → data/quant_tiles/A.npy


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: HOST SOFTWARE TRANSPORT (run_gemm.py)                      │
└─────────────────────────────────────────────────────────────────────┘

run_gemm.py (Host orchestrator):
    
    File: accel/python/host_uart/run_gemm.py
    
    Loads quantized matrices:
    • Reads "A.npy" (INT8 activation matrix - "A" in GEMM)
    • Reads "B.npy" (INT8 weight matrix from export_bsr.py)
    
    Sends via UART (ACTUAL PATH USED):
    ┌───────────────────────────────────────────────────────────┐
    │ uart_driver.py                                             │
    │ - Formats activation + weight data into UART packets       │
    │ - send_packet(CMD_WRITE, payload)                          │
    │ - Line 299: self.uart.send_packet(CMD_WRITE, a_payload)  │
    └────────┬────────────────────────────────────────────────┘
             ↓
    UART TX over serial (send bytes to FPGA via /dev/ttyUSB0)


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4a: HARDWARE RECEPTION (UART Path)                            │
└─────────────────────────────────────────────────────────────────────┘

On FPGA/Verilator:

    uart_rx.sv (UART receiver - converts serial to 8-bit words)
    ↓
    uart_rx_data[7:0], uart_rx_valid
    
    ↓
    Goes to: accel_top.sv internal wiring
    ↓
    csr.sv (CSR - parses packets, routes commands)
    ↓
    ╔═══════════════════════════════════════════════════════╗
    ║ CHOICE: Is this a WRITE to an address or a weight?   ║
    ║ - If ADDR = 0x1000: Activation buffer write          ║
    ║ - If ADDR = 0x2000: Weight buffer write              ║
    ║ - If ADDR = 0x5X: CSR register                       ║
    ╚═══════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4b: DATA ROUTING IN HARDWARE                                  │
└─────────────────────────────────────────────────────────────────────┘

ACTIVATION BUFFER PATH:

    csr.sv receives write command for address 0x1000
    ↓
    Routes to: axi_dma_bridge.sv (abstracts AXI writes to DMA FIFO)
              OR directly to: bsr_dma.sv 
    
    In accel_top.sv line ~400-450 (interconnect):
    - csr_wen → bsr_dma input
    - csr_wdata → bsr_dma input
    - bsr_dma.sv processes bytes and determines destination:
    
    bsr_dma.sv checks address bits:
    - If address 0x1000-0x1FFF: ACTIVATION buffer write
    - If address 0x2000-0x2FFF: WEIGHT buffer write
    - Extracts byte data from payload
    
    ↓
    block_we, block_waddr, block_wdata signals
    
    ↓
    act_buffer.sv (receives write)
    
    Address mapping:
    act_buffer base = 0x1000 (from uart_driver payload)
    act_waddr = (address - 0x1000) / word_width
    
    ↓
    act_buffer.sv (Dense INT8 activation storage)
    • Stores rows of A matrix for row-stationary dataflow
    • NOT block-sparse (no metadata)
    • Simple BRAM write: act_we, act_waddr, act_wdata


┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: COMPUTE (Activations flow to PE array)                     │
└─────────────────────────────────────────────────────────────────────┘

    act_buffer.sv (stores rows of A matrix)
    ↓ (via systolic_array.sv interface)
    
    systolic_array.sv (row-stationary dataflow):
    • Reads one row at a time from act_buffer
    • Broadcasts horizontally to all PE rows
    • Each PE receives: activation[i], weight[j]
    
    ↓
    pe.sv (Processing Element in each systolic cell):
    • Receives activation[i] (from left, broadcasts down)
    • Receives weight[j] (from top, broadcasts right)
    • Multiplies: temp = activation[i] * weight[j]
    • Accumulates: accumulator += temp
    
    ↓
    mac8.sv (8×8 Multiply-Accumulate in each PE):
    • INT8 multiplication
    • 32-bit accumulator
    • Result flows to next PE or accumulator BRAM
    
    ↓
    Partial sums → output BRAM for result readback
```

### COMPLETE FILE PATH TABLE

| Stage | Python File | Purpose |
|-------|-------------|---------|
| Training | `train_mnist.py` | Trains model, saves test batch to `mnist_inputs.npy` |
| Extraction | `quantize.py` | Loads inputs, forward pass, extracts layer outputs |
| Quantization | `quantize.py` | Converts float activations → INT8 |
| Serialization | `quantize.py` | Saves INT8 tiles to `data/quant_tiles/A.npy` |
| Host Orchestration | `run_gemm.py` | Loads A/B matrices, calls send_tile_data() |
| Transport | `uart_driver.py` | **ONLY PATH USED**: Sends via UART serial packets |
| Address Mapping | `csr_map.py` | Defines 0x1000 as act_buffer base address |

| Stage | Verilog File | Purpose |
|-------|-------------|---------|
| Reception | `uart_rx.sv` | **Receives UART serial bytes** |
| Command Parse | `csr.sv` | Parses packet, extracts address and data |
| DMA Router | `bsr_dma.sv` | Routes writes based on address (0x1000-0x1FFF = activation) |
| **Buffering** | **`act_buffer.sv`** | **Stores dense INT8 activations (NOT sparse)** |
| Distribution | `systolic_array.sv` | Broadcasts rows to PE array |
| Compute | `pe.sv` + `mac8.sv` | Multiplies activation × weight, accumulates |

### CLARIFICATION: Why NOT through AXI path?

**axi_driver.py** and **axi_lite_slave_v2.sv** exist but are NOT used for activation transfers:

✅ **Actually Used:**
- `uart_driver.py` → UART packets → `uart_rx.sv` → `csr.sv` → `bsr_dma.sv` → `act_buffer.sv`

❌ **NOT Used (for activations):**
- `axi_driver.py` (exists for future AXI DMA alternative)
- `axi_lite_slave_v2.sv` (CSR read/write interface, not data path)
- `axi_dma_bridge.sv` (bridges AXI writes to DMA FIFO, but not activated)
- `axi_master_sim.py` (simulator only, not synthesis)

**Why?** The project uses **UART** as the primary data transport because:
1. Simpler to implement in simulation (Verilator)
2. Works on all FPGA boards without external master
3. Sufficient bandwidth for MNIST CNN (28×28 = 784 bytes per image)

---

## KEY DESIGN PATTERNS

1. **Hierarchical Module Instantiation**: Top-level orchestrates all subsystems.
2. **Data Buffering**: Act/Wgt buffers decouple DMA from compute.
3. **DMA-driven Loading**: bsr_dma.sv handles metadata parsing and data movement.
4. **CSR Control Plane**: csr.sv separate from data path.
5. **Clock Domain Crossing**: sync_2ff/pulse_sync for FPGA deployment.
6. **Software/Hardware Split**: Python trains and exports; Verilog executes.
7. **Asymmetric Sparsity**: Weights are block-sparse (BSR), activations are dense (no sparsity).

---


