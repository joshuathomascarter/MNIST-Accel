# Test Results

## Test Inventory

### SystemVerilog Testbenches (`hw/sim/sv/`)

| Testbench | DUT | Description |
|-----------|-----|-------------|
| `pe_tb.sv` | `pe.sv` | PE weight loading, MAC accumulation, bypass |
| `systolic_tb.sv` | `systolic_array_sparse.sv` | 14x14 array weight load, activation streaming, output collection |
| `bsr_dma_tb.sv` | `bsr_dma.sv` | BSR metadata + weight DMA transfers |
| `meta_decode_tb.sv` | `meta_decode.sv` | BSR metadata BRAM read sequencing |
| `output_accumulator_tb.sv` | `output_accumulator.sv` | INT32 accumulation, requantization, DMA readout |
| `perf_tb.sv` | `perf.sv` | Performance counter increment, latch, clear |
| `tb_axi_lite_slave_enhanced.sv` | `axi_lite_slave.sv` | AXI4-Lite protocol, simultaneous R/W |
| `accel_top_tb.sv` | `accel_top.sv` | Basic top-level CSR + DMA integration |
| `accel_top_tb_full.sv` | `accel_top.sv` | Full inference flow: load weights, stream activations, read output |
| `integration_tb.sv` | `accel_top.sv` | Multi-layer sequential inference |

### Cocotb Tests (`hw/sim/cocotb/`)

| Test File | Description |
|-----------|-------------|
| `test_accel_top.py` | Python-driven top-level integration tests |
| `cocotb_axi_master_test.py` | AXI master protocol verification |

### C++ Tests (`sw/cpp/tests/`)

| Test File | Status | Description |
|-----------|--------|-------------|
| `test_bsr_encoder.cpp` | Implemented | BSR pack/unpack round-trip |
| `test_buffer_manager.cpp` | Implemented | Memory allocation, DMA buffer management |
| `test_dma.cpp` | Implemented | DMA transfer simulation |
| `test_tiling.cpp` | Implemented | Tile dimension calculation |
| `test_end_to_end.cpp` | Stub | End-to-end inference (not yet implemented) |

### Python Tests (`sw/ml_python/tests/`)

| Test File | Status | Description |
|-----------|--------|-------------|
| `test_golden_models.py` | Implemented | BSR INT8 GEMM correctness (pytest) |
| `test_exporters.py` | Implemented | BSR export format verification |
| `test_mac.py` | Implemented | MAC8 golden model vs RTL comparison |
| `test_edges.py` | Implemented | Edge case testing (zeros, saturation) |
| `test_csr_pack.py` | Partial | CSR pack/unpack (imports missing `host_uart`) |
| `test_integration.py` | Broken | Imports from nonexistent `host_uart/` directory |
| `post_training_quant_tests.py` | Implemented | Post-training quantization accuracy |

## Running Tests

### SystemVerilog (Verilator)

```bash
cd hw/sim
make          # Runs all SV testbenches via Verilator
```

Or individual testbenches:

```bash
cd hw/sim
make pe_tb
make systolic_tb
make accel_top_tb
```

### Cocotb

```bash
cd hw/sim/cocotb
make -f Makefile.accel_top
```

### C++

```bash
cd sw/cpp/build
cmake ..
make
ctest --output-on-failure
```

### Python

```bash
cd sw/ml_python
python -m pytest tests/ -v
```

## Known Issues

1. **`test_integration.py`** imports from `host_uart/` which does not exist. Needs migration to `host_axi/` imports.
2. **`test_csr_pack.py`** has the same `host_uart` import dependency.
3. **C++ `test_end_to_end.cpp`** is a stub that returns success without testing.
4. **RTL systolic output** currently reads as zero in cocotb simulation (scheduler timing bugs under investigation, see AUDIT.md R7-R9).