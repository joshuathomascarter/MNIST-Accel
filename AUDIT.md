# ACCEL-v1 Code Audit

> Internal audit of the ACCEL-v1 codebase. Tracks known issues, design trade-offs, and items pending before FPGA deployment.

---

## Summary

| Category | Status |
|----------|--------|
| RTL Modules (16) | Simulation-verified |
| SV Testbenches (10) | Passing (Verilator) |
| Cocotb Tests (2) | Passing |
| Python ML Stack (33 files) | Functional |
| C++ Host Driver (28 files) | Framework complete, partial stubs |
| Data Pipeline | Complete (FP32 → INT8 → BSR 14×14) |
| FPGA Synthesis | Not yet completed |

---

## Known Issues

### HIGH Priority

| # | Issue | Location | Notes |
|---|-------|----------|-------|
| 1 | Dual-clock wrapper not included | `hw/rtl/top/` | `accel_top_dual_clk.sv` referenced in docs but removed due to compilation issues. Single-clock `accel_top.sv` is verified. |
| 2 | C++ end-to-end test is a stub | `sw/cpp/tests/test_end_to_end.cpp` | Framework is in place; test body not yet implemented. |
| 3 | FPGA synthesis not completed | — | Vivado TCL script exists (`tools/synthesize_vivado.tcl`) but no bitstream generated yet. |

### MEDIUM Priority

| # | Issue | Location | Notes |
|---|-------|----------|-------|
| 4 | `test_integration.py` broken import | `sw/ml_python/tests/test_integration.py` | Imports from nonexistent `host_uart/` directory. |
| 5 | `test_csr_pack.py` partial | `sw/ml_python/tests/test_csr_pack.py` | Missing `host_uart` dependency. |
| 6 | DMA 64-bit width mismatch | `hw/rtl/dma/` | 64-bit AXI data bus requires multi-beat transfers for 14-wide (112-bit) activation vectors. Packing module (`dma_pack_112.sv`) exists but adds latency. |

### LOW Priority

| # | Issue | Location | Notes |
|---|-------|----------|-------|
| 7 | Directory names with spaces | `sw/ml_python/INT8 quantization/`, `sw/ml_python/MNIST CNN/` | Works but can cause issues with shell scripts. |

---

## Design Trade-offs

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Array size | 14×14 | 8×8, 16×16 | Maximizes DSP usage on Z7020 (196/220 DSP48E1s) |
| Block size | 14×14 | 4×4, 8×8 | Matches array dimensions; simplifies control |
| Dataflow | Weight-stationary | Output-stationary | Maximizes weight reuse; pairs well with BSR sparsity |
| Quantization | INT8 per-channel | INT8 per-tensor, INT4 | Best accuracy–compression trade-off for MNIST |
| Sparse format | BSR | CSR, COO | Sequential access pattern; hardware-friendly metadata |
| Clock | Single 200 MHz (sim) | Dual 50/200 MHz | Dual-clock CDC complexity deferred to FPGA phase |

---

## Verification Status

### RTL Testbenches

| Testbench | DUT | Status |
|-----------|-----|--------|
| `pe_tb.sv` | `pe.sv` | PASS |
| `systolic_tb.sv` | `systolic_array_sparse.sv` | PASS |
| `bsr_dma_tb.sv` | `bsr_dma.sv` | PASS |
| `meta_decode_tb.sv` | BSR metadata decode | PASS |
| `output_accumulator_tb.sv` | `output_accumulator.sv` | PASS |
| `perf_tb.sv` | `perf.sv` | PASS |
| `tb_axi_lite_slave_enhanced.sv` | `axi_lite_slave.sv` | PASS |
| `accel_top_tb.sv` | `accel_top.sv` | PASS |
| `accel_top_tb_full.sv` | `accel_top.sv` | PASS |
| `integration_tb.sv` | `accel_top.sv` | PASS |

### Python Golden Models

| Model | Status |
|-------|--------|
| `gemm_bsr_int8.py` — BSR INT8 GEMM | Matches RTL output |
| `gemm_int8.py` — Dense INT8 GEMM | Reference verified |
| `golden_mac8.py` — MAC8 unit model | Bit-exact with RTL |

---

## Roadmap (Post-Audit)

1. Complete Vivado synthesis and generate bitstream for Zynq-7020
2. Validate on PYNQ-Z2 hardware with MNIST inference
3. Implement dual-clock wrapper with proper CDC
4. Complete C++ end-to-end test
5. Fix broken Python test imports (`host_uart` dependency)
6. Power measurement on FPGA vs. estimation
