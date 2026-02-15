# ACCEL-v1 — Master Work Plan

> **When every item below is checked off, the project is done.**
>
> Architecture: 14×14 weight-stationary sparse systolic array, INT8×INT8→INT32, BSR format, PYNQ-Z2 (Zynq-7020)
> Model: MNIST CNN — conv1(1→32,3×3) → conv2(32→64,3×3) → maxpool(2) → fc1(9216→128) → fc2(128→10)

---

## Part 0 — Files to Read First

Read these in order. After this you'll understand the full architecture and data flow.

### Architecture & Dataflow (read in order)

| # | File | What You Learn |
|---|------|---------------|
| 1 | [docs/architecture/ROW_STATIONARY_DATAFLOW.md](docs/architecture/ROW_STATIONARY_DATAFLOW.md) | How weights are loaded and activations stream through the PE array |
| 2 | [docs/architecture/SPARSITY_FORMAT.md](docs/architecture/SPARSITY_FORMAT.md) | BSR binary format: row_ptr, col_idx, weight blocks, tile ordering |
| 3 | [docs/DEEP_DIVE.md](docs/DEEP_DIVE.md) | Complete architecture spec: timing, memory bandwidth, power estimates |
| 4 | [docs/critical_path_timing.md](docs/critical_path_timing.md) | Clock domain crossing, critical paths, timing closure |
| 5 | [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) | High-level overview (NOTE: still references UART and 2×2 — stale but gives conceptual foundation) |

### RTL — Read in Bottom-Up Order

| # | File | Lines | What It Does |
|---|------|-------|-------------|
| 6 | [hw/rtl/mac/mac8.sv](hw/rtl/mac/mac8.sv) | 484 | INT8 MAC unit mapping to DSP48E1 |
| 7 | [hw/rtl/systolic/pe.sv](hw/rtl/systolic/pe.sv) | 411 | Weight-stationary processing element wrapping mac8 |
| 8 | [hw/rtl/systolic/systolic_array_sparse.sv](hw/rtl/systolic/systolic_array_sparse.sv) | 426 | 14×14 PE grid with block_valid gating (THIS is what accel_top uses) |
| 9 | [hw/rtl/control/bsr_scheduler.sv](hw/rtl/control/bsr_scheduler.sv) | 462 | 10-state FSM that walks BSR metadata and dispatches blocks 







| 11 | [hw/rtl/dma/bsr_dma.sv](hw/rtl/dma/bsr_dma.sv) | 776 | AXI4 master for BSR weight transfer (header→row_ptr→col_idx→weights) |
| 12 | [hw/rtl/dma/act_dma.sv](hw/rtl/dma/act_dma.sv) | 608 | AXI4 master for activation reads |
| 13 | [hw/rtl/meta/meta_decode.sv](hw/rtl/meta/meta_decode.sv) | 55 | BSR metadata decoder (stub — no cache) |
possibly make a meat decode bram
| File | Lines | What It Does |
|------|-------|-------------|
| [hw/rtl/buffer/act_buffer.sv](hw/rtl/buffer/act_buffer.sv) | 290 | Double-buffered activation SRAM (NOT used by accel_top — inline BRAMs instead) |
| [hw/rtl/buffer/wgt_buffer.sv](hw/rtl/buffer/wgt_buffer.sv) | 243 | Double-buffered weight SRAM (NOT used by accel_top) |
| 14 | [hw/rtl/host_iface/axi_lite_slave.sv](hw/rtl/host_iface/axi_lite_slave.sv) | 308 | AXI4-Lite → CSR bus bridge |
| 15 | [hw/rtl/control/csr.sv](hw/rtl/control/csr.sv) | 527 | Control/status register block (0x00–0xDC) |
| 16 | [hw/rtl/host_iface/axi_dma_bridge.sv](hw/rtl/host_iface/axi_dma_bridge.sv) | 480 | 2:1 AXI4 read arbiter for bsr_dma + act_dma |

| [hw/rtl/monitor/perf.sv](hw/rtl/monitor/perf.sv) | 263 | Performance counters (6 registers) |
| [hw/rtl/buffer/output_accumulator.sv](hw/rtl/buffer/output_accumulator.sv) | 458 | Accumulator with ReLU + quantization (NOT used by accel_top) |



| 17 | [hw/rtl/top/accel_top.sv](hw/rtl/top/accel_top.sv) | 1352 | **Main integration top-level** — read this last, everything connects here |



| [hw/rtl/control/pulse_sync.sv](hw/rtl/control/pulse_sync.sv) | 90 | Toggle-based CDC pulse synchronizer |
| [hw/rtl/control/sync_2ff.sv](hw/rtl/control/sync_2ff.sv) | 190 | 2-FF synchronizer + gray-coded async FIFO |


## Part 1 — RTL Bugs (Fix These First)

### Tier 1 — Critical (blocks correct operation)

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| R1 | [accel_top.sv](hw/rtl/top/accel_top.sv) | ~1095–1110 | **BRAM width packing mismatch.** Inline BRAMs are 112 bits wide (14×8). DMA writes 64-bit data with zero-extension `{{48{1'b0}}, data}`. Only 8 of 14 INT8 values per row are populated. | Add a multi-beat packing FSM that assembles 2× 64-bit DMA beats into one 112-bit BRAM entry. First beat fills bits [63:0], second beat fills bits [111:64]. Track a beat counter per write address. |
| R2 | [bsr_dma.sv](hw/rtl/dma/bsr_dma.sv) | ~720 | **Weight block truncation.** `words_remaining = total_blocks * 8` transfers only 64 bytes per block. 14×14 blocks need 196 bytes (25 beats of 8 bytes). | Change to `words_remaining = total_blocks * 25` (ceil(196/8)). Add packing logic for the partial last beat (4 remaining bytes). |

| R5 | [csr.sv](hw/rtl/control/csr.sv) | ~165 | **Glitchy clock gating.** Uses `assign clk_gated = clk & csr_clk_en` (AND-gate). Other modules use latch-based ICG. A glitch on `csr_clk_en` while `clk` is high creates a false clock edge. | Replace with latch-based ICG: `always @(clk or csr_clk_en) if (!clk) en_latched <= csr_clk_en; assign clk_gated = clk & en_latched;` |
| R6 | [output_accumulator.sv](hw/rtl/buffer/output_accumulator.sv) | ~415 | **Scale factor truncation.** `quantize_relu` uses only `scale[15:0]` of a 32-bit Q16.16 value. The integer part (bits [31:16]) is ignored, so any scale ≥ 1.0 wraps. | Use full 32-bit `scale_factor` in the multiply: `relu_val * $signed({1'b0, scale_factor})`. Adjust the bit-select for the result accordingly. |
| R7 | [accel_top.sv](hw/rtl/top/accel_top.sv) | — | **No write DMA / result output path.** Results only accessible via CSR `result_data` (128 bits = 4 accumulators out of 196). No way to read all 196 PE outputs back to DDR. | Option A: Add a write DMA engine (new module). Option B: Wire all 196 accumulators to CSR with an index register for sequential readback. Option B is simpler for MNIST (small outputs). |
| R8 | [pe.sv](hw/rtl/systolic/pe.sv) | ~350 | **load_weight/en overlap.** Assertion commented out with TODO. If both signals are high simultaneously, MAC accumulates garbage during weight loading. | Fix `bsr_scheduler.sv` to guarantee 1-cycle gap between `load_weight` deassertion and `en` assertion. Then uncomment the assertion. |

### Tier 2 — High (correctness risk, will cause subtle bugs)

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| R9 | [meta_decode.sv](hw/rtl/meta/meta_decode.sv) | entire | **Cache not implemented.** `CACHE_DEPTH=64` parameter exists but no cache logic. Every metadata access hits BRAM with full latency. Comments in accel_top describe "64-entry direct-mapped cache" that doesn't exist. | Either: (A) Implement the direct-mapped cache (tag + valid + data arrays, hit/miss logic), or (B) remove the misleading parameter/comments and accept pass-through latency. For MNIST with small metadata, (B) is fine. |
| R10 | [bsr_scheduler.sv](hw/rtl/control/bsr_scheduler.sv) | ~238 | **Counter overflow risk.** `S_LOAD_WGT` checks `load_cnt == LOAD_CNT_MAX[4:0] + 1`. For BLOCK_SIZE > 16 this silently overflows 5 bits. | Widen `load_cnt` to at least `$clog2(BLOCK_SIZE)+1` bits. Use `load_cnt == BLOCK_SIZE` directly instead of the truncated expression. |
| R11 | [bsr_scheduler.sv](hw/rtl/control/bsr_scheduler.sv) | ~200–250 | **Counter reuse.** `load_cnt` is reused across `S_LOAD_WGT` and `S_WAIT_WGT` states for different purposes. A missed reset during state transition corrupts the count. | Use separate registers for weight-load count and wait count. Or add explicit reset of `load_cnt` in every state entry. |
| R12 | [csr.sv](hw/rtl/control/csr.sv) | ~335–345 | **Potential BSR register reset scoping error.** Inconsistent indentation suggests `r_bsr_config` reset may be outside the `if (!rst_n)` block. | Audit begin/end pairing in the reset block. Ensure all BSR registers (`r_bsr_config`, `r_bsr_base_addr`, etc.) are inside the same `if (!rst_n)` scope. |
| R13 | [act_dma.sv](hw/rtl/dma/act_dma.sv) | — | **No AXI 4KB boundary guard.** Only a simulation `$warning`. A burst crossing a 4KB boundary violates AXI protocol. On real Zynq interconnect this causes unpredictable hangs. | Add hardware logic: if `align_addr[11:0] + burst_bytes > 4096`, split into two bursts. Or cap `arlen` so the burst stays within the current 4KB page. |
| R14 | [accel_top.sv](hw/rtl/top/accel_top.sv) | ~1050 | **col_idx address underflow.** `col_idx_rd_addr = meta_mem_addr - 128` wraps if `meta_mem_addr < 128`. The `meta_is_col_idx_r` mux prevents using corrupt data, but BRAM reads at garbage address. | Gate the BRAM read enable: `col_idx_bram_ren = meta_is_col_idx & bsr_meta_req_valid`. |


### Tier 3 — Medium (cleanup, won't block functionality)

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| R16 | [systolic_array.sv](hw/rtl/systolic/systolic_array.sv) | ~108–118 | **Unsynthesizable `initial` blocks.** Dense array uses `initial begin` with runtime `integer` for bypass_mat computation. | Copy the `localparam` approach from `systolic_array_sparse.sv`. This module isn't used by `accel_top`, so low priority. |
| R17 | [act_buffer.sv](hw/rtl/buffer/act_buffer.sv) | ~248–260 | **Latency comment mismatch.** Comment says "1-cycle read latency" but implementation has 2-cycle (BRAM read + output register). | Either remove the extra output register (1-cycle), or update the comment to say 2-cycle. Module is unused by accel_top currently. |
| R18 | [block_reorder_buffer.sv](hw/rtl/control/block_reorder_buffer.sv) | ~142–155 | **EMIT state `out_valid` double-assignment.** Set unconditionally then conditionally cleared. Synthesis may have inconsistent behavior. | Use a single conditional assignment: `out_valid = (emit_cnt < max_count) ? 1'b1 : 1'b0;` |
| R19 | [bsr_dma.sv](hw/rtl/dma/bsr_dma.sv) | ~300–350 | **Dead code.** `READ_HEADER` and `WAIT_HEADER` states immediately return to IDLE. Vestigial header-parse path. | Remove the dead states. Update the state encoding if using one-hot. |
| R20 | [accel_top.sv](hw/rtl/top/accel_top.sv) | — | **Buffer modules never instantiated.** `act_buffer.sv`, `wgt_buffer.sv`, `output_accumulator.sv` are implemented but bypassed by inline BRAMs. | Either: (A) Wire them in to get ping-pong banking + clock gating + quantization, or (B) delete them and document inline BRAM as the design choice. |

### Delete / Ignore

| File | Reason |
|------|--------|
| [top/accel_top_dual_clk.sv](hw/rtl/top/accel_top_dual_clk.sv) | 371 lines. Missing comma syntax error, every module instantiation has wrong parameter/port names, references non-existent `axi_dma_master`, uses 2×2 array. **Will not elaborate.** Delete or rewrite from scratch. |
| [top/deprecated/accel_top_legacy.sv](hw/rtl/top/deprecated/accel_top_legacy.sv) | 1047 lines. Every major instantiation uses wrong port names. References non-existent `uart_rx`/`uart_tx`. 4MB block_mem exceeds BRAM. **Completely broken.** Delete. |

---

## Part 2 — C++ Work

### Tier 1 — Bugs That Block Compilation

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| C1 | [memory_manager.hpp](hw/sim/cpp/include/memory_manager.hpp) | ~935 | **Typo: `__sync_synchronze()`** — won't compile on non-ARM. | Change to `__sync_synchronize()`. |
| C2 | [golden_models.hpp](hw/sim/cpp/include/golden_models.hpp) | — | **`add_residual()` declaration** has wrong name and signature vs implementation `add_residual_int8()` in .cpp. Linker error. | Rename declaration to match: `void add_residual_int8(const int8_t*, const int8_t*, int8_t*, size_t, float, float, float)`. |
| C3 | [golden_models.hpp](hw/sim/cpp/include/golden_models.hpp) | — | **`conv2d_int8()` declared** but .cpp has `conv2d_int8_simple()` and `conv2d_int8_im2col()`. Linker error. | Either rename the declaration to `conv2d_int8_im2col`, or add a `conv2d_int8` wrapper that calls `conv2d_int8_im2col`. |
| C4 | [golden_models.hpp](hw/sim/cpp/include/golden_models.hpp) | — | **`conv2d_bsr_int8()`, `compare_buffers()`, `compute_mae()` declared but never implemented.** Linker error if any test or source references them. | Implement them in golden_models.cpp, or delete the declarations. |
| C5 | [axi_master.hpp](hw/sim/cpp/include/axi_master.hpp) | ~240 | **`SoftwareModelBackend` initializes array-size registers to 16.** Should be 14. | `registers_[0x10/4] = 14; registers_[0x14/4] = 14; registers_[0x18/4] = 14;` |
| C6 | [performance_counters.hpp](hw/sim/cpp/include/performance_counters.hpp) | ~134 | **Default `num_pes = 256`.** Should be 196 (14×14). | Change default to `196`. |

### Tier 2 — Functional Bugs

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| C7 | [accelerator_driver.cpp](hw/sim/cpp/src/accelerator_driver.cpp) | ~324 | **BSR config bit conflict.** `BSR_CONFIG_ENABLE = bit 0 = SCHED_MODE`. Setting it to 1 may select Dense mode when BSR (sparse) is intended. | Audit `csr_bsr.hpp`: `BSR_CONFIG_ENABLE` and `BSR_CONFIG_SCHED_MODE` both alias bit 0. Either separate them into different bits, or use a single field with clear semantics. Update `set_bsr_buffers()` to write the correct mode. |
| C8 | [accelerator_driver.cpp](hw/sim/cpp/src/accelerator_driver.cpp) | ~395 | **DMA loading commented out.** `run_layer()` never actually moves data — `memory_->load_activations(...)` and `memory_->read_outputs(...)` are commented out. | Uncomment and implement the DMA transfer calls. Requires `MemoryManager` methods to be functional. |
| C9 | [accelerator_driver.cpp](hw/sim/cpp/src/accelerator_driver.cpp) | ~126 | **Clock frequency mismatch.** `CLOCK_MHZ = 200.0` but `performance_config.hpp` PYNQ-Z2 has 100 MHz default. | Decide the target frequency. If 200 MHz is the datapath clock (from dual-clock), document it. If single-clock, change to 100. |

### Tier 3 — Stubs to Implement

Every one of these files is either empty (all functions return true / print "not implemented") or uses wrong API calls. They need to be written from scratch.

| # | File | Lines | What Needs Doing |
|---|------|-------|-----------------|
| C10 | [main.cpp](hw/sim/cpp/main.cpp) | 282 | **Implement all 4 command handlers:** `cmd_infer` (load BSR weights + run MNIST inference), `cmd_test` (run test suite), `cmd_bench` (run benchmarks), `cmd_sim` (Verilator simulation). Connect argument parsing. |
| C11 | [resnet_inference.cpp](hw/sim/cpp/src/resnet_inference.cpp) | 305 | **Either delete or rewrite as `mnist_inference.cpp`.** Currently returns hardcoded dummy results, calls non-existent `dense_to_bsr()`, divides by 256 (16×16). Every function is a stub. |
| C12 | [tests/test_bsr_packer.cpp](hw/sim/cpp/tests/test_bsr_packer.cpp) | 177 | **All 6 tests are commented-out TODOs.** Uses non-existent `BSRPacker` class. Rewrite using free functions `pack_to_bsr()`, `unpack_from_bsr()`, `serialize_for_hardware()`, `validate_bsr()`. Change block size to 14. |
| C13 | [tests/test_golden_models.cpp](hw/sim/cpp/tests/test_golden_models.cpp) | 168 | **All 8 tests are TODOs.** Implement: allocate INT8 matrices, run `matmul_int8` / `bsr_matmul_int8`, compare results, test edge cases (zero matrix, max values, overflow). |
| C14 | [tests/test_end_to_end.cpp](hw/sim/cpp/tests/test_end_to_end.cpp) | 148 | **All 5 tests are TODOs.** Implement: create `AcceleratorDriver` with `SoftwareModelBackend`, configure MNIST layer, run through driver, verify against golden model output. |
| C15 | [tests/test_axi_transactions.cpp](hw/sim/cpp/tests/test_axi_transactions.cpp) | 165 | **All 7 tests are TODOs.** Uses non-existent AXI methods (`write32_masked`, `read64`, `burst_write`). Rewrite using actual `AXIMaster` API (`write32`, `read32`). |
| C16 | [tests/test_performance.cpp](hw/sim/cpp/tests/test_performance.cpp) | 221 | **All 5 benchmarks print "not implemented".** Fix banner from "16x16" to "14x14". Implement timing loops for matmul, conv, BSR packing. |
| C17 | [tests/test_stress.cpp](hw/sim/cpp/tests/test_stress.cpp) | 218 | **All 8 tests are TODOs.** Fix `N = 16` → `14`. Implement stress tests for large matrices, rapid CSR writes, memory pressure, concurrent operations. |
| C18 | [verilator/tb_accel_top.cpp](hw/sim/cpp/verilator/tb_accel_top.cpp) | 226 | **All test functions are TODOs.** Verilator model include is commented out. Uncomment `#include "Vaccel_top.h"`, implement AXI-Lite helpers, write register read/write tests and a simple matmul. |
| C19 | [verilator/tb_mac8.cpp](hw/sim/cpp/verilator/tb_mac8.cpp) | 194 | **All 7 tests are TODOs.** Uncomment `#include "Vmac8.h"`. Golden helper `golden_mac8()` is implemented — use it. Test: zero inputs, max positive, max negative, overflow, accumulate, clear. |
| C20 | [verilator/tb_pe.cpp](hw/sim/cpp/verilator/tb_pe.cpp) | 220 | **All 6 tests are TODOs.** Uncomment `#include "Vpe.h"`. Test: weight load, MAC accumulate, clear, passthrough, back-to-back, pipeline depth. |
| C21 | [verilator/tb_systolic_array.cpp](hw/sim/cpp/verilator/tb_systolic_array.cpp) | 238 | **All 4 tests are TODOs.** Fix `N = 16` → `14`. Uncomment Verilator include. Test: identity matmul, random matmul, sparse matmul with block_valid, weight reload. |
| C22 | [examples/deploy_example.cpp](hw/sim/cpp/examples/deploy_example.cpp) | 110 | **Uses non-existent `BSRPacker` class and 16×16 buffers.** Rewrite using free functions from `bsr_packer.hpp`. Fix all buffer sizes to 14×14. |

### Delete

| File | Reason |
|------|--------|
| [include/resnet_inference.hpp](hw/sim/cpp/include/resnet_inference.hpp) | Describes ResNet-18 224×224 ImageNet inference. All of this is wrong for MNIST. Duplicate `#endif`. Delete. |
| [src/resnet_inference.cpp](hw/sim/cpp/src/resnet_inference.cpp) | Every function is a stub. Returns hardcoded "Tabby cat" result. Calls non-existent functions. Delete. |

---

## Part 3 — Python Fixes

### Bugs

| # | File | Line(s) | Bug | Fix |
|---|------|---------|-----|-----|
| P1 | [host/accel.py](sw/host/accel.py) | ~484 | **Test uses 16×16 blocks.** `weights = np.random.randint(-128, 127, (num_blocks, 16, 16), ...)` | Change to `(num_blocks, 14, 14)`. |
| P2 | [host_axi/run_gemm_axi.py](sw/host_axi/run_gemm_axi.py) | ~15 | **Imports from non-existent `host_uart`.** `from host_uart.csr_map import ...` | Change to `from host_axi.csr_map import Config, pack_u32, unpack_u32` (or use relative import). |
| P3 | [host_axi/run_gemm_axi.py](sw/host_axi/run_gemm_axi.py) | — | **`pack_u32(1.0)` fails.** `struct.pack('I', val)` rejects floats. | Cast to int first: `pack_u32(int(1.0))`, or use `struct.pack('f', val)` for float CSR writes. |
| P4 | [host_axi/csr_map.py](sw/host_axi/csr_map.py) | — | **DMA register addresses not 4-byte aligned.** `0x50, 0x51, 0x52, 0x53, 0x54` — AXI4-Lite requires 4-byte alignment. | Change to `0x50, 0x54, 0x58, 0x5C, 0x60`. Verify these match RTL `csr.sv` offsets. |
| P5 | [tests/test_csr_pack.py](sw/tests/test_csr_pack.py) | ~3 | **Imports from non-existent `host_uart`.** | Change to `from host_axi.csr_map import *`. |
| P6 | [tests/test_integration.py](sw/tests/test_integration.py) | ~20–22 | **All imports from non-existent `host_uart`**. Plus references 5 undefined constants (`STS_ERR_CRC`, `CMD_WRITE`, etc.). | Either rewrite for AXI host interface, or delete (742 lines of dead code for a UART interface that no longer exists). |
| P7 | [tests/test_edges.py](sw/tests/test_edges.py) | — | **Tests always report PASS.** Comparison block is missing — tests print "PASS ✓" unconditionally without checking results. | Add actual assertions: `np.testing.assert_array_equal(sparse_result, dense_result)`. |

### Delete

| File | Reason |
|------|--------|
| [INT8 quantization/quantize old.py](sw/INT8%20quantization/quantize%20old.py) | Broken `forward()` method, corrupted network definition, legacy copy. |
| [INT8 quantization/quantize_resnet18.py](sw/INT8%20quantization/quantize_resnet18.py) | ResNet-18 only — not used by MNIST pipeline. |
| [training/train_resnet18.py](sw/training/train_resnet18.py) | ResNet-18 training — not used. Has duplicate `BlockSparsePruner`. |
| [training/export_resnet18_bsr.py](sw/training/export_resnet18_bsr.py) | ResNet-18 BSR export — third copy of `build_bsr_from_dense`. |
| [tests/test_integration.py](sw/tests/test_integration.py) | 742 lines importing non-existent `host_uart` module. Every test fails with `ModuleNotFoundError`. Rewriting from scratch for AXI would be better than fixing. |

---

## Part 4 — SV Testbenches

### Fix Array Sizes (all testbenches use wrong dimensions)

| # | File | Current | Should Be | Impact |
|---|------|---------|-----------|--------|
| T1 | [sv/accel_top_tb.sv](hw/sim/sv/accel_top_tb.sv) | N_ROWS=8, N_COLS=8 | 14, 14 | Tests pass at wrong size; won't catch BRAM packing bug (R1) |
| T2 | [sv/accel_top_tb_full.sv](hw/sim/sv/accel_top_tb_full.sv) | N_ROWS=8, N_COLS=8 | 14, 14 | Scale factor format mismatch (writes IEEE 754 float, HW expects Q16.16) |
| T3 | [sv/integration_tb.sv](hw/sim/sv/integration_tb.sv) | N_ROWS=16, N_COLS=16 | 14, 14 | BSR block data packed as 256 bytes (should be 196) |
| T4 | [sv/output_accumulator_tb.sv](hw/sim/sv/output_accumulator_tb.sv) | N_ROWS=16, N_COLS=16 | 14, 14 | NUM_ACCS=256 instead of 196 |
| T5 | [sv/systolic_tb.sv](hw/sim/sv/systolic_tb.sv) | N_ROWS=8, N_COLS=8 | 14, 14 | Minor — tests still valid at smaller size |
| T6 | [sv/tb_axi_lite_slave_enhanced.sv](hw/sim/sv/tb_axi_lite_slave_enhanced.sv) | Tests CSR at 0x50–0x54 | 0x00–0x3C (per csr.sv) | **Testing completely wrong addresses** |
| T7 | [sv/meta_decode_tb.sv](hw/sim/sv/meta_decode_tb.sv) | 1-cycle BRAM model | 2-cycle | Won't catch real latency bugs |

### Missing Testbenches (write new)

| # | Module | Priority | Why |
|---|--------|----------|-----|
| T8 | `scheduler.sv` | **HIGH** | Has 4 known bugs (R3, R4, R15). 705 lines, 8-state FSM, untested. |
| T9 | `bsr_scheduler.sv` | **HIGH** | Has 3 known bugs (R8, R10, R11). 462 lines, 10-state FSM, handles entire sparse traversal. |
| T10 | `act_dma.sv` | **HIGH** | No 4KB boundary guard (R13). Untested with real AXI traffic patterns. |
| T11 | `csr.sv` | MEDIUM | Clock gating bug (R5), reset scoping bug (R12). Only tested indirectly through accel_top. |
| T12 | `axi_dma_bridge.sv` | MEDIUM | 2:1 arbiter — should test back-to-back requests, priority, watchdog timeout. |
| T13 | `act_buffer.sv` | LOW | Double-buffered SRAM. Not instantiated by accel_top. Test if you wire it in. |
| T14 | `block_reorder_buffer.sv` | LOW | Insertion-sort buffer for metadata reordering. |

---

## Part 5 — Build System & Tools (all paths broken)

Every script in `tools/` references a flat directory structure (`rtl/`, `testbench/`) that doesn't match the actual layout (`hw/rtl/`, `hw/sim/sv/`). Fix paths or rewrite.

| # | File | What's Wrong | Fix |
|---|------|-------------|-----|
| B1 | [tools/build.sh](tools/build.sh) | References `rtl/host_iface/axi_lite_slave_v2.sv` (doesn't exist), `accel_top.v` (wrong extension). | Update all paths to `hw/rtl/...` and `.sv` extension. |
| B2 | [tools/test.sh](tools/test.sh) | Uses `$PROJECT_ROOT/accel/python/host` (should be `sw/host/`), `testbench/sv` (should be `hw/sim/sv/`), undefined `$SIM_DIR`. | Rewrite all paths. Define missing variables. |
| B3 | [tools/Makefile.verilator](tools/Makefile.verilator) | `RTL_DIR = rtl`, `TB_DIR = testbench`. References non-existent `axi_dma_master.sv` and `rtl/uart/`. | Set `RTL_DIR = ../hw/rtl`, `TB_DIR = ../hw/sim`. Remove phantom module references. |
| B4 | [tools/ci_verilator.sh](tools/ci_verilator.sh) | Hardcoded `WORKSPACE="/workspaces/ACCEL-v1"`. | Use `$(git rev-parse --show-toplevel)` or `$PWD`. |
| B5 | [tools/synthesize_vivado.tcl](tools/synthesize_vivado.tcl) | Targets **Artix-7 XC7A100T** but project uses **Zynq-7020**. Also references non-existent `rtl/uart/`. | Change part to `xc7z020clg400-1`. Remove `rtl/uart/` glob. Update resource estimates. |
| B6 | [tools/run/run_accel_tb.sh](tools/run/run_accel_tb.sh) | All paths use flat `testbench/`, `rtl/`. | Prefix with `hw/`. |
| B7 | [tools/run/run_axi_demo.sh](tools/run/run_axi_demo.sh) | Hardcoded `/workspaces/ACCEL-v1`, uses `accel/python/host_axi/`. | Fix base path, change to `sw/host_axi/`. |
| B8 | [tools/run/run_coverage.sh](tools/run/run_coverage.sh) | All `testbench/verilator/` should be `hw/sim/sv/`. All `rtl/` should be `hw/rtl/`. | Fix all paths. Coverage merge/report logic is correct. |
| B9 | [hw/sim/CMakeLists.txt](hw/sim/CMakeLists.txt) | 8-line stub referencing non-existent `test_mac8_verilator.cpp`. | Either delete or rewrite with actual Verilator build flow. |
| B10 | [hw/sim/cocotb/Makefile.accel_top](hw/sim/cocotb/Makefile.accel_top) | References `perf_monitor.sv` (should be `perf.sv`). Missing several submodule sources. | Fix filename. Add all required RTL sources for accel_top elaboration. |
| B11 | [hw/sim/cocotb/Makefile.cocotb](hw/sim/cocotb/Makefile.cocotb) | Paths `verilog/`, `tb/`, `python/host` don't exist. | Rewrite with correct relative paths to `../../rtl/` and `../../sw/host/`. |

---

## Part 6 — Documentation Updates

| # | File | What's Wrong | Fix |
|---|------|-------------|-----|
| D1 | [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) | Describes UART as primary host interface, references 2×2 PEs and 8×8 blocks. | Rewrite for AXI4-Lite + 14×14 array. Or mark clearly as "historical — see DEEP_DIVE.md for current design". |
| D2 | [docs/architecture/HOST_RS_TILER.md](docs/architecture/HOST_RS_TILER.md) | Has two conflicting protocol specs (5-byte vs 7-byte packets). References non-existent `host_uart/` directory. Duplicate register map sections. | Rewrite for AXI host interface. Remove UART protocol sections. Consolidate register map. |
| D3 | [docs/architecture/SPARSITY_FORMAT.md](docs/architecture/SPARSITY_FORMAT.md) | Describes 8×8 BSR blocks. | Update to 14×14 or note that the format is parameterized and current hardware uses 14×14. |
| D4 | [STRUCTURE.md](STRUCTURE.md) | May have stale file references after deletions. | Verify all listed files still exist. |

---

## Part 7 — Suggested Work Order

### Phase 1: Clean Dead Code (1 day)
1. Delete: `accel_top_dual_clk.sv`, `accel_top_legacy.sv`
2. Delete: `resnet_inference.hpp`, `resnet_inference.cpp`
3. Delete: `quantize old.py`, `quantize_resnet18.py`, `train_resnet18.py`, `export_resnet18_bsr.py`
4. Delete: `test_integration.py` (dead UART code)
5. Delete or rewrite: `tools/convert_to_sv.sh` (obsolete)

### Phase 2: RTL Tier 1 Fixes (2–3 weeks)
6. Fix R1 (BRAM width packing) — this blocks all data-path testing
7. Fix R2 (bsr_dma weight truncation) — blocks BSR path
8. Fix R5 (CSR clock gating) — simple, do it early
9. Fix R6 (output_accumulator scale) — simple
10. Fix R3 + R4 (scheduler bit widths) — straightforward
11. Fix R7 (result output path) — design decision needed (write DMA vs CSR index)
12. Fix R8 (pe load_weight/en overlap) — needs bsr_scheduler timing analysis

### Phase 3: RTL Tier 2 Fixes + Testbenches (2 weeks)
13. Fix R9–R15
14. Write testbench for `scheduler.sv` (T8)
15. Write testbench for `bsr_scheduler.sv` (T9)
16. Update all existing testbenches to 14×14 (T1–T7)
17. Fix SV testbench scale factor format in `accel_top_tb_full.sv` (T2)
18. Fix `tb_axi_lite_slave_enhanced.sv` CSR addresses (T6)

### Phase 4: C++ Compilation Fixes (1 week)
19. Fix C1–C6 (compilation blockers + size errors)
20. Fix C7–C9 (functional bugs)
21. Delete `resnet_inference.*`
22. Fix `deploy_example.cpp` (C22)

### Phase 5: C++ Stubs → Real Code (2–3 weeks)
23. Implement `golden_models` missing functions (C4)
24. Implement test files: `test_bsr_packer`, `test_golden_models`, `test_end_to_end` (C12–C14)
25. Implement `main.cpp` command handlers (C10)
26. Implement Verilator testbenches: `tb_mac8`, `tb_pe`, `tb_systolic_array`, `tb_accel_top` (C18–C21)

### Phase 6: Python + Build + Docs (1 week)
27. Fix P1–P7 (Python bugs)
28. Delete Python dead files
29. Fix all tool scripts (B1–B11)
30. Fix Vivado TCL target part (B5)
31. Update docs (D1–D4)

### Phase 7: Integration + Deployment (1–2 weeks)
32. Run full pipeline: train → quantize → export_bsr_14x14 → Verilator sim → verify
33. Synthesize with Vivado for Zynq-7020
34. Deploy bitstream to PYNQ-Z2
35. Run MNIST inference on-board
36. Collect performance numbers and power measurements

---

## Quick Reference: File Counts

| Category | Total | Complete | Stub/Broken | Has Bugs |
|----------|-------|----------|-------------|----------|
| RTL (.sv) | 23 | 17 | 2 (delete) | 15 |
| C++ (.hpp/.cpp) | 33 | 10 | 15 stubs | 8 |
| Python (.py) | 38 | 28 | 3 stubs | 9 |
| SV Testbenches | 10 | 3 clean | 0 | 7 (wrong sizes) |
| Build/Tool scripts | 11 | 0 | 11 (broken paths) | 11 |
| Docs (.md) | ~12 | 4 | 0 | 4 stale |
| **TOTAL** | **~127** | **62** | **31** | **54** |
