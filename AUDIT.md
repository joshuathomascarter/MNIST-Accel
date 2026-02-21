# Code Audit — ACCEL-v1

**Date: February 9, 2026**

## Audit Scope
- **21 SystemVerilog RTL files** (`hw/rtl/`)
- **10 SV testbench files** (`hw/sim/sv/`)
- **~32 C++ files** (`sw/cpp/`)
- **~30 Python files** (`sw/ml_python/`)
- **Build system** (CMake, Makefiles, Vivado TCL)

---

## EXECUTIVE SUMMARY

| Category | Count |
|---|---|
| **RTL compile errors** | 3 |
| **RTL data corruption bugs** | 4 |
| **RTL logic bugs** | 8 |
| **RTL port mismatches** | 6 |
| **RTL multi-bit CDC violation** | 1 |
| **RTL stubs / dead modules** | 2 |
| **C++ build-breaking** | 5 |
| **C++ API mismatches** | 8 |
| **C++ logic bugs** | 6 |
| **C++ stub files** | 16 |
| **Python runtime errors** | 5 |
| **Python import errors** | 3 |
| **Cross-cutting mismatches** | 4 |
| **Total actionable issues** | **~71** |

**Bottom line:** The RTL core (mac8, pe, systolic arrays) is solid and likely synthesizable. The integration layer (`accel_top.sv`) has real bugs. The C++ stack (`sw/cpp/`) is heavily scaffolded (many stub files). The Python stack (`sw/ml_python/`) has two incompatible CSR maps and a missing `host_uart/` directory.

---

# PART 1: RTL AUDIT (`hw/rtl/`)

## RTL — CORE COMPUTE (Solid)

### `hw/rtl/mac/mac8.sv` — 484 lines [CLEAN]
**Status: CLEAN.** Well-designed signed 8×8→32 MAC with zero bypass, operand isolation, and ResNet residual support. DSP48E1 inference attributes present. SVA assertions for unknowns and clear priority. No bugs found.

### `hw/rtl/systolic/pe.sv` — 411 lines [CLEAN]
**Status: CLEAN.** Weight-stationary PE with pipelined activation forwarding, systolic load_weight propagation, and MAC8 instantiation. The `load_weight`/`en` mutual exclusion assertion is commented out with a TODO about scheduler timing — **needs resolution** but not a bug.

### `hw/rtl/systolic/systolic_array_sparse.sv` — 426 lines [CLEAN]
**Status: CLEAN.** Sparse-aware 14×14 array with correct row-selective weight loading (`load_ptr`), per-PE bypass for residuals, and `block_valid` gating. The `c_out_flat` output is directly connected via indexed part-select — correct.

### `hw/rtl/systolic/systolic_array.sv` — 274 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R1 | **Compile error** | `bypass_mat` is **used before declaration**. Line ~97 uses `bypass_mat[r_idx][c_idx]` in a generate block, but `bypass_mat` is declared at line ~172). In the UNPACK_BYPASS generate, `integer r_idx, c_idx; initial begin` uses runtime initialization inside a generate — this is non-portable and will fail on many synthesis tools. |
| R2 | **Bug** | `assign c_out_flat = { >> {acc_mat} };` uses SystemVerilog streaming operator on a 2D unpacked array. This is **not universally supported** — Vivado handles it but Verilator does not. The sparse variant uses explicit indexed part-select instead. |
| R3 | Minor | Activation skew uses `reg` inside a generate block with `integer k` — this works but is non-standard for synthesis; Verilator may warn about `integer` in synthesizable blocks. |

## RTL — BUFFERS

### `hw/rtl/buffer/act_buffer.sv` — 290 lines [CLEAN]
**Status: CLEAN.** Double-buffered activation SRAM with clock gating, ping-pong via bank_sel, 1-cycle read latency (with extra output register — actually 2 cycles). Assertions verify address bounds and bank_sel validity.

**NOTE:** The read path has 2 registers: `read_data` (BRAM output) and `a_vec` (output register). This is **2-cycle latency**, not 1. If the scheduler assumes 1 cycle, there's a pipeline bubble.

### `hw/rtl/buffer/wgt_buffer.sv` — 243 lines [ISSUES]
**Status: HAS BUG.**

| # | Severity | Issue |
|---|---|---|
| R4 | **Data corruption** | Same 2-register read path as act_buffer. Read latency is **2 cycles**, but comments say "1-cycle latency, same as act_buffer." The `bsr_scheduler` and `scheduler` both assume 1-cycle latency when prefetching weight data. The systolic array receives **stale weight data by 1 cycle**, meaning the first weight loaded into each PE row is from the *previous* read, not the current one. |

### `hw/rtl/buffer/output_accumulator.sv` — 458 lines [ISSUES]
**Status: HAS BUG.**

| # | Severity | Issue |
|---|---|---|
| R5 | **Logic bug** | `quantize_relu` function uses `$signed({1'b0, scale[15:0]})` — takes only the **lower 16 bits** of the 32-bit scale_factor, then sign-extends with a leading 0. This means the Q16.16 fixed-point value only uses the fractional part. If scale_factor > 65535 (i.e., integer part > 0), the upper bits are discarded, producing wrong quantization results. |
| R6 | Minor | DMA read address `{dma_rd_addr, 3'b000}` generates accumulator indices 0,8,16,... — meaning only every 8th accumulator is readable. For 196 accumulators, this needs `ceil(196/8)=25` addresses. With `ADDR_W=10`, this works, but edge alignment (196 % 8 = 4) means the last DMA read accesses 4 valid + 4 out-of-range accumulators. |

## RTL — CONTROL

### `hw/rtl/control/bsr_scheduler.sv` — 462 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R7 | **Logic bug** | `S_LOAD_WGT` transition condition: `load_cnt == LOAD_CNT_MAX[4:0] + 1` means the counter reaches 14 (BLOCK_SIZE), not 13. This means the weight load phase runs for **15 cycles** (0-14) instead of 14 (0-13). The extra cycle has `load_weight_r=1` while no valid weight data is being driven. |
| R8 | **Logic bug** | `S_WAIT_WGT` assumes the load_weight signal needs 14 cycles to propagate through all columns (due to systolic chaining). But `pe.sv` propagates `load_weight_out` with a 1-cycle register delay, so 14 columns = 14 cycles of propagation. However, the last PE receives load_weight on cycle 13 and needs 1 more cycle to capture the weight. Total wait should be 14 cycles, and the counter resets at `LOAD_CNT_MAX` (13). The wait is actually only 13 cycles — **1 cycle short**. The last column may not finish loading. |
| R9 | Bug | `S_STREAM_ACT` streams for only `BLOCK_SIZE` (14) cycles, but the activation pipeline has 14 stages of delay (PIPE=1). The first PE gets valid data on cycle 0, but PE[13] doesn't get its first valid activation until cycle 13. Only 1 cycle of actual valid compute happens for the last PE. Should stream for `BLOCK_SIZE + N_COLS - 1 = 27` cycles (or use different activation addressing). |
| R10 | Bug | `meta_raddr <= 128 + blk_ptr` in `S_FETCH_COL` — magic number 128 as col_idx offset. If `row_ptr` array has more than 128 entries, the offset overlaps. This should be dynamically calculated from `num_rows + 1`. |

### `hw/rtl/control/scheduler.sv` — 705 lines [ISSUES]
**Status: HAS ISSUES.**

| # | Severity | Issue |
|---|---|---|
| R11 | Bug | `ceil_div_N` and `ceil_div_K` return `{M_W{1'b0}}` on divide-by-zero — should be `{N_W{1'b0}}` and `{K_W{1'b0}}` respectively. Uses wrong width constant for the zero fill. |
| R12 | Bug | `en_mask_row` and `en_mask_col` initial values use `{M_W{1'b0}}` — should be `{MAX_TM{1'b0}}` and `{MAX_TN{1'b0}}` respectively. Masks are `MAX_TM` and `MAX_TN` bits wide, but zeroed with `M_W` bits. |
| R13 | Minor | k_tile update triggering `state[5]` in the registered block creates a one-hot decoding of the `state` register — this is correct but fragile if additional states are added. |

### `hw/rtl/control/csr.sv` — 527 lines [ISSUES]
**Status: HAS ISSUES.**

| # | Severity | Issue |
|---|---|---|
| R14 | **Bug** | Clock gating: `assign clk_gated = clk & csr_clk_en` — this is **glitch-prone** combinational clock gating without a latch. CSR registers can corrupt on glitches. Other modules (scheduler, buffers) correctly use latch-based gating. |
| R15 | Bug | BSR defaults in reset block are **not indented** inside the `if (!rst_n)` block. Looking at the code structure: the `r_bsr_*` reset assignments at lines ~350-360 are at a different indentation level. In SystemVerilog, this is fine syntactically (they're still inside the `begin`/`end`), but it's confusing and error-prone for maintenance. |
| R16 | Stub | `BSR_ERROR_CODE` always reads as `32'd0` — no actual error tracking. |
| R17 | Stub | `ACT_DMA_CTRL` busy bit hardcoded to `1'b0`. |

### `hw/rtl/control/block_reorder_buffer.sv` — 231 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R18 | **Logic bug** | In EMIT state, `out_valid` is set to 1, then overwritten to 0 in the same always_ff evaluation when `emit_idx >= count - 1`. The last beat of sorted output is **never visible** to downstream logic. |
| R19 | Bug | `count == 0` edge case: if `in_row_done` fires with 0 blocks, `count - 1` underflows, causing the FSM to emit garbage from uninitialized memory. |
| R20 | Dead code | `SORT_SHIFT` state (3'd3) is declared but never transitioned to. |

### `hw/rtl/control/multi_layer_buffer.sv` — 57 lines [ISSUES]
**Status: PROBLEMATIC.**

| # | Severity | Issue |
|---|---|---|
| R21 | **Synthesis issue** | `unified_mem` with `ADDR_WIDTH=16` = 65536 × 32 bits = 2 Mbit — consumes ~40% of Zynq-7020 BRAM. |
| R22 | **Synthesis issue** | `assign rd_data = unified_mem[absolute_addr]` is an **async read** — won't infer BRAM (requires registered reads). Will use LUTs instead, which can't hold 2 Mbit. |

### `hw/rtl/control/pulse_sync.sv` — 78 lines [CLEAN]
**Status: CLEAN.** Toggle-based pulse synchronizer with ASYNC_REG attributes.

### `hw/rtl/control/sync_2ff.sv` — 197 lines [ISSUES]
**Status: MOSTLY CLEAN, 1 ISSUE.**

| # | Severity | Issue |
|---|---|---|
| R23 | **CDC violation** | `accel_top_dual_clk.sv` uses `sync_2ff #(.WIDTH(32))` for a 32-bit counter `blocks_processed`. Multi-bit bus through a 2-FF synchronizer causes **metastability and data corruption**. Needs gray-coding or handshake-based CDC. |

## RTL — DMA

### `hw/rtl/dma/act_dma.sv` — 608 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R24 | **Bug** | `act_we` is set to 1 in READ_DATA but **never cleared** in SEND_ADDR, DONE_STATE, or IDLE. Since this is a registered always_ff with no default assignments, `act_we` stays high indefinitely after the first data beat, causing **spurious writes** to the activation buffer. |
| R25 | **Bug** | Partial burst: if `bytes_remaining == 0`, `arlen = ((0+7) >> 3) - 1 = -1 = 255`, issuing a 256-beat burst for 0 bytes. Edge case, but violates AXI spec. |
| R26 | Bug | No 4KB boundary crossing check — AXI4 spec violation. Only has a simulation `$warning`. |

### `hw/rtl/dma/bsr_dma.sv` — 776 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R27 | **Bug** | Same `row_ptr_we`/`col_idx_we`/`wgt_we` not-cleared issue as act_dma. Write enables stay asserted across state transitions. |
| R28 | **Bug** | `SETUP_WEIGHTS: words_remaining = total_blocks * 8` assumes 64-byte blocks (8 beats). Actual 14×14 blocks are 196 bytes (25 beats). **Weight data is truncated** — only 64 of 196 bytes are loaded per block. |
| R29 | Bug | Same `words_remaining == 0` underflow as act_dma for col_idx burst length calculation. |

## RTL — HOST INTERFACE

### `hw/rtl/host_iface/axi_lite_slave.sv` — 290 lines [ISSUES]
**Status: HAS BUG.**

| # | Severity | Issue |
|---|---|---|
| R30 | **Data corruption** | Simultaneous read+write: `csr_addr` mux prioritizes read address, but `csr_wen` is also asserted. CSR module receives write command with **wrong address** (the read address instead of write address). |

### `hw/rtl/host_iface/axi_dma_bridge.sv` — 480 lines [CLEAN]
**Status: MOSTLY CLEAN.** Watchdog timeout doesn't set error flag (minor). Otherwise correct 2:1 AXI arbiter.

## RTL — METADATA

### `hw/rtl/meta/meta_decode.sv` — 56 lines [ISSUES]
**Status: HAS BUG.**

| # | Severity | Issue |
|---|---|---|
| R31 | **Data corruption** | `S_WAIT` captures `mem_rdata` after 1 cycle, but BRAM has 2 cycles of read latency (registered output + output register). The captured data is from the **previous** read, not the current one. This silently corrupts all BSR metadata lookups. |
| R32 | Stub | `CACHE_DEPTH` parameter exists but caching was removed — dead parameter. |

## RTL — MONITOR

### `hw/rtl/monitor/perf.sv` — 204 lines [ISSUES]
**Status: MINOR BUG.**

| # | Severity | Issue |
|---|---|---|
| R33 | Minor | Latched counter values are off by 1 due to `prev_state` edge detection delay. |

## RTL — TOP-LEVEL INTEGRATION

### `hw/rtl/top/accel_top.sv` — 1352 lines [ISSUES]
**Status: HAS BUGS.**

| # | Severity | Issue |
|---|---|---|
| R34 | **Data corruption** | `act_buf_wdata` (64-bit from DMA) is zero-extended to 112 bits: `{{(N_ROWS*DATA_W-64){1'b0}}, act_buf_wdata}`. Only 8 of 14 INT8 activations are loaded per write — **top 6 values are always zero**. Same issue for weight buffer writes. |
| R35 | **Bug** | `col_idx_rd_addr = meta_mem_addr[...] - 128` — underflows if `meta_mem_addr < 128`, reading garbage BRAM data. |
| R36 | Bug | `sched_mode = cfg_bsr_config[0]` defaults to 0 (BSR mode) on reset. If no BSR data is loaded, the BSR scheduler reads uninitialized BRAM. |

### `hw/rtl/top/accel_top_dual_clk.sv` — 310 lines [BROKEN]
**Status: WON'T COMPILE.**

| # | Severity | Issue |
|---|---|---|
| R37 | **Compile error** | Missing comma: `.clk(clk_data)` before `.rst_n(rst_n)` at line ~249. |
| R38 | **Compile error** | Trailing `);` after `assign blocks_processed_data = 32'd0` — stray syntax. |
| R39 | **Module not found** | Instantiates `axi_dma_master` — this module doesn't exist in the codebase. |
| R40 | Port mismatch | `act_buffer` and `wgt_buffer` instantiated with wrong port names (`.wen`, `.raddr`, `.rdata` vs actual `.we`, `.k_idx`, `.b_vec`). |
| R41 | CDC violation | 32-bit `sync_2ff` for counter (see R23 above). |

### `hw/rtl/top/deprecated/accel_top_legacy.sv` — 1047 lines [BROKEN]
**Status: CANNOT COMPILE.** Multiple driver conflicts, port mismatches with every module it instantiates, references UART interfaces that no longer exist. **Should be deleted** — it provides no value and causes confusion.

---

# PART 2: C++ AUDIT

---

## C++ Files (`sw/cpp/`)

> Key findings: Many files are stubs, CMake has issues, block size 14 vs 16 mismatch throughout.

### 1. `sw/cpp/CMakeLists.txt` (~200 lines) — **BUILD CONFIG**
**Status:** Real but BROKEN

| # | Severity | Issue |
|---|---|---|
| 1 | **BUILD BREAK** | Lists source files that **don't exist**: `src/axi_master.cpp`, `src/bsr_packer.cpp`, `src/memory_manager.cpp`. These are header-only. |
| 2 | **BUILD BREAK** | Defines `-DBLOCK_SIZE=16 -DN_ROWS=16 -DN_COLS=16` but `csr_map.hpp` defines `SYSTOLIC_ROWS=14`, `SYSTOLIC_COLS=14`, `BLOCK_SIZE=14`. Hardware is 14×14. |
| 3 | **BUILD BREAK** | Missing `src/performance_counters.cpp` and `src/test_utils.cpp` from `CORE_SOURCES` — these exist but aren't linked. |
| 4 | Bug | Verilator integration is `# TODO` — commented out, no RTL simulation possible. |

---

### 2. `sw/cpp/apps/main.cpp` (~324 lines) — **CLI ENTRY POINT**
**Status:** STUB

| # | Severity | Issue |
|---|---|---|
| 5 | Stub | `parse_args()` is entirely commented out — only does `argv[1]` check. |
| 6 | Stub | All 4 command handlers (`cmd_infer`, `cmd_test`, `cmd_bench`, `cmd_sim`) print "Not yet implemented" and return. |

---

### 3. `sw/cpp/include/driver/accelerator_driver.hpp` (~640 lines) — **DRIVER HEADER**
**Status:** Real, complete

- `AcceleratorError` exception class fully defined (lines ~315-360).
- `AcceleratorDriver` class fully declared with all methods and private members including `layer_configs_`, `layer_weights_`, `axi_`, `memory_`, `perf_`.
- `float_to_bits()` uses union type-punning — technically UB in C++ (should use `std::memcpy` or `std::bit_cast`).

---

### 4. `sw/cpp/src/driver/accelerator_driver.cpp` (~835 lines) — **DRIVER IMPL**
**Status:** Real, with bugs

| # | Severity | Issue |
|---|---|---|
| 7 | **Logic bug** | `set_bsr_buffers()` writes `BSR_CONFIG` with `BSR_CONFIG_ENABLE \| BSR_CONFIG_VERIFY \| BSR_CONFIG_ZERO_SKIP`, **overwriting** the scheduler mode that `set_scheduler_mode()` just set moments before in `configure_layer()`. `BSR_CONFIG_ENABLE` and `BSR_CONFIG_SCHED_MODE` are both bit 0 per `csr_bsr.hpp`. |
| 8 | **Logic bug** | `run_layer()` has commented-out memory operations: `// memory_->load_activations(...)` and `// memory_->read_outputs(...)` — data is **never actually transferred** to/from DMA buffers. |
| 9 | Bug | Constructor hardcodes clock to 200 MHz but `performance_config.hpp` defaults PYNQ-Z2 to 100 MHz. |
| 10 | Info | `set_layer_weights(layer_idx, bsr)` serializes BSR with `BSR_BLOCK_ELEMENTS` constant which must match `bsr_packer.hpp`'s value (196 for 14×14). |

---

### 5. `sw/cpp/include/driver/axi_master.hpp` (~625 lines) — **AXI INTERFACE**
**Status:** Real, complete

- `AXIBackend` abstract class ✓
- `DevMemBackend` ✓
- `SoftwareModelBackend` ✓
- `VerilatorBackend<VModel>` template ✓
- `AXIMaster` class fully defined with `write_reg()`, `read_reg()`, `set_bits()`, `clear_bits()`, `test_bits()`, `write_burst()`, `read_burst()`, `write_memory()`, `read_memory()` ✓
- No bugs found.

---

### 6. `sw/cpp/include/compute/bsr_packer.hpp` (~683 lines) — **BSR CONVERSION**
**Status:** Real, complete, with inline tests

- Standalone functions: `pack_to_bsr()`, `unpack_from_bsr()`, `is_block_nonzero()`, `validate_bsr()`, `get_sparsity()`, `serialize_for_hardware()`, `deserialize_from_hardware()`, `verify_round_trip()`, `verify_serialization()` ✓
- `BSRMatrix` struct with `compression_ratio()` ✓
- Inline unit tests (`test_all_zeros`, `test_serialization`, `test_non_aligned`, `run_all_tests`) ✓

| # | Severity | Issue |
|---|---|---|
| 11 | **API mismatch** | `deploy_example.cpp` uses a `BSRPacker` **class** with methods `.dense_to_bsr()`, `.pack_for_hardware()` — these don't exist. The real API is standalone `pack_to_bsr()` and `serialize_for_hardware()`. |

---

### 7. `sw/cpp/include/driver/csr_bsr.hpp` (~65 lines) — **BSR CSR DEFS**
**Status:** Real

| # | Severity | Issue |
|---|---|---|
| 12 | **Design conflict** | `BSR_CONFIG_ENABLE` (bit 0) is aliased with `BSR_CONFIG_SCHED_MODE` (bit 0). Comment says "same as SCHED_MODE" but `set_bsr_buffers()` and `set_scheduler_mode()` use them with conflicting semantics. |

---

### 8. `sw/cpp/include/driver/csr_map.hpp` (~353 lines) — **CSR ADDRESS MAP**
**Status:** Real, complete. Defines `SYSTOLIC_ROWS=14`, `SYSTOLIC_COLS=14`, `BLOCK_SIZE=14`. Includes all control, dimension, tile, buffer, scale, status, perf, result, DMA, and BSR registers.

---

### 9. `sw/cpp/include/compute/golden_models.hpp` (~317 lines) — **GOLDEN DECLS**
**Status:** Real, with mismatches

| # | Severity | Issue |
|---|---|---|
| 13 | **API mismatch** | Declares `add_residual(int32_t* main, int8_t* residual, ...)` but `.cpp` implements `add_residual_int8(int8_t* main, int8_t* residual, ...)` — **different name AND different first param type**. |
| 14 | **Missing impl** | Declares `conv2d_int8()` and `conv2d_bsr_int8()` — these are **NOT implemented** in `golden_models.cpp`. The `.cpp` has `conv2d_int8_simple()` and `conv2d_int8_im2col()` instead. |

---

### 10. `sw/cpp/src/compute/golden_models.cpp` (~936 lines) — **GOLDEN IMPL**
**Status:** Real, substantial

Working implementations:
- `matmul_int8()` (dense GEMM) ✓
- `bsr_matmul_int8()` (sparse BSR GEMM) ✓
- `relu_int8()`, `relu_int32()`, `relu6_int8()` ✓
- `requantize_int32_to_int8()` ✓
- `add_residual_int8()` ✓ (but name doesn't match header)
- `maxpool2d_int8()` ✓
- `avgpool_global_int8()` ✓
- `conv2d_int8_simple()` (6-loop direct) ✓
- `im2col_int8()` + `conv2d_int8_im2col()` ✓

---

### 11. `sw/cpp/include/utils/npy_loader.hpp` (~120 lines) — **NPY LOADER**
**Status:** Real

| # | Severity | Issue |
|---|---|---|
| 15 | Bug | NPY v2 files use 4-byte header length but code casts to `uint16_t`: `header_len = static_cast<uint16_t>(header_len32)` — **truncates headers >65535 bytes**. |
| 16 | Bug | No dtype validation — template type `T` reinterprets bytes regardless of file's actual dtype. `load_npy<int8_t>("float32_file.npy")` silently produces garbage. |

---

### 12. `sw/cpp/include/utils/performance_config.hpp` (~330 lines) — **PERF CONFIG**
**Status:** Real

| # | Severity | Issue |
|---|---|---|
| 17 | **Compile error** | Uses C99 designated initializers (`.name = "PYNQ-Z2"`) which require **C++20**, but CMakeLists.txt specifies `CMAKE_CXX_STANDARD 17`. |

---

### 13. `sw/cpp/include/utils/performance_counters.hpp` (~200 lines) + `performance_counters.cpp` (~200 lines) — **PERF COUNTERS**
**Status:** Real

| # | Severity | Issue |
|---|---|---|
| 18 | Bug | Default `num_pes = 256` (16×16) but actual array is 14×14 = **196 PEs**. Utilization calculations will be wrong. |

---

### 14. `sw/cpp/include/compute/resnet_inference.hpp` (~260 lines) — **INFERENCE HEADER**
**Status:** Declared, with syntax error

| # | Severity | Issue |
|---|---|---|
| 19 | **Compile error** | Duplicate `#endif` at end of file: `#endif  // RESNET_INFERENCE_HPP` followed by another `#endif // RESNET_INFERENCE_HPP`. |
| 20 | Design | `LayerWeights::weight_bsr` is a raw `BSRMatrix*` pointer with manual `delete` in destructor — fragile ownership, should be `unique_ptr`. |

---

### 15. `sw/cpp/src/compute/resnet_inference.cpp` (~300 lines) — **INFERENCE IMPL**
**Status:** STUB (entirely fake)

| # | Severity | Issue |
|---|---|---|
| 21 | **Stub** | `run_inference()` returns **hardcoded** dummy result (class 281 "tabby cat", 85% confidence). |
| 22 | **Stub** | `load_model()` creates dummy weights filled with `1`s — doesn't load `.npy` files. |
| 23 | **Stub** | `run_conv_layer()`, `run_basic_block()`, `run_fc_layer()` are **empty**. |
| 24 | **Stub** | `verify_accuracy()` always returns `true`. |
| 25 | **API mismatch** | Calls `resnet_accel::dense_to_bsr()` — no such function; should be `pack_to_bsr()`. |
| 26 | Bug | `run_inference_file()` creates dummy 224×224×3 image instead of loading the file. |

---

### 16. `sw/cpp/include/memory/memory_manager.hpp` (~1746 lines) — **MEMORY MGMT**
**Status:** Real, complete, with typo

Classes: `IMemoryAllocator` (interface), `SimulationAllocator`, `DevMemAllocator`, `DMABuffer`, `MemoryManager` — all fully implemented.

| # | Severity | Issue |
|---|---|---|
| 27 | **Compile error** | Line 865: `__sync_synchronze()` — **typo**, should be `__sync_synchronize()`. This is in `DevMemAllocator::cache_flush()` on non-ARM platforms. |
| 28 | Info | `create_memory_allocator()` references `ddr::ACT_BUFFER_BASE` and `ddr::REGION_SIZE` which are correctly defined in the same file. ✓ |

`MemoryManager::allocate_for_layer()` **does exist** — takes 4 size params. Note: `deploy_example.cpp` calls it differently (see issue #32).

---

### 17. `sw/cpp/include/utils/test_utils.hpp` (~120 lines) + `test_utils.cpp` (~15 lines)
**Status:** Real (utils), effectively empty (.cpp)

---

### 18. `sw/cpp/tests/deploy_example.cpp` (~140 lines) — **DEPLOYMENT EXAMPLE**
**Status:** WON'T COMPILE

| # | Severity | Issue |
|---|---|---|
| 29 | **API mismatch** | `BSRPacker packer; packer.dense_to_bsr(...)` — `BSRPacker` class doesn't exist. Functions are standalone: `pack_to_bsr()`. |
| 30 | **API mismatch** | `bsr.sparsity()` — no such member. Use `get_sparsity(bsr)`. |
| 31 | **API mismatch** | `packer.pack_for_hardware(bsr)` — no such function. Use `serialize_for_hardware(bsr)`. |
| 32 | **API mismatch** | `mem.allocate_for_layer(layer_name, act_size, wgt_size, bsr_size)` — actual signature is `allocate_for_layer(act_size, wgt_size, out_size, bsr_size)`. No `layer_name` param and missing `out_size`. |

---

### 19-25. Test Files (7 files) — **ALL STUBS**

| File | Lines | Status |
|---|---|---|
| `test_end_to_end.cpp` | ~180 | All test cases return `true` without testing. |
| `test_golden_models.cpp` | ~200 | All test cases return `true` without testing. |
| `test_bsr_packer.cpp` | ~180 | References non-existent `BSRPacker` class; all stubs. |
| `test_axi_transactions.cpp` | ~200 | All test cases return `true` without testing. |
| `test_performance.cpp` | ~280 | All stubs; uses `N = 16` (should be 14). |
| `test_stress.cpp` | ~260 | All 8 test functions return `true` without testing. |
| `test_platform_performance.cpp` | ~120 | Display-only, prints tables, no actual testing. |

`test_virtual_layer.cpp` (~494 lines) is the **only test with real logic**: defines `ComputeModelBackend` with actual GEMM computation, tests CSR read/write, and runs a virtual layer compute. But still uses `N=16`.

---

### 26-29. Verilator Testbenches (4 files) — **ALL STUBS**

| File | Lines | Status |
|---|---|---|
| `tb_accel_top.cpp` | ~270 | `#include "Vaccel_top.h"` commented out. All logic commented out. |
| `tb_mac8.cpp` | ~260 | `#include "Vmac8.h"` commented out. Prints "Not yet implemented". |
| `tb_pe.cpp` | ~290 | `#include "Vpe.h"` commented out. Prints "Not yet implemented". |
| `tb_systolic_array.cpp` | ~310 | Uses `N = 16` (should be 14). All logic commented out. |

---

## PYTHON FILES (`sw/ml_python/`)

### 30. `sw/ml_python/golden/gemm_bsr_int8.py` (~170 lines) — **BSR GEMM GOLDEN**
**Status:** Real, with logic issue

| # | Severity | Issue |
|---|---|---|
| 33 | **Logic bug** | In `gemm_bsr_int8()`, the dequantization loop at the end iterates `for local_row in range(block_h)` and applies the **same `C_tile_int32`** to output for each `local_row`. This accumulates the same tile result `block_h` times. The output mapping `C[:, col_start:col_end] += C_tile_fp32` is repeated for each `local_row`, resulting in incorrect scaling. |
| 34 | Import error | `__main__` block imports `from training.export_bsr import build_bsr_from_dense` and `from golden_models.gemm_int8 import gemm_int8_per_channel` — these require specific `sys.path` setup that isn't present (the script doesn't add parent dirs). |

---

### 31. `sw/ml_python/golden/golden_fc1_test.py` (~140 lines) — **FC1 GOLDEN TEST**
**Status:** Real

| # | Severity | Issue |
|---|---|---|
| 35 | Bug | Output save path uses `os.path.join(os.path.dirname(__file__), "../data/bsr_export/fc1/golden_output.npy")` — relative path will likely resolve incorrectly from `sw/golden/` since `../data/` doesn't exist there. Should use `project_root`. |

---

### 32. `sw/ml_python/golden_models/gemm_int8.py` (~85 lines) — **GEMM INT8 CHECKER**
**Status:** Real, but limited

- Reads CSV files from Verilator testbench output and verifies GEMM correctness.
- Works only with CSV format from `tb/integration/` (a directory that doesn't seem to exist in the repo).

---

### 33. `sw/ml_python/golden_models/golden_mac8.py` (~180 lines) — **MAC8 GOLDEN**
**Status:** Real, with undefined function

| # | Severity | Issue |
|---|---|---|
| 36 | **Runtime error** | `GoldenMAC8.cycle()` calls `sign_extend_8_to_32(a)` in bypass mode (line 107), but this function is **never defined** in the file. Only `interpret_8bit_as_signed()` and `to_s32()` are defined. Will crash with `NameError`. |

---

### 34. `sw/ml_python/host/accel.py` (~300 lines) — **PYNQ DRIVER**
**Status:** Real, with issues

| # | Severity | Issue |
|---|---|---|
| 37 | **Format mismatch** | `set_scale_factors()` converts scales to Q16.16 fixed-point (`int(Sa * 65536)`), but C++ driver uses IEEE 754 float bits (`float_to_bits()`). **Incompatible formats** — hardware expects one or the other, not both. |
| 38 | Bug | `__main__` example creates weights with block size 16×16 (`(num_blocks, 16, 16)`) but `BLOCK_SIZE = 14`. |
| 39 | Bug | `SimulatedCSR.write()` doesn't simulate the scheduler-mode writes correctly — `BSR_CONFIG` register isn't properly modeled. |

---

### 35. `sw/ml_python/host/axi_driver.py` (~250 lines) — **AXI DMA DRIVER**
**Status:** Real

- `AXIDriver` wraps `AXIMasterSim` for CSR writes, burst DMA, and status polling.
- `SparseMatrixLoader` is synthetic demo data, not real weight loading.
- Uses a **different CSR address map** (`DMA_LAYER=0x50`, `DMA_CTRL=0x51`, etc.) from the main `csr_map.py` (`DMA_LAYER=0x50`, but main CSR map has `DMA_SRC_ADDR=0x90`). These are two different register interfaces.

---

### 36. `sw/ml_python/host/axi_master_sim.py` (~340 lines) — **AXI MASTER SIM**
**Status:** Real, self-contained simulator

- `AXIMasterSim` simulates AXI4-Lite with write/read, burst, DMA FIFO, metrics.
- Valid CSR addresses hardcoded to `{0x50, 0x51, 0x52, 0x53, 0x54}` — writing to any other address returns `SLVERR`. This will reject writes to the main CSR registers (0x00, 0x04, etc.) used by `run_gemm_axi.py`.

| # | Severity | Issue |
|---|---|---|
| 40 | **Logic bug** | `AXIMasterSim.csr_valid_addrs = {0x50, 0x51, 0x52, 0x53, 0x54}`. But `run_gemm_axi.py`'s `configure_accelerator()` writes to 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x28, 0x30 — **ALL of these will SLVERR**. |

---

### 37. `sw/ml_python/host/memory.py` (~250 lines) — **MEMORY UTILS**
**Status:** Real, clean

- `DMABuffer` wrapper (PYNQ or simulation) ✓
- `BSRMatrix` class with `from_dense()`, `to_dense()`, `pack_for_dma()`, `sparsity()` ✓
- `pack_activations()` ✓
- No bugs found.

---

### 38. `sw/ml_python/host_axi/csr_map.py` (~280 lines) — **CSR MAP (PYTHON)**
**Status:** Real, with issues

| # | Severity | Issue |
|---|---|---|
| 41 | **Address error** | `DMA_CTRL = 0x51` and `DMA_COUNT = 0x52`, `DMA_STATUS = 0x53` — these are **NOT 4-byte aligned**. Every CSR register should be at a 4-byte boundary (0x00, 0x04, 0x08...). The C++ `csr_map.hpp` has `DMA_CTRL = 0x9C`. These appear to be byte addresses mixed with word addresses. |
| 42 | Missing | `STS_ERR_CRC`, `STS_ERR_ILLEGAL`, `CMD_WRITE`, `CMD_READ`, `SOF` are imported by `test_integration.py` from `host_uart.csr_map`, but the `host_uart/` directory **doesn't exist** in the workspace. |

---

### 39. `sw/ml_python/host_axi/run_gemm_axi.py` (~310 lines) — **AXI GEMM RUNNER**
**Status:** Real, with bugs

| # | Severity | Issue |
|---|---|---|
| 43 | **Type error** | `self.csr_write(0x28, pack_u32(1.0))` — `pack_u32()` takes an `int`, not `float`. `pack_u32(1.0)` returns bytes, but `csr_write()` passes bytes as the `value` arg to `axi.csr_write()` which expects `int`. |
| 44 | **CSR address mismatch** | `configure_accelerator()` writes DIMS to 0x00/0x04/0x08 and tiles to 0x0C/0x10/0x14 — but `csr_map.py` defines `DIMS_M=0x04`, `DIMS_N=0x08`, `DIMS_K=0x0C`, `TILES_Tm=0x10`. The 0x00 write goes to CTRL (START), not DIMS_M. |
| 45 | **Import error** | `from host_uart.csr_map import Config, pack_u32, unpack_u32` — `host_uart/` directory doesn't exist. Should import from `host_axi.csr_map`. |
| 46 | Bug | `read_results()` uses `np.unpackbits()` to convert uint32 to int8, which doesn't make sense — produces bit arrays, not int8 values. Should reinterpret bytes directly. |
| 47 | Logic | `csr_write(0x40, 0x1)` for START — but `CTRL` is at offset 0x00, not 0x40. 0x40 is `PERF_TOTAL`. |

---

### 40. `sw/ml_python/training/export_bsr.py` (~330 lines) — **BSR EXPORT (8×8)**
**Status:** Real, with bug

| # | Severity | Issue |
|---|---|---|
| 48 | **NameError** | `export_model_bsr()` references `quantizer.scale_matrix` — `quantizer` is **never defined** in this file. Will crash if `quantize=True`. |

- `build_bsr_from_dense()` ✓
- `save_bsr_binary()`, `save_bsr_binary_int8()`, `save_bsr_metadata()` ✓
- `load_sparse_model_from_npz()` ✓
- Uses 8×8 blocks (`layer_block_cfg` returns `(8,8)` for linear layers) — different from hardware's 14×14.

---

### 41. `sw/ml_python/training/export_bsr_14x14.py` (~803 lines) — **BSR EXPORT (14×14)**
**Status:** Real, most complete export script

- `build_bsr_14x14()`, `build_bsr_14x14_int8_direct()` ✓
- `save_bsr_binary_int8()`, `save_bsr_metadata()` ✓
- `export_layer_14x14()`, `export_int8_layer_14x14()` ✓
- `export_from_int8_dir()`, `export_model_14x14()` ✓
- Properly handles padding to 14×14 blocks.
- No significant bugs found.

---

### 42. `sw/ml_python/tests/test_integration.py` (~742 lines) — **INTEGRATION TESTS**
**Status:** Real tests, but won't run

| # | Severity | Issue |
|---|---|---|
| 49 | **Import error** | Imports from `host_uart.run_gemm`, `host_uart.uart_driver`, `host_uart.csr_map` — the `host_uart/` directory **doesn't exist** in this workspace. All tests will fail at import. |

The test logic itself is well-structured if the imports could be resolved: `TestGEMMConfig`, `TestMatrixGeneration`, `TestHostRSTilerUnit`, `TestTilingAlgorithms`, `TestProtocolCommunication`, `TestGEMMIntegration`, `TestErrorHandling`, `TestPerformance`, `TestStreamParser`, `TestCommandLineInterface`.

---

### 43. `sw/ml_python/tests/test_golden_models.py` (~140 lines) — **GOLDEN MODEL TESTS**
**Status:** Real, should work

- Uses `pytest` framework.
- Tests BSR INT8 GEMM with dense, sparse, and zero-column inputs.
- Tests BSR structure (indptr indexing, block count, sparsity calculation).
- Imports from `golden.gemm_bsr_int8` and `training.export_bsr` — requires correct `sys.path`.

---

## CROSS-CUTTING ISSUES

### 1. Block Size Mismatch (14 vs 16)
The hardware is 14×14 (196 PEs, fitting in 220 DSP48E1s). Multiple files use 16:

| File | Value Used | Should Be |
|---|---|---|
| `CMakeLists.txt` (`sw/cpp/`) | `BLOCK_SIZE=16`, `N_ROWS=16`, `N_COLS=16` | 14 |
| `performance_counters.hpp` (`sw/cpp/`) | `num_pes = 256` (16×16) | 196 |
| `test_performance.cpp` | `N = 16` | 14 |
| `test_stress.cpp` | Implicit 16×16 | 14 |
| `tb_systolic_array.cpp` | `N = 16` | 14 |
| `accel.py` (`sw/ml_python/host/`) `__main__` | Block size 16×16 in example | 14 |

### 2. Missing `host_uart/` Directory
Multiple files import from `host_uart.run_gemm`, `host_uart.uart_driver`, `host_uart.csr_map`:
- `sw/ml_python/tests/test_integration.py`
- `sw/ml_python/host_axi/run_gemm_axi.py`
- `sw/ml_python/tests/test_csr_pack.py`

This directory doesn't exist in the workspace. Either it was deleted, not committed, or lives in a different branch.

### 3. Scale Format Disagreement
- C++ `accelerator_driver.cpp`: Stores scales as **IEEE 754 float bits** via `float_to_bits()`.
- Python `accel.py`: Stores scales as **Q16.16 fixed-point** via `int(Sa * 65536)`.
- Hardware `output_accumulator.sv`: Uses Q16.16 in `quantize_relu` function.
- **Resolution: Python is correct, C++ is wrong.** The hardware expects Q16.16.

### 4. CSR Address Map Drift
- `csr.sv` (RTL): `DMA_SRC_ADDR = 0x90`, `DMA_CTRL = 0x9C` (authoritative)
- `csr_map.hpp` (C++): `DMA_SRC_ADDR = 0x90`, `DMA_CTRL = 0x9C` [CLEAN] (matches RTL)
- `accel.py` (Python): `DMA_SRC_ADDR = 0x90`, `DMA_CTRL = 0x9C` [CLEAN] (matches RTL)
- `csr_map.py` (`sw/ml_python/host_axi/`): `DMA_LAYER = 0x50`, `DMA_CTRL = 0x51` [BROKEN] (wrong, not 4-byte aligned)
- `axi_driver.py` (`sw/ml_python/host/`): Uses `csr_map.py` addresses [BROKEN] (wrong)
- `axi_master_sim.py` (`sw/ml_python/host/`): Valid addresses = `{0x50-0x54}` [BROKEN] (rejects real CSR writes)

**Resolution: `csr_map.py` and `axi_driver.py` must be updated to match RTL/C++ map.**

### 5. Buffer Read Latency Mismatch (RTL)
- `act_buffer.sv` and `wgt_buffer.sv`: Both have **2-cycle read latency** (BRAM + output register).
- `scheduler.sv`: Assumes **1-cycle latency** (PREPRIME parameter exists but doesn't fully compensate).
- `bsr_scheduler.sv`: Assumes **1-cycle latency** (prefetch in S_WAIT_WGT is off by 1).
- **Result:** Systolic array receives stale data on first cycle of each tile.

### 6. Activation/Weight Data Width (RTL → DMA)
- DMA buses are **64-bit** (AXI_DATA_W=64).
- Buffers store **112-bit** vectors (14 × 8-bit).
- `accel_top.sv` zero-extends 64→112 bits, meaning **only 8 of 14 values are loaded per DMA write**.
- **Resolution:** Need either multiple DMA writes per buffer entry, or a wider AXI bus.

---

## PRIORITY FIX ORDER

### Tier 1: Won't Compile / Data Corruption (Fix First)
1. **R37-R41** `accel_top_dual_clk.sv` — Fix syntax errors or **delete** (use `accel_top.sv` instead)
2. **R34** `accel_top.sv` — Fix 64→112 bit zero-extension (needs proper DMA packing logic)
3. **R24,R27** `act_dma.sv`/`bsr_dma.sv` — Add `act_we <= 0` / `*_we <= 0` default in all states
4. **R28** `bsr_dma.sv` — Fix `words_remaining = total_blocks * 25` (not 8) for weight phase
5. **R31** `meta_decode.sv` — Add extra wait cycle for 2-cycle BRAM latency
6. **R30** `axi_lite_slave.sv` — Fix simultaneous R/W address conflict
7. **C++ CMakeLists.txt** — Fix source list, change BLOCK_SIZE to 14

### Tier 2: Logic Bugs (Fix Next)
8. **R7-R9** `bsr_scheduler.sv` — Fix load cycle count, wait duration, stream duration
9. **R5** `output_accumulator.sv` — Use full 32-bit scale in quantize_relu
10. **R11-R12** `scheduler.sv` — Fix width constants in ceil_div and mask init
11. **R14** `csr.sv` — Replace combinational clock gating with latch-based
12. **R4** `wgt_buffer.sv` — Document 2-cycle latency; update scheduler to match
13. **R18-R19** `block_reorder_buffer.sv` — Fix last-beat visibility and count==0 edge case
14. **Python** `golden_mac8.py` — Add missing `sign_extend_8_to_32()` function
15. **Python** `csr_map.py` — Update addresses to match RTL (4-byte aligned)

### Tier 3: Stubs / Incomplete (Fill In or Delete)
16. Delete `accel_top_legacy.sv` — it's broken and misleading
17. Delete or fix `accel_top_dual_clk.sv` — if dual-clock isn't needed
18. Fix `resnet_inference.cpp` — either implement or clearly mark as demo stub
19. Fix `deploy_example.cpp` — update API calls to match real `bsr_packer.hpp`
20. Fill in or remove C++ test stubs (7 test files)
21. Fill in or remove Verilator testbench stubs (4 files)
22. Resolve `host_uart/` missing directory — update imports to `host_axi/`
