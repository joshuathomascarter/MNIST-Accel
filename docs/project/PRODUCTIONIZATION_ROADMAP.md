# ACCEL-v1 Productionization Roadmap

**Author**: Joshua Carter  
**Date**: November 19, 2025  
**Status**: Gap Analysis & Implementation Plan

## Executive Summary

This document addresses critical gaps identified in the ACCEL-v1 sparse matrix accelerator project and provides a systematic roadmap to production-grade hardware.

**Current State**: Functional RTL simulation with incomplete sparse path, no hardware validation  
**Target State**: Synthesizable, FPGA-verified accelerator with formal verification and multi-layer support

---

## Critical Gaps Identified

### 1. Incomplete Sparse Path ❌ **BLOCKING**

**Issues:**
- `block_reorder_buffer.sv`: Missing sorting algorithm (line 58: "TODO: Implement sorting logic")
- `accel_top.v`: Stubbed multi-layer buffer (line 653-654)
- `top_sparse.v`: Disconnected block storage (line 183, 221)

**Impact:** Cannot claim "sparse acceleration" when core sorting is unimplemented

**Resolution Plan:**
- **Week 1**: Implement insertion sort in block_reorder_buffer.sv
- **Week 1**: Wire block_reorder_buffer into scheduler pipeline
- **Week 2**: Add multi-layer metadata buffering
- **Week 2**: Full end-to-end sparse matmul test (DMA → sorted blocks → systolic)

---

### 2. No Hardware Validation ❌ **CRITICAL**

**Issues:**
- Zero synthesis reports (no LUT/BRAM/DSP counts)
- No timing closure data (target: 100 MHz)
- No FPGA bitstream generation
- No power measurements

**Impact:** Simulation-only claims don't hold up in industry reviews

**Resolution Plan:**

#### Synthesis Flow (Week 3-4)
```tcl
# Target: Xilinx Artix-7 XC7A100T-1CSG324C
# Tools: Vivado 2023.2

create_project accel_v1 ./vivado_proj -part xc7a100tcsg324-1
add_files -fileset sources_1 [glob rtl/**/*.v rtl/**/*.sv]
set_property top accel_top [current_fileset]

# Constraints
create_clock -period 10.0 [get_ports clk]  # 100 MHz
set_input_delay -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]

# Synthesize
synth_design -top accel_top -part xc7a100tcsg324-1
report_utilization -file reports/utilization.rpt
report_timing_summary -file reports/timing.rpt
report_power -file reports/power.rpt

# Implementation
opt_design
place_design
route_design
```

**Expected Results (Artix-7 XC7A100T):**
| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUTs | ~15,000 | 63,400 | 24% |
| FFs | ~12,000 | 126,800 | 9% |
| BRAM (36Kb) | ~40 | 135 | 30% |
| DSPs | 20 | 240 | 8% |

**Timing Goals:**
- Setup slack: > 0.5 ns @ 100 MHz (10 ns period)
- Hold slack: > 0.1 ns
- Worst path: systolic_array → accumulator

**Power Budget:**
- Dynamic: < 1.5 W
- Static: < 0.3 W
- **Total**: < 2 W @ 100 MHz

---

### 3. UART Bottleneck ❌ **PERFORMANCE**

**Current State:**
- UART: 115,200 baud = **14.4 KB/s**
- Systolic array: 2×2 PEs × 100 MHz = **400M MACs/sec**
- **Utilization**: 0.3% (I/O starved!)

**Root Cause Analysis:**
```
MNIST FC1 layer:
- Weights: 128 × 9216 × INT8 = 1.18 MB
- UART transfer time: 1,180,000 bytes / 14,400 bytes/s = **82 seconds**
- Compute time: 128 × 9216 / 400M = **0.003 seconds**
- **Speedup lost**: 27,333× slower than compute!
```

**Resolution Plan:**

#### Phase 1: AXI4 DMA (Week 5-6)
Replace UART with AXI4 memory-mapped DMA:
- Bandwidth: 32-bit @ 100 MHz = **400 MB/s**
- MNIST FC1 load time: **3 ms** (vs 82 sec)
- **Speedup**: 27,000× faster

```systemverilog
// rtl/dma/axi_dma_master.sv
module axi_dma_master #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter BURST_LEN = 256  // 1 KB bursts
)(
    // AXI4 Master interface
    output logic [ADDR_WIDTH-1:0] m_axi_araddr,
    output logic [7:0]            m_axi_arlen,
    output logic [2:0]            m_axi_arsize,
    output logic [1:0]            m_axi_arburst,
    output logic                  m_axi_arvalid,
    input  logic                  m_axi_arready,
    input  logic [DATA_WIDTH-1:0] m_axi_rdata,
    input  logic                  m_axi_rvalid,
    output logic                  m_axi_rready,
    
    // Internal BRAM write
    output logic         bram_we,
    output logic [15:0]  bram_waddr,
    output logic [31:0]  bram_wdata
);
    // Burst transfer FSM: IDLE → READ_REQ → READ_DATA → WRITE_BRAM
endmodule
```

#### Phase 2: PCIe Integration (Week 7-8)
For desktop FPGA deployment:
- **Gen3 x4**: 4 GB/s theoretical
- **Realistic**: 2 GB/s sustained
- **MNIST FC1**: <1 ms load time

---

### 4. No Formal Verification ❌ **QUALITY**

**Current State:**
- Zero SVA assertions
- No model checking
- No constraint randomization

**Industry Standard** (AMD/Tenstorrent):
- **Coverage**: >95% functional coverage
- **Assertions**: 10-20 per module
- **Formal**: Property checking on FSMs

**Resolution Plan:**

#### SVA Assertions (Week 9)

```systemverilog
// rtl/systolic/systolic_array.v
module systolic_array #(...) (...);
    // Existing logic...
    
    // === FORMAL VERIFICATION ASSERTIONS ===
    
    // Property: Enable must remain stable during computation
    property p_enable_stable;
        @(posedge clk) disable iff (!rst_n)
        en && !clr |=> $stable(en) throughout (c_out_valid[->1]);
    endproperty
    assert_enable_stable: assert property (p_enable_stable)
        else $error("Enable changed during computation");
    
    // Property: Clear resets accumulator
    property p_clear_resets;
        @(posedge clk) disable iff (!rst_n)
        clr |=> (c_out == 0);
    endproperty
    assert_clear_resets: assert property (p_clear_resets)
        else $error("Clear did not reset accumulator");
    
    // Property: Output valid follows input valid by PE_PIPELINE_DEPTH cycles
    property p_valid_delay;
        @(posedge clk) disable iff (!rst_n)
        en && !clr |-> ##PE_PIPELINE_DEPTH c_out_valid;
    endproperty
    assert_valid_delay: assert property (p_valid_delay);
    
    // Constraint: Row/col enables must not exceed array dimensions
    a_row_enable_bounds: assert property (
        @(posedge clk) $countones(en_mask_row) <= N_ROWS
    ) else $fatal("Row enable mask exceeds array size");
    
    a_col_enable_bounds: assert property (
        @(posedge clk) $countones(en_mask_col) <= N_COLS
    ) else $fatal("Column enable mask exceeds array size");
    
    // Coverage: Track enable patterns
    covergroup cg_enable_patterns @(posedge clk);
        cp_row_mask: coverpoint en_mask_row {
            bins all_on = {2'b11};
            bins partial[] = {2'b01, 2'b10};
            bins all_off = {2'b00};
        }
        cp_col_mask: coverpoint en_mask_col;
        cross cp_row_mask, cp_col_mask;
    endgroup
    cg_enable_patterns cg_inst = new();
    
endmodule
```

#### Formal Model Checking (Week 10)
```tcl
# Cadence JasperGold / Synopsys VC Formal
analyze -sv rtl/systolic/systolic_array.v
elaborate -top systolic_array
clock clk
reset -expression !rst_n

# Deadlock freedom
prove -property {en && !clr |-> ##[1:$] c_out_valid}

# Safety: No overflow
prove -property {c_out <= MAX_ACCUMULATOR_VALUE}

# Liveness: System responds
prove -property {start |-> ##[1:100] done}
```

---

### 5. Single-Layer Focus ❌ **SCALABILITY**

**Current State:**
- `bsr_scheduler.sv`: Processes ONE layer
- `block_reorder_buffer.sv`: Sorting TODO → can't handle multi-layer
- No layer-to-layer pipelining

**Real CNNs** (ResNet-18):
- **18 layers** (16 CONV + 2 FC)
- Need: Layer buffering, pipeline stages, layer switching

**Resolution Plan:**

#### Multi-Layer Buffer (Week 11-12)

```systemverilog
// rtl/control/multi_layer_buffer.sv
module multi_layer_buffer #(
    parameter MAX_LAYERS = 8,
    parameter MAX_BLOCKS_PER_LAYER = 65536
)(
    input  logic clk,
    input  logic rst_n,
    
    // Layer configuration from DMA
    input  logic [2:0]   cfg_layer_id,
    input  logic         cfg_layer_start,
    input  logic [31:0]  cfg_num_blocks,
    
    // Metadata write (from DMA)
    input  logic         wr_valid,
    input  logic [2:0]   wr_layer,
    input  logic [31:0]  wr_row_ptr,
    input  logic [15:0]  wr_col_idx,
    
    // Metadata read (to scheduler)
    input  logic [2:0]   rd_layer,
    input  logic [15:0]  rd_addr,
    output logic [31:0]  rd_row_ptr,
    output logic [15:0]  rd_col_idx,
    output logic         rd_valid,
    
    // Layer switching control
    input  logic         switch_layer,
    output logic         switch_ready
);

    // Per-layer metadata storage
    typedef struct packed {
        logic [31:0] num_blocks;
        logic [31:0] row_ptr_base;  // Start address in shared BRAM
        logic [31:0] col_idx_base;
        logic        valid;
    } layer_meta_t;
    
    layer_meta_t layer_table [0:MAX_LAYERS-1];
    
    // Shared BRAM for all layers (partitioned)
    logic [31:0] row_ptr_mem [0:MAX_BLOCKS_PER_LAYER-1];
    logic [15:0] col_idx_mem [0:MAX_BLOCKS_PER_LAYER-1];
    
    // Write logic: Partition BRAM by layer
    always_ff @(posedge clk) begin
        if (wr_valid) begin
            automatic logic [31:0] abs_addr = layer_table[wr_layer].row_ptr_base + wr_addr;
            row_ptr_mem[abs_addr] <= wr_row_ptr;
            col_idx_mem[abs_addr] <= wr_col_idx;
        end
    end
    
    // Read logic: Translate layer-relative to absolute address
    always_comb begin
        automatic logic [31:0] abs_addr = layer_table[rd_layer].row_ptr_base + rd_addr;
        rd_row_ptr = row_ptr_mem[abs_addr];
        rd_col_idx = col_idx_mem[abs_addr];
        rd_valid = layer_table[rd_layer].valid;
    end
    
    // Layer configuration
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_LAYERS; i++) begin
                layer_table[i].valid <= 1'b0;
            end
        end else if (cfg_layer_start) begin
            layer_table[cfg_layer_id].num_blocks <= cfg_num_blocks;
            layer_table[cfg_layer_id].valid <= 1'b1;
            // Allocate BRAM partition (cumulative)
            if (cfg_layer_id == 0) begin
                layer_table[0].row_ptr_base <= 0;
            end else begin
                layer_table[cfg_layer_id].row_ptr_base <= 
                    layer_table[cfg_layer_id-1].row_ptr_base + 
                    layer_table[cfg_layer_id-1].num_blocks;
            end
        end
    end
    
    // Layer switch ready: Wait for current computation to finish
    always_comb begin
        switch_ready = !scheduler_busy && layer_table[rd_layer+1].valid;
    end
    
endmodule
```

---

## Implementation Timeline

### Phase 1: Core Sparse Path (Weeks 1-2) 🔥 **URGENT**
- [ ] Implement insertion sort in `block_reorder_buffer.sv`
- [ ] Wire block_reorder_buffer into scheduler
- [ ] End-to-end sparse matmul test (MNIST FC1 layer)
- [ ] **Milestone**: Demo 9× speedup vs dense on 90% sparse matrix

### Phase 2: Hardware Validation (Weeks 3-4)
- [ ] Vivado synthesis flow setup
- [ ] Generate utilization/timing/power reports
- [ ] FPGA bitstream generation (Artix-7)
- [ ] On-board testing with UART (baseline)
- [ ] **Milestone**: FPGA demo @ 100 MHz, <2W power

### Phase 3: I/O Acceleration (Weeks 5-6)
- [ ] AXI4 DMA master implementation
- [ ] Replace UART with AXI memory-mapped transfers
- [ ] Benchmark: MNIST FC1 load time <5ms
- [ ] **Milestone**: Achieve >10% systolic utilization

### Phase 4: Formal Verification (Weeks 9-10)
- [ ] Add SVA assertions to all modules (10+ per module)
- [ ] Functional coverage groups (target: >90%)
- [ ] JasperGold/VC Formal model checking
- [ ] **Milestone**: Zero assertion failures in 1M cycle run

### Phase 5: Multi-Layer Support (Weeks 11-12)
- [ ] Implement `multi_layer_buffer.sv`
- [ ] Layer switching protocol in scheduler
- [ ] ResNet-18 layer sequence test
- [ ] **Milestone**: Run 3-layer CNN end-to-end

---

## Success Metrics

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| **Sparse Path Complete** | ❌ 60% | ✅ 100% | All TODOs resolved |
| **FPGA Synthesis** | ❌ None | ✅ Pass @ 100MHz | Vivado timing report |
| **Resource Usage** | ❌ Unknown | ✅ <30% LUTs | Utilization report |
| **Power** | ❌ Unknown | ✅ <2W | FPGA measurements |
| **I/O Bandwidth** | ❌ 14.4 KB/s | ✅ >10 MB/s | AXI DMA throughput |
| **Systolic Utilization** | ❌ 0.3% | ✅ >50% | Perf counters |
| **Assertions** | ❌ 0 | ✅ >100 | SVA count |
| **Coverage** | ❌ 0% | ✅ >90% | Functional cov |
| **Multi-Layer** | ❌ 1 layer | ✅ 8 layers | ResNet test |

---

## Risk Mitigation

### Technical Risks

1. **Timing Closure Failure @ 100 MHz**
   - Mitigation: Insert pipeline stages in critical paths (systolic array, scheduler)
   - Fallback: Reduce clock to 50 MHz (still 200M MACs/sec)

2. **BRAM Shortage (Artix-7 has 135 BRAMs)**
   - Mitigation: Use external DDR3 for weight storage, BRAM for cache only
   - Calculation: 1.18 MB weights = 327 BRAMs (EXCEEDS!)
   - Solution: 16 KB L1 cache (4 BRAMs) + DDR3 backing store

3. **AXI DMA Complexity**
   - Mitigation: Use Xilinx AXI DMA IP core (pre-verified)
   - Alternative: Simple AXI lite master (slower but simpler)

### Schedule Risks

1. **Underestimated Synthesis Debug Time**
   - Buffer: Add 1 week contingency per phase
   - Early start: Begin synthesis in Week 2 (parallel with sparse path)

2. **Formal Tools Learning Curve**
   - Mitigation: Start with basic assertions (Week 1), advanced formal later
   - Training: Allocate 2 days for JasperGold tutorial

---

## Deliverables

### Week 4 Checkpoint
1. `SYNTHESIS_REPORT.md` - Full Vivado utilization/timing/power data
2. `block_reorder_buffer.sv` - Fully implemented sorting (no TODOs)
3. `test_sparse_end2end.sv` - 10,000-cycle test passing
4. FPGA bitstream + demo video

### Week 8 Checkpoint
1. `axi_dma_master.sv` - AXI4 DMA implementation
2. Bandwidth benchmarks: >100 MB/s demonstrated
3. Systolic utilization: >25% on real workload

### Week 12 Final
1. `multi_layer_buffer.sv` - 8-layer support
2. `FORMAL_VERIFICATION_REPORT.md` - 100+ assertions, >90% coverage
3. ResNet-18 layer execution trace
4. Production-ready RTL (zero TODOs, full documentation)

---

## Conclusion

**Current Assessment**: Promising architecture, incomplete implementation (academic project stage)

**Path to Production**:
1. ✅ **Fix TODOs** (2 weeks) → Credible sparse claim
2. ✅ **Hardware validation** (2 weeks) → Real performance data
3. ✅ **I/O acceleration** (2 weeks) → Usable system
4. ✅ **Formal verification** (2 weeks) → Industry quality
5. ✅ **Multi-layer** (2 weeks) → Real CNN deployment

**Timeline**: 12 weeks to production-grade accelerator  
**Effort**: 1 engineer full-time

**Bottom Line**: This project has solid bones but needs muscle. The critiques are valid - let's fix them systematically.

---

**Next Action**: Implement block sorting (highest priority blocker)

