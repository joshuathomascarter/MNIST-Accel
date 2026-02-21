# =============================================================================
# POWER_ANALYSIS.md — Clock Gating and Power Optimization Guide
# =============================================================================
# Author: Joshua Carter
# Date: November 19, 2025
# Target: <1.5W @ 100 MHz (vs current estimated ~2W)
#
# This guide addresses:
#   1. Clock gating for idle modules
#   2. Vivado power analysis flow
#   3. Dynamic vs static power breakdown
#   4. Optimization techniques (operand isolation, multi-Vt cells)
# =============================================================================

## Current Power Estimate (No Optimization)

Based on Artix-7 XC7A100T-1 @ 100 MHz:

| Component | Dynamic Power | Static Power | Total |
|-----------|---------------|--------------|-------|
| Systolic Array (20 DSPs) | ~0.6W | ~0.05W | 0.65W |
| BRAM (40×36Kb) | ~0.4W | ~0.03W | 0.43W |
| Logic (15K LUTs) | ~0.5W | ~0.02W | 0.52W |
| I/O (AXI) | ~0.3W | ~0.01W | 0.31W |
| Clock Network | ~0.1W | — | 0.1W |
| **TOTAL** | **~1.9W** | **~0.11W** | **~2.01W** |

**Problem**: Exceeds 1.5W target by 34%

**Root Cause**: Systolic array and BRAMs run continuously, even when idle

---

## Solution 1: Clock Gating

### Concept
Disable clock to idle modules → **reduce dynamic power by 40-60%**

```systemverilog
// Example: Gate systolic array clock when not computing
logic systolic_clk_en;
logic systolic_clk_gated;

assign systolic_clk_en = (state == COMPUTE) || (state == DRAIN);

// BUFGCE: Clock gating primitive (Xilinx)
BUFGCE systolic_clk_gate (
    .I(clk),           // Input clock
    .CE(systolic_clk_en),  // Clock enable
    .O(systolic_clk_gated) // Gated output
);

systolic_array u_systolic (
    .clk(systolic_clk_gated),  // Use gated clock
    .rst_n(rst_n),
    // ... other ports
);
```

### Implementation Plan

#### 1. Identify Idle Conditions

| Module | Idle Condition | Gating Signal |
|--------|----------------|---------------|
| `systolic_array` | No rows enabled (`row_enable == 0`) | `systolic_clk_en` |
| `act_buffer` | No reads pending (`rd_en == 0`) | `act_buf_clk_en` |
| `wgt_buffer` | No reads pending (`rd_en == 0`) | `wgt_buf_clk_en` |
| `bsr_dma` | Transfer complete (`state == IDLE`) | `dma_clk_en` |
| `meta_decode` | No cache access (`cache_ren == 0`) | `cache_clk_en` |

#### 2. Add Clock Gating to `accel_top.sv`

```systemverilog
// =============================================================================
// Clock Gating Logic
// =============================================================================
logic systolic_clk_en, systolic_clk;
logic act_buf_clk_en, act_buf_clk;
logic wgt_buf_clk_en, wgt_buf_clk;

// Systolic enable: Active when any row is enabled
assign systolic_clk_en = (row_enable != 2'b00) || (state == COMPUTE);

// Buffer enables: Active when scheduler is reading
assign act_buf_clk_en = scheduler_rd_en;
assign wgt_buf_clk_en = scheduler_rd_en;

// Clock gating cells (Xilinx BUFGCE primitive)
BUFGCE u_systolic_gate (
    .I(clk),
    .CE(systolic_clk_en),
    .O(systolic_clk)
);

BUFGCE u_act_buf_gate (
    .I(clk),
    .CE(act_buf_clk_en),
    .O(act_buf_clk)
);

BUFGCE u_wgt_buf_gate (
    .I(clk),
    .CE(wgt_buf_clk_en),
    .O(wgt_buf_clk)
);

// Instantiate modules with gated clocks
systolic_array u_systolic (
    .clk(systolic_clk),  // Gated clock
    // ...
);

act_buffer u_act_buf (
    .clk(act_buf_clk),  // Gated clock
    // ...
);
```

#### 3. Expected Power Reduction

| Component | Before (W) | After (W) | Savings |
|-----------|------------|-----------|---------|
| Systolic (idle 50% time) | 0.60 | 0.30 | **50%** |
| Act Buffer (idle 70% time) | 0.20 | 0.06 | **70%** |
| Wgt Buffer (idle 70% time) | 0.20 | 0.06 | **70%** |
| **Total Dynamic** | 1.9W | **1.1W** | **42%** |

**New Total: ~1.2W** (within 1.5W target )

---

## Solution 2: Vivado Power Analysis

### Step 1: Generate SAIF File (Switching Activity)

```tcl
# Run simulation with activity logging
vsim -c work.tb_accel_top -do "
    log -r /*
    run 100us
    coverage save -onexit coverage.ucdb
    quit
"

# Convert to SAIF (Switching Activity Interchange Format)
vcd2saif -input trace.vcd -output activity.saif
```

### Step 2: Vivado Power Report

Add to `synthesize_vivado.tcl`:

```tcl
# =============================================================================
# Power Analysis with SAIF
# =============================================================================
puts "========================================="
puts "Running Power Analysis"
puts "========================================="

# Read switching activity
read_saif -strip_path tb_accel_top/dut activity.saif

# Set operating conditions
set_operating_conditions -voltage 1.0 -grade commercial -model typical

# Generate power report
report_power \
    -file reports/power_detailed.rpt \
    -hierarchical \
    -verbose

# Extract key metrics
set total_power [get_property TOTAL_POWER [current_design]]
set dynamic_power [get_property DYNAMIC_POWER [current_design]]
set static_power [get_property STATIC_POWER [current_design]]

puts "Total Power:   [format %.3f $total_power] W"
puts "Dynamic Power: [format %.3f $dynamic_power] W"
puts "Static Power:  [format %.3f $static_power] W"

# Check against target
if {$total_power > 1.5} {
    puts "WARNING: Power exceeds 1.5W target!"
    puts "Consider:"
    puts "  - Add clock gating to idle modules"
    puts "  - Use Multi-Vt cells (HVT for non-critical paths)"
    puts "  - Reduce clock frequency to 50 MHz"
}
```

### Step 3: Hierarchical Power Breakdown

```tcl
# Generate per-module power report
report_power -hierarchical -levels 3 -file reports/power_hierarchy.rpt
```

Example output:
```
Module                        | Dynamic (W) | Static (W) | Total (W)
------------------------------|-------------|------------|----------
accel_top                     | 1.12        | 0.11       | 1.23
  systolic_array              | 0.32        | 0.05       | 0.37
  act_buffer                  | 0.08        | 0.02       | 0.10
  wgt_buffer                  | 0.08        | 0.02       | 0.10
  bsr_scheduler               | 0.15        | 0.01       | 0.16
  bsr_dma                     | 0.25        | 0.01       | 0.26
  axi_lite                     | 0.02        | 0.00       | 0.02
```

---

## Solution 3: Advanced Techniques

### 1. Operand Isolation

Prevent unnecessary switching in datapath when module is idle:

```systemverilog
// Isolate MAC inputs when not computing
logic [7:0] mac_a_gated, mac_b_gated;

assign mac_a_gated = compute_en ? mac_a : 8'h00;
assign mac_b_gated = compute_en ? mac_b : 8'h00;

mac8 u_mac (
    .a(mac_a_gated),  // Isolated input
    .b(mac_b_gated),
    .out(mac_out)
);
```

**Benefit**: Reduces toggle rate → **5-10% dynamic power savings**

### 2. Multi-Vt Cell Optimization

Use high-Vt (HVT) cells for non-critical paths → **reduce leakage by 20%**

```tcl
# Vivado synthesis directive
set_property IOSTANDARD LVCMOS18 [get_ports *]
set_property DRIVE 12 [get_ports *]

# Use HVT cells for registers in control logic
set_property CELL_TYPE HVT [get_cells -hierarchical -filter {REF_NAME =~ FD*}]
```

### 3. Memory Power Gating

BRAM sleep mode when not accessed:

```systemverilog
// BRAM with sleep enable
bram_controller #(
    .DATA_WIDTH(128),
    .ADDR_WIDTH(10)
) u_cache (
    .clk(clk),
    .sleep(cache_sleep),  // Power down when idle
    .ren(cache_ren),
    .wen(cache_wen),
    // ...
);

assign cache_sleep = (state == IDLE) && !cache_ren && !cache_wen;
```

**Benefit**: **30-50% BRAM static power reduction**

---

## Validation Checklist

### Pre-Silicon
- [ ] Run Vivado power analysis with SAIF activity
- [ ] Verify total power <1.5W @ 100 MHz
- [ ] Check dynamic/static breakdown (target: 80%/20%)
- [ ] Simulate clock gating correctness (no glitches)
- [ ] Verify isolated operands don't break datapath

### Post-Silicon (FPGA)
- [ ] Measure actual power with ammeter (VCC rail)
- [ ] Thermal imaging (ensure <60°C junction temp)
- [ ] Compare measured vs projected (±10% tolerance)
- [ ] Stress test: 100% utilization for 10 minutes

---

## Summary Table

| Optimization | Power Savings | Effort | Risk |
|--------------|---------------|--------|------|
| Clock gating | **40-50%** | Medium | Low (well-tested) |
| Operand isolation | 5-10% | Low | Very low |
| Multi-Vt cells | 10-20% | Low | Very low (tool-driven) |
| BRAM sleep | 5-15% | Medium | Medium (timing check) |
| Reduce to 50 MHz | **50%** | Very low | High (performance hit) |

**Recommended Strategy:**
1. Implement clock gating (immediate 40% savings → 1.2W)
2. Add operand isolation (incremental 5% → 1.14W)
3. Run Vivado power analysis to validate
4. Only reduce frequency if still >1.5W

---

## Next Steps

1. **Add clock gating**: Modify `accel_top.sv` per examples above
2. **Simulate**: Verify gated clocks don't introduce functional bugs
3. **Synthesize**: Run Vivado with power analysis
4. **Measure**: Deploy to FPGA and measure actual power
5. **Document**: Update PRODUCTIONIZATION_ROADMAP.md with results

---

## References

- [Xilinx UG907: Clock Resources](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2023_2/ug907-vivado-power-methodology.pdf)
- [UG835: Vivado Power Analysis](https://docs.xilinx.com/v/u/en-US/ug835-vivado-power-analysis-optimization)
- [Clock Gating Best Practices](https://www.synopsys.com/glossary/what-is-clock-gating.html)
