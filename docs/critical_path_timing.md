# Critical Path Analysis and Pipeline Planning
**Timing Analysis for 8×8 Systolic Array @ 100 MHz**

Date: November 25, 2025
Target: Zynq-7020, Speed Grade -1
Clock Period: 10 ns (100 MHz)

---

## 1. What is a Critical Path?

**Critical path = the longest delay path in your circuit.**

It determines the maximum clock frequency:
```
f_max = 1 / T_critical

If T_critical = 10 ns, then f_max = 100 MHz
If T_critical = 15 ns, then f_max = 67 MHz
```

**Your goal:** Keep T_critical < 10 ns to run at 100 MHz.

---

## 2. Critical Path in PE (Processing Element)

### 2.1 PE Datapath

Let's trace the path from inputs to outputs in one clock cycle:

```
        ┌─────────────────────────────────────┐
        │          Processing Element          │
        │                                      │
 a_in───┼──┐                                  │
  8b    │  │                                  │
        │  │    ┌──────────┐                  │
        │  └───►│          │                  │
        │       │  8×8     │   16-bit         │
weight──┼──────►│  Mult    ├────────┐         │
 (reg)  │       │          │        │         │
  8b    │       └──────────┘        │         │
        │                           │         │
        │       ┌──────────┐        │         │
        │       │  32-bit  │◄───────┘         │
        │   ┌──►│  Adder   │                  │
        │   │   └────┬─────┘                  │
        │   │        │                        │
        │   │   ┌────▼────┐                   │
        │   └───┤  32-bit │                   │
        │       │   Reg   │                   │
        │       │  (acc)  │                   │
        │       └────┬────┘                   │
        │            │                        │
        │            └───────────────────────►│acc_out
        │                                     │  32b
        │                                     │
        └─────────────────────────────────────┘

Critical path: a_in → Mult → Adder → acc register
```

### 2.2 Delay Breakdown

**Using Xilinx 7-series timing models (28nm, -1 speed grade):**

```
Component               Delay (ns)  Notes
─────────────────────────────────────────────────────────
Input routing           0.5         FPGA routing delay
8×8 Multiplier          2.5         Combinational logic
16→32 bit extension     0.2         Sign extension
32-bit Adder            2.8         Ripple-carry or fast
Register setup time     0.4         FF setup (Tsu)
Output routing          0.3         To next stage
─────────────────────────────────────────────────────────
TOTAL (worst case)      6.7 ns
```

**Wait, this is under 10 ns!** ✓ Should meet timing at 100 MHz.

**BUT:** This assumes you're using PURE combinational logic. Let's check if DSP slices are better...

---

## 3. Using DSP48E1 Slices

### 3.1 DSP48E1 Architecture

Zynq-7000 has **DSP48E1** slices optimized for multiply-accumulate:

```
         ┌─────────────────────────────────────────┐
         │           DSP48E1 Slice                  │
         │                                          │
    A────┼─►┌──────┐                               │
   25b   │  │  Pre │                               │
         │  │ Add  │     ┌──────────┐              │
    D────┼─►└──┬───┘     │          │              │
   25b   │     │         │  25×18   │              │
         │     └────────►│   Mult   │              │
    B────┼───────────────►│          │   48-bit    │
   18b   │               └────┬─────┘   product    │
         │                    │                     │
         │               ┌────▼─────┐               │
         │               │  48-bit  │               │
    C────┼──────────────►│   ALU    │               │
   48b   │               │ (P+M+C)  │               │
         │               └────┬─────┘               │
         │                    │                     │
         │               ┌────▼────┐                │
         │               │  48-bit │                │
         │               │   Reg   │                │
         │               │   (P)   │                │
         │               └────┬────┘                │
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              ▼
                           P output
                            48-bit
```

**Key features:**
- Hardened multiply-accumulate
- Pipelined (up to 4 stages)
- Dedicated routing (fast!)

### 3.2 DSP48E1 Timing

**Operating modes:**

**Mode 1: Combinational (no pipeline)**
```
A, B inputs → Multiply → Add → P output
Delay: 4.5 ns (from datasheet)

Can run at: 1/4.5ns = 222 MHz max
```

**Mode 2: 1-stage pipeline (register after multiply)**
```
Cycle 0: A, B → Multiply → [REG]
Cycle 1: [REG] → Add → P output

Critical path: 2.5 ns (just the adder stage)
Can run at: 1/2.5ns = 400 MHz!
```

**Mode 3: 2-stage pipeline (register after mult + add)**
```
Cycle 0: A, B → Multiply → [REG1]
Cycle 1: [REG1] → Add → [REG2]
Cycle 2: [REG2] → P output

Critical path: ~2 ns per stage
Can run at: 500 MHz+ (overkill for us)
```

### 3.3 Recommendation for Your Design

**Use DSP48E1 in Mode 1 (combinational) for now:**
- Latency: 1 cycle (simple!)
- Timing: 4.5 ns < 10 ns (plenty of margin)
- No pipeline stalls to manage

**Later optimization (if needed):**
- Switch to Mode 2 (1-stage pipeline)
- Increases throughput potential
- Adds 1 cycle latency (but doesn't affect total time much)

---

## 4. Critical Paths in Other Modules

### 4.1 Sparse Controller FSM

**State machine critical path:**

```
Current_state → Next_state_logic → Register
              → Output_decode      → Control signals

Typical delay:
  State register output:        0.5 ns
  Combinational logic (LUTs):   3.0 ns (assume 4-5 LUT levels)
  Control signal routing:       1.0 ns
  Setup time:                   0.4 ns
  ────────────────────────────────────
  Total:                        4.9 ns
```

**Under 10 ns ✓** Should be fine.

### 4.2 AXI Master Read Path

**Critical path:**

```
AXI_rvalid → FIFO write_enable → FIFO address_gen → FIFO write

Delay:
  Input routing:           0.5 ns
  FIFO control logic:      2.0 ns
  BRAM write setup:        1.5 ns
  ──────────────────────────────
  Total:                   4.0 ns
```

**Under 10 ns ✓** No problem.

### 4.3 Activation Propagation (Systolic Array)

**Longest path: a_in[row 0] → PE00 → PE01 → ... → PE07**

If activations flow through 8 PEs in one cycle:
```
PE0_out → routing → PE1_in → PE1_reg

Per hop:
  PE output:         0.3 ns
  Routing:           1.0 ns (could be long!)
  Next PE input:     0.4 ns
  ──────────────────────────
  Total per hop:     1.7 ns

For 8 hops: 8 × 1.7 = 13.6 ns ❌ TOO SLOW!
```

**WAIT!** This is a misunderstanding of systolic arrays...

### 4.4 Systolic Array Timing (Corrected)

**Activations DON'T propagate through all 8 PEs in one cycle!**

**Correct timing:**
```
Cycle 0: a0 enters PE00
Cycle 1: a0 moves PE00→PE01, a1 enters PE00
Cycle 2: a0 moves PE01→PE02, a1 moves PE00→PE01, a2 enters PE00
...

Each PE only sees its input in one cycle!
```

**Critical path is just ONE PE hop:**
```
PE_out → routing → PE_in → register

Delay: 1.7 ns (from above)
```

**Under 10 ns ✓✓✓** Plenty of margin!

---

## 5. Complete System Critical Paths

### 5.1 Summary of All Paths

| Path Description | Delay (ns) | Slack @ 100 MHz | Status |
|------------------|------------|-----------------|--------|
| PE MAC operation (DSP) | 4.5 | 5.5 ns | ✓ Pass |
| Sparse FSM logic | 4.9 | 5.1 ns | ✓ Pass |
| AXI FIFO write | 4.0 | 6.0 ns | ✓ Pass |
| Systolic array hop | 1.7 | 8.3 ns | ✓ Pass |
| Weight broadcast | 2.0 | 8.0 ns | ✓ Pass |
| CSR register access | 3.5 | 6.5 ns | ✓ Pass |

**Worst case: 4.9 ns (Sparse FSM)**

**Timing margin: 10 - 4.9 = 5.1 ns (51%!)** ✓✓✓

### 5.2 Clock Domain Crossing

**Your design has ONE clock domain:**
- All logic runs at 100 MHz from PS
- No CDC (Clock Domain Crossing) issues!

**Simplification: No need for:**
- Async FIFOs
- Handshake synchronizers
- Metastability protection

---

## 6. Pipeline Planning

### 6.1 Do You Need Pipelining?

**Short answer: NO, not yet.**

**Reasons:**
1. Critical path (4.9 ns) has 51% margin at 100 MHz
2. DSP slices are already fast enough (4.5 ns)
3. Systolic array is naturally pipelined (wave propagation)
4. Adding pipelines increases complexity

**When you WOULD need pipelining:**
- Target frequency > 200 MHz
- Using slower combinational multipliers (not DSP slices)
- Critical path > 10 ns in synthesis

### 6.2 Future Pipeline Stages (If Needed)

If you later want to run at 200 MHz, here's how to pipeline:

**Stage 1: Multiply**
```
Cycle 0:
  Input: a_in, weight
  Compute: product = a_in × weight
  Register: product_reg

Cycle 1:
  Input: product_reg, acc
  Compute: sum = product_reg + acc
  Output: acc_out
```

**Stage 2: Add**
```
Cycle 0: A × W → [product_reg]
Cycle 1: [product_reg] + acc → [sum_reg]
Cycle 2: [sum_reg] → output
```

**Impact:**
- Latency: 2 cycles instead of 1
- Throughput: Still 1 MAC/cycle (no change!)
- Timing: Each stage now ~2.5 ns → 400 MHz capable

### 6.3 Systolic Array Pipeline Depth

**Current design:**
```
Depth = 8 rows + 8 columns - 1 = 15 cycles

Example:
  Cycle 0:  a0 enters
  Cycle 7:  a0 reaches PE07 (rightmost)
  Cycle 15: a7 finishes at PE77
```

**This is ALREADY a 15-stage pipeline!**
- No changes needed
- Naturally efficient
- Utilization ramps up over 15 cycles, then steady-state

---

## 7. Timing Optimization Techniques

### 7.1 Register Retiming

**Idea:** Move registers through combinational logic to balance delays.

**Before:**
```
[REG] → 8ns logic → 2ns logic → [REG]
        \_________10ns__________/  (critical path)
```

**After retiming:**
```
[REG] → 5ns logic → [REG] → 5ns logic → [REG]
        \___5ns___/     \___5ns___/  (balanced!)
```

**Vivado does this automatically!** Use `phys_opt_design -retime`.

### 7.2 Placement Constraints

**Idea:** Place PEs close together to reduce routing delay.

**In your XDC file:**
```tcl
# Place PE array in a rectangular region
create_pblock pblock_systolic_array
resize_pblock pblock_systolic_array -add {SLICE_X50Y50:SLICE_X100Y100}
add_cells_to_pblock pblock_systolic_array [get_cells systolic_array_8x8/*]

# Place DSPs nearby
create_pblock pblock_dsp
resize_pblock pblock_dsp -add {DSP48_X2Y20:DSP48_X5Y27}
add_cells_to_pblock pblock_dsp [get_cells systolic_array_8x8/pe_array*]
```

**Benefit:** Reduces routing delay from 1.0 ns to 0.5 ns.

### 7.3 Logic Replication

**Idea:** Duplicate high-fanout signals to reduce load.

**Example:**
```
enable signal drives 64 PEs
→ Routing delay = 1.5 ns (long wire!)

After replication:
enable → [REG] → enable_0 (drives 32 PEs)
              → enable_1 (drives 32 PEs)

Routing delay = 0.8 ns (shorter wires)
```

**Vivado directive:**
```tcl
set_property MAX_FANOUT 32 [get_nets enable]
```

---

## 8. Setup and Hold Time Analysis

### 8.1 Setup Time

**Setup time (Tsu) = data must be stable BEFORE clock edge.**

```
         ┌─────┐
Data ────┤     │
       ┌─┴─┐   │
Clock ─┤   │───┤
       └───┘   │
       ↑       │
       │   Tsu │
       └───────┘

Tsu = 0.4 ns (typical for Zynq FF)
```

**Setup check:**
```
T_clk ≥ T_logic + T_routing + Tsu

10 ns ≥ 4.5 ns + 1.0 ns + 0.4 ns
10 ns ≥ 5.9 ns ✓ OK (4.1 ns slack)
```

### 8.2 Hold Time

**Hold time (Th) = data must remain stable AFTER clock edge.**

```
         ┌─────┐
Data ────┤     │
         │   ┌─┴─┐
Clock ───┤   │   │
         │   └───┘
         │   ↑
         │ Th│
         └───┘

Th = 0.1 ns (typical)
```

**Hold check:**
```
T_logic + T_routing ≥ Th

Typical: 1.0 + 0.5 = 1.5 ns ≥ 0.1 ns ✓ OK

Hold violations are RARE in FPGAs (lots of routing delay).
```

---

## 9. Clock Skew and Jitter

### 9.1 Clock Skew

**Clock skew = difference in clock arrival time at two FFs.**

```
        Clock
          │
      ┌───┴───┐
      │       │
   0.2ns   0.5ns  (skew = 0.3 ns)
      │       │
     FF1     FF2
```

**Xilinx clock buffers (BUFG) minimize skew:**
- Typical skew: 50-200 ps (0.05-0.2 ns)
- Negligible compared to 10 ns period

**No action needed!** Just use proper clock buffers.

### 9.2 Clock Jitter

**Jitter = variation in clock period cycle-to-cycle.**

```
Ideal:     ____      ____      ____
          |    |    |    |    |    |
         _|    |____|    |____|    |___
          10ns  10ns  10ns

Real:      ____      ____       ____
          |    |    |    |     |    |
         _|    |____|    |_____|    |____
          10ns  10.1ns  9.9ns  (±0.1ns jitter)
```

**PS clock (from PYNQ) has typical jitter:**
- RMS jitter: 50 ps
- Peak-to-peak: 200 ps

**Impact on timing:**
```
Effective period = 10 ns - 0.2 ns = 9.8 ns
Still enough margin (4.9 ns critical path)
```

---

## 10. Post-Synthesis Timing Analysis

### 10.1 Vivado Timing Reports

**After synthesis, check timing:**

```tcl
# In Vivado Tcl console
report_timing_summary -file timing_summary.rpt
report_timing -path_type full -max_paths 10 -file timing_worst.rpt
```

**Look for:**
1. **Worst Negative Slack (WNS):** Should be > 0 ns
   - If WNS < 0: Timing violation! Must fix.
   - If WNS > 1 ns: Good margin ✓

2. **Total Negative Slack (TNS):** Sum of all negative slacks
   - Should be 0 ns

3. **Failing Endpoints:** Number of paths failing timing
   - Should be 0

**Example good report:**
```
WNS: 5.123 ns ✓
TNS: 0.000 ns ✓
Failing Endpoints: 0 ✓
```

### 10.2 Critical Path Report

**Example Vivado output:**

```
Slack: 5.1 ns (MET)
Source: systolic_array_8x8/pe_00_00/acc_reg[15]/C
Destination: systolic_array_8x8/pe_00_00/acc_reg[23]/D

Path breakdown:
  Clock path delay:     0.825 ns
  Logic delay:          2.145 ns (DSP48E1 multiply-add)
  Net delay:            1.923 ns (routing)
  Setup time:           0.412 ns
  ─────────────────────────────
  Total:                4.893 ns

Required time:         10.000 ns
Arrival time:          -4.893 ns
Slack:                  5.107 ns ✓
```

**This tells you:**
- Critical path is in PE[0][0]
- DSP multiply-add takes 2.1 ns
- Routing takes 1.9 ns
- 5.1 ns slack (51% margin) ✓

---

## 11. Timing Closure Strategy

### 11.1 If You Have Timing Violations

**Step 1: Identify the problem**
```tcl
report_timing -max_paths 100 -slack_less_than 0
```

**Common causes:**
- Long routing paths (poor placement)
- Too many logic levels (deep combinational logic)
- High fanout nets (one signal drives many gates)

**Step 2: Apply fixes**

For routing issues:
```tcl
set_property LOC SLICE_X50Y50 [get_cells problematic_cell]
```

For logic depth:
```tcl
# Add pipeline stage
set_property PIPELINE_STAGES 1 [get_cells dsp_cell]
```

For fanout:
```tcl
set_property MAX_FANOUT 32 [get_nets high_fanout_net]
```

**Step 3: Re-run synthesis**
```tcl
reset_run synth_1
launch_runs synth_1
wait_on_run synth_1
```

### 11.2 Design Iteration Process

```
Initial design
     ↓
Synthesize
     ↓
Check timing ────→ Violations? ───→ Apply fixes ───┐
     ↓                                              │
     No                                             │
     ↓                                              │
Implementation ←───────────────────────────────────┘
     ↓
Check timing again
     ↓
Pass? → Generate bitstream
```

**Expected iterations: 2-4** (typical for first-time design)

---

## 12. Timing Constraints (XDC File)

### 12.1 Basic Clock Constraint

**File: `constraints/timing.xdc`**

```tcl
# Create clock constraint for 100 MHz
create_clock -period 10.000 -name aclk [get_ports aclk]

# Set input delay (data from PS)
set_input_delay -clock aclk 2.0 [get_ports s_axi_*]

# Set output delay (data to PS)
set_output_delay -clock aclk 2.0 [get_ports m_axi_*]

# False paths (signals that don't need timing analysis)
set_false_path -from [get_ports aresetn]
```

### 12.2 Advanced Constraints

**If you add pipelining later:**

```tcl
# Multi-cycle path (operation takes 2 cycles instead of 1)
set_multicycle_path 2 -from [get_cells systolic_array/pe_*/mult_reg*] \
                        -to [get_cells systolic_array/pe_*/acc_reg*]

# Max delay constraint (custom requirement)
set_max_delay 5.0 -from [get_pins sparse_controller/state_reg*/C] \
                   -to [get_pins systolic_array/pe_*/enable]
```

---

## 13. Timing vs. Area Trade-offs

### 13.1 Option 1: Slow & Small

```
Clock: 50 MHz (20 ns period)
DSPs: Combinational mode (no pipeline)
Resources: 64 DSPs, 10,000 LUTs

Performance: 3.2 GOPS
Timing margin: 15 ns (huge!)
```

**Pros:** Easy timing closure, low power
**Cons:** Half the performance

### 13.2 Option 2: Fast & Medium (YOUR DESIGN)

```
Clock: 100 MHz (10 ns period)
DSPs: Combinational mode
Resources: 64 DSPs, 15,000 LUTs

Performance: 6.4 GOPS
Timing margin: 5 ns (good)
```

**Pros:** Good balance
**Cons:** None really!

### 13.3 Option 3: Fastest & Large

```
Clock: 200 MHz (5 ns period)
DSPs: 1-stage pipeline
Resources: 64 DSPs, 20,000 LUTs, more FFs

Performance: 12.8 GOPS
Timing margin: 2.5 ns (tight)
```

**Pros:** 2× performance
**Cons:** Harder timing closure, more power, added latency

**Recommendation: Stick with Option 2 (100 MHz) for now.**

---

## 14. Summary and Checklist

### 14.1 Critical Path Summary

✓ **PE MAC operation:** 4.5 ns (DSP48E1)
✓ **Sparse FSM:** 4.9 ns (worst case)
✓ **AXI logic:** 4.0 ns
✓ **Systolic propagation:** 1.7 ns per hop

**Worst case: 4.9 ns < 10 ns ✓**

### 14.2 Pipeline Decision

✓ **No pipelining needed for 100 MHz**
✓ **Systolic array is naturally pipelined (15 stages)**
✓ **Can add DSP pipeline later if targeting 200 MHz+**

### 14.3 Timing Closure Checklist

- [ ] Create XDC file with clock constraint (10 ns period)
- [ ] Run synthesis in Vivado
- [ ] Check `report_timing_summary` for WNS > 0
- [ ] If violations exist, apply fixes (placement, retiming)
- [ ] Re-run and verify timing closure
- [ ] Generate bitstream once timing passes

### 14.4 Expected Results

**After first synthesis (Week 2, Dec 1):**
- WNS: 3-6 ns (good margin)
- No timing violations
- Ready for implementation

**If you see violations:**
- Most likely cause: Poor placement of PEs
- Fix: Add pblock constraints to group PEs together
- Should resolve in 1-2 iterations

---

## 15. Next Steps (Nov 26+)

**Nov 26-30 (Week 1): RTL Coding**
- Don't worry about timing yet
- Focus on functional correctness
- Use DSP48E1 inference (let Vivado choose pipeline depth)

**Dec 1 (Week 2): First Synthesis**
- Run synthesis
- Check timing report
- If WNS > 0: You're good! ✓
- If WNS < 0: Apply placement constraints

**Dec 8 (Week 3): Timing Optimization**
- Fine-tune placement if needed
- Verify timing margin is 10-20%
- Lock down timing constraints

**You're ready to start coding tomorrow!** 🚀
