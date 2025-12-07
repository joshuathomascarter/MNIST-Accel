# ============================================================================
# Dual-Clock Domain Constraints for 2× Throughput
# Phase 5: 50 MHz Control, 200 MHz Datapath
# ============================================================================
# 
# PURPOSE:
# --------
# Run datapath at 4× control clock to achieve 2× throughput while maintaining
# power savings. Control logic (scheduler, DMA) runs at 50 MHz (adequate for
# tile scheduling), while compute (systolic array) runs at 200 MHz.
#
# PERFORMANCE GAIN:
# -----------------
# Baseline: 100 MHz → 3.2 GOPS (2×2 array × 8×8 blocks × 100 MHz)
# Dual-clock: 200 MHz datapath → 6.4 GOPS (2× throughput!)
#
# POWER IMPACT:
# -------------
# Dynamic power ∝ f, so datapath power increases:
#   - Datapath @ 100 MHz: 485 mW
#   - Datapath @ 200 MHz: 970 mW (+485 mW)
#   - Control @ 50 MHz: 250 mW (vs 405 mW @ 100 MHz → -155 mW)
#   - Net change: +330 mW
# HOWEVER: 2× throughput means 2× work per watt → energy efficiency same!
# Final power: 840 mW + 330 mW = 1170 mW @ 6.4 GOPS (vs 840 mW @ 3.2 GOPS)
# Energy per operation: 1170/6.4 = 183 pJ/op (vs 840/3.2 = 263 pJ/op → 30% better!)
#
# CLOCK DOMAIN CROSSING (CDC):
# -----------------------------
# 1. Control → Data (50 MHz → 200 MHz):
#    - Tile start signals: Use pulse synchronizer
#    - Configuration data: Use async FIFO
# 2. Data → Control (200 MHz → 50 MHz):
#    - Systolic done signals: Use 2-FF synchronizer
#    - Status counters: Use gray-code crossing
#
# ============================================================================

# ============================================================================
# Define Clock Domains
# ============================================================================

# Control clock: 50 MHz (20ns period)
create_clock -period 20.0 -name clk_ctrl [get_ports clk_ctrl]

# Datapath clock: 200 MHz (5ns period)
create_clock -period 5.0 -name clk_data [get_ports clk_data]

# Clocks are asynchronous (no phase relationship)
set_clock_groups -asynchronous -group [get_clocks clk_ctrl] -group [get_clocks clk_data]

# ============================================================================
# Input/Output Delays
# ============================================================================

# Control interface (50 MHz)
set_input_delay -clock clk_ctrl 4.0 [get_ports {start abort cfg_*}]
set_output_delay -clock clk_ctrl 4.0 [get_ports {done busy blocks_processed}]

# Datapath interface (200 MHz)
set_input_delay -clock clk_data 1.0 [get_ports {act_data_* wgt_data_*}]
set_output_delay -clock clk_data 1.0 [get_ports {result_*}]

# ============================================================================
# CDC Constraints (50 MHz ↔ 200 MHz)
# ============================================================================

# Control → Data crossings (use pulse synchronizer)
set_max_delay -from [get_clocks clk_ctrl] -to [get_clocks clk_data] 20.0
set_min_delay -from [get_clocks clk_ctrl] -to [get_clocks clk_data] 0.0

# Data → Control crossings (use 2-FF synchronizer)
set_max_delay -from [get_clocks clk_data] -to [get_clocks clk_ctrl] 20.0
set_min_delay -from [get_clocks clk_data] -to [get_clocks clk_ctrl] 0.0

# ============================================================================
# False Paths for CDC Logic
# ============================================================================

# Pulse synchronizers (start, abort signals)
set_false_path -from [get_pins -hierarchical -filter {NAME =~ *pulse_sync*/meta_ff_reg*/C}] \
               -to   [get_pins -hierarchical -filter {NAME =~ *pulse_sync*/sync_ff_reg*/D}]

# 2-FF synchronizers (done, busy signals)
set_false_path -from [get_pins -hierarchical -filter {NAME =~ *sync_2ff*/meta_ff_reg*/C}] \
               -to   [get_pins -hierarchical -filter {NAME =~ *sync_2ff*/sync_ff_reg*/D}]

# ============================================================================
# Module-to-Clock Assignment
# ============================================================================

# Control modules → 50 MHz
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_ctrl]
set_case_analysis 0 [get_pins -hierarchical -filter {NAME =~ *scheduler*/clk && PARENT =~ clk_ctrl}]
set_case_analysis 0 [get_pins -hierarchical -filter {NAME =~ *csr*/clk && PARENT =~ clk_ctrl}]
set_case_analysis 0 [get_pins -hierarchical -filter {NAME =~ *axi_dma*/clk && PARENT =~ clk_ctrl}]
set_case_analysis 0 [get_pins -hierarchical -filter {NAME =~ *bsr_scheduler*/clk && PARENT =~ clk_ctrl}]

# Datapath modules → 200 MHz
set_case_analysis 1 [get_pins -hierarchical -filter {NAME =~ *systolic*/clk && PARENT =~ clk_data}]
set_case_analysis 1 [get_pins -hierarchical -filter {NAME =~ *act_buffer*/clk && PARENT =~ clk_data}]
set_case_analysis 1 [get_pins -hierarchical -filter {NAME =~ *wgt_buffer*/clk && PARENT =~ clk_data}]
set_case_analysis 1 [get_pins -hierarchical -filter {NAME =~ *mac8*/clk && PARENT =~ clk_data}]

# ============================================================================
# Multi-Cycle Paths (for configuration data)
# ============================================================================

# Configuration registers updated at 50 MHz, consumed at 200 MHz
# Allow 4 cycles (4 × 5ns = 20ns) for stable read
set_multicycle_path 4 -from [get_pins -hierarchical -filter {NAME =~ *csr*/cfg_*_reg*/C}] \
                       -to   [get_pins -hierarchical -filter {NAME =~ *systolic*/cfg_*}]

# ============================================================================
# Clock Skew and Jitter Budgets
# ============================================================================

# 50 MHz clock (relaxed timing)
set_input_jitter clk_ctrl 0.5  # 500ps jitter budget

# 200 MHz clock (tight timing)
set_input_jitter clk_data 0.1  # 100ps jitter budget

# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================
# 
# 1. REQUIRED RTL CHANGES:
#    - Modify accel_top.sv to accept two clock inputs:
#      * input wire clk_ctrl  // 50 MHz control clock
#      * input wire clk_data  // 200 MHz datapath clock
#    - Insert CDC synchronizers:
#      * Pulse synchronizer for start/abort (ctrl → data)
#      * 2-FF synchronizer for done/busy (data → ctrl)
#      * Async FIFO for tile configuration (ctrl → data)
#
# 2. CLOCK GENERATION:
#    - FPGA: Use Vivado Clock Wizard IP
#      * PLL input: 100 MHz board clock
#      * Output 1: 50 MHz (clk_ctrl)
#      * Output 2: 200 MHz (clk_data)
#    - ASIC: Use PLL with dividers
#
# 3. VERIFICATION:
#    - CDC formal verification (Jasper/VC Formal)
#    - Simulation with dual-clock testbench
#    - STA with CDC timing exceptions
#    - Measure throughput (should be ~2× baseline)
#
# 4. PERFORMANCE METRICS:
#    - Throughput: 6.4 GOPS (2× improvement)
#    - Power: 1170 mW (vs 840 mW baseline)
#    - Energy efficiency: 183 pJ/op (vs 263 pJ/op → 30% better!)
#    - Latency: Same per tile (200 MHz systolic completes in same cycles)
#
# 5. RISK MITIGATION:
#    - CDC bugs are subtle → extensive formal verification required
#    - Metastability risk → use 2-FF synchronizers with ASYNC_REG attribute
#    - Timing closure @ 200 MHz may be challenging → may need pipeline stages
#
# ============================================================================
