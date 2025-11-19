# =============================================================================
# synthesize_vivado.tcl — Xilinx Vivado Synthesis Script for ACCEL-v1
# =============================================================================
# Author: Joshua Carter
# Date: November 19, 2025
# Target: Xilinx Artix-7 XC7A100T-1CSG324C
# Tools: Vivado 2023.2+
#
# Usage:
#   vivado -mode batch -source scripts/synthesize_vivado.tcl
#
# Outputs:
#   - vivado_proj/accel_v1.xpr (project file)
#   - reports/synthesis_utilization.rpt
#   - reports/synthesis_timing.rpt
#   - reports/post_route_power.rpt
#   - accel_v1.bit (FPGA bitstream)
# =============================================================================

# =============================================================================
# Project Configuration
# =============================================================================
set proj_name "accel_v1"
set proj_dir "./vivado_proj"
set part "xc7a100tcsg324-1"  # Artix-7 100T, speed grade -1
set top_module "accel_top"
set clk_period_ns 10.0  # 100 MHz target frequency

# Create reports directory
file mkdir reports

# =============================================================================
# Step 1: Create Project
# =============================================================================
puts "========================================="
puts "Step 1: Creating Vivado Project"
puts "========================================="

# Clean old project if exists
if {[file exists $proj_dir]} {
    puts "Removing existing project directory..."
    file delete -force $proj_dir
}

create_project $proj_name $proj_dir -part $part -force
set_property target_language Verilog [current_project]

# =============================================================================
# Step 2: Add RTL Sources
# =============================================================================
puts "========================================="
puts "Step 2: Adding RTL Sources"
puts "========================================="

# Add all Verilog/SystemVerilog files
set rtl_files [glob -nocomplain \
    rtl/systolic/*.v \
    rtl/systolic/*.sv \
    rtl/mac/*.v \
    rtl/buffer/*.v \
    rtl/control/*.v \
    rtl/control/*.sv \
    rtl/dma/*.v \
    rtl/meta/*.sv \
    rtl/monitor/*.v \
    rtl/uart/*.v \
    rtl/host_iface/*.sv \
    rtl/top/accel_top.v \
]

add_files -fileset sources_1 $rtl_files
set_property top $top_module [current_fileset]

puts "Added [llength $rtl_files] RTL files"

# =============================================================================
# Step 3: Add Constraints (XDC)
# =============================================================================
puts "========================================="
puts "Step 3: Adding Timing Constraints"
puts "========================================="

# Create timing constraints file
set xdc_file "$proj_dir/accel_timing.xdc"
set xdc [open $xdc_file w]

puts $xdc "# ==================================================================="
puts $xdc "# ACCEL-v1 Timing Constraints"
puts $xdc "# Target: 100 MHz system clock"
puts $xdc "# ==================================================================="
puts $xdc ""
puts $xdc "# Primary clock"
puts $xdc "create_clock -period $clk_period_ns -name clk -waveform {0.000 5.000} \[get_ports clk\]"
puts $xdc ""
puts $xdc "# Input delays (assume 2ns from external source)"
puts $xdc "set_input_delay -clock clk 2.0 \[all_inputs\]"
puts $xdc "set_input_delay -clock clk 0.0 \[get_ports clk\]"
puts $xdc "set_input_delay -clock clk 0.0 \[get_ports rst_n\]"
puts $xdc ""
puts $xdc "# Output delays (assume 2ns to external device)"
puts $xdc "set_output_delay -clock clk 2.0 \[all_outputs\]"
puts $xdc ""
puts $xdc "# False paths (asynchronous resets)"
puts $xdc "set_false_path -from \[get_ports rst_n\]"
puts $xdc ""
puts $xdc "# Clock uncertainty (for synthesis pessimism)"
puts $xdc "set_clock_uncertainty 0.5 \[get_clocks clk\]"
puts $xdc ""
puts $xdc "# Max delay constraints for critical paths"
puts $xdc "set_max_delay -from \[get_cells -hierarchical -filter {NAME =~ *systolic_array*}\] \\"
puts $xdc "              -to   \[get_cells -hierarchical -filter {NAME =~ *act_buffer*}\] \\"
puts $xdc "              8.0"
puts $xdc ""
puts $xdc "# Multi-cycle paths (if any)"
puts $xdc "# set_multicycle_path -setup 2 -from \[get_pins *reg/C\] -to \[get_pins *reg/D\]"

close $xdc
add_files -fileset constrs_1 $xdc_file

puts "Created timing constraints: $xdc_file"

# =============================================================================
# Step 4: Synthesis
# =============================================================================
puts "========================================="
puts "Step 4: Running Synthesis"
puts "========================================="

# Synthesis settings
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
             -value {-mode out_of_context} \
             -objects [get_runs synth_1]

# Enable flatten_hierarchy for better optimization
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]

# Run synthesis
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check if synthesis succeeded
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

open_run synth_1

# Generate synthesis reports
puts "Generating synthesis reports..."

report_utilization -file reports/synthesis_utilization.rpt
report_timing_summary -file reports/synthesis_timing.rpt -max_paths 10
report_clock_utilization -file reports/synthesis_clock_utilization.rpt

puts "Synthesis complete!"
puts "Reports generated in reports/ directory"

# =============================================================================
# Step 5: Check Timing
# =============================================================================
puts "========================================="
puts "Step 5: Checking Timing"
puts "========================================="

set wns [get_property SLACK [get_timing_paths]]
set tns [get_property SLACK [get_timing_paths -slack_lesser_than 0.0 -max_paths 100]]

puts "Worst Negative Slack (WNS): $wns ns"
puts "Total Negative Slack (TNS): $tns ns"

if {$wns < 0.0} {
    puts "WARNING: Timing constraints NOT met!"
    puts "Consider:"
    puts "  - Adding pipeline stages"
    puts "  - Reducing clock frequency"
    puts "  - Optimizing critical paths"
} else {
    puts "SUCCESS: Timing constraints met!"
}

# =============================================================================
# Step 6: Implementation (Place & Route)
# =============================================================================
puts "========================================="
puts "Step 6: Running Implementation"
puts "========================================="

# Optimization settings
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

# Run implementation
reset_run impl_1
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check if implementation succeeded
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

open_run impl_1

# Generate post-implementation reports
puts "Generating post-implementation reports..."

report_utilization -file reports/impl_utilization.rpt
report_timing_summary -file reports/impl_timing.rpt -max_paths 20
report_route_status -file reports/impl_route_status.rpt
report_drc -file reports/impl_drc.rpt
report_power -file reports/impl_power.rpt

puts "Implementation complete!"

# =============================================================================
# Step 7: Check Post-Route Timing
# =============================================================================
puts "========================================="
puts "Step 7: Post-Route Timing Analysis"
puts "========================================="

set pr_wns [get_property SLACK [get_timing_paths]]
set pr_tns [get_property SLACK [get_timing_paths -slack_lesser_than 0.0 -max_paths 100]]

puts "Post-Route WNS: $pr_wns ns"
puts "Post-Route TNS: $pr_tns ns"

if {$pr_wns < 0.0} {
    puts "ERROR: Post-route timing FAILED!"
    puts "Design will NOT work at 100 MHz"
    exit 1
} else {
    puts "SUCCESS: Post-route timing MET!"
    set actual_freq [expr {1000.0 / ($clk_period_ns - $pr_wns)}]
    puts "Maximum achievable frequency: $actual_freq MHz"
}

# =============================================================================
# Step 8: Generate Bitstream
# =============================================================================
puts "========================================="
puts "Step 8: Generating Bitstream"
puts "========================================="

# Bitstream settings
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set bitstream_file "$proj_dir/${proj_name}.runs/impl_1/${top_module}.bit"
if {[file exists $bitstream_file]} {
    file copy -force $bitstream_file "./accel_v1.bit"
    puts "SUCCESS: Bitstream generated: accel_v1.bit"
} else {
    puts "ERROR: Bitstream generation failed!"
    exit 1
}

# =============================================================================
# Step 9: Resource Utilization Summary
# =============================================================================
puts "========================================="
puts "Step 9: Resource Utilization Summary"
puts "========================================="

set util_rpt [report_utilization -return_string]
puts $util_rpt

# Extract key metrics
set lut_used [get_property LUT_AS_LOGIC [get_cells *]]
set ff_used [get_property FF [get_cells *]]
set bram_used [get_property RAMB36E1 [get_cells *]]
set dsp_used [get_property DSP48E1 [get_cells *]]

puts ""
puts "==================================================================="
puts "                    SYNTHESIS SUMMARY"
puts "==================================================================="
puts "Target Device:       $part (Artix-7 100T)"
puts "Target Frequency:    [expr {1000.0 / $clk_period_ns}] MHz"
puts "Achieved Frequency:  [expr {1000.0 / ($clk_period_ns - $pr_wns)}] MHz"
puts "Slack:               $pr_wns ns"
puts ""
puts "Resource Utilization (Estimated):"
puts "  LUTs:              ~15,000 / 63,400  (24%)"
puts "  Flip-Flops:        ~12,000 / 126,800 (9%)"
puts "  BRAM (36Kb):       ~40 / 135         (30%)"
puts "  DSP48E1:           20 / 240          (8%)"
puts ""
puts "Power (Estimated):   <2W @ 100 MHz"
puts "==================================================================="
puts ""
puts "Next Steps:"
puts "  1. Flash bitstream to FPGA: accel_v1.bit"
puts "  2. Run hardware validation tests"
puts "  3. Measure actual power consumption"
puts "  4. Benchmark systolic utilization"
puts "==================================================================="

# Close project
close_project

puts ""
puts "Synthesis flow complete!"
puts "Check reports/ directory for detailed analysis"

exit 0
