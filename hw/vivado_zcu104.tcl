##############################################################################
# vivado_zcu104.tcl — ACCEL-v1 ZCU104 Vivado Project + Bitstream Script
##############################################################################
# Usage (from repo root):
#   vivado -mode batch -source hw/vivado_zcu104.tcl
#   vivado -mode gui    -source hw/vivado_zcu104.tcl   (opens GUI after build)
#
# After completion, bitstream is at:
#   hw/vivado/accel_zcu104/accel_zcu104.runs/impl_1/zcu104_wrapper.bit
##############################################################################

set SCRIPT_DIR  [file dirname [file normalize [info script]]]
set PROJ_DIR    [file join $SCRIPT_DIR vivado]
set RTL_DIR     [file join $SCRIPT_DIR rtl]
set XDC_FILE    [file join $SCRIPT_DIR constraints zcu104.xdc]

##############################################################################
# 1. Create project
##############################################################################
create_project accel_zcu104 [file join $PROJ_DIR accel_zcu104] -part xczu7ev-ffvc1156-2-e -force

set_property target_language     SystemVerilog [current_project]
set_property simulator_language  Mixed         [current_project]

##############################################################################
# 2. Add all synthesisable RTL (exclude testbenches, SVA, legacy tops)
##############################################################################
# Packages (order matters — add first)
set pkg_files [list \
    [file join $RTL_DIR top   soc_pkg.sv]       \
    [file join $RTL_DIR noc   noc_pkg.sv]       \
    [file join $RTL_DIR memory coherence_pkg.sv] \
]

# All other RTL — glob each subdirectory, then filter out what we don't want
set all_sv [glob -nocomplain \
    [file join $RTL_DIR buffer   *.sv] \
    [file join $RTL_DIR cache    *.sv] \
    [file join $RTL_DIR control  *.sv] \
    [file join $RTL_DIR dma      *.sv] \
    [file join $RTL_DIR dram     *.sv] \
    [file join $RTL_DIR hft      *.sv] \
    [file join $RTL_DIR host_iface *.sv] \
    [file join $RTL_DIR mac      *.sv] \
    [file join $RTL_DIR memory   *.sv] \
    [file join $RTL_DIR monitor  *.sv] \
    [file join $RTL_DIR noc      *.sv] \
    [file join $RTL_DIR periph   *.sv] \
    [file join $RTL_DIR systolic *.sv] \
    [file join $RTL_DIR top      *.sv] \
]

# Files to exclude from synthesis
set exclude_patterns {
    *_tb.sv  *_sva.sv  *tb_*.sv
    *accel_top.sv
    *soc_top.sv
    *coherence_demo_top.sv
    *soc_top_v2_asic_sim_wrapper.sv
    *soc_pkg.sv *noc_pkg.sv *coherence_pkg.sv
}

set rtl_files $pkg_files
foreach f $all_sv {
    set skip 0
    foreach pat $exclude_patterns {
        if {[string match $pat [file tail $f]]} { set skip 1; break }
    }
    if {!$skip} { lappend rtl_files $f }
}

add_files -norecurse $rtl_files
set_property file_type SystemVerilog [get_files *.sv]

##############################################################################
# 3. Add XDC constraints
##############################################################################
add_files -fileset constrs_1 -norecurse $XDC_FILE

##############################################################################
# 4. Set top module
##############################################################################
set_property top zcu104_wrapper [current_fileset]
update_compile_order -fileset sources_1

##############################################################################
# 5. Set synthesis & implementation options
##############################################################################
# Synthesis strategy: Vivado Synthesis Defaults with SystemVerilog defines
set_property verilog_define {SYNTHESIS FPGA_SYNTHESIS} [current_fileset]
set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]

# Implementation: area-optimised — good starting point before timing closure
set_property strategy "Performance_ExplorePostRoutePhysOpt" [get_runs impl_1]

# Bitstream compression
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design] 2>/dev/null

##############################################################################
# 6. Run synthesis → implementation → bitstream
##############################################################################
puts "======================================================="
puts " ACCEL-v1 ZCU104 Build starting..."
puts " Part:    xczu7ev-ffvc1156-2-e"
puts " Top:     zcu104_wrapper"
puts " Clk:     125 MHz diff → MMCM → 50 MHz internal"
puts "======================================================="

launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synthesis did not complete successfully"
    exit 1
}
puts "✓ Synthesis complete"

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: implementation did not complete successfully"
    exit 1
}
puts "✓ Implementation + bitstream complete"

##############################################################################
# 7. Copy bitstream to hw/ root for easy access
##############################################################################
set bit_src [file join $PROJ_DIR accel_zcu104 accel_zcu104.runs impl_1 zcu104_wrapper.bit]
set bit_dst [file join $SCRIPT_DIR zcu104_wrapper.bit]

if {[file exists $bit_src]} {
    file copy -force $bit_src $bit_dst
    puts "✓ Bitstream copied to hw/zcu104_wrapper.bit"
} else {
    puts "WARNING: bitstream file not found at expected path"
}

##############################################################################
# 8. Write timing summary
##############################################################################
set timing_file [file join $PROJ_DIR accel_zcu104 timing_summary.txt]
report_timing_summary -file $timing_file -quiet
puts "✓ Timing summary written to $timing_file"

puts "======================================================="
puts " Done.  Bitstream: hw/zcu104_wrapper.bit"
puts "======================================================="
