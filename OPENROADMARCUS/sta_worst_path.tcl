read_liberty runs/timing_run/issue_reproducible/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_liberty macros/sram_1rw_wrapper.lib
read_verilog runs/timing_run/results/synthesis/soc_top_v2.v
link_design soc_top_v2
read_sdc constraints/soc_top_v2.sdc
set_false_path -from [get_cells _3299_]
set_false_path -from [get_cells _3300_]
report_wns
report_tns
puts ""
puts "=== WORST SETUP PATH ==="
report_checks -path_delay max -nworst 1 -digits 3
puts "DONE"
