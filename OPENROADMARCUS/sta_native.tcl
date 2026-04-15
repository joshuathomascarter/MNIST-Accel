read_liberty runs/timing_run/issue_reproducible/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_liberty macros/sram_1rw_wrapper.lib
read_verilog runs/timing_run/results/synthesis/soc_top_v2.v
link_design soc_top_v2
read_sdc constraints/soc_top_v2.sdc

# ── Pre-CTS false paths ────────────────────────────────────────────────────
# Reset synchronizer cells: rst_sync_ff[0] and rst_sync_ff[1] fan out to 306K
# async-reset pins with no reset distribution tree (pre-CTS artifact).
# Set false path from these so the pre-CTS timing noise doesn't dominate.
# In P&R, a proper reset distribution tree is built during CTS.
set_false_path -from [get_cells _3299_]
set_false_path -from [get_cells _3300_]

# ── Timing analysis ────────────────────────────────────────────────────────
report_wns
report_tns
puts ""
puts "=== SETUP top-5 violating endpoints ==="
report_checks -path_delay max -group_name core_clk -nworst 5 -digits 3
puts ""
puts "=== HOLD top-3 violating endpoints ==="
report_checks -path_delay min -group_name core_clk -nworst 3 -digits 3
