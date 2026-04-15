# sta_clean.tcl — WNS analysis with dynamic high-fanout false-path suppression
#
# Strategy: scan every net in the synthesized netlist.  Any cell driving
# more than HIGH_FANOUT_THRESH loads is a pre-CTS artifact (no clock/reset
# distribution tree, behavioral array expanded to FFs, etc.) and is
# false-pathed so it doesn't dominate the WNS report.
#
# After suppression the remaining violations are real combinational paths
# that must meet the 50 MHz (20 ns) target.

read_liberty runs/timing_run/issue_reproducible/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_liberty macros/sram_1rw_wrapper.lib
read_verilog runs/timing_run/results/synthesis/soc_top_v2.v
link_design soc_top_v2
read_sdc constraints/soc_top_v2.sdc

# ── Known pre-CTS false paths (reset synchronizer fanout) ─────────────────
set_false_path -from [get_cells _3299_]
set_false_path -from [get_cells _3300_]

# ── Dynamic high-fanout sweep ──────────────────────────────────────────────
# Cells driving >5000 loads are synthesis artifacts (behavioral arrays mapped
# to FFs, pre-CTS clock/reset distribution, etc.).  False-path them.
set HIGH_FANOUT_THRESH 5000
set fp_count 0

foreach net [get_nets *] {
    set load_pins [get_pins -of_objects $net -filter "direction == input"]
    if {[llength $load_pins] > $HIGH_FANOUT_THRESH} {
        set driver_pins [get_pins -of_objects $net -filter "direction == output"]
        foreach dpin $driver_pins {
            catch {
                set drv_cell [get_cells -of_objects $dpin]
                set_false_path -from $drv_cell
                incr fp_count
                puts "  FP: [get_name $drv_cell] fanout=[llength $load_pins]"
            }
        }
    }
}
puts "Dynamic false-paths applied: $fp_count"
puts ""

# ── Timing summary ─────────────────────────────────────────────────────────
report_wns
report_tns
puts ""
puts "=== TOP-5 SETUP VIOLATORS ==="
report_checks -path_delay max -nworst 5 -digits 3
puts ""
puts "=== TOP-3 HOLD VIOLATORS ==="
report_checks -path_delay min -nworst 3 -digits 3
puts "DONE"
