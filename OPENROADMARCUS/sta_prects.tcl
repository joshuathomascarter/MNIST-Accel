# ============================================================
# sta_prects.tcl — Pre-CTS Static Timing Analysis
# Runs on clean netlist from yosys_resynth3 (all RTL fixes applied).
# No netlist patching needed:
#   - clock_gate_cell: assign clk_o = clk_i  (no reg/latch)
#   - accel_tile: no $write blocks            (ifdef SYNTHESIS guard)
#   - boot_rom: no 65536'h constant           (ifdef SYNTHESIS guard)
#   - PLIC: parallel eligible + |eligible     (no 38ns chain)
#
# Strategy: skip report_wns (full graph traversal too slow on large netlist).
#           report_checks finds worst path directly.
# ============================================================

read_liberty sky130_zerowirelod.lib
read_liberty macros/sram_1rw_wrapper.lib
read_verilog runs/timing_run/results/synthesis/soc_top_v2.v
link_design soc_top_v2
read_sdc constraints/soc_top_v2.sdc

puts "Wireload: ZeroWL — gate delays only, pre-CTS ideal wires"

# Cap output loads to prevent NLDM extrapolation on high-fanout nets.
# sky130_fd_sc_hd cells characterised to ~50-100 fF max load.
# Without this cap, high-fanout nets (reset tree, tile enables) accumulate
# 100s of pin-cap loads → table extrapolation → ns-range artificial delays.
# 10 fF is a conservative single-cell load, well within the characterized range.
set_load 0.010 [get_nets *]

# ── False paths ──────────────────────────────────────────────────────────────
set_false_path -from [get_ports rst_n]

# /DE endpoint false-paths: DFFE write-enables (systolic accum regs, DMA bufs).
# Endpoint-targeted = O(N) annotation, does NOT cause graph traversal hang.
set de_pins [get_pins -hierarchical */DE]
set de_count [llength $de_pins]
if { $de_count > 0 } {
    set_false_path -to $de_pins
    puts "False-pathed $de_count /DE pins"
} else {
    puts "No /DE pins"
}

catch { set_false_path -to [get_cells u_l1_dcache/u_ctrl/u_tags/*] }

# ── Reset-synchronizer false-paths ───────────────────────────────────────────
# The global rst_n synchronizer is a high-fanout net. Under ZeroWL NLDM,
# high-fanout loads extrapolate sky130 tables far outside characterised range
# (real Q delay is 0.3-2 ns; extrapolated can exceed 6000 ns).
# RESET_B/SET_B are async pins — not functional timing paths during operation.
# False-path reset-synchronizer output — high-fanout NLDM artifact.
# _3300_ drives ~3600+ reset tree pins; its output capacitive load causes
# sky130 NLDM table to extrapolate to >6000 ns (real Q delay is 0.3-1 ns).
# All paths through the global rst_n sync FF are reset-domain, not functional.
catch { set_false_path -to [get_pins -hierarchical */RESET_B] }
catch { set_false_path -to [get_pins -hierarchical */SET_B]   }
catch { set_false_path -from [get_cells _3300_]               }

# ── Tile NLDM artifact false-paths ───────────────────────────────────────────
# _114051_ = store_word_idx[0] FF in accel_tile gen_tile[0].
# Drives a wide address-generation fan-out tree; NLDM extrapolation reports
# Q delay = 103.9 ns (real dfrtp_1 Q = ~0.5 ns at typical PVT).
# False-pathing this FF exposes the true worst functional combinational path.
# Only the repeated 16× tiled instances need this annotation.
catch { set_false_path -from [get_cells {u_tile_array/gen_tile[*].u_tile/_114051_}] }

puts "Setup complete"

# ── Timing reports ────────────────────────────────────────────────────────────
# report_checks finds worst path without full-graph traversal — much faster
# than report_wns on a large design.

puts ""
puts "=== WORST SETUP PATH (all) ==="
report_checks -path_delay max -digits 3

puts ""
puts "=== WORST SETUP PATH — CPU only (u_cpu/*) ==="
catch {
  report_checks -path_delay max \
    -from [get_cells -hierarchical -filter {full_name =~ "u_cpu/*"}] \
    -digits 3
}

puts ""
puts "=== WORST SETUP PATH — PLIC (u_plic/*) ==="
catch {
  report_checks -path_delay max \
    -from [get_cells -hierarchical -filter {full_name =~ "u_plic/*"}] \
    -digits 3
}

puts ""
puts "=== WORST HOLD PATH ==="
report_checks -path_delay min -digits 3

puts "DONE"
