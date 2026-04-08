set ::env(DESIGN_NAME) soc_top_v2
set ::env(CLOCK_PORT) clk
set ::env(CLOCK_PERIOD) 20.0
set ::env(VERILOG_DEFINES) [list SYNTHESIS ASIC_SYNTHESIS]
set ::env(BASE_SDC_FILE) $::env(DESIGN_DIR)/constraints/soc_top_v2.sdc

set ::env(VERILOG_FILES) [concat \
  [list \
    $::env(DESIGN_DIR)/rtl/top/soc_pkg.sv \
    $::env(DESIGN_DIR)/rtl/noc/noc_pkg.sv \
  ] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/cache/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/control/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/dram/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/mac/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/memory/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/monitor/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/noc/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/periph/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/rtl/systolic/*.sv]] \
  [list \
    $::env(DESIGN_DIR)/rtl/top/axi_addr_decoder.sv \
    $::env(DESIGN_DIR)/rtl/top/axi_arbiter.sv \
    $::env(DESIGN_DIR)/rtl/top/axi_crossbar.sv \
    $::env(DESIGN_DIR)/rtl/top/simple_cpu.sv \
    $::env(DESIGN_DIR)/rtl/top/soc_top_v2.sv \
  ] \
]

# ── Floorplan ────────────────────────────────────────────────────────────────
set ::env(FP_SIZING) relative
set ::env(FP_CORE_UTIL) 35
set ::env(PL_TARGET_DENSITY) 0.45

# ── SRAM Macro Binding ───────────────────────────────────────────────────────
# sram_1rw_wrapper.sv is guarded with `ifndef SYNTHESIS so the behavioral body
# is excluded during synthesis.  The stub LEF/LIB here give OpenROAD the
# geometry and timing arcs it needs to place and time the macro.
#
# For a real tapeout: replace these stubs with the foundry-signed OpenRAM
# sky130 macro views (e.g. sky130_sram_1kbyte_1r1w) once PDK is selected.
set ::env(EXTRA_LEFS) [list $::env(DESIGN_DIR)/macros/sram_1rw_wrapper.lef]
set ::env(EXTRA_LIBS) [list $::env(DESIGN_DIR)/macros/sram_1rw_wrapper.lib]

# Tell Yosys to treat any cell found in EXTRA_LIBS as a black box (no mapping)
set ::env(SYNTH_READ_BLACKBOX_LIB) 1
set ::env(SYNTH_ELABORATE_ONLY)    0

# ── Power delivery network ───────────────────────────────────────────────────
# Single VDD / VSS domain per TAPEOUT_TARGET.md freeze rules
set ::env(VDD_NETS)             [list VDD]
set ::env(GND_NETS)             [list VSS]
set ::env(FP_PDN_RAILS_LAYER)   met1
set ::env(FP_PDN_HORIZONTAL_LAYER) met2
set ::env(FP_PDN_VERTICAL_LAYER)   met3

# ── DFT / Scan ───────────────────────────────────────────────────────────────
# Scan insertion is planned per DFT_SCAN_PLAN.md.
# Set RUN_DFT=1 once the scan-enable port is added to soc_top_v2.sv:
#   input  logic scan_en_i,
#   input  logic scan_in_i,
#   output logic scan_out_o,
set ::env(RUN_DFT) 0

# ── IO Pad placeholders ──────────────────────────────────────────────────────
# Padring is defined in IO_PAD_PLAN.md.  Enable once padring RTL is added:
# set ::env(USE_GPIO_PADS) 1
# set ::env(FP_IO_HMETAL) met5
# set ::env(FP_IO_VMETAL) met4
set ::env(FP_IO_MODE) 1    ;# matched-length, no pads yet

# ── Verification tools ───────────────────────────────────────────────────────
set ::env(RUN_KLAYOUT) 0
set ::env(RUN_MAGIC)   0
set ::env(RUN_LVS)     0    ;# enable after macro LIB/LEF are bound
