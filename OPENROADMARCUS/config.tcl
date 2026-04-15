set ::env(DESIGN_NAME) soc_top_v2
set ::env(CLOCK_PORT) clk
set ::env(CLOCK_PERIOD) 20.0
set ::env(VERILOG_DEFINES) [list SYNTHESIS ASIC_SYNTHESIS]
set ::env(BASE_SDC_FILE) $::env(DESIGN_DIR)/constraints/soc_top_v2.sdc

set ::env(VERILOG_FILES) [list \
    $::env(DESIGN_DIR)/macros/sram_1rw_wrapper_bb.v \
    $::env(DESIGN_DIR)/rtl_v2k/soc_top_v2_all.v \
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

# Tell Yosys to treat sram_1rw_wrapper as a black box via the Verilog stub
# (sram_1rw_wrapper_bb.v, prepended to VERILOG_FILES).  The Liberty file is
# consumed only by OpenSTA for timing arcs — disable Liberty-based blackboxing
# so Yosys does not create a conflicting parameterless cell from the .lib.
set ::env(SYNTH_READ_BLACKBOX_LIB) 0
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
set ::env(QUIT_ON_LINTER_ERRORS) 0

# Keep module hierarchy during synthesis — prevents OOM during FLATTEN on large SoCs.
# OpenROAD P&R works fine with hierarchical netlists.
set ::env(SYNTH_FLAT_TOP) 0
