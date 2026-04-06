set ::env(DESIGN_NAME) soc_top_v2
set ::env(CLOCK_PORT) clk
set ::env(CLOCK_PERIOD) 20.0
set ::env(VERILOG_DEFINES) [list SYNTHESIS ASIC_SYNTHESIS]
set ::env(BASE_SDC_FILE) $::env(DESIGN_DIR)/constraints/soc_top_v2.sdc

set ::env(VERILOG_FILES) [concat \
  [list \
    $::env(DESIGN_DIR)/src/rtl/top/soc_pkg.sv \
    $::env(DESIGN_DIR)/src/rtl/noc/noc_pkg.sv \
    $::env(DESIGN_DIR)/src/rtl/memory/coherence_pkg.sv \
  ] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/buffer/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/cache/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/control/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/dma/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/dram/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/hft/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/host_iface/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/mac/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/memory/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/monitor/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/noc/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/periph/*.sv]] \
  [lsort [glob -nocomplain $::env(DESIGN_DIR)/src/rtl/systolic/*.sv]] \
  [list \
    $::env(DESIGN_DIR)/src/rtl/top/axi_addr_decoder.sv \
    $::env(DESIGN_DIR)/src/rtl/top/axi_arbiter.sv \
    $::env(DESIGN_DIR)/src/rtl/top/axi_crossbar.sv \
    $::env(DESIGN_DIR)/src/rtl/top/obi_to_axi.sv \
    $::env(DESIGN_DIR)/src/rtl/top/simple_cpu.sv \
    $::env(DESIGN_DIR)/src/rtl/top/soc_top_v2.sv \
  ] \
]

set ::env(FP_SIZING) relative
set ::env(FP_CORE_UTIL) 35
set ::env(PL_TARGET_DENSITY) 0.45
set ::env(RUN_KLAYOUT) 0
set ::env(RUN_MAGIC) 0
