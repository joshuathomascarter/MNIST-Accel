TOPLEVEL_LANG = verilog
TOPLEVEL = accel_top
MODULE = test_accel_top

SIM = verilator
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += -Wno-fatal

VERILOG_SOURCES = \
    $(PWD)/../../rtl/top/accel_top.sv \
    $(PWD)/../../rtl/host_iface/axi_lite_slave.sv \
    $(PWD)/../../rtl/host_iface/axi_dma_bridge.sv \
    $(PWD)/../../rtl/control/csr.sv \
    $(PWD)/../../rtl/control/bsr_scheduler.sv \
    $(PWD)/../../rtl/dma/act_dma.sv \
    $(PWD)/../../rtl/dma/bsr_dma.sv \
    $(PWD)/../../rtl/dma/out_dma.sv \
    $(PWD)/../../rtl/dma/dma_pack_112.sv \
    $(PWD)/../../rtl/buffer/output_accumulator.sv \
    $(PWD)/../../rtl/buffer/act_buffer.sv \
    $(PWD)/../../rtl/buffer/wgt_buffer.sv \
    $(PWD)/../../rtl/systolic/pe.sv \
    $(PWD)/../../rtl/systolic/systolic_array_sparse.sv \
    $(PWD)/../../rtl/mac/mac8.sv \
    $(PWD)/../../rtl/monitor/perf.sv

include $(shell cocotb-config --makefiles)/Makefile.sim