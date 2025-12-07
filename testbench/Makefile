# =============================================================================
# Makefile — Verilator Build System for ACCEL-v1
# =============================================================================
# Author: Joshua Carter
# Date: November 19, 2025
# Target: Verilator 5.0+
#
# Usage:
#   make              # Build all RTL into C++ simulator
#   make run          # Run simulation (1000 cycles)
#   make wave         # View waveforms in GTKWave
#   make clean        # Remove build artifacts
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================
VERILATOR = verilator
VERILATOR_ROOT ?= $(shell verilator --getenv VERILATOR_ROOT)

# Verilator flags
VERILATOR_FLAGS = \
	--cc \
	--exe \
	--build \
	-Wall \
	--trace \
	-Wno-WIDTH \
	-Wno-UNUSED \
	-Wno-UNOPTFLAT \
	--top-module accel_top \
	-CFLAGS "-std=c++14 -O3" \
	-LDFLAGS "-lpthread"

# RTL source files
RTL_DIR = ../../rtl
RTL_SOURCES = \
	$(RTL_DIR)/systolic/pe.sv \
	$(RTL_DIR)/systolic/systolic_array.sv \
	$(RTL_DIR)/mac/mac8.sv \
	$(RTL_DIR)/buffer/act_buffer.sv \
	$(RTL_DIR)/buffer/wgt_buffer.sv \
	$(RTL_DIR)/control/scheduler.sv \
	$(RTL_DIR)/control/csr.sv \
	$(RTL_DIR)/dma/bsr_dma.sv \
	$(RTL_DIR)/dma/dma_lite.sv \
	$(RTL_DIR)/uart/uart_rx.sv \
	$(RTL_DIR)/uart/uart_tx.sv \
	$(RTL_DIR)/monitor/perf.sv \
	$(RTL_DIR)/top/accel_top.sv

# C++ testbench
CPP_SOURCES = test_accel_verilator.cpp

# Output executable
EXEC = obj_dir/Vaccel_top

# =============================================================================
# Targets
# =============================================================================

.PHONY: all run wave clean help

all: $(EXEC)

$(EXEC): $(RTL_SOURCES) $(CPP_SOURCES)
	@echo "=========================================="
	@echo "Building with Verilator..."
	@echo "=========================================="
	$(VERILATOR) $(VERILATOR_FLAGS) $(RTL_SOURCES) $(CPP_SOURCES)
	@echo ""
	@echo "✅ Build complete: $(EXEC)"
	@echo ""

run: $(EXEC)
	@echo "=========================================="
	@echo "Running simulation (1000 cycles @ 100 MHz)"
	@echo "=========================================="
	./$(EXEC)
	@echo ""
	@echo "✅ Simulation complete!"
	@echo "Generated: trace.vcd (waveform)"
	@echo ""

wave: trace.vcd
	@echo "Opening GTKWave..."
	gtkwave trace.vcd &

trace.vcd: $(EXEC)
	./$(EXEC)

clean:
	@echo "Cleaning build artifacts..."
	rm -rf obj_dir *.vcd *.log

help:
	@echo "ACCEL-v1 Verilator Build System"
	@echo "==============================="
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build RTL simulator"
	@echo "  make run      - Run simulation"
	@echo "  make wave     - View waveforms"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Performance:"
	@echo "  Verilator is 10-100× faster than iverilog"
	@echo "  Typical: 10K cycles in ~0.1s vs 5s"
	@echo ""
