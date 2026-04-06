# =============================================================================
# pynq_z2.xdc — Pin constraints for PYNQ-Z2 board (xc7z020clg400-1)
# =============================================================================
# Companion to hw/rtl/top/pynq_z2_wrapper.sv
# Reference: Digilent PYNQ-Z2 schematic rev C, TUL master XDC
#
# Active pin set:
#   clk_125m   — 125 MHz PL oscillator
#   btn0_n     — reset (directly active press)
#   pmoda_tx   — SoC UART TX (connect USB-TTL adapter)
#   pmoda_rx   — SoC UART RX
#   led[3:0]   — LD0–LD3
#   led4_r/g   — RGB LED 4 red/green
#   sw[1:0]    — slide switches
#   btn2, btn3 — push buttons
# =============================================================================

# ---- 125 MHz PL reference clock -------------------------------------------
set_property -dict {PACKAGE_PIN H16 IOSTANDARD LVCMOS33} [get_ports clk_125m]
create_clock -period 8.000 -name clk_125m -waveform {0.000 4.000} [get_ports clk_125m]

# ---- Reset (active press = low) -------------------------------------------
set_property -dict {PACKAGE_PIN D19 IOSTANDARD LVCMOS33} [get_ports btn0_n]

# ---- UART on PMODA (directly active — active press accent) ----------------
# Pin 1 (top row, leftmost) = TX out
# Pin 2 (top row, second)   = RX in
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports pmoda_tx]
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports pmoda_rx]

# ---- LEDs ------------------------------------------------------------------
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# ---- RGB LED 4 (accent) — accent active-low on board ----------------------
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports led4_r]
set_property -dict {PACKAGE_PIN G17 IOSTANDARD LVCMOS33} [get_ports led4_g]

# ---- Switches --------------------------------------------------------------
set_property -dict {PACKAGE_PIN M20 IOSTANDARD LVCMOS33} [get_ports {sw[0]}]
set_property -dict {PACKAGE_PIN M19 IOSTANDARD LVCMOS33} [get_ports {sw[1]}]

# ---- Buttons ---------------------------------------------------------------
set_property -dict {PACKAGE_PIN L20 IOSTANDARD LVCMOS33} [get_ports btn2]
set_property -dict {PACKAGE_PIN L19 IOSTANDARD LVCMOS33} [get_ports btn3]

# ---- Timing ----------------------------------------------------------------
# Derived 50 MHz clock is generated internally by MMCM; Vivado will infer it
# from the MMCME2_BASE output in pynq_z2_wrapper.  No create_generated_clock
# needed here unless you override the auto-derived constraint.

# False-path the async reset
set_false_path -from [get_ports btn0_n]

# I/O delay — conservative for PMODA (3.3 V, short traces)
set_input_delay  -clock clk_125m 3.0 [get_ports {pmoda_rx sw[*] btn2 btn3}]
set_output_delay -clock clk_125m 3.0 [get_ports {pmoda_tx led[*] led4_r led4_g}]
