# =============================================================================
# zcu104.xdc — Pin constraints for ZCU104 board (xczu7ev-ffvc1156-2-e)
# =============================================================================
# Companion to hw/rtl/top/zcu104_wrapper.sv
# Reference: AMD UG1267 (ZCU104 Evaluation Board User Guide v1.1)
#
# IMPORTANT: Verify all PACKAGE_PIN assignments against the ZCU104 schematic
# and the AMD master XDC for your board revision before synthesis.
#
# Active pin set:
#   sysclk_125_p/n — 125 MHz PL differential oscillator
#   cpu_reset      — CPU reset pushbutton (center, active-high)
#   pmod_tx/rx     — SoC UART TX/RX on PMOD header J160
#   led[3:0]       — User LEDs 0–3
#   dip_sw[3:0]    — 4-position DIP switch
#   btn_n/s/e/w    — Directional push buttons
# =============================================================================

# ---- 125 MHz PL differential reference clock -------------------------------
# Fixed oscillator → PL pins  (Bank 64, DIFF_SSTL12)
set_property PACKAGE_PIN E12 [get_ports sysclk_125_p]
set_property PACKAGE_PIN D12 [get_ports sysclk_125_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports sysclk_125_p]
set_property IOSTANDARD DIFF_SSTL12 [get_ports sysclk_125_n]
create_clock -period 8.000 -name sysclk_125 -waveform {0.000 4.000} [get_ports sysclk_125_p]

# ---- CPU Reset (active-high press) -----------------------------------------
set_property -dict {PACKAGE_PIN M11 IOSTANDARD LVCMOS33} [get_ports cpu_reset]

# ---- UART on PMOD J160 (directly active — attach USB-TTL adapter) ----------
# Top row pin 1 = TX out, pin 2 = RX in
# Bank 65 (HD bank, 3.3 V PMOD)
# NOTE: Verify exact PMOD pins against ZCU104 schematic for your board rev.
set_property -dict {PACKAGE_PIN D7  IOSTANDARD LVCMOS33} [get_ports pmod_tx]
set_property -dict {PACKAGE_PIN F8  IOSTANDARD LVCMOS33} [get_ports pmod_rx]

# ---- User LEDs (active-high) -----------------------------------------------
# Bank 65 (HD bank, 3.3 V)
set_property -dict {PACKAGE_PIN D5  IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN D6  IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN A5  IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN B5  IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# ---- DIP Switches -----------------------------------------------------------
# Bank 64 (HP bank, 1.8 V)
set_property -dict {PACKAGE_PIN A17 IOSTANDARD LVCMOS18} [get_ports {dip_sw[0]}]
set_property -dict {PACKAGE_PIN A16 IOSTANDARD LVCMOS18} [get_ports {dip_sw[1]}]
set_property -dict {PACKAGE_PIN B16 IOSTANDARD LVCMOS18} [get_ports {dip_sw[2]}]
set_property -dict {PACKAGE_PIN B15 IOSTANDARD LVCMOS18} [get_ports {dip_sw[3]}]

# ---- Push Buttons (active-high) --------------------------------------------
# Bank 65 (HD bank, 3.3 V)
set_property -dict {PACKAGE_PIN C4  IOSTANDARD LVCMOS33} [get_ports btn_n]
set_property -dict {PACKAGE_PIN B3  IOSTANDARD LVCMOS33} [get_ports btn_s]
set_property -dict {PACKAGE_PIN C3  IOSTANDARD LVCMOS33} [get_ports btn_e]
set_property -dict {PACKAGE_PIN A4  IOSTANDARD LVCMOS33} [get_ports btn_w]

# ---- Timing -----------------------------------------------------------------
# The 50 MHz derived clock is generated internally by MMCME4_ADV; Vivado will
# infer it from the MMCME4_ADV output in zcu104_wrapper.  No explicit
# create_generated_clock needed unless you override the auto-derived constraint.

# False-path the async reset button
set_false_path -from [get_ports cpu_reset]

# False-path DIP switches (asynchronous input, directly sampled)
set_false_path -from [get_ports {dip_sw[*]}]
set_false_path -from [get_ports {btn_n btn_s btn_e btn_w}]

# I/O delay — conservative for PMOD (3.3 V, short traces)
set_input_delay  -clock sysclk_125 3.0 [get_ports pmod_rx]
set_output_delay -clock sysclk_125 3.0 [get_ports pmod_tx]
set_output_delay -clock sysclk_125 3.0 [get_ports {led[*]}]
