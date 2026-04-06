# soc_top_v2 ASIC Bringup Bundle

This directory is the staged pre-OpenROAD ASIC-side handoff for `soc_top_v2`.

## Included

- `openlane/` — regenerated source bundle, constraints, config, UPF, and planning docs
- `rtl_reference/` — raw RTL snapshot for correlation against the filtered OpenLane bundle
- `sim_reference/` — source-side simulation collateral used to derive the ASIC bundle
- `fw_reference/` and `hw_reference/firmware.hex` — bring-up software context
- `tools_reference/` — source-prep and Yosys helper scripts
- `tools_reference/asic_uart_block_demo.py` + `tools_reference/soc_top_v2_uart_host.py` — UART preload / block-execute / readback flow
- `docs_reference/` — architecture, timing, verification, and microarchitecture docs

## Not Included

- OpenROAD or OpenLane run outputs
- generated DEF/GDS/LEF/LIB artifacts
- backend signoff reports

## What Is Already Done In This Workspace

- `openlane/src/` has been regenerated from the live simulation filelist
- `openlane/config.tcl` is aligned to the current `soc_top_v2` source set
- reset, power, pad, macro, and DFT planning collateral are all staged together

## External Steps Still Required

1. Run the backend flow in the target OpenROAD/OpenLane environment.
2. Bind real foundry SRAM macros and tech libraries.
3. Complete scan/DFT insertion and package-aware pad timing.
4. Review DRC, LVS, STA, IR-drop, and congestion results.
5. Use `rtl_reference/top/soc_top_v2_asic_sim_wrapper.sv` plus `tools_reference/asic_uart_block_demo.py` for pre-silicon preload / compute / readback validation.

Use `MANIFEST.txt` for the exact file list.
