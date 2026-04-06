# soc_top_v2 FPGA Bringup Bundle

This directory is the staged FPGA-side handoff for `soc_top_v2`.

## Included

- `rtl/` — live SystemVerilog RTL snapshot
- `sim/` — simulation sources and filelists with generated `obj_dir*` folders pruned
- `fw/` — bare-metal firmware sources plus the current built firmware artifacts
- `tools/` — Vivado and local verification helper scripts
- `tools/fpga_uart_block_demo.py` + `tools/soc_top_v2_uart_host.py` — UART preload / block-execute / readback flow
- `docs/` — architecture, simulation, verification, and port-reference documents
- `docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md` — board-side integration punch list
- `hw/firmware.hex` — boot ROM image mirrored next to the bundle

## Not Included

- FPGA bitstream output
- Vivado-generated implementation reports
- board-specific wrapper, block design, or pin-assignment outputs

## What Is Already Done In This Workspace

- `soc_top_v2` Verilator simulation is passing on the main SoC testbench
- tile-level Verilator regression coverage is present in `sim/`
- firmware image and source are already staged for UART-led smoke bring-up

## External Steps Still Required

1. Run `tools/run_synthesis.sh` in a Vivado environment.
2. Work through `docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md`.
3. Generate the bitstream and board reports.
4. Run `tools/fpga_uart_block_demo.py --port <tty>` for the preload / compute / readback loop.

Use `MANIFEST.txt` for the exact file list.
