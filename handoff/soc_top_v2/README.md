# soc_top_v2 Handoff Bundles

- `fpga_bringup/` contains the staged FPGA-side source bundle for `soc_top_v2`.
- `fpga_bringup/rtl/top/pynq_z2_wrapper.sv` is the FPGA bringup top with a persistent local DRAM backing store.
- `fpga_bringup/tools/fpga_uart_block_demo.py` is the FPGA preload / compute / readback host entrypoint.
- `fpga_bringup/docs/SOC_TOP_V2_FPGA_BOARD_INTEGRATION_CHECKLIST.md` is the concrete board-side punch list.
- `asic_bringup/` contains the staged pre-OpenROAD ASIC-side source bundle for `soc_top_v2`.
- `asic_bringup/rtl_reference/top/soc_top_v2_asic_sim_wrapper.sv` is the ASIC pre-silicon bringup wrapper with the same backing-store model.
- `asic_bringup/tools_reference/asic_uart_block_demo.py` is the ASIC preload / compute / readback host entrypoint.

Each subdirectory includes a `README.md` with usage notes and a `MANIFEST.txt`
with the exact staged files.
