# soc_top_v2 FPGA Board Integration Checklist

This checklist is the remaining FPGA-side punch list for `soc_top_v2` after the
repo-local source prep is complete.

## Inputs Already Ready In This Workspace

- `soc_top_v2` RTL: `hw/rtl/top/soc_top_v2.sv`
- firmware smoke test: `fw/main.c`
- tile/NoC CSR interface: `fw/hal_accel.h`
- simulation and filelists: `hw/sim/sv/`
- Vivado driver scripts: `tools/run_synthesis.sh`, `tools/synthesize_vivado.tcl`
- staged FPGA handoff bundle: `handoff/soc_top_v2/fpga_bringup/`

## External Artifacts Still Required

- board wrapper RTL around `soc_top_v2`
- board XDC constraints for clock, reset, UART, GPIO, and debug pins
- Vivado project outputs, implementation reports, and the final bitstream
- optional PS/block-design plumbing if a Zynq PS-assisted demo is required

## Checklist

1. Freeze the board-demo boundary.
   Use `soc_top_v2` as the core top, keep `SPARSE_VC_ALLOC=0` and `INNET_REDUCE=0`, and do not add new architectural features during board integration.
2. Add a board wrapper around `soc_top_v2`.
   Feed a single board clock into `clk`, generate active-low reset for `rst_n`, route `uart_rx` and `uart_tx`, and decide which board pins or debug nets will expose `gpio_o`, `gpio_i`, `gpio_oe`, `accel_busy`, and `accel_done`.
3. Resolve the DRAM-interface mismatch explicitly.
   `soc_top_v2` exports raw `dram_phy_*` signals, which do not directly match the PYNQ-Z2 PS DDR interface. For a first board demo, implement a shim or wrapper that accepts the SoC DRAM commands and returns deterministic `dram_phy_rdata` plus `dram_phy_rdata_valid`. If direct PS DDR use is required, that bridge must be written as a separate integration block.
4. Satisfy the current firmware expectations.
   `fw/main.c` expects UART output, GPIO writes, a full-tile barrier, one DMA load from `0x40000000`, and one compute command to complete. The board integration must allow that sequence to progress without hanging the DRAM path.
5. Freeze the clock plan before Vivado implementation.
   The RTL and ASIC handoff currently assume a 50 MHz baseline, while `tools/synthesize_vivado.tcl` still targets 100 MHz. Choose one frequency before synthesis, update constraints consistently, and keep firmware UART settings aligned with that choice.
6. Create board constraints.
   Add an XDC covering `clk`, `rst_n`, `uart_rx`, `uart_tx`, chosen GPIO pins, and any debug pins or ILAs. If some SoC outputs are consumed only inside the wrapper, constrain the wrapper ports rather than the raw SoC signals.
7. Build the boot image used by the Boot ROM.
   Rebuild `fw/firmware.hex` from the current firmware sources, confirm the image is the one referenced by `BOOT_ROM_FILE`, and make sure the Vivado project picks up the same file.
8. Run the first bringup with a narrow success criterion.
   A successful first boot is: UART prints `ACCEL-v1 SoC`, `Barrier OK`, `DMA OK`, `Compute OK`, and `Done.`; GPIO reaches `0xF0`; and `accel_done` or tile status confirms progress.
9. Instrument the first debug pass.
   If the board does not boot cleanly, probe `rst_core_n`, UART transmit, Boot ROM fetch progress, `dram_phy_read`, `dram_phy_rdata_valid`, `dram_ctrl_busy`, tile 0 status, `accel_busy`, and `accel_done`.

## Minimum Wrapper Behavior For A Smoke Demo

- clock and reset delivered cleanly to `soc_top_v2`
- UART routed to a real serial endpoint
- GPIO visible on LEDs, PMOD, or ILA
- DRAM shim accepts the load/store traffic needed by tile 0
- `dram_phy_rdata_valid` asserted often enough for the DMA load in `fw/main.c`
- no permanent backpressure on the SoC DRAM side

## Deliverables To Produce Outside This Repo

- wrapper RTL or block-design glue for the board
- board XDC
- Vivado synthesis and implementation reports
- `.bit` file and any hardware handoff files used by the board workflow