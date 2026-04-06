# OpenLane Handoff Notes

This repo now has a credible ASIC cleanup package, but it is still not ready
for clean full-chip signoff on `soc_top_v2`. Treat the current state as a
serious handoff starting point, not a finished tapeout flow.

## Frozen Recommendation

- **Best chance first silicon:** harden `accel_top` first as a reusable macro.
- **Stretch full-chip target:** baseline `soc_top_v2` with
  `SPARSE_VC_ALLOC=0` and `INNET_REDUCE=0`.
- **Do not tape out the novelty-on NoC configuration first.** Keep the INR and
  sparse-allocator variants for publishable comparison and later silicon spins.

## Current Design Identity

- RTL format: SystemVerilog
- Full-chip functional top: `soc_top_v2`
- Smaller existing subsystem top: `accel_top`
- Primary RTL root: `hw/rtl/`
- Current source inventory: `hw/sim/sv/filelist.f`
- Preferred current-top prep flow: `hw/openlane/soc_top_v2/prep_openlane_sources.sh`
- Primary clock assumption: `clk` at `20.0 ns` (`50 MHz`)
- Current full-chip handoff area: `hw/openlane/soc_top_v2/`

## What Is Already Cleaned Up

- `csr.sv` bypasses its manual latch-based clock gate under `ASIC_SYNTHESIS`.
- `soc_top_v2` now uses asynchronous reset assertion with synchronized reset
  release before distributing `rst_core_n`.
- FPGA-only `ram_style` hints are now guarded behind `XILINX_FPGA` in the live
  inferred-memory blocks that were still carrying them.
- The `hw/openlane/soc_top_v2/` directory now includes a pin-order file,
  stronger starter SDC, target-freeze doc, macro plan, IO plan, reset plan,
  DFT plan, and a blunt readiness checklist.

## Current Full-Chip Blockers

1. There is still no clean backend run in an environment with a stronger
   SystemVerilog frontend than the local Yosys 0.61 setup.
2. The memory macro plan is now defined, but macro wrappers plus LEF/LIB
   integration are not implemented yet.
3. Pad cells, padring generation, package-aware DRAM timing, and final IO ESD
   choices are still planning-level collateral only.
4. Scan insertion, chain balancing, and MBIST are not integrated into the RTL
   or backend flow yet.
5. If the target remains `soc_top_v2`, the CPU/cache/TLB/PTW/crossbar/DRAM side
   is still the dominant timing and congestion risk.

## What Marcus Should Use

For the current full-chip baseline, start from these paths:

- `hw/rtl/`
- `hw/sim/sv/filelist.f`
- `hw/openlane/soc_top_v2/prep_openlane_sources.sh`
- `hw/openlane/soc_top_v2/config.tcl`
- `hw/openlane/soc_top_v2/constraints/soc_top_v2.sdc`
- `hw/openlane/soc_top_v2/constraints/soc_top_v2.pin_order.cfg`

Do not use these as backend sources:

- `hw/sim/`
- testbenches
- SVA-only files
- Vivado XDC/TCL files as ASIC constraints
- `firmware.hex`
- comparison/demo RTL that is not instantiated by the chosen target

## Practical Path

1. If schedule risk matters, harden `accel_top` first and use that macro as the
   first taped-out artifact.
2. If the project must stay full-chip, use the `hw/openlane/soc_top_v2/`
   collateral and keep the NoC novelty toggles off.
3. Run backend work in a stronger OpenROAD/OpenLane environment than the local
   Yosys-only setup.
4. Implement the macro wrappers from `MACRO_PLAN.md` before taking full-chip
   timing numbers seriously.
5. Treat the current reset, DFT, and IO docs as the frozen first-pass policy,
   then let backend turn them into foundry-specific implementation collateral.
