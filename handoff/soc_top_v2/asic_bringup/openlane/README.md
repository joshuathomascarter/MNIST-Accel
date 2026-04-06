# soc_top_v2 OpenLane Starter

This directory is the current ASIC/OpenLane handoff starting point for
`soc_top_v2`.

## What It Contains

- `prep_openlane_sources.sh` — builds a cleaned source bundle from the real
  simulation filelist
- `constraints/soc_top_v2.sdc` — starter clock/reset timing constraints
- `constraints/soc_top_v2.pin_order.cfg` — side-based IO grouping for the
  current full-SoC top
- `POWER_ARCHITECTURE.md` — first-pass single-domain recommendation
- `MACRO_PLAN.md` — frozen memory macro migration plan
- `TAPEOUT_TARGET.md` — frozen `soc_top_v2` scope and backend handoff target
- `TAPEOUT_READINESS_CHECKLIST.md` — blunt ordered risk checklist
- `IO_PAD_PLAN.md` — pad/pin grouping plan for the full SoC top
- `RESET_STRATEGY.md` — reset policy for ASIC bring-up
- `DFT_SCAN_PLAN.md` — first-pass DFT/scan strategy
- `soc_top_v2_single_domain.upf` — minimal one-domain power-intent placeholder

## Recommended Flow

1. Run `bash prep_openlane_sources.sh`
2. Review generated `src/` and `config.tcl`
3. Treat the generated bundle as the real `soc_top_v2` source set; it filters
  out known comparison/demo RTL that is not instantiated in the current SoC
4. Start with a single-domain OpenLane run using `clk` at `20.0 ns`
5. Only add macros/power domains after the generic flow is stable
6. Keep `SPARSE_VC_ALLOC=0` and `INNET_REDUCE=0` for the default handoff run;
   treat novelty-enabled builds as experimental comparison configurations
7. Keep the handoff frozen on `soc_top_v2`; reduce risk by holding optional
  features off and closing the baseline full-chip path before exploring more
  aggressive variants

## Current Assumptions

- RTL language: SystemVerilog
- Top module: `soc_top_v2`
- Clock port: `clk`
- Starter clock period: `20.0 ns` (`50 MHz`)
- ASIC synthesis define: `ASIC_SYNTHESIS`
- Default NoC config: `SPARSE_VC_ALLOC=0`, `INNET_REDUCE=0`
- Full-chip pin grouping file: `constraints/soc_top_v2.pin_order.cfg`

## Local Tooling Note

The local Yosys 0.61 install in this workspace is currently a frontend
limitation, not a trustworthy pass/fail gate for `soc_top_v2` readiness. Use
the generated bundle for handoff, but expect final backend bring-up to happen
in an environment with a stronger SystemVerilog frontend.
