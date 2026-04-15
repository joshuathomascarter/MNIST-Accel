# IO and Pad Plan

This file freezes the first-pass IO grouping for `soc_top_v2`. It is a planning
document, not a final padring netlist.

## Scope Split

- For the best-chance taped-out artifact (`accel_top`), prefer macro hardening
  without a full padring.
- For full-chip `soc_top_v2`, use the grouping below and the matching
  `constraints/soc_top_v2.pin_order.cfg` as the floorplan starting point.

## Side Assignment

- **North:** DRAM-facing bus signals
  - `dram_phy_act[*]`
  - `dram_phy_read[*]`
  - `dram_phy_write[*]`
  - `dram_phy_pre[*]`
  - `dram_phy_row[*]`
  - `dram_phy_col[*]`
  - `dram_phy_ref`
  - `dram_phy_wdata[*]`
  - `dram_phy_wstrb[*]`
  - `dram_phy_rdata[*]`
  - `dram_phy_rdata_valid`
  - `dram_ctrl_busy`
- **South:** software-visible low-speed pins
  - `gpio_i[*]`
  - `gpio_o[*]`
  - `gpio_oe[*]`
  - `uart_rx`
  - `uart_tx`
- **East:** infrastructure pins
  - `clk`
  - `rst_n`
- **West:** status and interrupt pins
  - `irq_external`
  - `irq_timer`
  - `accel_busy`
  - `accel_done`

## Pad Guidance

- Use a dedicated clock input pad for `clk`.
- Treat `rst_n` as a POR-qualified digital input, not just a random GPIO-style
  pad.
- Keep the DRAM-facing signals on the same side to limit long top-level routes.
- Group low-speed control/status pads away from the DRAM bundle.
- Distribute `VDD`/`VSS` pads around all four sides, with extra supply density
  near the DRAM side because that side carries the widest bus cluster.

## What Is Still Missing

- exact pad cell list for the target PDK
- ESD and corner pad selection
- bump or bondout map
- package-aware timing numbers for the DRAM side
- foundry-specific padring implementation file