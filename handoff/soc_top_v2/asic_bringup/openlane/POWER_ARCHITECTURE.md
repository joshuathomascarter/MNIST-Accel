# soc_top_v2 Power Architecture Starter

For the first OpenLane/OpenROAD bring-up, use a single power domain.

Recommended first-pass assumptions:

- One top-level `VDD` and one top-level `VSS`
- No multi-voltage partitioning in the initial run
- No explicit clock gating cells in the initial run
- No power gating in the initial run
- Revisit multi-domain power intent only after the full SoC is stable in
  generic synthesis and floorplanning

Rationale:

- The existing `accel_top.upf` is for the older accelerator-centric partition,
  not the current `soc_top_v2` integration.
- OpenLane bring-up is simpler and less fragile with a single-domain top.
- The current source tree was proven functionally in FPGA-style simulation, not
  in a full multi-domain ASIC flow.

Recommended phase order:

1. Single-domain synth/floorplan/place route on `soc_top_v2`
2. Measure area and congestion hot spots
3. Decide whether the full SoC remains monolithic or needs partitioning
4. Add integrated clock-gating cells and re-run CTS/timing
5. Only then define multi-domain power intent, level shifters, and isolation
