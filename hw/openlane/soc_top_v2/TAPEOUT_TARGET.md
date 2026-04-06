# Tapeout Target Freeze

## Frozen Answer

- **Target top:** `soc_top_v2`
- **Frozen configuration:** baseline full-chip build with
  `SPARSE_VC_ALLOC=0` and `INNET_REDUCE=0`
- **Backend policy:** reduce risk inside the current SoC scope rather than
  pivoting to a different top-level

## What Is In Scope

The current handoff is the full `soc_top_v2` chip, including:

- CPU, OBI-to-AXI bridge, TLB, and page-table walker
- L1 and L2 cache hierarchy
- tile array, scratchpads, DMA gateway, and NoC fabric
- DRAM front-end and PHY-facing top-level interface
- UART, timer, GPIO, PLIC, Ethernet, and performance monitor blocks

## First-Silicon Freeze Rules

To keep the backend problem bounded while staying on `soc_top_v2`, freeze to:

- single `clk` domain
- single `VDD` and single `VSS`
- baseline NoC feature set with novelty options disabled
- current pad, reset, DFT, and macro plans in this directory as the only
  implementation reference set

## Risk Reduction Strategy

The way to de-risk this handoff is to hold the chip boundary fixed and close the
remaining implementation gaps in order:

1. Replace inferred SRAMs with foundry-backed macros and bindings.
2. Complete padframe, package, and DRAM timing implementation.
3. Add scan/DFT infrastructure and review ATPG evidence.
4. Run full backend closure in the intended OpenROAD/OpenLane environment.

## Out Of Scope For The Frozen Handoff

Do not expand the scope with:

- novelty-on NoC experiments for first silicon
- new top-level wrappers or alternate chip targets
- multi-voltage partitioning before single-domain closure is stable