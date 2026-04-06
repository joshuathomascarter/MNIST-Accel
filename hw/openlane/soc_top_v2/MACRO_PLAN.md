# soc_top_v2 Memory Macro Plan

The repo no longer treats inferred memories as an acceptable tapeout end state.
Generic memory inference is allowed only for the first sizing and floorplanning
passes. Before signoff, the blocks below need explicit wrappers plus LIB/LEF
views from the chosen SRAM compiler.

Current status:

- The accel-top-local banks and BSR metadata tables now sit behind the
  technology-neutral `sram_1rw_wrapper` RTL abstraction.
- Foundry macro binding is still pending. The current wrapper implementation is
  a generic synchronous model, not a final compiled SRAM instance.

## Frozen Scope Assumptions

- Stretch full-chip target: `soc_top_v2` with `SPARSE_VC_ALLOC=0` and
  `INNET_REDUCE=0`
- Best-chance first-silicon target: `accel_top` hardened as a reusable macro
- First-pass power architecture: one `VDD`, one `VSS`, one clock

## Macro Decisions

| Logical storage | Current RTL location | Recommended macro shape | Notes |
| --- | --- | --- | --- |
| Activation ping-pong banks | `act_buffer` | 2 x single-port 1RW SRAMs | Read and write happen on opposite banks, so each physical bank does not need a true dual-port macro |
| Weight ping-pong banks | `wgt_buffer` | 2 x single-port 1RW SRAMs | Same banking argument as `act_buffer` |
| Output ping-pong banks | `output_bram_buffer` | 2 x single-port 1RW SRAMs | One bank drains while the other fills |
| BSR row pointer table | `accel_top` `row_ptr_bram` | 1 x small 1RW SRAM | Size is modest, but move it behind a wrapper so timing is modeled correctly |
| BSR column index table | `accel_top` `col_idx_bram` | 1 x small 1RW SRAM | Same wrapper plan as row pointers |
| Weight block staging store | `accel_top` weight block RAM | 1 x 1RW SRAM or banked 1RW SRAMs | Choose banking based on final burst width and compiler options |
| Tile-local scratchpad | `accel_scratchpad` | 1R1W macro if retained, else banked 1RW replacement | Only needed if this block remains in the taped-out scope |
| Boot ROM | `boot_rom` | ROM or foundry-supported compiled memory | Keep generic only for bring-up builds |
| On-chip SRAM peripheral | `sram_ctrl` backing array | Single-port SRAM macro | Freeze byte-write behavior in the wrapper |
| L1 cache arrays | cache data/tag arrays | Separate tag and data macros | Do not attempt full-chip signoff on `soc_top_v2` without real cache macros |
| L2 cache arrays and MSHR storage | L2/cache hierarchy | Separate macros by function | This is one reason `soc_top_v2` is the stretch target, not the best-chance target |

## Bring-Up Order

1. Run generic synthesis/place on the chosen target only to estimate area,
   congestion, and macro counts.
2. Replace accelerator-local banks first: `act_buffer`, `wgt_buffer`,
   `output_bram_buffer`, and the BSR metadata stores.
3. If the target remains `accel_top`, stop there and harden the macro cleanly.
4. Only then decide whether full-chip `soc_top_v2` remains realistic enough to
   justify cache, SRAM, and boot ROM macro integration.
5. Add macro hooks, LEF/LIB views, and explicit floorplan constraints before
   asking backend to close timing.

## Wrapper Rules

- Put technology-neutral wrappers in the RTL tree instead of instantiating
  compiler cells directly in datapath modules.
- Keep functional ports stable so the generic inferred-memory model and the
  ASIC macro implementation share the same interface.
- Gate macro selection with one top-level define or parameter, not ad hoc
  per-module edits.
- Treat timing from generic inferred memories as provisional only.

## Wrapper Progress

- Implemented now:
  - `act_buffer`
  - `wgt_buffer`
  - `output_bram_buffer`
  - `accel_top` row-pointer and column-index metadata tables
- Still pending:
  - explicit foundry macro binding inside the wrapper layer
  - any scratchpad-specific 1R1W or banked replacement for `accel_scratchpad`
  - cache, ROM, and SRAM-peripheral wrapper integration for the full SoC path

Without this migration, a full `soc_top_v2` run may elaborate, but the timing,
area, and power numbers will not be credible.
