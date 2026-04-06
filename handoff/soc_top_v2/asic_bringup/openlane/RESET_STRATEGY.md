# Reset Strategy

## Frozen First-Pass Policy

- One external active-low reset input: `rst_n`
- Asynchronous assertion from the pad or POR source
- Synchronized deassertion onto the core clock before distribution through the
  chip
- No combinational reset generation inside the design
- No reset-domain partitioning in the first pass

## What Is Implemented Now

- `soc_top_v2` now synchronizes reset release internally and distributes a
  cleaned `rst_core_n`.
- The ASIC build of `csr.sv` no longer depends on manual clock gating, which
  removes one reset-propagation hazard from the baseline path.

## What Backend Still Needs To Do

- Replace the abstract top-level `rst_n` source with the actual POR/pad hookup.
- If `accel_top` becomes the taped-out macro, apply the same asynchronous-assert
  and synchronized-release policy at that top.
- Mark the reset synchronizer flops according to backend DFT policy if they are
  to be excluded from normal scan stitching.
- Keep the current false-path treatment from the raw reset input to normal data
  timing analysis.

## What We Are Explicitly Not Doing Yet

- separate debug reset
- separate CPU and accelerator reset domains
- power-gated reset islands
- reset compression or scan-reset muxing beyond what backend requires