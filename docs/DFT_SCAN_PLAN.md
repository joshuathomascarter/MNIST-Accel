# DFT and Scan Plan

## First-Pass Goal

The immediate goal is a scan-ready digital baseline, not an aggressive,
compressed production-test architecture.

## Frozen Policy

- single scan clock based on `clk`
- single `scan_en` control
- regular muxed scan insertion, no scan compression on first pass
- memories handled as non-scannable macros or black boxes
- MBIST deferred until the memory wrappers are in place

## Recommended Chain Counts

- `accel_top` macro target: start with 2 to 4 scan chains
- `soc_top_v2` full-chip target: expect at least 8 to 16 chains, depending on
  backend congestion and tester limits

## Required Implementation Steps

1. Freeze the taped-out top before adding DFT ports.
2. Add only the minimum scan ports to that chosen top.
3. Insert scan after the reset and clock policies are frozen.
4. Keep inferred memories out of the scan plan; replace them with wrappers or
   macros first.
5. Run ATPG review only after synthesis is happening in the intended backend
   environment.

## Assumptions

- First-pass ASIC flow uses one clock domain.
- Manual latch-based clock gates are not part of the ASIC baseline.
- Reset synchronizers and any explicit CDC flops may need backend-specific
  `dont_scan` handling.

## Out Of Scope For This Pass

- scan compression
- LBIST
- JTAG controller integration
- MBIST controller insertion
- multi-clock scan stitching