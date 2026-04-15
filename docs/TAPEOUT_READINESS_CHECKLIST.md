# Tapeout Readiness Checklist

Ordered by risk, bluntly.

1. **PARTIAL:** Accelerator-local SRAM wrappers now exist in RTL, but foundry
   macro binding and LIB/LEF integration do not. Full-chip timing and area
   numbers are still not credible until that second step is done.
2. **OPEN:** There is no clean backend run yet in a strong SystemVerilog
   frontend environment. Local Yosys limitations are still a real blocker.
3. **OPEN:** DFT is still planning-level only. No scan ports, no inserted scan
   chains, no ATPG evidence, and no MBIST integration yet.
4. **OPEN:** Pad cells, padring generation, ESD choices, and package-aware DRAM
   timing are not implemented.
5. **PARTIAL:** Reset policy is now defined, and `soc_top_v2` uses synchronized
   reset release internally, but top-level POR/pad integration is still pending.
6. **PARTIAL:** The ASIC path no longer depends on the CSR manual clock gate,
   and FPGA-only memory hints are isolated, but real library ICG insertion is
   intentionally deferred.
7. **PARTIAL:** OpenLane collateral is now materially better than the original
   starter set: SDC, pin order, scope freeze, macro plan, IO plan, reset plan,
   and DFT plan are present.
8. **FROZEN:** `soc_top_v2` is the only taped-out artifact for this handoff.
   Risk reduction must happen inside the current full-chip scope, not by
   pivoting to a different top-level.
9. **FROZEN:** If `soc_top_v2` is taped out, keep `SPARSE_VC_ALLOC=0` and
   `INNET_REDUCE=0` for first silicon.

## Exit Criteria Before Calling The Full SoC Tapeout-Ready

- Macro wrappers compiled and integrated
- clean synthesis/place/route in the intended backend environment
- scan inserted and ATPG reviewed
- padframe and package timing frozen
- power/clock tree strategy signed off by backend
- full-chip DRC/LVS/timing reports reviewed instead of inferred