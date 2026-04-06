# Address Interleaving Analysis — DRAM Controller

## Summary

This document analyses the two address interleaving modes supported by
`dram_addr_decoder.sv` and their impact on access-pattern efficiency.

## Interleaving Modes

### Mode 0: RBC (Row → Bank → Column)

```
 [  ROW (14b)  |  BANK (3b)  | COL (10b) | BYTE (1b) ]
  Bit 27       Bit 13        Bit 10       Bit 0
```

- **Sequential accesses** traverse *columns within one bank* before moving to the next bank.
- Good for **streaming / DMA** workloads: a single long burst stays in one row of one bank → high row-hit rate.
- Poor for **random** access across many banks (limited bank-level parallelism).

### Mode 1: BRC (Bank → Row → Column)

```
 [  BANK (3b)  |  ROW (14b)  | COL (10b) | BYTE (1b) ]
  Bit 27       Bit 24        Bit 10       Bit 0
```

- **Sequential accesses** interleave across *banks* first → maximises bank-level parallelism.
- Ideal for **random / multi-master** traffic mixes (accelerator + CPU both active).
- Higher precharge frequency per bank since row changes more often within each bank.

## Expected Performance

| Metric              | RBC (Mode 0)      | BRC (Mode 1)      |
|---------------------|--------------------|--------------------|
| Row-hit rate (DMA)  | ~80–95 %          | ~30–60 %           |
| Row-hit rate (rand) | ~5–15 %           | ~5–15 %            |
| Bank-level ||ism    | Low                | High               |
| Avg. read latency   | Low (streaming)    | Low (random)       |
| Refresh impact      | Moderate           | Spread             |

## Architectural Recommendation

For the MNIST accelerator workload:
- **Tile loads** (Conv/FC weight rows) are sequential in memory → **RBC** gives
  highest row-hit rate and lowest average latency.
- **Multi-layer overlap** (CPU triggers next layer while DMA fills current) benefits
  from BRC's bank parallelism.

**Default: Mode 0 (RBC)** — optimised for the dominant streaming pattern.
Mode can be switched at runtime via the CSR register `DRAM_ADDR_MODE`.

## Verification

1. `test_dram_addr_interleave.py` validates both decode modes against a Python
   reference model.
2. Bank-crossing and row-crossing boundary checks confirm correct field extraction.
3. Full-system DRAM simulation CSVs fed to `tools/plot_dram_experiments.py`
   produce row-hit-rate and bank-utilisation plots for visual comparison.

## Resource Cost

The address decoder is purely combinational:
- **~30 LUTs** (bit-field extraction + MUX for mode select).
- 0 FFs, 0 BRAM, 0 DSP.
