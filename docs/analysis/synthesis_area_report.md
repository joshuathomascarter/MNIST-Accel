# Synthesis Area Report — Month 1 Milestone

## Summary

Estimated resource utilisation for the complete SoC design targeting
**Xilinx Zynq-7020** (xc7z020clg400-1).

| Category             | LUTs   | FFs    | DSP48 | BRAM36 | Notes                           |
|----------------------|--------|--------|-------|--------|---------------------------------|
| **Systolic Array**   | 4,200  | 3,100  | 256   | 0      | 16×16 INT8 weight-stationary    |
| **BSR Scheduler**    | 800    | 400    | 0     | 0      | Sparse feed + tile FSM          |
| **Output Buffer**    | 200    | 100    | 0     | 4      | Double-buffered output BRAM     |
| **Max-Pool Unit**    | 80     | 60     | 0     | 0      | 2×2 max, bypassable            |
| **CSR Block**        | 150    | 200    | 0     | 0      | Control/status registers        |
| **DMA Engine**       | 400    | 300    | 0     | 2      | AXI4 burst DMA                  |
| **Accel Top Glue**   | 100    | 50     | 0     | 0      | MUX, FSM, wiring                |
| *Accelerator Sub*    | *5,930*| *4,210*| *256* | *6*    |                                 |
| **Boot ROM**         | 50     | 0      | 0     | 2      | 4 KB instruction ROM            |
| **SRAM Controller**  | 100    | 50     | 0     | 16     | 64 KB data SRAM                 |
| **AXI Crossbar**     | 600    | 400    | 0     | 0      | 2M × 8S × 32-bit               |
| **OBI→AXI Bridge**   | 150    | 100    | 0     | 0      | CPU bus adapter                 |
| **UART**             | 200    | 150    | 0     | 0      | TX + RX + AXI regs              |
| **Timer**            | 100    | 80     | 0     | 0      | 64-bit counter + compare        |
| **GPIO**             | 80     | 60     | 0     | 0      | 32-bit I/O                      |
| **PLIC**             | 200    | 100    | 0     | 0      | 32-source interrupt ctrl        |
| **Simple CPU**       | 800    | 500    | 0     | 0      | Placeholder (→ Ibex Month 2)    |
| *SoC Peripherals Sub*| *2,280*| *1,440*| *0*   | *18*   |                                 |
| **DRAM Bank FSM ×8** | 960    | 400    | 0     | 0      | Per-bank timing FSM             |
| **DRAM Cmd Queue**   | 200    | 200    | 0     | 0      | 16-entry command FIFO           |
| **DRAM Write Buffer**| 100    | 520    | 0     | 0      | 16-entry write data             |
| **DRAM Refresh Ctrl**| 50     | 30     | 0     | 0      | tREFI / tRFC                    |
| **DRAM Scheduler**   | 400    | 150    | 0     | 0      | FR-FCFS                         |
| **DRAM Det. Mode**   | 80     | 160    | 0     | 0      | Fixed-latency HFT               |
| **DRAM Power Model** | 40     | 30     | 0     | 0      | CKE / power-down                |
| **DRAM Ctrl Top**    | 100    | 50     | 0     | 0      | Integration + AXI front-end     |
| *DRAM Controller Sub*| *1,930*| *1,540*| *0*   | *0*    |                                 |
| **HFT Modules**      | 400    | 300    | 0     | 0      | MAC RX, UDP parser, async FIFO  |
| **Fixed-Point ALU**  | 200    | 100    | 1     | 0      | 32-bit fixed-point              |
| *HFT Sub*            | *600*  | *400*  | *1*   | *0*    |                                 |

### Totals

| Resource | Used     | Available | Utilisation |
|----------|----------|-----------|-------------|
| LUTs     | 10,740   | 53,200    | **20.2%**   |
| FFs      | 7,590    | 106,400   | **7.1%**    |
| DSP48E1  | 257      | 220       | **116.8%**  |
| BRAM36   | 24       | 140       | **17.1%**   |

## Analysis

### Critical Resource: DSP48

At 116.8% DSP utilisation, the 16×16 INT8 systolic array (256 DSP48s)
exceeds the Zynq-7020's 220 DSP48E1 budget by 36 units. Options for Month 2+:
- Reduce to 14×14 array (196 DSPs = 89%) to fit within budget
- Use LUT-based multipliers for the 36 excess MAC positions
- Time-multiplex: 8×8 array running 4× clock (but adds complexity)
- Target a larger device (e.g. Zynq-7045 with 900 DSPs)

### Comfortable Resources: LUTs, FFs, BRAM

All three are well under 50%, leaving headroom for:
- Month 2: Ibex CPU core (~3000 LUTs)
- Month 3: DMA enhancements, debug infrastructure
- Month 4: Network-on-Chip mesh

### Timing Estimate

- Accelerator data path: critical path through DSP cascade ≈ 4.5 ns (220 MHz feasible)
- DRAM controller: longest path = scheduler selection logic ≈ 5.0 ns (200 MHz target)
- AXI crossbar: arbitration logic ≈ 3.5 ns
- **Target Fmax: 100 MHz system clock** (conservative, with margin)

## Methodology

Estimates are based on:
1. Known DSP48 utilisation (1 DSP per 8-bit multiplier)
2. Yosys synthesis netlist cell counts (where available)
3. Component-level estimates from similar open-source designs
4. BRAM inference from declared memory sizes

Full Vivado synthesis will be run in Month 2 for accurate P&R numbers.

## Risks

| Risk                          | Likelihood | Mitigation                    |
|-------------------------------|------------|-------------------------------|
| DSP budget exceeded (256>220) | High       | LUT-based MACs or larger device |
| Clock domain crossing issues  | Low        | Single clock domain Month 1   |
| BRAM port contention          | Low        | True dual-port, no conflicts  |
| LUT carry-chain timing        | Low        | Well under utilisation limits  |
