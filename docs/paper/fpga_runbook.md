# ZCU104 FPGA Runbook

Complete step-by-step instructions to synthesize, implement, and demonstrate
the MNIST-Accel SoC on the AMD ZCU104 (xczu7ev-ffvc1156-2-e).

---

## Prerequisites

| Requirement | Details |
|------------|---------|
| OS | Linux (Ubuntu 20.04 or 22.04 recommended) |
| Vivado | 2022.2 or 2023.2, **Design Edition** (not WebPACK — ZCU104 device not in free tier) |
| License | Vivado Design Edition node-locked or floating license |
| RAM | ≥ 32 GB (implementation uses ~20–28 GB peak) |
| Disk | ≥ 50 GB free for project + reports |
| Runtime | ~2–4 hours total (synth ≈ 45 min, impl ≈ 90 min, bitstream ≈ 30 min) |

---

## Step 1 — Clone the Repo

```bash
git clone <repo-url> MNIST-Accel
cd MNIST-Accel
```

---

## Step 2 — Generate the DRAM Hex Init File

The FPGA bitstream pre-loads MNIST model weights into on-chip memory.
The DRAM init hex must exist at `data/dram_init.hex`.

```bash
# If already present, skip this step.
ls data/dram_init.hex

# If not present, regenerate from model weights:
python3 tools/gen_dram_init.py \
    --weights data/int8/mnist_int8_weights.npz \
    --output  data/dram_init.hex
```

---

## Step 3 — Check RTL Filelist

The Vivado script reads `hw/sim/sv/filelist.f` to find all RTL sources plus
`hw/rtl/top/zcu104_wrapper.sv` (added separately as the FPGA-only top).

```bash
wc -l hw/sim/sv/filelist.f          # should be ~55+ files
ls hw/rtl/top/zcu104_wrapper.sv     # board wrapper
ls hw/constraints/zcu104.xdc        # pin constraints
```

---

## Step 4 — Run Vivado Synthesis + Implementation

```bash
vivado -mode batch -source tools/synthesize_vivado.tcl 2>&1 | tee vivado_run.log
```

This script performs all 8 steps automatically:

| Step | Description |
|------|-------------|
| 1 | Create Vivado project in `hw/vivado_proj/` |
| 2 | Add all RTL sources + `zcu104_wrapper.sv` |
| 3 | Add `zcu104.xdc` pin constraints + timing constraints |
| 4 | Run synthesis (`synth_1`) |
| 5 | Check post-synthesis WNS/TNS |
| 6 | Run implementation (`impl_1`) with `Explore` directive |
| 7 | Check post-route WNS/TNS — exits with error if timing fails |
| 8 | Generate bitstream; copy to `hw/zcu104_wrapper.bit` |

**The script exits 1 if post-route timing fails** (WNS < 0). Target: 50 MHz.

**Expected outputs:**
```
hw/zcu104_wrapper.bit                  ← flash this to the board
hw/reports/impl_utilization.rpt
hw/reports/impl_timing.rpt             ← contains true WNS at 50 MHz
hw/reports/impl_power.rpt              ← power estimate in Watts
hw/reports/synthesis_summary.json      ← machine-readable WNS + freq
```

---

## Step 5 — Flash Bitstream to ZCU104

### Option A — Vivado Hardware Manager (GUI)

```bash
vivado
# → Open Hardware Manager
# → Open Target → Auto Connect
# → Program Device → select hw/zcu104_wrapper.bit
```

### Option B — OpenOCD / xsdb

```bash
xsdb -interactive
connect
targets -set -filter {name =~ "xczu7ev*"}
fpga hw/zcu104_wrapper.bit
```

### Option C — SD Card (persistent)

Copy the bitstream to SD card as `BOOT.BIN` using Xilinx bootgen (requires
a `.bif` file). Only needed if you want the FPGA to load on power-up without
a PC.

---

## Step 6 — Connect UART

Attach a USB-to-TTL adapter (3.3 V logic level, e.g. CP2102 or FTDI232) to
PMOD J160 on ZCU104:

| USB-TTL wire | PMOD J160 pin | Signal |
|-------------|--------------|--------|
| RX (adapter) | Pin 1 (top-left) | SoC UART TX |
| TX (adapter) | Pin 2 | SoC UART RX |
| GND | Pin 5 or 6 | Ground |

Open a terminal at **115200 baud, 8N1**:

```bash
screen /dev/ttyUSB0 115200
# or
minicom -D /dev/ttyUSB0 -b 115200
# or (macOS)
screen /dev/cu.usbserial-* 115200
```

---

## Step 7 — Power on and Demo

1. Power on ZCU104.
2. If the bitstream was flashed, the SoC boots automatically.
3. Otherwise, flash via Vivado Hardware Manager first.
4. Press the center pushbutton (M11) to reset if needed.

**Expected UART output:**

```
[BOOT] MNIST-Accel SoC v2 — sky130/ZCU104 demo
[BOOT] Firmware loaded. Running MNIST inference...
[ACCEL] Tile array initialized (16 tiles, 4x4 mesh)
[INFER] Loading digit 7 from DRAM...
[INFER] Running Conv1... done
[INFER] Running FC1... done
[INFER] Running FC2... done
Predicted: 7
True label: 7
PASS: matches golden
Cycles: 0000a029
```

**Expected LED state:**

| LED | Meaning |
|-----|---------|
| LED0 | Accel busy (flashes during compute) |
| LED1 | Accel done (solid after inference) |
| LED2 | GPIO[2] — firmware controlled |
| LED3 | GPIO[3] — firmware controlled |

GPIO[7:4] = `0xF` (done), GPIO[3:0] = predicted digit (e.g. `7`).

---

## Step 8 — Capture Results

After a successful run, record:

1. **Post-route WNS** from `hw/reports/impl_timing.rpt`:
   ```
   grep "WNS" hw/reports/impl_timing.rpt | head -3
   ```

2. **Power** from `hw/reports/impl_power.rpt`:
   ```
   grep -A5 "Total On-Chip Power" hw/reports/impl_power.rpt
   ```

3. **Utilization** from `hw/reports/impl_utilization.rpt`:
   ```
   grep -E "(LUT|FF|BRAM|URAM|DSP)" hw/reports/impl_utilization.rpt | head -10
   ```

4. **Throughput**: already measured at **53.6 inf/sec @ 50 MHz** from Verilator
   E2E simulation — the FPGA run confirms this in hardware.

---

## Expected Resource Utilization

Based on pre-synthesis estimates (see `docs/paper/design_report.md` §5.2):

| Resource | ZCU104 Available | Estimated Used | Est. % |
|----------|-----------------|----------------|--------|
| LUT6 | 230,400 | ~170,000 | ~74% |
| FF | 460,800 | ~120,000 | ~26% |
| BRAM36 | 312 | ~141 | ~45% |
| URAM (288 Kb) | 96 | ~56 | ~58% |
| DSP58E2 | 1,728 | ~1,728 + ~320 LUT | ~100% |

---

## Troubleshooting

**Vivado exits with "Timing constraints NOT met":**
- Check `impl_timing.rpt` for the critical path — likely in NoC switch allocator or systolic address generation
- Add `set_multicycle_path -setup 2` for paths you know are multi-cycle
- Reduce clock to 40 MHz: change `clk_period_ns` to `25.0` in `synthesize_vivado.tcl` and update MMCM divider to 25.0

**UART shows garbage:**
- Verify baud rate is exactly 115200 — the SoC uses `CLK_FREQ / (16 × BAUD)` divider
- Check TX/RX are not swapped (common mistake with USB-TTL adapters)

**Board doesn't boot after bitstream:**
- Verify MMCM `LOCKED` — check LED behavior; if no LEDs light, the 125 MHz clock may not be getting to the PL
- Press and hold the center button (cpu_reset = M11) for 1 second then release

**Implementation runs out of RAM:**
- Use `set_property STEPS.IMPL_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]` instead of `Explore`
- Reduce to `-jobs 2` in the TCL script

---

*All files referenced above are in the MNIST-Accel repository root.*
