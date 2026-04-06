# PYNQ Host Drivers

Python drivers for controlling ACCEL-v1 on Zynq-7020 FPGA via PYNQ.

## Files

| File | Description |
|------|-------------|
| `accel.py` | Main accelerator driver class |
| `memory.py` | DMA buffer allocation, BSR format utilities |
| `benchmark_sparse.py` | Sparse matrix benchmarks |
| `power_profiling.py` | Zynq power measurement |

## Quick Start

### On PYNQ Board

```python
from accel import SparseAccelerator
import numpy as np

# Initialize accelerator
accel = SparseAccelerator("/path/to/overlay.bit")

# Create sparse weight matrix (BSR format)
weights = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
weights[np.abs(weights) < 100] = 0  # 75% sparsity

# Create activations
activations = np.random.randint(-128, 127, (64, 16), dtype=np.int8)

# Run sparse MatMul
result = accel.sparse_matmul(weights, activations, sparsity=0.75)
print(f"Output shape: {result.shape}")
print(f"Utilization: {accel.get_utilization():.1%}")
```

### Simulation Mode (No FPGA)

```bash
# Runs in simulation mode automatically
python3 accel.py
python3 benchmark_sparse.py
python3 power_profiling.py
```

## CSR Register Map

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | CTRL | R/W | Control: bit0=start, bit1=reset |
| 0x04 | STATUS | R | Status: bit0=busy, bit1=done |
| 0x08 | M_DIM | R/W | M dimension (rows) |
| 0x0C | N_DIM | R/W | N dimension (columns) |
| 0x10 | K_DIM | R/W | K dimension (inner) |
| 0x20 | CYCLE_CNT | R | Total cycles |
| 0x24 | ACTIVE_CNT | R | Active compute cycles |
| 0x28 | STALL_CNT | R | Stall cycles |

## Benchmarks

```bash
python3 benchmark_sparse.py
```

Output:
```
================================================================================
ACCEL-v1 Sparse Matrix Multiply Benchmarks
================================================================================
  spmm_64x64x64_s50   | CPU: 0.15ms | Accel: 0.005ms | Speedup: 28.8x | Util: 80%
  spmm_128x128x128_s50| CPU: 1.33ms | Accel: 0.005ms | Speedup: 258x  | Util: 80%
```

## Power Profiling

```python
from power_profiling import ZynqPowerMonitor

with ZynqPowerMonitor() as monitor:
    # Run workload
    result = accel.sparse_matmul(weights, activations)
    
profile = monitor.get_profile("inference")
print(f"Average Power: {profile.avg_power_mw:.1f} mW")
print(f"Energy: {profile.energy_mj:.3f} mJ")
print(f"Efficiency: {profile.gops_per_watt:.2f} GOPS/W")
```

## Dependencies

**On PYNQ:**
- pynq >= 2.7
- numpy >= 1.20
- scipy (optional, for sparse utilities)

**Simulation (no FPGA):**
- numpy >= 1.20 only
