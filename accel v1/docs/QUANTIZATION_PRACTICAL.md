# ACCEL-v1 Quantization - Practical Examples and CSR Programming

This supplement contains hands-on examples for packing scales, programming CSRs, troubleshooting, and numeric validation.

## Table of Contents

- [Practical CSR Programming Examples](#practical-csr-programming-examples-and-endianness)
- [pack_f32 and endianness](#pack_f32-and-endianness)
- [Writing Scales to CSRs](#writing-scales-to-csrs-host-side-example)
- [Packing Fixed-Point Scale](#packing-fixed-point-scale-hardware-fixed-point-representation)
- [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
- [Numeric Example](#numeric-example-scale-combination-and-overflow)
- [Checklist for Deployment](#checklist-for-deployment)

---

## Practical CSR Programming Examples and Endianness

The accelerator expects scale factors and CSR values packed as little-endian 32-bit words. Below are precise host-side examples showing how to pack scales and write them into CSRs using Python.

### pack_f32 and endianness

Use Python's `struct` packing with little-endian format (`'<f'`) to convert FP32 scale values into 4 bytes matching the hardware CSR layout:

```python
import struct

def pack_f32_le(value: float) -> bytes:
    """Pack a Python float into 4 bytes little-endian (IEEE-754 single precision)."""
    return struct.pack('<f', float(value))

# Example usage
>>> pack_f32_le(0.125)
b"\x00\x00\x00?"  # little-endian bytes (illustrative)
```

**Note**: The exact byte string will differ; always use `struct.pack('<f', value)` to ensure hardware-compatible byte order.

### Writing Scales to CSRs (host-side example)

This example demonstrates writing activation and weight scales to the accelerator using the existing UART packet format. Replace `uart` with an instance of `UARTDriver` (or `MockUARTDriver` in tests).

```python
from host_uart.uart_driver import UARTDriver
from host_uart.csr_map import CMD_WRITE, CSR_SCALE_ACT, CSR_SCALE_WGT
import struct

# Initialize UART driver
uart = UARTDriver('/dev/ttyUSB0', baudrate=115200)

# Scale values (example)
Sa = 0.125   # activation scale
Sw = 0.0625  # weight scale

# Pack scale values and CSR addresses into payloads
payload_sa = CSR_SCALE_ACT.to_bytes(4, 'little') + struct.pack('<f', Sa)
payload_sw = CSR_SCALE_WGT.to_bytes(4, 'little') + struct.pack('<f', Sw)

# Send CSR write commands
uart.send_packet(CMD_WRITE, payload_sa)
uart.send_packet(CMD_WRITE, payload_sw)

# Optional: read back CSR to verify
# uart.send_packet(CMD_READ, CSR_SCALE_ACT.to_bytes(4, 'little'))
# resp = uart.recv_packet()
```

### Packing Fixed-Point Scale (hardware fixed-point representation)

If the hardware expects a fixed-point representation rather than a float CSR, convert using a chosen fractional width (e.g., 24 fractional bits):

```python
def encode_scale_fixed(scale: float, frac_bits: int = 24) -> int:
    # Saturate to 32-bit unsigned
    fixed = int(round(scale * (1 << frac_bits)))
    fixed &= 0xFFFFFFFF
    return fixed

# Pack into little-endian 32-bit
fixed_bytes = encode_scale_fixed(0.125).to_bytes(4, 'little')
```

Check `csr_map.py` for whether scales are written as FP32 or fixed-point. Use the matching packing function.

## Common Pitfalls and Troubleshooting

### Endianness Issues
- **Problem**: Writing big-endian bytes to a little-endian CSR will cause the hardware to interpret an incorrect scale (often yielding extremely large or zero values).
- **Solution**: Always use `'<f'` format and `to_bytes(..., 'little')` for consistency.

### Missing Scale Programming
- **Problem**: Forgetting to program `Sa`/`Sw` will produce incorrect output. Results can be orders of magnitude off.
- **Solution**: Always verify scale CSR writes before starting computation.

### Accumulator Overflow
- **Problem**: INT32 accumulator width may be insufficient for large K dimensions and tile sizes.
- **Solution**: Ensure scale selection or shifting accounts for headroom. Monitor accumulator range.

### Communication Errors
- **Problem**: Packets get dropped due to CRC mismatch or communication issues.
- **Solution**: Enable verbose logging on the host and check the stream parser behavior in `host_uart/uart_driver.py`.

## Numeric Example: Scale Combination and Overflow

Consider small example values to illustrate the impact of scales:

```python
import numpy as np

A_fp32 = np.array([[0.5, -0.25]], dtype=np.float32)  # 1×2
B_fp32 = np.array([[0.25], [0.125]], dtype=np.float32)  # 2×1

Sa = np.max(np.abs(A_fp32)) / 127.0  # example scale
Sw = np.max(np.abs(B_fp32)) / 127.0

A_q = np.round(A_fp32 / Sa).astype(np.int8)
B_q = np.round(B_fp32 / Sw).astype(np.int8)

C_int32 = (A_q.astype(np.int32) @ B_q.astype(np.int32))
combined_scale = Sa * Sw
C_fp32 = C_int32.astype(np.float32) * combined_scale

print('A_q', A_q, 'Sa', Sa)
print('B_q', B_q, 'Sw', Sw)
print('C_int32', C_int32, 'C_fp32', C_fp32)
```

This shows how the INT32 accumulator output `C_int32` maps back to the floating-point domain via `combined_scale`.

## Checklist for Deployment

1. Calibrate weight scales (`Sw`) using per-layer weight max or percentile.
2. Calibrate activation scales (`Sa`) using representative calibration dataset.
3. Decide on output scale or implement right-shift/scale combination on host/hardware.
4. Pack scales using `struct.pack('<f', scale)` or fixed-point encoder as required by `csr_map.py`.
5. Write scales to CSRs and verify readback (use `MockUARTDriver` for local tests).
6. Run bit-exact tests with small matrices to confirm behavior before full model deployment.

---

If you'd like, I can now automatically insert example scripts into `python/INT8 quantization/` and a short notebook demonstrating the calibration workflow. Say "add scripts and notebook" and I'll create them next.
