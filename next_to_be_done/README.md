# Next To Be Done - ACCEL-v1 Future Development

This folder contains stub files for future development tasks. All files are empty stubs ready for implementation.

## Development Tasks

### 🧩 **IM2COL & Convolution Support**
- `python/tiler/im2col_tiles.py` - IM2COL tiling aligned to (Tm,Tn,Tk)
- `python/tests/test_im2col_tiles.py` - IM2COL tiling tests

### 🔢 **Quantization & Post-Processing**
- `python/post/requant.py` - Fixed-point INT32→INT8 with ties-to-even
- `python/train_qat.py` - Quantization-Aware Training (PyTorch)
- `data/checkpoints/mnist_qat.pt.placeholder` - QAT model checkpoint

### **Performance & Analysis**
- `python/utils/perf_counters.py` - MAC utilization, reuse metrics
- `python/utils/plot_roofline.py` - Roofline model visualization
- `python/utils/plot_perf_w.py` - Power/performance analysis
- `rtl/monitors/perf.v` - Hardware performance counters

### **Testing & Validation**
- `python/tests/test_edges.py` - Edge cases and CRC testing
- `tb/fuzz/tb_uart_crc_fuzz.sv` - UART CRC fuzz testing
- `tb/unit/tb_dma_lite.sv` - DMA unit tests
- `tb/integration/tb_accel_dma.sv` - DMA integration tests

### **Hardware Scaling**
- `rtl/systolic/systolic_array.v` - Scale to 4x4 or 8x4 array
- `rtl/control/scheduler.v` - Double-buffer overlap (ping/pong)
- `rtl/dma/dma_lite.v` - AXI-Stream DMA engine
- `rtl/accel_top.v` - DMA-integrated top level

### 💾 **DMA Integration**
- `python/host_dma/run_dma.py` - DMA-based host driver
- Alternative to UART for high-bandwidth transfers

### **Documentation**
- `docs/power_scaling.md` - FPGA power analysis
- `docs/perf_overlap.md` - Double-buffer overlap details
- `docs/RESULTS.md` - Performance results and scaling

### 🔄 **CI/CD Pipeline**
- `.github/workflows/ci.yml` - GitHub Actions CI
- `Makefile` - Build and test automation

## **Implementation Priority**

1. **IM2COL tiling** - Enable convolution support
2. **Performance counters** - Hardware monitoring
3. **Array scaling** - Improve throughput
4. **DMA engine** - Replace UART bottleneck
5. **QAT training** - Improve accuracy
6. **CI pipeline** - Automate testing
7. **Power analysis** - Performance/Watt metrics

## 📝 **Notes**

All files are **empty stubs** with TODO comments indicating what needs to be implemented. Each file includes the original requirements from the development roadmap.

Start with any task based on your current priorities!