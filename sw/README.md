# ACCEL-v1 Software Stack

Host-side software for the ACCEL-v1 sparse CNN accelerator. Includes a C++ driver framework for Zynq deployment and Python tooling for model training, INT8 quantization, BSR export, and golden-model verification.

---

## Directory Structure

```
sw/
├── cpp/                           # C++ host driver framework
│   ├── include/
│   │   ├── compute/               #   BSR encoder, golden models, tiling
│   │   ├── driver/                #   Accelerator driver, CSR interface
│   │   ├── memory/                #   Address map, buffer manager, DMA
│   │   └── utils/                 #   NPY loader, logging
│   ├── src/                       #   Implementation files
│   ├── apps/
│   │   ├── run_mnist_inference.cpp #   MNIST inference application
│   │   └── benchmark.cpp          #   Performance benchmarking
│   └── tests/                     #   Unit tests (BSR, DMA, tiling, e2e)
│
└── ml_python/                     # Python ML & verification tooling
    ├── training/
    │   ├── export_bsr_14x14.py    #   14×14 BSR weight export (production)
    │   ├── export_bsr.py          #   Generic BSR export utilities
    │   └── blocksparse_train.py   #   Block-sparse aware training
    ├── exporters/
    │   ├── export_conv.py         #   Convolution weight export
    │   └── export_transformer.py  #   Transformer weight export
    ├── golden/
    │   ├── gemm_bsr_int8.py       #   BSR INT8 GEMM golden model
    │   └── golden_fc1_test.py     #   FC1 layer verification
    ├── golden_models/
    │   ├── gemm_int8.py           #   Dense INT8 GEMM reference
    │   └── golden_mac8.py         #   MAC8 unit reference model
    ├── host/
    │   ├── accel.py               #   PYNQ accelerator driver
    │   ├── axi_driver.py          #   AXI transaction layer
    │   ├── axi_master_sim.py      #   AXI master simulation model
    │   └── memory.py              #   DMA buffer management
    ├── host_axi/
    │   ├── csr_map.py             #   CSR register definitions
    │   └── run_gemm_axi.py        #   AXI-based GEMM execution
    ├── demo/
    │   └── classify_digit.py      #   MNIST digit classification demo
    ├── tests/                     #   Unit and integration tests
    ├── INT8 quantization/
    │   └── quantize.py            #   Post-training quantization
    └── MNIST CNN/
        └── train_mnist.py         #   MNIST CNN training (98.9% FP32)
```

---

## C++ Host Driver

The C++ framework targets Zynq bare-metal or Linux deployment via `/dev/mem` mapped AXI registers. It provides BSR encoding, DMA buffer management, and golden-model verification.

### Build

```bash
cd sw/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Key Components

| Module | Header | Description |
|--------|--------|-------------|
| BSR Encoder | `compute/bsr_encoder.hpp` | Dense-to-BSR conversion, hardware serialization |
| Golden Model | `compute/golden_model.hpp` | Bit-exact INT8 GEMM reference (dense + BSR) |
| Tiling | `compute/tiling.hpp` | Matrix tiling for 14×14 systolic array |
| Accelerator | `driver/accelerator.hpp` | CSR control, layer configuration, inference |
| DMA | `memory/dma_controller.hpp` | AXI DMA buffer allocation and transfers |
| Buffer Mgr | `memory/buffer_manager.hpp` | Physically contiguous memory management |
| Address Map | `memory/address_map.hpp` | DDR region layout for weights/activations/output |

### Tests

```bash
cd sw/cpp/build
ctest --output-on-failure

# Individual tests:
./test_bsr_encoder
./test_buffer_manager
./test_dma
./test_tiling
./test_end_to_end
```

---

## Python ML Tooling

### Training and Export

```bash
# Train MNIST CNN (FP32 baseline, 98.9% accuracy)
cd sw/ml_python/"MNIST CNN"
python train_mnist.py

# Quantize to INT8 (per-channel, 98.7% accuracy)
cd sw/ml_python/"INT8 quantization"
python quantize.py

# Export to 14×14 BSR format for hardware
cd sw/ml_python/training
python export_bsr_14x14.py --from-int8
```

### Golden Models

Bit-exact Python reference implementations for verifying hardware results:

```bash
# Verify BSR INT8 GEMM against dense reference
cd sw/ml_python/golden
python gemm_bsr_int8.py

# Verify FC1 layer output
python golden_fc1_test.py
```

### MNIST Demo

```bash
cd sw/ml_python/demo
python classify_digit.py
```

Loads quantized weights, runs inference through all four CNN layers (conv1 → conv2 → fc1 → fc2), and prints the predicted digit class.

### PYNQ Driver

The Python host driver (`sw/ml_python/host/accel.py`) targets the PYNQ-Z2 overlay interface:

```python
from host.accel import AccelDriver

driver = AccelDriver(bitstream="accel_top.bit")
driver.load_bsr_weights("data/bsr_export_14x14/fc1/")
driver.load_activations(activation_tensor)
driver.start()
driver.wait_done()
result = driver.read_output()
```

### Tests

```bash
cd sw/ml_python/tests
python -m pytest test_golden_models.py -v
python -m pytest test_exporters.py -v
python -m pytest test_mac.py -v
```

---

## Data Formats

### INT8 Quantized Weights (`data/int8/`)

Per-layer files:
- `{layer}_weight_int8.npy` — INT8 weight tensor
- `{layer}_weight_scales.npy` — Per-channel scale factors
- `{layer}_bias_int8.npy` — INT8 bias values
- `{layer}_bias_scale.json` — Bias scale metadata

### BSR Export (`data/bsr_export_14x14/`)

Per-layer directories containing:
- `weights.bsr` — Packed non-zero 14×14 blocks (INT8)
- `row_ptr.npy` — BSR row pointer array
- `col_idx.npy` — BSR column index array
- `weights.meta.json` — Block dimensions, sparsity, shape metadata

---

## Known Issues

- C++ driver has partial stub implementations (inference pipeline, some tests)
- `host_axi/csr_map.py` uses non-standard register offsets (needs alignment with RTL CSR map)
- Some Python test files reference a removed `host_uart/` module
- See top-level [AUDIT.md](../AUDIT.md) for a complete code audit