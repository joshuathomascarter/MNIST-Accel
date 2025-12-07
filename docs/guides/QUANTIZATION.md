# ACCEL-v1 INT8 Quantization Methodology

## Overview

This document describes the INT8 quantization scheme implemented in ACCEL-v1, including the mathematical formulation, calibration procedures, and implementation details for achieving efficient CNN inference on the systolic array accelerator.

## Quantization Theory

### Mathematical Foundation

#### Symmetric Quantization
ACCEL-v1 uses **symmetric quantization** with zero-point at zero for both weights and activations:

```
Q = round(R / S)
R = Q × S
```

Where:
- `Q`: Quantized INT8 value [-128, 127]
- `R`: Real (FP32) value  
- `S`: Scale factor (FP32)

#### Scale Factor Calculation
```python
def calculate_scale(tensor, dtype=np.int8):
    """Calculate symmetric scale factor"""
    abs_max = np.max(np.abs(tensor))
    if dtype == np.int8:
        q_max = 127  # Symmetric range for INT8
    else:
        q_max = (2**(bits-1)) - 1
    
    scale = abs_max / q_max
    return scale
```

#### Quantization Function
```python
def quantize_symmetric(tensor, scale, dtype=np.int8):
    """Apply symmetric quantization"""
    quantized = np.round(tensor / scale)
    
    # Clamp to valid range
    if dtype == np.int8:
        quantized = np.clip(quantized, -128, 127)
    
    return quantized.astype(dtype), scale
```

## GEMM Quantization Pipeline

### Input Quantization
For matrix multiplication `C = A × B`:

1. **Activation Matrix A**: Quantized to INT8 with scale `Sa`
2. **Weight Matrix B**: Quantized to INT8 with scale `Sw`  
3. **Output Matrix C**: Accumulated in INT32, then requantized to INT8

### Mathematical Derivation

#### FP32 Reference
```
C_fp32[i,j] = Σ(k=0 to K-1) A_fp32[i,k] × B_fp32[k,j]
```

#### INT8 Computation
```
C_int32[i,j] = Σ(k=0 to K-1) A_int8[i,k] × B_int8[k,j]
```

#### Scale Relationship
```
C_fp32[i,j] = C_int32[i,j] × Sa × Sw
```

#### Output Requantization
```python
def requantize_output(C_int32, Sa, Sw, Sc):
    """Requantize INT32 accumulator to INT8 output"""
    # Combined scale factor
    combined_scale = Sa * Sw / Sc
    
    # Scale and round
    C_scaled = C_int32 * combined_scale
    C_rounded = np.round(C_scaled)
    
    # Clamp to INT8 range
    C_int8 = np.clip(C_rounded, -128, 127).astype(np.int8)
    
    return C_int8
```

## Calibration Methodology

### Dataset-Based Calibration

#### Weight Calibration
```python
def calibrate_weights(weight_tensor):
    """Per-tensor weight calibration"""
    # Calculate symmetric scale
    abs_max = np.max(np.abs(weight_tensor))
    scale = abs_max / 127.0
    
    # Quantize weights
    weights_int8 = np.round(weight_tensor / scale)
    weights_int8 = np.clip(weights_int8, -128, 127).astype(np.int8)
    
    return weights_int8, scale
```

#### Activation Calibration
```python
def calibrate_activations(activation_samples, percentile=99.99):
    """Percentile-based activation calibration"""
    # Collect statistics across calibration dataset
    abs_values = []
    for sample in activation_samples:
        abs_values.extend(np.abs(sample.flatten()))
    
    # Use percentile to handle outliers
    abs_max = np.percentile(abs_values, percentile)
    scale = abs_max / 127.0
    
    return scale
```

### Per-Channel vs Per-Tensor

#### Current Implementation: Per-Tensor
- **Weights**: Single scale factor per weight matrix
- **Activations**: Single scale factor per activation tensor
- **Advantages**: Simple hardware implementation, minimal storage overhead

#### Future Enhancement: Per-Channel
- **Weights**: Scale factor per output channel
- **Benefits**: Better precision for unbalanced weight distributions
- **Hardware Cost**: Additional scale storage and multipliers

## Hardware Implementation

### PE-Level Quantization

#### MAC Operation
```verilog
// INT8 × INT8 → INT32 MAC in PE
always @(posedge clk) begin
    if (rst_n) begin
        acc_reg <= 0;
    end else if (mac_enable) begin
        // Signed multiplication: INT8 × INT8 → INT16
        product = $signed(act_in) * $signed(wgt_stored);
        // Accumulation: INT16 + INT32 → INT32  
        acc_reg <= acc_reg + $signed(product);
    end
end
```

#### Requantization Unit
```verilog
module requantizer (
    input [31:0] acc_in,     // INT32 accumulator
    input [31:0] scale,      // Combined scale factor (fixed-point)
    input [7:0]  shift,      // Right shift amount
    output [7:0] result_out  // INT8 output
);

// Fixed-point multiply and shift
wire [63:0] scaled = acc_in * scale;
wire [31:0] shifted = scaled >> shift;

// Saturation clamp
assign result_out = (shifted > 127)  ? 8'd127 :
                   (shifted < -128) ? 8'd-128 :
                                      shifted[7:0];
endmodule
```

### Scale Factor Representation

#### Fixed-Point Encoding
```python
def encode_scale_factor(scale_fp32, integer_bits=8, fractional_bits=24):
    """Convert FP32 scale to fixed-point representation"""
    scale_fixed = int(scale_fp32 * (1 << fractional_bits))
    return scale_fixed & ((1 << 32) - 1)  # 32-bit fixed-point
```

#### CSR Programming
```python
# CSR register layout for quantization parameters
CSR_SCALE_ACT    = 0x20  # Activation scale factor  
CSR_SCALE_WGT    = 0x24  # Weight scale factor
CSR_SCALE_OUT    = 0x28  # Output scale factor
CSR_SHIFT_AMOUNT = 0x2C  # Right shift for requantization
```

## Accuracy Analysis

### Quantization Error Sources

#### 1. Clipping Error
```python
def calculate_clipping_error(tensor, scale):
    """Calculate error from clipping to [-128, 127]"""
    quantized = np.round(tensor / scale)
    clipped = np.clip(quantized, -128, 127)
    clipping_error = np.mean(np.abs(quantized - clipped))
    return clipping_error
```

#### 2. Rounding Error  
```python
def calculate_rounding_error(tensor, scale):
    """Calculate error from FP32 → INT8 rounding"""
    exact = tensor / scale
    rounded = np.round(exact)
    rounding_error = np.mean(np.abs(exact - rounded))
    return rounding_error
```

#### 3. Accumulation Error
The INT32 accumulator prevents overflow for typical CNN workloads:
```python
# Maximum accumulation without overflow
max_K = 2**23  # For INT8 × INT8 → INT32 accumulation
typical_K = 512  # CNN kernel sizes << max_K
```

### Precision Analysis

#### Signal-to-Quantization-Noise Ratio (SQNR)
```python
def calculate_sqnr(original, quantized):
    """Calculate SQNR in dB"""
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - quantized)**2)
    sqnr_db = 10 * np.log10(signal_power / noise_power)
    return sqnr_db
```

#### Expected SQNR for INT8
- **Theoretical**: ~48 dB for uniform distribution
- **Practical**: 35-45 dB for CNN tensors with calibration

## Software Integration

### Quantization Workflow

#### Training Phase
```python
def quantization_aware_training():
    """QAT workflow for ACCEL-v1 deployment"""
    # 1. Train FP32 model
    model_fp32 = train_model()
    
    # 2. Insert fake quantization nodes
    model_qat = insert_fake_quantization(model_fp32)
    
    # 3. Fine-tune with quantization simulation
    model_qat = fine_tune_quantized(model_qat)
    
    # 4. Export quantized weights and scales
    export_quantized_model(model_qat, "model_int8.npz")
```

#### Inference Phase
```python
def deploy_to_accel_v1(model_path):
    """Deploy quantized model to ACCEL-v1"""
    # 1. Load quantized model
    model_data = np.load(model_path)
    weights_int8 = model_data['weights']
    scales = model_data['scales']
    
    # 2. Program CSR registers
    program_scale_factors(scales)
    
    # 3. Upload weights to accelerator
    upload_weights(weights_int8)
    
    # 4. Ready for inference
    return True
```

### Host-Side Quantization

#### Pre-Processing
```python
def preprocess_input(image, activation_scale):
    """Quantize input image for ACCEL-v1"""
    # Normalize to [0, 1] or [-1, 1] range
    normalized = preprocess_image(image)
    
    # Apply quantization
    quantized = np.round(normalized / activation_scale)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    
    return quantized
```

#### Post-Processing  
```python
def postprocess_output(output_int8, output_scale):
    """Dequantize output from ACCEL-v1"""
    output_fp32 = output_int8.astype(np.float32) * output_scale
    return output_fp32
```

## Performance Optimizations

### Scale Factor Optimization

#### Power-of-2 Scales
```python
def optimize_scale_power_of_2(scale_fp32):
    """Approximate scale with power-of-2 for shift-only requantization"""
    log2_scale = np.log2(scale_fp32)
    power_of_2 = 2 ** np.round(log2_scale)
    shift_amount = int(np.round(log2_scale))
    return power_of_2, shift_amount
```

#### Benefits:
- Replace multiplication with bit shift
- Reduced hardware complexity
- Faster computation

### Batch Quantization
```python
def batch_quantize_weights(weight_tensors):
    """Efficient batch quantization for model deployment"""
    quantized_weights = []
    scale_factors = []
    
    for weights in weight_tensors:
        w_int8, scale = quantize_symmetric(weights)
        quantized_weights.append(w_int8)
        scale_factors.append(scale)
    
    return quantized_weights, scale_factors
```

## Validation and Testing

### Numerical Validation

#### Bit-Exact Testing
```python
def validate_quantization_accuracy():
    """Validate quantization against golden reference"""
    # Generate test matrices
    A_fp32 = np.random.randn(64, 128).astype(np.float32)
    B_fp32 = np.random.randn(128, 256).astype(np.float32)
    
    # FP32 reference
    C_fp32_ref = A_fp32 @ B_fp32
    
    # Quantized computation
    A_int8, Sa = quantize_symmetric(A_fp32)
    B_int8, Sw = quantize_symmetric(B_fp32)
    C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
    
    # Requantize output
    Sc = calculate_scale(C_fp32_ref)
    C_int8 = requantize_output(C_int32, Sa, Sw, Sc)
    C_fp32_quant = C_int8.astype(np.float32) * Sc
    
    # Calculate error metrics
    mse = np.mean((C_fp32_ref - C_fp32_quant)**2)
    sqnr = calculate_sqnr(C_fp32_ref, C_fp32_quant)
    
    print(f"MSE: {mse:.6f}, SQNR: {sqnr:.2f} dB")
    return mse < 1e-3  # Acceptable error threshold
```

#### Statistical Analysis
```python
def analyze_quantization_statistics():
    """Analyze quantization error distributions"""
    errors = []
    for _ in range(1000):  # Monte Carlo trials
        result = validate_quantization_accuracy()
        errors.append(result)
    
    # Statistical summary
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    percentiles = np.percentile(errors, [50, 90, 95, 99])
    
    return {
        'mean': mean_error,
        'std': std_error,
        'percentiles': percentiles
    }
```

## Future Enhancements

### Advanced Quantization Schemes

#### Mixed Precision
- **Weights**: 4-bit/8-bit adaptive quantization
- **Activations**: 8-bit/16-bit based on layer sensitivity
- **Accumulators**: 16-bit/32-bit based on accumulation depth

#### Dynamic Quantization
- **Runtime Scale Adjustment**: Adapt scales based on input statistics
- **Channel-wise Scaling**: Per-channel scales for improved accuracy
- **Outlier Detection**: Special handling for activation outliers

#### Sparse Quantization
- **Structured Sparsity**: Block-sparse patterns for hardware efficiency
- **Magnitude-based Pruning**: Remove small weights before quantization
- **Zero-skip Logic**: Hardware acceleration for sparse computations

### Hardware Optimizations

#### Advanced Requantization
```verilog
// Multi-stage requantization pipeline
module advanced_requantizer (
    input [31:0] acc_in,
    input [15:0] scale_mantissa,
    input [7:0]  scale_exponent,
    output [7:0] result_out
);
// Floating-point scale factor handling
// Pipeline stages for high-frequency operation
endmodule
```

#### Adaptive Precision
- **Layer-wise Precision**: Different quantization per layer
- **Workload-dependent**: Adjust precision based on workload characteristics
- **Quality-performance Trade-off**: Runtime precision adjustment

---

*This quantization methodology document provides the theoretical foundation and practical implementation details for INT8 quantization in ACCEL-v1. For software implementation, refer to the quantization utilities in `python/INT8 quantization/`.*

## See also

- Practical, runnable examples and CSR programming tips: `QUANTIZATION_PRACTICAL.md` (same docs folder)
