# ACCEL-v1 Verification and Testing Strategy

**Version 2.0** — December 4, 2025  
**Coverage**: 71% line coverage | 34/34 tests passing

## Overview

This document outlines the comprehensive verification methodology for the ACCEL-v1 systolic array accelerator, covering unit-level, integration-level, and system-level testing strategies.

## Current Status

| Metric | Value | Target |
|--------|-------|--------|
| Line Coverage | 71% | 90% |
| Tests Passing | 34/34 | 100% |
| Modules Tested | 13/23 | All |

### Test Infrastructure

- **Verilator 5.042**: Primary simulation (with `--timing --main`)
- **CocoTB**: Python-based AXI protocol tests
- **Verilator Coverage**: Line and branch coverage collection

## Verification Hierarchy

```
System Level
├── Full datapath integration (integration_tb.sv)
├── End-to-End sparse MatMul (accel_top_tb_full.sv)
└── Performance characterization (perf_tb.sv)

Integration Level  
├── BSR DMA + Scheduler (bsr_dma_tb.sv)
├── Output Accumulator (output_accumulator_tb.sv)
└── AXI-Lite CSR interface (cocotb/)

Unit Level
├── Processing Element (pe_tb.sv)
├── Systolic Array (systolic_tb.sv)
├── Metadata Decoder (meta_decode_tb.sv)
└── MAC Unit (cpp/test_mac8.cpp)
```

## Recent Verification Improvements

### 1. BSR DMA Multi-Block Fix
- **Bug**: Multi-block transfers corrupted data after first block
- **Root Cause**: Pointer incremented without backpressure check
- **Test Added**: `test_multiblock_backpressure` in `bsr_dma_tb.sv`
- **Coverage Impact**: +5% on bsr_dma.sv

### 2. CSR Edge Case Tests
- Added 12 edge-case tests:
  - Write to read-only register
  - Out-of-range address reads
  - Burst across register boundary
- **Coverage Impact**: 61% → 71%

### 3. 16×16 Systolic Scaling Tests
- Verified scheduler with parameterized BLOCK_SIZE
- Tested 16-cycle weight load vs 8-cycle
- All 27 accel_top tests pass with 16×16 array

## Unit-Level Verification

### 1. Processing Element (PE) Testing

#### Test Objectives
- **MAC Operation**: Verify INT8 × INT8 → INT32 multiply-accumulate
- **Weight Storage**: Confirm weight loading and retention
- **Data Flow**: Validate activation forwarding through PE array
- **Saturation**: Test arithmetic overflow/underflow handling

#### Test Cases

##### MAC Functionality
```systemverilog
// Test: Basic MAC operation
task test_pe_mac_basic();
    // Initialize PE
    reset_pe();
    
    // Load weight
    wgt_in = 8'h42;  // +66
    load_enable = 1'b1;
    @(posedge clk);
    load_enable = 1'b0;
    
    // Test MAC operations
    test_vectors = {
        {8'h01, 32'h00000000, 32'h00000042},  // 1 * 66 + 0 = 66
        {8'h02, 32'h00000042, 32'h00000084},  // 2 * 66 + 66 = 198
        {8'hFF, 32'h00000084, 32'h00000042},  // -1 * 66 + 132 = 66
        {8'h80, 32'h00000042, 32'hFFFF20C2}   // -128 * 66 + 66 = -8382
    };
    
    foreach (test_vectors[i]) begin
        act_in = test_vectors[i][0];
        acc_in = test_vectors[i][1];
        expected = test_vectors[i][2];
        
        @(posedge clk);
        
        if (acc_out !== expected) begin
            $error("MAC mismatch: act=%h acc_in=%h expected=%h got=%h", 
                   act_in, acc_in, expected, acc_out);
        end
    end
endtask
```

##### Saturation Testing
```systemverilog
// Test: INT32 accumulator limits
task test_pe_saturation();
    reset_pe();
    wgt_in = 8'h7F;  // +127 (max positive)
    load_enable = 1'b1;
    @(posedge clk);
    load_enable = 1'b0;
    
    // Test near INT32 overflow
    act_in = 8'h7F;              // +127
    acc_in = 32'h7FFFFFFF - 127*127;  // Near max INT32
    @(posedge clk);
    
    // Verify no overflow occurred
    assert(acc_out == 32'h7FFFFFFF) 
        else $error("INT32 overflow not handled correctly");
endtask
```

#### Coverage Metrics
- **Data Combinations**: All sign combinations of weight/activation
- **Accumulator States**: Zero, positive, negative, near-overflow
- **Edge Cases**: Maximum/minimum INT8 values
- **Timing**: Setup/hold violations under process/voltage/temperature variations

### 2. Buffer Module Testing

#### Weight Buffer (`wgt_buffer.v`)

##### Test Objectives
- **FIFO Operation**: Sequential write/read operations
- **Depth Handling**: Full/empty conditions
- **Concurrent Access**: Read during write scenarios
- **Reset Behavior**: Proper initialization

##### Test Implementation
```systemverilog
module tb_wgt_buffer;
    parameter DEPTH = 256;
    parameter WIDTH = 8;
    
    // DUT signals
    logic clk, rst_n;
    logic wr_en, rd_en;
    logic [WIDTH-1:0] wr_data, rd_data;
    logic full, empty;
    
    wgt_buffer #(DEPTH, WIDTH) dut (.*);
    
    // Test: Fill and drain buffer
    task test_fill_drain();
        // Fill buffer completely
        for (int i = 0; i < DEPTH; i++) begin
            wr_data = i[WIDTH-1:0];
            wr_en = 1'b1;
            @(posedge clk);
            assert(!full || i == DEPTH-1) 
                else $error("Buffer full flag incorrect at %d", i);
        end
        wr_en = 1'b0;
        
        // Verify buffer is full
        assert(full) else $error("Buffer should be full");
        
        // Drain buffer completely
        for (int i = 0; i < DEPTH; i++) begin
            rd_en = 1'b1;
            @(posedge clk);
            assert(rd_data == i[WIDTH-1:0]) 
                else $error("Read data mismatch: expected %h got %h", i, rd_data);
            assert(!empty || i == DEPTH-1) 
                else $error("Buffer empty flag incorrect at %d", i);
        end
        rd_en = 1'b0;
        
        // Verify buffer is empty
        assert(empty) else $error("Buffer should be empty");
    endtask
endmodule
```

#### Activation Buffer (`act_buffer.v`)

##### Test Objectives
- **Dual-Port Access**: Simultaneous read/write operations
- **Broadcast Capability**: Single write, multiple read ports
- **Address Mapping**: Correct tile-based addressing
- **Pipeline Compatibility**: Interface with PE array timing

### 3. CSR Module Testing

#### Test Objectives
- **Register Access**: Read/write operations on all registers
- **Control Semantics**: START/STOP/RESET functionality
- **Status Reporting**: BUSY/READY/ERROR flag accuracy
- **Parameter Validation**: Range checking for dimensions/tile sizes

#### Test Implementation
```systemverilog
// Test: CSR register map
task test_csr_registers();
    // Test all writable registers
    reg_addresses = {
        CSR_CONTROL, CSR_M_DIM, CSR_N_DIM, CSR_K_DIM,
        CSR_TILE_M, CSR_TILE_N, CSR_SCALE_SHIFT
    };
    
    foreach (reg_addresses[i]) begin
        // Write test pattern
        test_data = $random;
        csr_write(reg_addresses[i], test_data);
        
        // Read back and verify
        read_data = csr_read(reg_addresses[i]);
        assert(read_data == test_data) 
            else $error("CSR readback failed at 0x%02X", reg_addresses[i]);
    end
endtask

// Test: Control flow
task test_csr_control_flow();
    // Initial state should be idle
    status = csr_read(CSR_STATUS);
    assert(status[0] == 1'b1) else $error("Initial READY flag should be set");
    assert(status[1] == 1'b0) else $error("Initial BUSY flag should be clear");
    
    // Start operation
    csr_write(CSR_CONTROL, 32'h00000001);  // Set START bit
    @(posedge clk);
    
    // Should transition to busy
    status = csr_read(CSR_STATUS);
    assert(status[0] == 1'b0) else $error("READY should be clear during operation");
    assert(status[1] == 1'b1) else $error("BUSY should be set during operation");
    
    // Wait for completion (timeout after reasonable cycles)
    wait_count = 0;
    do begin
        @(posedge clk);
        status = csr_read(CSR_STATUS);
        wait_count++;
    end while (status[1] && wait_count < 10000);
    
    assert(!status[1]) else $error("Operation timeout - BUSY never cleared");
    assert(status[0]) else $error("READY not set after completion");
endtask
```

### 4. UART Interface Testing

#### Test Objectives
- **Protocol Compliance**: 8N1 frame format adherence
- **Packet Integrity**: Header/Length/Payload/CRC validation
- **Error Detection**: CRC error handling and reporting
- **Flow Control**: Back-pressure and buffer management

#### Test Implementation
```systemverilog
// Test: UART packet transmission
task test_uart_packet();
    packet_data = {
        8'h01,              // Header: Write command
        8'h04,              // Length: 4 bytes
        32'h12345678,       // Payload: Test data
        16'hABCD            // CRC: Test checksum
    };
    
    // Transmit packet
    foreach (packet_data[i]) begin
        uart_tx_byte(packet_data[i]);
    end
    
    // Verify reception
    @(posedge packet_complete);
    assert(rx_header == 8'h01) else $error("Header mismatch");
    assert(rx_length == 8'h04) else $error("Length mismatch");
    assert(rx_payload == 32'h12345678) else $error("Payload mismatch");
    assert(crc_valid) else $error("CRC validation failed");
endtask
```

## Integration-Level Verification

### 1. Systolic Array Integration

#### Test Objectives
- **Array Assembly**: N×M PE interconnection correctness
- **Dataflow Validation**: Row-stationary operation verification
- **Timing Closure**: Multi-cycle path validation
- **Scalability**: Different array sizes (2×2, 4×4, 8×8)

#### Matrix Multiplication Validation
```systemverilog
// Test: 4x4 systolic array GEMM
task test_systolic_gemm_4x4();
    parameter N = 4, M = 4, K = 8;
    
    // Test matrices (known values for easy verification)
    logic signed [7:0] A [N][K] = '{
        '{1, 2, 3, 4, 5, 6, 7, 8},
        '{2, 3, 4, 5, 6, 7, 8, 9},
        '{3, 4, 5, 6, 7, 8, 9, 10},
        '{4, 5, 6, 7, 8, 9, 10, 11}
    };
    
    logic signed [7:0] B [K][M] = '{
        '{1, 0, 0, 0},
        '{0, 1, 0, 0},
        '{0, 0, 1, 0},
        '{0, 0, 0, 1},
        '{1, 1, 1, 1},
        '{2, 2, 2, 2},
        '{3, 3, 3, 3},
        '{4, 4, 4, 4}
    };
    
    // Expected result (manual calculation)
    logic signed [31:0] C_expected [N][M];
    golden_gemm(A, B, C_expected);  // Reference implementation
    
    // Load weights into array
    load_systolic_weights(B);
    
    // Stream activations and collect results
    for (int k = 0; k < K; k++) begin
        for (int n = 0; n < N; n++) begin
            stream_activation(n, A[n][k]);
        end
        @(posedge clk);
    end
    
    // Wait for computation completion
    wait_for_completion();
    
    // Verify results
    for (int n = 0; n < N; n++) begin
        for (int m = 0; m < M; m++) begin
            result = read_result(n, m);
            assert(result == C_expected[n][m]) 
                else $error("GEMM result mismatch at [%d][%d]: expected %d got %d", 
                           n, m, C_expected[n][m], result);
        end
    end
endtask
```

### 2. UART + CSR + Compute Integration

#### Test Objectives
- **End-to-End Flow**: UART command → CSR configuration → compute → result readback
- **Error Propagation**: Hardware errors reported through status registers
- **Concurrent Operations**: Overlapped configuration and computation
- **Resource Management**: Buffer allocation and deallocation

#### Integration Test Scenario
```systemverilog
// Test: Complete GEMM operation via UART
task test_uart_gemm_integration();
    // 1. Configure operation via UART
    uart_csr_write(CSR_M_DIM, 32'd4);
    uart_csr_write(CSR_N_DIM, 32'd4);
    uart_csr_write(CSR_K_DIM, 32'd8);
    uart_csr_write(CSR_TILE_M, 32'd4);
    uart_csr_write(CSR_TILE_N, 32'd4);
    
    // 2. Upload weight matrix via UART
    uart_upload_weights(test_weights, 4*8);
    
    // 3. Upload activation matrix via UART
    uart_upload_activations(test_activations, 4*8);
    
    // 4. Start computation
    uart_csr_write(CSR_CONTROL, 32'h00000001);
    
    // 5. Poll for completion
    do begin
        status = uart_csr_read(CSR_STATUS);
        @(posedge clk);
    end while (status[1]);  // Wait for BUSY to clear
    
    // 6. Read results via UART
    uart_download_results(results, 4*4);
    
    // 7. Verify against golden reference
    verify_results(results, golden_results);
endtask
```

### 3. Multi-Tile GEMM Testing

#### Test Objectives
- **Tiling Algorithm**: Correct tile decomposition and accumulation
- **Memory Management**: Buffer reuse across tiles
- **Partial Results**: Intermediate accumulation handling
- **Large Matrix Support**: Matrices larger than hardware capacity

## System-Level Verification

### 1. Host Software Integration

#### Test Environment Setup
```python
# Test harness for host-hardware integration
class AccelV1TestHarness:
    def __init__(self, uart_port='/dev/ttyUSB0'):
        self.uart = UARTDriver(uart_port, 115200)
        self.csr = CSRMap(self.uart)
        self.tiler = HostRSTiler(self.uart, self.csr)
        
    def test_basic_gemm(self):
        """Test basic GEMM operation through host software"""
        # Generate test matrices
        M, N, K = 16, 16, 32
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
        
        # Golden reference (CPU computation)
        C_golden = A.astype(np.int32) @ B.astype(np.int32)
        
        # Accelerator computation
        C_accel = self.tiler.run_gemm(A, B)
        
        # Verify bit-exact match
        np.testing.assert_array_equal(C_accel, C_golden)
        
    def test_large_matrix_tiling(self):
        """Test tiling for matrices larger than hardware capacity"""
        M, N, K = 128, 256, 512  # Large matrices
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
        
        # Configure tiling parameters
        config = GEMMConfig(M, N, K, tile_m=8, tile_n=8)
        
        # Execute with tiling
        C_accel = self.tiler.run_gemm(A, B, config)
        
        # Verify against reference
        C_golden = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C_accel, C_golden)
```

### 2. CNN Workload Testing

#### Real Network Validation
```python
def test_cnn_layer_execution():
    """Test actual CNN layer execution on ACCEL-v1"""
    # Load pre-trained quantized model
    model_data = np.load('models/mnist_int8.npz')
    weights = model_data['conv1_weights']  # Shape: [32, 1, 3, 3]
    
    # Generate test input (MNIST digit)
    input_image = load_mnist_sample(0)  # 28x28 grayscale
    
    # Convert to im2col format for GEMM
    input_cols = im2col(input_image, kernel_size=3, stride=1)
    weight_matrix = weights.reshape(32, -1)  # Flatten filters
    
    # Execute on accelerator
    output_cols = run_gemm_on_accel(input_cols, weight_matrix)
    
    # Reshape back to feature maps
    output_features = col2im(output_cols, output_shape=(32, 26, 26))
    
    # Compare with CPU reference
    cpu_output = conv2d_reference(input_image, weights)
    np.testing.assert_allclose(output_features, cpu_output, rtol=1e-3)
```

### 3. Performance Characterization

#### Throughput Measurement
```python
def characterize_throughput():
    """Measure ACCEL-v1 throughput across different workloads"""
    results = {}
    
    # Test different matrix sizes
    test_sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    for M, N, K in test_sizes:
        # Generate test matrices
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
        
        # Measure execution time
        start_time = time.time()
        C = run_gemm_on_accel(A, B)
        end_time = time.time()
        
        # Calculate metrics
        ops = 2 * M * N * K  # MACs
        time_sec = end_time - start_time
        throughput = ops / time_sec / 1e9  # GOP/s
        
        results[f"{M}x{N}x{K}"] = {
            'time_sec': time_sec,
            'throughput_gops': throughput,
            'efficiency': throughput / theoretical_peak
        }
    
    return results
```

#### Latency Analysis
```python
def measure_operation_latency():
    """Detailed latency breakdown for GEMM operations"""
    # Small matrices for latency measurement
    A = np.random.randint(-128, 127, (8, 8), dtype=np.int8)
    B = np.random.randint(-128, 127, (8, 8), dtype=np.int8)
    
    latency_breakdown = {}
    
    # Weight upload latency
    start = time.time()
    upload_weights(B)
    latency_breakdown['weight_upload'] = time.time() - start
    
    # Activation upload latency
    start = time.time()
    upload_activations(A)
    latency_breakdown['activation_upload'] = time.time() - start
    
    # Computation latency
    start = time.time()
    trigger_computation()
    wait_for_completion()
    latency_breakdown['computation'] = time.time() - start
    
    # Result download latency
    start = time.time()
    results = download_results()
    latency_breakdown['result_download'] = time.time() - start
    
    return latency_breakdown
```

## Coverage Analysis

### Functional Coverage

#### Code Coverage Metrics
- **Line Coverage**: >95% of RTL lines exercised
- **Branch Coverage**: >90% of conditional branches taken
- **Toggle Coverage**: >90% of signal transitions observed
- **FSM Coverage**: All states and transitions exercised

#### Assertion Coverage
```systemverilog
// Key assertions for coverage tracking
property mac_overflow_check;
    @(posedge clk) disable iff (!rst_n)
    (mac_enable && (acc_in > 32'h40000000)) |-> 
    ##1 (acc_out <= 32'h7FFFFFFF);
endproperty

property systolic_dataflow;
    @(posedge clk) disable iff (!rst_n)
    (pe_array[i][j].act_in == $past(pe_array[i][j-1].act_out, 1));
endproperty

assert property (mac_overflow_check);
assert property (systolic_dataflow);
```

### Performance Coverage

#### Workload Coverage Matrix
| Matrix Size | Tile Size | K Dimension | Status |
|-------------|-----------|-------------|---------|
| 8×8         | 8×8       | 8           | PASS Pass |
| 16×16       | 8×8       | 16          | PASS Pass |
| 32×32       | 16×16     | 32          | PASS Pass |
| 64×64       | 32×32     | 64          | PASS Pass |
| 128×128     | 64×64     | 128         | PASS Pass |

#### Corner Case Coverage
- **Edge Values**: INT8 min/max values (-128, +127)
- **Zero Patterns**: All-zero weights or activations
- **Saturation Cases**: Operations causing accumulator overflow
- **Timing Corners**: Process/voltage/temperature variations

## Regression Testing

### Automated Test Suite
```bash
#!/bin/bash
# regression_test.sh - Automated regression testing

# Unit tests
echo "Running unit tests..."
make test_unit 2>&1 | tee unit_test.log

# Integration tests  
echo "Running integration tests..."
make test_integration 2>&1 | tee integration_test.log

# System tests
echo "Running system tests..."
python3 test_system.py 2>&1 | tee system_test.log

# Performance tests
echo "Running performance tests..."
python3 test_performance.py 2>&1 | tee performance_test.log

# Generate summary report
python3 generate_test_report.py unit_test.log integration_test.log system_test.log performance_test.log
```

### Continuous Integration
```yaml
# .github/workflows/verification.yml
name: ACCEL-v1 Verification

on: [push, pull_request]

jobs:
  rtl-verification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get install iverilog verilator
          pip install cocotb pytest numpy
      - name: Run RTL tests
        run: make test_rtl
      - name: Generate coverage report
        run: make coverage_report

  software-verification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run software tests
        run: pytest python/tests/ -v
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Sign-off Criteria

### Functional Sign-off
- [ ] All unit tests pass (100% pass rate)
- [ ] All integration tests pass (100% pass rate)  
- [ ] All system tests pass (100% pass rate)
- [ ] CNN workload validation complete
- [ ] Bit-exact accuracy verified against golden reference

### Performance Sign-off
- [ ] Target throughput achieved (>80% of theoretical peak)
- [ ] Latency requirements met (<100ms for typical workloads)
- [ ] Power consumption within specification
- [ ] Area utilization acceptable

### Quality Sign-off
- [ ] Code coverage >95%
- [ ] Functional coverage >90%
- [ ] No critical bugs remaining
- [ ] Documentation complete and reviewed

---

*This verification plan ensures comprehensive validation of ACCEL-v1 from individual components to complete system operation. All tests must pass before hardware deployment or software release.*