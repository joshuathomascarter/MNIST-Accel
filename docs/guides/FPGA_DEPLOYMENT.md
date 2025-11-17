# ACCEL-v1 FPGA Deployment Guide

## Overview

This guide provides step-by-step instructions for synthesizing, implementing, and deploying the ACCEL-v1 systolic array accelerator on Intel/Altera FPGA platforms. The design targets Cyclone V FPGAs but can be adapted to other Intel FPGA families.

## Prerequisites

### Required Software
- **Intel Quartus Prime** (version 20.1 or later)
  - Standard Edition (free) is sufficient for Cyclone V
  - Download from Intel FPGA Software Center
- **ModelSim-Intel FPGA Edition** (optional, for simulation)
- **System Console** (included with Quartus Prime)
- **NIOS II Software Build Tools** (if using soft processor)

### Supported Hardware
- **Primary Target**: Cyclone V Development Kit
  - Part: 5CGXFC7C7F23C8
  - Package: F484
  - Speed Grade: -8
- **Alternative Targets**: 
  - Cyclone V SoC Development Kit
  - Custom Cyclone V boards with UART interface

### Development Environment
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Disk Space**: 20 GB for Quartus installation + 5 GB for project
- **USB**: For JTAG programming and UART communication

## Project Setup

### 1. Clone Repository
```bash
git clone https://github.com/joshuathomascarter/ACCEL-v1.git
cd ACCEL-v1
```

### 2. Directory Structure
```
ACCEL-v1/
├── accel v1/
│   ├── verilog/          # RTL source files
│   │   ├── top/          # Top-level modules
│   │   ├── systolic/     # Systolic array implementation
│   │   ├── buffer/       # Memory buffers
│   │   ├── control/      # Control logic
│   │   └── uart/         # UART interface
│   ├── constraints/      # Timing and pin constraints (will create)
│   ├── tcl/             # Quartus TCL scripts (will create)
│   └── docs/            # Documentation
```

### 3. Create Quartus Project

#### Method 1: Using TCL Script (Recommended)
```tcl
# Create project creation script
# File: create_project.tcl

# Set project name and location
set project_name "accel_v1"
set project_dir "./quartus_project"

# Create project
project_new -revision accel_v1 -overwrite $project_name

# Set device family and part
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CGXFC7C7F23C8
set_global_assignment -name TOP_LEVEL_ENTITY accel_top

# Add source files
set_global_assignment -name VERILOG_FILE "../verilog/top/accel_top.v"
set_global_assignment -name VERILOG_FILE "../verilog/systolic/systolic_array.v"
set_global_assignment -name VERILOG_FILE "../verilog/systolic/pe.v"
set_global_assignment -name VERILOG_FILE "../verilog/mac/mac8.v"
set_global_assignment -name VERILOG_FILE "../verilog/buffer/act_buffer.v"
set_global_assignment -name VERILOG_FILE "../verilog/buffer/wgt_buffer.v"
set_global_assignment -name VERILOG_FILE "../verilog/control/csr.v"
set_global_assignment -name VERILOG_FILE "../verilog/control/scheduler.v"
set_global_assignment -name VERILOG_FILE "../verilog/uart/uart_rx.v"
set_global_assignment -name VERILOG_FILE "../verilog/uart/uart_tx.v"

# Set compilation options
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name MIN_CORE_JUNCTION_TEMP 0
set_global_assignment -name MAX_CORE_JUNCTION_TEMP 85
set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 256
set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top
set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top
set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top

# Save project
export_assignments
project_close
```

Run the script:
```bash
cd "accel v1"
mkdir quartus_project
cd quartus_project
quartus_sh -t ../create_project.tcl
```

#### Method 2: Manual Creation
1. Open Quartus Prime
2. File → New Project Wizard
3. Set working directory: `accel v1/quartus_project`
4. Project name: `accel_v1`
5. Add source files from `verilog/` directory
6. Select device: Cyclone V 5CGXFC7C7F23C8
7. Set top-level entity: `accel_top`

## Pin Assignments

### 1. Create Pin Assignment File
Create `pin_assignments.tcl`:

```tcl
# Clock and Reset
set_location_assignment PIN_M9 -to clk_50mhz
set_location_assignment PIN_P22 -to rst_n_button
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk_50mhz
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to rst_n_button

# UART Interface (assuming USB-UART bridge)
set_location_assignment PIN_B12 -to uart_tx
set_location_assignment PIN_A12 -to uart_rx
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to uart_tx
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to uart_rx

# LEDs for status indication
set_location_assignment PIN_A8 -to led_status[0]
set_location_assignment PIN_A9 -to led_status[1]
set_location_assignment PIN_A10 -to led_status[2]
set_location_assignment PIN_B10 -to led_status[3]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_status[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_status[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_status[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_status[3]

# GPIO expansion (optional, for debugging)
set_location_assignment PIN_C3 -to gpio_debug[0]
set_location_assignment PIN_C2 -to gpio_debug[1]
set_location_assignment PIN_G2 -to gpio_debug[2]
set_location_assignment PIN_G1 -to gpio_debug[3]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to gpio_debug[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to gpio_debug[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to gpio_debug[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to gpio_debug[3]
```

Apply pin assignments:
```bash
quartus_sh -t pin_assignments.tcl accel_v1
```

### 2. Top-Level Module Modification
Update `accel_top.v` to include I/O pins:

```verilog
module accel_top (
    // Clock and reset
    input  wire        clk_50mhz,      // 50 MHz board clock
    input  wire        rst_n_button,   // Reset button (active low)
    
    // UART interface
    input  wire        uart_rx,        // UART receive
    output wire        uart_tx,        // UART transmit
    
    // Status LEDs
    output wire [3:0]  led_status,     // Status indicators
    
    // Debug GPIO (optional)
    output wire [3:0]  gpio_debug      // Debug signals
);

    // Internal signals
    wire clk_100mhz;    // PLL-generated 100 MHz clock
    wire rst_n_sync;    // Synchronized reset
    
    // Clock generation
    clock_pll u_clock_pll (
        .inclk0(clk_50mhz),
        .c0(clk_100mhz),        // 100 MHz output
        .locked(pll_locked)
    );
    
    // Reset synchronization
    reset_sync u_reset_sync (
        .clk(clk_100mhz),
        .rst_n_async(rst_n_button & pll_locked),
        .rst_n_sync(rst_n_sync)
    );
    
    // Status LED assignments
    assign led_status[0] = pll_locked;     // PLL lock indicator
    assign led_status[1] = ~rst_n_sync;    // Reset active indicator
    assign led_status[2] = uart_busy;      // UART activity
    assign led_status[3] = compute_done;   // Computation status
    
    // Debug GPIO assignments
    assign gpio_debug[0] = systolic_start;
    assign gpio_debug[1] = buffer_write_enable;
    assign gpio_debug[2] = csr_write_strobe;
    assign gpio_debug[3] = interrupt_pending;
    
    // ACCEL-v1 core instantiation
    accel_core u_accel_core (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        .uart_busy(uart_busy),
        .compute_done(compute_done),
        .systolic_start(systolic_start),
        .buffer_write_enable(buffer_write_enable),
        .csr_write_strobe(csr_write_strobe),
        .interrupt_pending(interrupt_pending)
    );

endmodule
```

## Timing Constraints

### 1. Create SDC File
Create `accel_v1_timing.sdc`:

```tcl
# ACCEL-v1 Timing Constraints

# Clock definitions
create_clock -period 20.000 -name clk_50mhz [get_ports clk_50mhz]
create_generated_clock -name clk_100mhz -source [get_ports clk_50mhz] \
    -divide_by 1 -multiply_by 2 [get_pins u_clock_pll|altpll_component|auto_generated|pll1|clk[0]]

# Clock uncertainty and jitter
set_clock_uncertainty -rise_from [get_clocks clk_50mhz] -rise_to [get_clocks clk_100mhz] 0.1
set_clock_uncertainty -fall_from [get_clocks clk_50mhz] -fall_to [get_clocks clk_100mhz] 0.1

# Input delays (relative to board clock)
set_input_delay -clock clk_50mhz -max 2.0 [get_ports rst_n_button]
set_input_delay -clock clk_50mhz -min 0.5 [get_ports rst_n_button]

# UART timing constraints
set_input_delay -clock clk_100mhz -max 3.0 [get_ports uart_rx]
set_input_delay -clock clk_100mhz -min -1.0 [get_ports uart_rx]
set_output_delay -clock clk_100mhz -max 5.0 [get_ports uart_tx]
set_output_delay -clock clk_100mhz -min -2.0 [get_ports uart_tx]

# Output delays for LEDs and GPIO
set_output_delay -clock clk_100mhz -max 8.0 [get_ports led_status[*]]
set_output_delay -clock clk_100mhz -min -2.0 [get_ports led_status[*]]
set_output_delay -clock clk_100mhz -max 8.0 [get_ports gpio_debug[*]]
set_output_delay -clock clk_100mhz -min -2.0 [get_ports gpio_debug[*]]

# Multicycle paths (for slower operations)
set_multicycle_path -setup -end 2 -from [get_registers *csr_reg*] -to [get_registers *buffer_control*]
set_multicycle_path -hold -end 1 -from [get_registers *csr_reg*] -to [get_registers *buffer_control*]

# False paths (asynchronous signals)
set_false_path -from [get_ports rst_n_button] -to [get_registers *]
set_false_path -from [get_registers *interrupt_flag*] -to [get_ports led_status[*]]

# Maximum delay constraints for critical paths
set_max_delay -from [get_registers *systolic_array*mac_result*] \
              -to [get_registers *accumulator*] 8.0

# Minimum delay constraints
set_min_delay -from [get_registers *buffer_addr*] \
              -to [get_registers *buffer_data*] 1.0
```

Add SDC file to project:
```tcl
set_global_assignment -name SDC_FILE accel_v1_timing.sdc
```

### 2. Clock Configuration
Create PLL using Quartus IP Catalog:

1. Tools → IP Catalog
2. Search for "ALTPLL" or "Phase-Locked Loop"
3. Configure:
   - Input Clock: 50 MHz
   - Output Clock 0: 100 MHz (2x multiplication)
   - Enable clock0 output
   - Generate Verilog instantiation template

## Synthesis and Implementation

### 1. Analysis and Synthesis
```bash
# Command line synthesis
quartus_map accel_v1

# Or use GUI: Processing → Start → Start Analysis & Synthesis
```

Expected output:
- Logic utilization: ~15% of Cyclone V
- Memory blocks: ~25% for buffers
- DSP blocks: 4 (for MAC units)

### 2. Fitter (Place and Route)
```bash
# Command line fitting
quartus_fit accel_v1

# Or use GUI: Processing → Start → Start Fitter
```

Review fitter report:
- Timing analysis summary
- Resource utilization
- Pin assignments verification

### 3. Timing Analysis
```bash
# Generate timing analysis
quartus_sta accel_v1

# Or use GUI: Processing → Start → Start Timing Analyzer
```

Key timing checks:
- Setup slack: Should be positive (>0.5 ns preferred)
- Hold slack: Should be positive
- Clock skew: Should be minimal (<1 ns)

### 4. Generate Programming Files
```bash
# Generate .sof file for JTAG programming
quartus_asm accel_v1
```

## Optimization Strategies

### 1. Performance Optimization

#### Register Pipelining
Add pipeline registers in critical paths:
```verilog
// Example: Pipeline MAC operation
always @(posedge clk) begin
    if (!rst_n) begin
        mult_reg <= 0;
        acc_reg <= 0;
    end else begin
        mult_reg <= a_in * b_in;      // Stage 1: Multiply
        acc_reg <= acc_reg + mult_reg; // Stage 2: Accumulate
    end
end
```

#### Parallel Processing
Enable parallel MAC operations:
```verilog
// Unroll inner loop for better throughput
genvar i;
generate
    for (i = 0; i < PE_ARRAY_SIZE; i = i + 1) begin : pe_array
        pe #(.PIPE(1)) u_pe (
            .clk(clk), .rst_n(rst_n),
            .a_in(a_vec[i]), .b_in(b_vec[i]),
            .acc_out(acc_array[i])
        );
    end
endgenerate
```

### 2. Resource Optimization

#### Memory Optimization
Use efficient memory configurations:
```tcl
# Enable memory optimization
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"
set_global_assignment -name OPTIMIZE_POWER_DURING_SYNTHESIS "EXTRA EFFORT"

# Use dedicated memory blocks
set_instance_assignment -name RAMSTYLE "M10K" -to buffer_memory
```

#### DSP Block Utilization
Force use of dedicated multipliers:
```tcl
set_instance_assignment -name DSP_BLOCK_BALANCING "AUTO" -to mac_unit
```

### 3. Power Optimization

#### Clock Gating
Implement clock gating for unused modules:
```verilog
// Clock gating for buffer when not active
wire buffer_clock;
assign buffer_clock = clk & buffer_enable;

always @(posedge buffer_clock or negedge rst_n) begin
    if (!rst_n) begin
        buffer_data <= 0;
    end else begin
        buffer_data <= write_data;
    end
end
```

#### Power-Aware Synthesis
```tcl
set_global_assignment -name OPTIMIZE_POWER_DURING_SYNTHESIS ON
set_global_assignment -name OPTIMIZE_POWER_DURING_FITTING "EXTRA EFFORT"
```

## Programming and Testing

### 1. Programming the FPGA

#### Using Quartus Programmer
```bash
# Program via JTAG
quartus_pgm -c USB-Blaster -m JTAG -o "p;output_files/accel_v1.sof"

# Or use GUI: Tools → Programmer
```

#### Using System Console
```tcl
# Open System Console
system_console

# Connect to target
set masters [get_service_paths master]
set master [lindex $masters 0]
open_service master $master

# Program device
device_load_sof output_files/accel_v1.sof
```

### 2. UART Configuration
Set up host UART interface:
```bash
# Linux/macOS
screen /dev/ttyUSB0 115200

# Windows (use PuTTY or similar)
# Port: COM3 (check Device Manager)
# Baud: 115200, 8N1
```

### 3. Basic Connectivity Test
```python
# Python test script
import serial
import time

# Open UART connection
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Send test packet
test_packet = b'\xa5\x5a\x04\x01\x00\x00\x00\x00\x42'
ser.write(test_packet)

# Read response
response = ser.read(13)
print(f"Response: {response.hex()}")

ser.close()
```

### 4. Full System Test
```bash
# Run complete test suite
cd "accel v1/python"
python -m pytest tests/ -v

# Run loopback test
python host_uart/run_gemm.py --use_loopback --M 4 --N 4 --K 4 --verbose
```

## Debugging and Troubleshooting

### 1. Common Issues

#### Timing Violations
```bash
# Check timing report
quartus_sta accel_v1 -c accel_v1
```
Solutions:
- Add pipeline stages
- Reduce clock frequency
- Optimize critical paths

#### Resource Over-utilization
Check resource usage in fitter report:
- Reduce buffer sizes
- Optimize PE array dimensions
- Use compression techniques

#### UART Communication Errors
- Verify baud rate settings
- Check pin assignments
- Test with oscilloscope
- Validate CRC calculation

### 2. Debug Features

#### SignalTap Logic Analyzer
```tcl
# Add SignalTap instance
set_global_assignment -name ENABLE_SIGNALTAP ON
set_global_assignment -name USE_SIGNALTAP_FILE signaltap.stp

# Configure probe points
set_instance_assignment -name CONNECT_TO_SLD_NODE_ENTITY_PORT \
    acq_clk -to clk_100mhz -section_id auto_signaltap_0
```

#### LED Debug Indicators
```verilog
// Debug LED assignments
assign led_status[0] = systolic_array_busy;
assign led_status[1] = uart_rx_valid;
assign led_status[2] = csr_write_enable;
assign led_status[3] = buffer_full_flag;
```

#### UART Debug Messages
```verilog
// Debug UART transmission
always @(posedge clk) begin
    if (debug_enable) begin
        uart_tx_data <= debug_counter[7:0];
        uart_tx_valid <= 1'b1;
    end
end
```

### 3. Performance Monitoring

#### Cycle Count Measurement
```verilog
// Performance counters
reg [31:0] cycle_counter;
reg [31:0] operation_counter;

always @(posedge clk) begin
    if (!rst_n) begin
        cycle_counter <= 0;
        operation_counter <= 0;
    end else begin
        cycle_counter <= cycle_counter + 1;
        if (mac_enable)
            operation_counter <= operation_counter + 1;
    end
end
```

#### Throughput Analysis
```python
# Host-side performance measurement
import time

start_time = time.time()
result = tiler.run_gemm(A, B, config)
end_time = time.time()

elapsed = end_time - start_time
operations = M * N * K * 2  # Multiply-accumulate ops
throughput = operations / elapsed / 1e9  # GOPS

print(f"Throughput: {throughput:.2f} GOPS")
```

## Advanced Features

### 1. Partial Reconfiguration
Enable partial reconfiguration for runtime updates:
```tcl
set_global_assignment -name ENABLE_PARTIAL_RECONFIGURATION ON
set_instance_assignment -name PARTITION_HIERARCHY pr_partition_name \
    -to systolic_array_inst -section_id pr_partition
```

### 2. HPS Integration (for SoC variants)
```tcl
# Add HPS component for Cyclone V SoC
set_global_assignment -name HPS_ENABLE_SDRAM_ECC OFF
set_global_assignment -name HPS_DAP_SPLIT_MODE "SDM"
```

### 3. PCIe Interface (for high-end FPGAs)
```verilog
// PCIe endpoint for high-bandwidth communication
pcie_endpoint u_pcie (
    .clk_pcie(pcie_clk),
    .rst_n_pcie(pcie_rst_n),
    .rx_data(pcie_rx_data),
    .tx_data(pcie_tx_data)
);
```

## Validation and Verification

### 1. Post-Implementation Simulation
```bash
# Generate simulation netlist
quartus_eda --simulation --tool=modelsim_oem --format=verilog \
    --output_directory=simulation/modelsim accel_v1

# Run simulation
cd simulation/modelsim
vsim -do simulate.do
```

### 2. Hardware-in-the-Loop Testing
```python
# Automated hardware validation
import pytest
from accel_v1_test import ACCELTestSuite

test_suite = ACCELTestSuite(uart_port='/dev/ttyUSB0')
test_suite.run_all_tests()
```

### 3. Regression Testing
```bash
# Automated regression suite
./scripts/run_regression.sh --fpga --board cyclone5_dev_kit
```

## Deployment Checklist

- [ ] RTL compilation successful
- [ ] Timing constraints met
- [ ] Resource utilization acceptable
- [ ] Pin assignments verified
- [ ] Programming file generated
- [ ] FPGA programmed successfully
- [ ] UART communication verified
- [ ] Basic functionality tested
- [ ] Performance benchmarks completed
- [ ] Full test suite passed

## Support and Troubleshooting

### Resources
- **Intel FPGA Documentation**: [Intel FPGA Documentation Center](https://www.intel.com/content/www/us/en/programmable/documentation/)
- **Quartus Prime User Guide**: Comprehensive synthesis and implementation guide
- **ModelSim User Manual**: For simulation debugging
- **Development Board Documentation**: Board-specific pin assignments and constraints

### Common Support Contacts
- **Intel FPGA Support**: Submit cases through Intel Support Portal
- **Development Board Vendor**: For hardware-specific issues
- **ACCEL-v1 Community**: GitHub issues and discussions

---

*This deployment guide provides comprehensive instructions for FPGA implementation of ACCEL-v1. For software setup, refer to the main README.md and HOST_RS_TILER.md documentation.*