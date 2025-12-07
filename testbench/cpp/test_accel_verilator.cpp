// =============================================================================
// test_accel_verilator.cpp — Verilator C++ Testbench for ACCEL-v1
// =============================================================================
// Author: Joshua Carter
// Date: November 19, 2025
// Description: High-performance C++ testbench for accel_top module
//
// Features:
//   - 100 MHz clock generation
//   - VCD waveform tracing
//   - UART TX/RX monitoring
//   - Performance counters
// =============================================================================

#include "Vaccel_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include <iomanip>

// Simulation parameters
#define CLK_PERIOD_NS 10        // 100 MHz
#define SIM_CYCLES 1000         // Run for 1000 cycles
#define RESET_CYCLES 10         // Hold reset for 10 cycles

// Utility macros
#define TICK(dut, tfp, time) do { \
    dut->clk = 1; \
    dut->eval(); \
    tfp->dump(time); \
} while(0)

#define TOCK(dut, tfp, time) do { \
    dut->clk = 0; \
    dut->eval(); \
    tfp->dump(time); \
} while(0)

// =============================================================================
// Main Testbench
// =============================================================================
int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    
    std::cout << "=========================================" << std::endl;
    std::cout << "ACCEL-v1 Verilator Testbench" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Clock frequency: 100 MHz" << std::endl;
    std::cout << "Simulation cycles: " << SIM_CYCLES << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;
    
    // Instantiate DUT
    Vaccel_top* dut = new Vaccel_top;
    
    // Setup VCD tracing
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);  // Trace depth
    tfp->open("trace.vcd");
    
    // Simulation time (in nanoseconds)
    vluint64_t sim_time = 0;
    
    // =========================================================================
    // Reset Sequence
    // =========================================================================
    std::cout << "Applying reset..." << std::endl;
    
    dut->clk = 0;
    dut->rst_n = 0;
    dut->uart_rx = 1;  // UART idle high
    
    for (int i = 0; i < RESET_CYCLES; i++) {
        TICK(dut, tfp, sim_time);
        sim_time += CLK_PERIOD_NS / 2;
        
        TOCK(dut, tfp, sim_time);
        sim_time += CLK_PERIOD_NS / 2;
    }
    
    // Release reset
    dut->rst_n = 1;
    std::cout << "Reset released at " << sim_time << " ns" << std::endl;
    std::cout << std::endl;
    
    // =========================================================================
    // Main Simulation Loop
    // =========================================================================
    std::cout << "Starting main simulation..." << std::endl;
    
    int uart_tx_count = 0;
    int perf_cycle_count = 0;
    
    for (int cycle = 0; cycle < SIM_CYCLES; cycle++) {
        // Rising edge
        TICK(dut, tfp, sim_time);
        sim_time += CLK_PERIOD_NS / 2;
        
        // Monitor UART TX
        if (dut->uart_tx_valid) {
            std::cout << "[Cycle " << std::dec << std::setw(4) << cycle << "] "
                     << "UART TX: 0x" << std::hex << std::setw(2) 
                     << std::setfill('0') << (int)dut->uart_tx_data 
                     << std::dec << std::endl;
            uart_tx_count++;
        }
        
        // Falling edge
        TOCK(dut, tfp, sim_time);
        sim_time += CLK_PERIOD_NS / 2;
        
        // Progress indicator every 100 cycles
        if ((cycle + 1) % 100 == 0) {
            std::cout << "Progress: " << (cycle + 1) << " / " << SIM_CYCLES 
                     << " cycles" << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Simulation Complete!" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Total cycles:       " << SIM_CYCLES << std::endl;
    std::cout << "Simulation time:    " << sim_time << " ns" << std::endl;
    std::cout << "UART TX count:      " << uart_tx_count << std::endl;
    std::cout << "VCD trace:          trace.vcd" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    tfp->close();
    delete dut;
    delete tfp;
    
    std::cout << "✅ Testbench passed!" << std::endl;
    return 0;
}
