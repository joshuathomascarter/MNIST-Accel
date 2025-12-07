#include <iostream>
#include <cstdint>
#include <verilated.h> // Verilator Library
#include <verilated_cov.h> // Verilator Coverage
#include "Vpe.h"       // The Verilog Model (Generated later)

// ==========================================
// 1. The Custom Stack Allocator (Unchanged)
// ==========================================
class StackAllocator {
public:
    StackAllocator(size_t size_bytes) {
        total_size = size_bytes;
        memory_pool = new uint8_t[size_bytes];
        offset = 0;
    }

    ~StackAllocator() {
        delete[] memory_pool;
    }

    template <typename T>
    T* alloc(size_t count = 1) {
        size_t size_needed = sizeof(T) * count;
        if (offset + size_needed > total_size) return nullptr;
        void* ptr = memory_pool + offset;
        offset += size_needed;
        return static_cast<T*>(ptr);
    }

    void reset() {
        offset = 0;
    }

private:
    uint8_t* memory_pool;
    size_t total_size;
    size_t offset;
};

// ==========================================
// 2. Test Vector Structure
// ==========================================
// This matches the input pins of your PE module
struct TestCase {
    int8_t a_in;
    int8_t b_in;
    bool load_weight;
    bool en;
    bool clr;
    const char* description;
};

// ==========================================
// 3. Main Simulation
// ==========================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    std::cout << "--- Starting PE Hardware Simulation ---" << std::endl;

    // 1. Setup Allocator & DUT (Device Under Test)
    StackAllocator mem(4096);
    Vpe* dut = new Vpe; // Instantiate the Verilog module

    // 2. Create Test Scenarios using our Allocator
    int num_tests = 5;
    TestCase* tests = mem.alloc<TestCase>(num_tests);

    // Cycle 0: Reset
    tests[0] = {0, 0, 0, 0, 1, "Reset"};
    // Cycle 1: Load Weight (Weight = 5)
    tests[1] = {0, 5, 1, 0, 0, "Load Weight 5"};
    // Cycle 2: Compute (Input 2 * Weight 5 = 10)
    tests[2] = {2, 0, 0, 1, 0, "Compute 2 * 5"};
    // Cycle 3: Compute (Input 3 * Weight 5 = 15) -> Accumulator should be 10 + 15 = 25
    tests[3] = {3, 0, 0, 1, 0, "Compute 3 * 5"};
    // Cycle 4: Hold (Do nothing)
    tests[4] = {0, 0, 0, 0, 0, "Hold"};

    // 3. Run Simulation Loop
    dut->clk = 0;
    dut->rst_n = 0; // Assert Reset
    dut->eval();    // Update hardware
    dut->rst_n = 1; // Release Reset

    for (int i = 0; i < num_tests; ++i) {
        // A. Drive Inputs from our Test Case
        dut->a_in = tests[i].a_in;
        dut->b_in = tests[i].b_in;
        dut->load_weight = tests[i].load_weight;
        dut->en = tests[i].en;
        dut->clr = tests[i].clr;

        // B. Toggle Clock (Rising Edge)
        dut->clk = 1;
        dut->eval(); // Solve logic

        // C. Print Status
        std::cout << "[Cycle " << i << "] " << tests[i].description 
                  << " | Output Acc: " << (int)dut->acc << std::endl;

        // D. Toggle Clock (Falling Edge)
        dut->clk = 0;
        dut->eval();
    }

    // Cleanup
    dut->final();
    VerilatedCov::write("logs/pe_coverage.dat");
    delete dut;
    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}