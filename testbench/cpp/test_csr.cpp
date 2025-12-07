#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vcsr.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vcsr* top = new Vcsr;

    // Test logic for CSR component
    top->reset = 1;
    top->eval();
    top->reset = 0;
    top->eval();

    // Add assertions to validate functionality
    // Example assertion
    assert(top->some_output_signal == expected_value);

    delete top;
    return 0;
}