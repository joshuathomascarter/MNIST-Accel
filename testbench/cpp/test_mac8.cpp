#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vmac8.h"

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vmac8* top = new Vmac8;

    // Test logic here
    top->eval();

    // Assertions to validate functionality
    assert(top->some_output_signal == expected_value);

    delete top;
    return 0;
}