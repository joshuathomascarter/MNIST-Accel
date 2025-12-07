#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vaccel_top.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vaccel_top* top = new Vaccel_top;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}