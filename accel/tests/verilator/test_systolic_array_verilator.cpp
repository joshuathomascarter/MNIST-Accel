#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vsystolic_array.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vsystolic_array* top = new Vsystolic_array;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}