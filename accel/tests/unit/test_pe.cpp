#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vpe.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vpe* top = new Vpe;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    assert(top->some_signal == expected_value);

    delete top;
    return 0;
}