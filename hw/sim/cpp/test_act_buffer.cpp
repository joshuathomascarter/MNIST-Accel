#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vact_buffer.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vact_buffer* top = new Vact_buffer;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}