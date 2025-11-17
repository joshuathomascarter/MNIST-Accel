#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vwgt_buffer.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vwgt_buffer* top = new Vwgt_buffer;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}