#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vuart_rx.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vuart_rx* top = new Vuart_rx;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}