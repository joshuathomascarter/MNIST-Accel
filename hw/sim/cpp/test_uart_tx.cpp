#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vuart_tx.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vuart_tx* top = new Vuart_tx;

    top->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(top->some_signal == expected_value);

    delete top;
    return 0;
}