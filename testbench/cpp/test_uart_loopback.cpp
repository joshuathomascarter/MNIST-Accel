#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vuart_rx.h"
#include "Vuart_tx.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vuart_rx* uart_rx = new Vuart_rx;
    Vuart_tx* uart_tx = new Vuart_tx;

    uart_rx->eval();
    uart_tx->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(uart_rx->some_signal == expected_value);

    delete uart_rx;
    delete uart_tx;
    return 0;
}