#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vact_buffer.h"
#include "Vwgt_buffer.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vact_buffer* act_buf = new Vact_buffer;
    Vwgt_buffer* wgt_buf = new Vwgt_buffer;

    act_buf->eval();
    wgt_buf->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(act_buf->some_signal == expected_value);

    delete act_buf;
    delete wgt_buf;
    return 0;
}