#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vcsr.h"
#include "Vscheduler.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vcsr* csr = new Vcsr;
    Vscheduler* scheduler = new Vscheduler;

    csr->eval();
    scheduler->eval();
    
    // Add your test cases here
    // Example assertion
    // assert(csr->some_signal == expected_value);

    delete csr;
    delete scheduler;
    return 0;
}