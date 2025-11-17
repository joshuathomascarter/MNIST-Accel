#include <verilated.h>
#include <verilated_vcd_c.h>
#include <iostream>
#include "Vmac8.h"

void run_test() {
    Vmac8* top = new Vmac8;
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("waveform.vcd");

    top->clk = 0;
    top->eval();
    tfp->dump(0);

    top->clk = 1;
    top->eval();
    tfp->dump(5);

    top->clk = 0;
    top->eval();
    tfp->dump(10);

    top->clk = 1;
    top->eval();
    tfp->dump(15);

    tfp->close();
    delete top;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    run_test();
    return 0;
}