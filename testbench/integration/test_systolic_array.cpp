#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vyour_systolic_array.h"

class TestSystolicArray {
public:
    Vyour_systolic_array* top;
    VerilatedContext* context;

    TestSystolicArray() {
        context = new VerilatedContext;
        top = new Vyour_systolic_array{context};
    }

    void run() {
        // Setup
        top->reset = 1;
        top->eval();
        top->reset = 0;

        // Test case
        top->input_signal = 1; // Example input
        top->eval();
        assert(top->output_signal == expected_output); // Replace with expected output

        // Teardown
        delete top;
        delete context;
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    TestSystolicArray test;
    test.run();
    return 0;
}