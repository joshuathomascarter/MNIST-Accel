// =============================================================================
// PE Unit Testbench for Coverage
// Tests the Processing Element (PE) and MAC8 module independently
// =============================================================================

`timescale 1ns/1ps

module pe_tb;
    // Signals
    reg clk;
    reg rst_n;
    reg signed [7:0] a_in;
    reg signed [7:0] b_in;
    reg en;
    reg clr;
    reg load_weight;
    wire signed [7:0] a_out;
    wire load_weight_out;
    wire signed [31:0] acc;
    
    // DUT
    pe #(
        .PIPE(1),
        .SAT(1)
    ) u_pe (
        .clk(clk),
        .rst_n(rst_n),
        .a_in(a_in),
        .b_in(b_in),
        .en(en),
        .clr(clr),
        .load_weight(load_weight),
        .a_out(a_out),
        .load_weight_out(load_weight_out),
        .acc(acc)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        $dumpfile("pe_tb.vcd");
        $dumpvars(0, pe_tb);
        
        // Initialize
        rst_n = 0;
        a_in = 0;
        b_in = 0;
        en = 0;
        clr = 0;
        load_weight = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        $display("=== PE Unit Test ===");
        
        // Test 1: Load weight
        $display("Test 1: Load weight");
        b_in = 8'd10;
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
        @(posedge clk);
        
        // Test 2: MAC operation
        $display("Test 2: MAC operation");
        a_in = 8'd5;
        en = 1;
        @(posedge clk);
        $display("  acc = %d (expected 50)", acc);
        
        a_in = 8'd3;
        @(posedge clk);
        $display("  acc = %d (expected 80)", acc);  // 50 + 30
        en = 0;  // Disable before next test
        
        // Test 3: Clear accumulator
        $display("Test 3: Clear accumulator");
        clr = 1;
        @(posedge clk);
        clr = 0;
        $display("  acc = %d (expected 0)", acc);
        
        // Test 4: Negative numbers
        $display("Test 4: Negative numbers");
        b_in = -8'd5;
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
        
        en = 1;  // Re-enable for MAC
        a_in = 8'd4;
        @(posedge clk);
        $display("  acc = %d (expected -20)", acc);
        
        // Test 5: Accumulation with negatives
        a_in = -8'd3;
        @(posedge clk);
        $display("  acc = %d (expected -5)", acc);  // -20 + 15
        
        // Test 6: Enable/disable
        $display("Test 6: Enable/disable");
        en = 0;
        a_in = 8'd100;
        @(posedge clk);
        @(posedge clk);
        $display("  acc should not change when en=0: %d", acc);
        
        // Test 7: Multiple MAC cycles
        $display("Test 7: Multiple MAC cycles");
        clr = 1;
        @(posedge clk);
        clr = 0;
        
        b_in = 8'd2;
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
        
        en = 1;
        for (int i = 1; i <= 10; i++) begin
            a_in = i;
            @(posedge clk);
        end
        $display("  acc = %d (expected sum of 2*i for i=1..10 = 110)", acc);
        en = 0;  // Disable before passthrough test
        
        // Test 8: Passthrough of a_out
        $display("Test 8: Activation passthrough");
        a_in = 8'd42;
        @(posedge clk);
        $display("  a_out = %d (expected 42)", a_out);
        
        // Test 9: load_weight passthrough (with en=0)
        $display("Test 9: Load weight passthrough");
        load_weight = 1;
        @(posedge clk);
        $display("  load_weight_out = %d (expected 1)", load_weight_out);
        load_weight = 0;
        @(posedge clk);
        $display("  load_weight_out = %d (expected 0)", load_weight_out);
        
        // Test 10: Edge case - max values
        $display("Test 10: Edge values");
        clr = 1;
        @(posedge clk);
        clr = 0;
        
        // en is already 0 from above
        b_in = 8'd127;  // Max positive
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
        
        a_in = 8'd127;
        en = 1;
        @(posedge clk);
        $display("  acc = %d (expected 16129)", acc);  // 127 * 127
        
        // Test 11: Saturation/Overflow detection
        $display("Test 11: Overflow detection");
        en = 0;
        @(posedge clk);
        clr = 1;
        @(posedge clk);
        clr = 0;
        
        // Load max weight
        load_weight = 1;
        b_in = 8'd127;
        @(posedge clk);
        load_weight = 0;
        
        // Accumulate many large values to cause overflow
        a_in = 8'd127;  // max positive
        en = 1;
        repeat(2000) @(posedge clk);  // 127*127*2000 will overflow 32-bit
        $display("  After 2000 cycles: acc = %d", acc);
        
        // Try with negative numbers for negative overflow
        en = 0;
        @(posedge clk);
        clr = 1;
        @(posedge clk);
        clr = 0;
        
        load_weight = 1;
        b_in = -8'd128;  // Most negative
        @(posedge clk);
        load_weight = 0;
        
        a_in = 8'd127;
        en = 1;
        repeat(500) @(posedge clk);  // -128*127*500 should cause negative overflow
        $display("  After negative overflow: acc = %d", acc);
        
        repeat(5) @(posedge clk);
        
        $display("=== PE Unit Test Complete ===");
        $finish;
    end
    
    // Timeout
    initial begin
        #100000;  // Increased for overflow tests
        $display("TIMEOUT");
        $finish;
    end
    
endmodule
