// =============================================================================
// Systolic Array Unit Testbench for Coverage
// Directly tests the sparse systolic array
// =============================================================================

`timescale 1ns/1ps

module systolic_tb;
    // Parameters (match the actual design)
    localparam N_ROWS = 8;
    localparam N_COLS = 8;
    localparam DATA_W = 8;
    localparam ACC_W = 32;
    
    // Signals
    reg clk;
    reg rst_n;
    reg block_valid;
    reg load_weight;
    reg [N_ROWS*DATA_W-1:0] a_in_flat;
    reg [N_COLS*DATA_W-1:0] b_in_flat;
    wire [N_ROWS*N_COLS*ACC_W-1:0] c_out_flat;
    
    // DUT
    systolic_array_sparse #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_systolic (
        .clk(clk),
        .rst_n(rst_n),
        .block_valid(block_valid),
        .load_weight(load_weight),
        .a_in_flat(a_in_flat),
        .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Helper function to pack activations
    function automatic [N_ROWS*DATA_W-1:0] pack_activations;
        input [7:0] a [0:N_ROWS-1];
        integer i;
        begin
            pack_activations = 0;
            for (i = 0; i < N_ROWS; i++) begin
                pack_activations[i*DATA_W +: DATA_W] = a[i];
            end
        end
    endfunction
    
    // Helper function to pack weights
    function automatic [N_COLS*DATA_W-1:0] pack_weights;
        input [7:0] b [0:N_COLS-1];
        integer i;
        begin
            pack_weights = 0;
            for (i = 0; i < N_COLS; i++) begin
                pack_weights[i*DATA_W +: DATA_W] = b[i];
            end
        end
    endfunction
    
    // Test sequence
    initial begin
        integer i, j;
        reg [7:0] a_arr [0:N_ROWS-1];
        reg [7:0] b_arr [0:N_COLS-1];
        
        $dumpfile("systolic_tb.vcd");
        $dumpvars(0, systolic_tb);
        
        // Initialize
        rst_n = 0;
        block_valid = 0;
        load_weight = 0;
        a_in_flat = 0;
        b_in_flat = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        $display("=== Systolic Array Unit Test ===");
        $display("Array Size: %0dx%0d", N_ROWS, N_COLS);
        
        // Test 1: Load weights into all PEs
        // IMPORTANT: block_valid must be LOW when loading weights (PE assertion)
        $display("\nTest 1: Loading weights");
        block_valid = 0;  // Must be 0 during weight load
        load_weight = 1;
        
        // Load weight for each column - takes N_ROWS cycles to fill
        for (i = 0; i < N_ROWS; i++) begin
            // Set weights for this row of PEs
            for (j = 0; j < N_COLS; j++) begin
                b_arr[j] = (i * N_COLS + j + 1) & 8'hFF;  // Weight = 1, 2, 3, ...
            end
            b_in_flat = pack_weights(b_arr);
            @(posedge clk);
        end
        load_weight = 0;
        // Wait for load_weight to propagate through all columns
        repeat(N_COLS + 2) @(posedge clk);
        $display("  Weights loaded: 1 to %0d", N_ROWS * N_COLS);
        
        // Test 2: Matrix multiply with identity-like activations
        $display("\nTest 2: Matrix multiply");
        block_valid = 1;
        
        // Send N_ROWS activation vectors
        for (i = 0; i < N_ROWS + N_COLS; i++) begin
            for (j = 0; j < N_ROWS; j++) begin
                // Each row gets activation = 1 only when it matches the cycle
                if (i >= j && i < j + N_ROWS) begin
                    a_arr[j] = 8'd1;
                end else begin
                    a_arr[j] = 8'd0;
                end
            end
            a_in_flat = pack_activations(a_arr);
            @(posedge clk);
        end
        block_valid = 0;
        
        $display("  Computation complete");
        
        // Test 3: Check some outputs
        $display("\nTest 3: Checking outputs");
        for (i = 0; i < 4; i++) begin
            $display("  out[0][%0d] = %0d", i, $signed(c_out_flat[(i*ACC_W) +: ACC_W]));
        end
        
        // Test 4: Clear and reload
        $display("\nTest 4: Reset and reload");
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Load different weights (block_valid must be 0)
        block_valid = 0;
        load_weight = 1;
        for (i = 0; i < N_ROWS; i++) begin
            for (j = 0; j < N_COLS; j++) begin
                b_arr[j] = 8'd2;  // All weights = 2
            end
            b_in_flat = pack_weights(b_arr);
            @(posedge clk);
        end
        load_weight = 0;
        // Wait for load_weight to propagate through all columns
        repeat(N_COLS + 2) @(posedge clk);
        
        // Compute with all-ones activations
        block_valid = 1;
        for (i = 0; i < N_ROWS + N_COLS; i++) begin
            for (j = 0; j < N_ROWS; j++) begin
                a_arr[j] = 8'd1;
            end
            a_in_flat = pack_activations(a_arr);
            @(posedge clk);
        end
        block_valid = 0;
        
        // Each output should be 2 * 8 = 16 (weight * activation_sum)
        $display("  Expected each output = 16 (2 * 8 activations)");
        $display("  out[0][0] = %0d", $signed(c_out_flat[0 +: ACC_W]));
        
        // Test 5: Negative values
        $display("\nTest 5: Negative values");
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Load negative weights (block_valid must be 0)
        block_valid = 0;
        load_weight = 1;
        for (i = 0; i < N_ROWS; i++) begin
            for (j = 0; j < N_COLS; j++) begin
                b_arr[j] = -8'd3;  // Weight = -3
            end
            b_in_flat = pack_weights(b_arr);
            @(posedge clk);
        end
        load_weight = 0;
        // Wait for load_weight to propagate through all columns
        repeat(N_COLS + 2) @(posedge clk);
        
        // Compute with positive activations
        block_valid = 1;
        for (i = 0; i < N_ROWS + N_COLS; i++) begin
            for (j = 0; j < N_ROWS; j++) begin
                a_arr[j] = 8'd2;  // Activation = 2
            end
            a_in_flat = pack_activations(a_arr);
            @(posedge clk);
        end
        block_valid = 0;
        
        $display("  Expected each output = -48 (-3 * 2 * 8)");
        $display("  out[0][0] = %0d", $signed(c_out_flat[0 +: ACC_W]));
        
        repeat(10) @(posedge clk);
        
        $display("\n=== Systolic Array Test Complete ===");
        $finish;
    end
    
    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end
    
endmodule
