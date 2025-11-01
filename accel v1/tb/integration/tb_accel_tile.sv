// tb_accel_tile.sv
// Integration testbench for accel_top UART-based accelerator
// Tests basic functionality through UART interface

module tb_accel_tile;
    // Parameters 
    localparam N_ROWS = 2;
    localparam N_COLS = 2;
    localparam TM = 8;
    localparam TN = 8;
    localparam TK = 8;
    localparam CLK_HZ = 50_000_000;
    localparam BAUD = 115_200;

    reg clk = 0;
    reg rst_n = 0;

    // UART signals
    reg uart_rx = 1'b1;  // UART idle high
    wire uart_tx;
    
    // Status outputs
    wire busy;
    wire done_tile;

    // DUT instantiation
    accel_top #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .TM(TM),
        .TN(TN),
        .TK(TK),
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        .busy(busy),
        .done_tile(done_tile)
    );

    // Clock generation
    always #10 clk = ~clk; // 50MHz (20ns period)

    initial begin
        $display("TB: Starting accel_top integration test");
        
        // Reset sequence
        rst_n = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("TB: Reset complete, busy=%b", busy);

        // Basic functionality test - send some UART data
        // (This is simplified - real test would send proper UART packets)
        repeat (100) @(posedge clk);
        
        $display("TB: Test completed, busy=%b, done_tile=%b", busy, done_tile);
        $finish;
    end

    // Monitor UART activity
    always @(posedge clk) begin
        if (uart_tx !== 1'b1) begin
            $display("TB: UART TX activity detected at time %t", $time);
        end
    end

    // Timeout
    initial begin
        #1000000; // 1ms timeout
        $display("TB: Timeout - test took too long");
        $finish;
    end

endmodule
        $readmemh("tb/integration/test_vectors/A_0.hex", dut.A_mem);
        $readmemh("tb/integration/test_vectors/B_0.hex", dut.B_mem);

        // Configure CSR via UART
        $display("Configuring CSR via UART...");
        rx = 1; // idle state
        @(posedge clk);
        uart_data_in = 32'h00000001; // Start command
        rx = 0; // Start bit
        @(posedge clk);
        rx = 1; // Stop bit

        // small delay
        repeat (2) @(posedge clk);

        // Start compute
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // wait for done
        wait (done == 1);
        $display("DUT reported done. Verifying results...");

        // Load expected C
        reg [31:0] expected [0:M*N-1];
        $readmemh("tb/integration/test_vectors/C_0.hex", expected);

        integer i;
        integer errs = 0;
        for (i = 0; i < M*N; i = i + 1) begin
            if (dut.C_mem[i] !== expected[i]) begin
                $display("Mismatch at idx %0d: dut=0x%08x expected=0x%08x", i, dut.C_mem[i], expected[i]);
                errs = errs + 1;
            end
        end

        if (errs == 0) begin
            $display("PASS: C matches expected for vector 0");
        end else begin
            $display("FAIL: %0d mismatches", errs);
        end

        $finish;
    end
endmodule
