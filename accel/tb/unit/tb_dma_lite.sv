// =============================================================================
// tb_dma_lite.sv — Testbench for Lightweight DMA Engine
// =============================================================================
// Tests:
//   - FIFO write and read
//   - 8-bit to 32-bit word assembly
//   - Packet transfer
//   - Status signals (done, bytes_transferred)
//   - Backpressure handling
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_dma_lite;

    localparam CLK_PERIOD = 20;  // 50 MHz
    
    reg clk, rst_n;
    
    // Input
    reg [7:0] in_data;
    reg       in_valid;
    wire      in_ready;
    
    // Output
    wire [31:0] out_data;
    wire        out_valid;
    reg         out_ready;
    
    // Control
    reg [15:0] cfg_pkt_len;
    reg        cfg_enable;
    
    // Status
    wire       dma_done;
    wire       dma_error;
    wire [31:0] dma_bytes_transferred;
    
    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    
    dma_lite #(
        .DATA_WIDTH(8),
        .FIFO_DEPTH(64),
        .FIFO_PTR_W(6)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(in_data),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .out_data(out_data),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .cfg_pkt_len(cfg_pkt_len),
        .cfg_enable(cfg_enable),
        .dma_done(dma_done),
        .dma_error(dma_error),
        .dma_bytes_transferred(dma_bytes_transferred)
    );
    
    // ========================================================================
    // Clock Generation
    // ========================================================================
    
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ========================================================================
    // Test Stimulus
    // ========================================================================
    
    initial begin
        // Initialize
        rst_n = 1'b0;
        in_data = 8'd0;
        in_valid = 1'b0;
        out_ready = 1'b0;
        cfg_pkt_len = 16'd64;
        cfg_enable = 1'b1;
        
        // Release reset
        @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 1: Write single 32-bit word (4 bytes)
        // ====================================================================
        $display("\n[TEST 1] Write and assemble single 32-bit word");
        
        // Byte 0
        in_data <= 8'h11;
        in_valid <= 1'b1;
        @(posedge clk);
        $display("  In: 0x%02x", in_data);
        
        // Byte 1
        in_data <= 8'h22;
        @(posedge clk);
        $display("  In: 0x%02x", in_data);
        
        // Byte 2
        in_data <= 8'h33;
        @(posedge clk);
        $display("  In: 0x%02x", in_data);
        
        // Byte 3
        in_data <= 8'h44;
        @(posedge clk);
        $display("  In: 0x%02x", in_data);
        
        in_valid <= 1'b0;
        
        // Wait for assembly
        repeat(10) @(posedge clk);
        
        $display("  Out valid: %b, Out data: 0x%08x (expect 0x44332211)", out_valid, out_data);
        
        // ====================================================================
        // TEST 2: Read output word with backpressure
        // ====================================================================
        $display("\n[TEST 2] Read output with ready handshake");
        
        // Write 4 more bytes
        for (int i = 0; i < 4; i = i + 1) begin
            in_data <= 8'h55 + i;
            in_valid <= 1'b1;
            @(posedge clk);
            $display("  In: 0x%02x", in_data);
        end
        
        in_valid <= 1'b0;
        repeat(5) @(posedge clk);
        
        // Read with ready signal
        out_ready <= 1'b1;
        @(posedge clk);
        $display("  Out valid: %b, Out data: 0x%08x", out_valid, out_data);
        
        out_ready <= 1'b0;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 3: Bulk transfer
        // ====================================================================
        $display("\n[TEST 3] Bulk transfer (16 words)");
        
        out_ready <= 1'b1;
        
        // Write 64 bytes (16 words)
        for (int i = 0; i < 64; i = i + 1) begin
            in_data <= i;
            in_valid <= 1'b1;
            @(posedge clk);
        end
        
        in_valid <= 1'b0;
        
        // Wait for all words to be transferred
        repeat(100) @(posedge clk);
        
        $display("  Total bytes transferred: %0d (expect 64)", dma_bytes_transferred);
        $display("  DMA done: %b", dma_done);
        
        out_ready <= 1'b0;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // Done
        // ====================================================================
        $display("\n[DONE] DMA lite testbench complete\n");
        $finish;
    end
    
    // ========================================================================
    // Waveform Dump
    // ========================================================================
    
    initial begin
        $dumpfile("tb_dma_lite.vcd");
        $dumpvars(0, tb_dma_lite);
    end

endmodule

`default_nettype wire
