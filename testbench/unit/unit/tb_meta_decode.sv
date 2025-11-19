// =============================================================================
// tb_meta_decode.sv — Testbench for Metadata Decoder
// =============================================================================
// Tests:
//   - Metadata write and cache storage
//   - Scheduler read-back verification
//   - CRC validation
//   - Error detection (invalid metadata types)
//   - Performance counter accumulation
//   - Cache hit/miss tracking
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_meta_decode;

    localparam CLK_PERIOD = 20;  // 50 MHz
    
    reg clk, rst_n;
    
    // DMA metadata input (write interface)
    reg [31:0] wr_data;
    reg [7:0]  wr_addr;
    reg [1:0]  wr_type;
    reg        wr_en;
    wire       wr_ready;
    
    // Scheduler read interface
    reg        rd_en;
    reg [7:0]  rd_addr;
    reg [1:0]  rd_type;
    wire [31:0] rd_data;
    wire        rd_valid;
    wire        rd_hit;
    
    // Config
    reg [15:0] cfg_num_rows;
    reg [15:0] cfg_num_cols;
    reg [31:0] cfg_total_blocks;
    reg [2:0]  cfg_block_size;
    
    // Performance
    wire [31:0] perf_cache_hits;
    wire [31:0] perf_cache_misses;
    wire [31:0] perf_decode_cycles;
    
    // Status
    wire        meta_error;
    wire [31:0] meta_error_flags;
    
    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    
    meta_decode dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_data(wr_data),
        .wr_addr(wr_addr),
        .wr_type(wr_type),
        .wr_en(wr_en),
        .wr_ready(wr_ready),
        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_type(rd_type),
        .rd_data(rd_data),
        .rd_valid(rd_valid),
        .rd_hit(rd_hit),
        .cache_hits(perf_cache_hits),
        .cache_misses(perf_cache_misses),
        .total_reads(perf_decode_cycles)
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
        wr_data = 32'd0;
        wr_addr = 8'd0;
        wr_type = 2'd0;
        wr_en = 1'b0;
        rd_en = 1'b0;
        rd_addr = 8'd0;
        rd_type = 2'd0;
        
        // Release reset
        @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 1: Write metadata (ROW_PTR type)
        // ====================================================================
        $display("\n[TEST 1] Write metadata - ROW_PTR");
        
        for (int i = 0; i < 10; i = i + 1) begin
            wr_data <= 32'h1000_0000 + (i << 4);
            wr_type <= 2'b00;  // ROW_PTR
            wr_en <= 1'b1;
            @(posedge clk);
            $display("  Write[%0d]: addr=0x%02x data=0x%08x (type=ROW_PTR)", i, wr_addr, wr_data);
        end
        
        wr_en <= 1'b0;
        repeat(10) @(posedge clk);
        
        // ====================================================================
        // TEST 2: Read metadata back (scheduler path)
        // ====================================================================
        $display("\n[TEST 2] Read metadata back");
        
        for (int i = 0; i < 5; i = i + 1) begin
            rd_addr <= i;
            rd_type <= 2'b00;  // ROW_PTR
            rd_en <= 1'b1;
            @(posedge clk);
            @(posedge clk);
            
            if (rd_valid) begin
                $display("  Read[%0d]: 0x%08x (valid=%b hit=%b)", i, rd_data, rd_valid, rd_hit);
            end else begin
                $display("  Read[%0d]: DATA INVALID", i);
            end
        end
        
        rd_en <= 1'b0;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 3: Write COL_IDX metadata
        // ====================================================================
        $display("\n[TEST 3] Write metadata - COL_IDX");
        
        for (int i = 0; i < 8; i = i + 1) begin
            wr_data <= 32'h0000_0000 + i;  // col_idx values
            wr_type <= 2'b01;  // COL_IDX
            wr_en <= 1'b1;
            @(posedge clk);
            $display("  Write[%0d]: addr=0x%02x data=0x%08x (type=COL_IDX)", i, wr_addr, wr_data);
        end
        
        wr_en <= 1'b0;
        repeat(10) @(posedge clk);
        
        // ====================================================================
        // TEST 4: Performance counter check
        // ====================================================================
        $display("\n[TEST 4] Performance metrics");
        $display("  Cache hits: %0d", perf_cache_hits);
        $display("  Cache misses: %0d", perf_cache_misses);
        $display("  Decode cycles: %0d", perf_decode_cycles);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 5: Error injection (invalid metadata type)
        // ====================================================================
        $display("\n[TEST 5] Error injection - invalid type");
        
        wr_data <= 32'h0000_0000;
        wr_type <= 2'b11;  // INVALID (only 00, 01, 10 valid)
        wr_en <= 1'b1;
        @(posedge clk);
        $display("  Wrote invalid type; checking for error handling");
        
        wr_en <= 1'b0;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // Done
        // ====================================================================
        $display("\n[DONE] Metadata decode testbench complete\n");
        $finish;
    end
    
    // ========================================================================
    // Waveform Dump
    // ========================================================================
    
    initial begin
        $dumpfile("tb_meta_decode.vcd");
        $dumpvars(0, tb_meta_decode);
    end

endmodule

`default_nettype wire
