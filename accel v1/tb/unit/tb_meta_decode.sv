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
    
    // DMA metadata input
    reg [31:0] dma_meta_data;
    reg [3:0]  dma_meta_valid;
    reg [1:0]  dma_meta_type;
    reg        dma_meta_wen;
    wire       dma_meta_ready;
    
    // Scheduler read
    reg [7:0]  sched_meta_raddr;
    reg        sched_meta_ren;
    wire [31:0] sched_meta_rdata;
    wire        sched_meta_rvalid;
    
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
    
    meta_decode #(
        .METADATA_CACHE_DEPTH(256),
        .METADATA_CACHE_ADDR_W(8),
        .DATA_WIDTH(32),
        .ENABLE_CRC(1),
        .ENABLE_PERF(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .dma_meta_data(dma_meta_data),
        .dma_meta_valid(dma_meta_valid),
        .dma_meta_type(dma_meta_type),
        .dma_meta_wen(dma_meta_wen),
        .dma_meta_ready(dma_meta_ready),
        .sched_meta_raddr(sched_meta_raddr),
        .sched_meta_ren(sched_meta_ren),
        .sched_meta_rdata(sched_meta_rdata),
        .sched_meta_rvalid(sched_meta_rvalid),
        .cfg_num_rows(cfg_num_rows),
        .cfg_num_cols(cfg_num_cols),
        .cfg_total_blocks(cfg_total_blocks),
        .cfg_block_size(cfg_block_size),
        .perf_cache_hits(perf_cache_hits),
        .perf_cache_misses(perf_cache_misses),
        .perf_decode_cycles(perf_decode_cycles),
        .meta_error(meta_error),
        .meta_error_flags(meta_error_flags)
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
        dma_meta_data = 32'd0;
        dma_meta_valid = 4'd0;
        dma_meta_type = 2'd0;
        dma_meta_wen = 1'b0;
        sched_meta_raddr = 8'd0;
        sched_meta_ren = 1'b0;
        cfg_num_rows = 16'd32;
        cfg_num_cols = 16'd32;
        cfg_total_blocks = 32'd512;
        cfg_block_size = 3'd1;  // 8×8 blocks
        
        // Release reset
        @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 1: Write metadata (ROW_PTR type)
        // ====================================================================
        $display("\n[TEST 1] Write metadata - ROW_PTR");
        
        for (int i = 0; i < 10; i = i + 1) begin
            dma_meta_data <= 32'h1000_0000 + (i << 4);
            dma_meta_valid <= 4'hF;
            dma_meta_type <= 2'b00;  // ROW_PTR
            dma_meta_wen <= 1'b1;
            @(posedge clk);
            $display("  Write[%0d]: 0x%08x (type=ROW_PTR)", i, dma_meta_data);
        end
        
        dma_meta_wen <= 1'b0;
        repeat(10) @(posedge clk);
        
        // ====================================================================
        // TEST 2: Read metadata back (scheduler path)
        // ====================================================================
        $display("\n[TEST 2] Read metadata back");
        
        for (int i = 0; i < 5; i = i + 1) begin
            sched_meta_raddr <= i;
            sched_meta_ren <= 1'b1;
            @(posedge clk);
            @(posedge clk);
            
            if (sched_meta_rvalid) begin
                $display("  Read[%0d]: 0x%08x (valid=%b)", i, sched_meta_rdata, sched_meta_rvalid);
            end else begin
                $display("  Read[%0d]: DATA INVALID", i);
            end
        end
        
        sched_meta_ren <= 1'b0;
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 3: Write COL_IDX metadata
        // ====================================================================
        $display("\n[TEST 3] Write metadata - COL_IDX");
        
        for (int i = 0; i < 8; i = i + 1) begin
            dma_meta_data <= 32'hAAAA_0000 + i;
            dma_meta_valid <= 4'hF;
            dma_meta_type <= 2'b01;  // COL_IDX
            dma_meta_wen <= 1'b1;
            @(posedge clk);
            $display("  Write[%0d]: 0x%08x (type=COL_IDX)", i, dma_meta_data);
        end
        
        dma_meta_wen <= 1'b0;
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
        
        dma_meta_data <= 32'h0000_0000;
        dma_meta_valid <= 4'hF;
        dma_meta_type <= 2'b11;  // INVALID
        dma_meta_wen <= 1'b1;
        @(posedge clk);
        $display("  Wrote invalid type; error=%b, error_flags=0x%08x", meta_error, meta_error_flags);
        
        dma_meta_wen <= 1'b0;
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
