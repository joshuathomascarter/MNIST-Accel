// =============================================================================
// Meta Decode Unit Testbench for Coverage
// Directly tests the metadata decoder module
// =============================================================================

`timescale 1ns/1ps

module meta_decode_tb;
    // Parameters
    localparam DATA_WIDTH = 32;
    localparam CACHE_DEPTH = 64;
    
    // Signals
    reg clk;
    reg rst_n;
    
    // Interface to BSR Scheduler
    reg                     req_valid;
    reg  [31:0]             req_addr;
    wire                    req_ready;
    
    // Interface to Memory (simulated BRAM)
    wire                    mem_en;
    wire [31:0]             mem_addr;
    reg  [DATA_WIDTH-1:0]   mem_rdata;
    
    // Output to Scheduler
    wire                    meta_valid;
    wire [DATA_WIDTH-1:0]   meta_rdata;
    reg                     meta_ready;
    
    // Simulated memory
    reg [DATA_WIDTH-1:0] memory [0:255];
    
    // DUT
    meta_decode #(
        .DATA_WIDTH(DATA_WIDTH),
        .CACHE_DEPTH(CACHE_DEPTH)
    ) u_meta_decode (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_addr(req_addr),
        .req_ready(req_ready),
        .mem_en(mem_en),
        .mem_addr(mem_addr),
        .mem_rdata(mem_rdata),
        .meta_valid(meta_valid),
        .meta_rdata(meta_rdata),
        .meta_ready(meta_ready)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Memory response model (1-cycle latency BRAM)
    always @(posedge clk) begin
        if (mem_en) begin
            mem_rdata <= memory[mem_addr[7:0]];
        end
    end
    
    // Test sequence
    initial begin
        integer i;
        
        $dumpfile("meta_decode_tb.vcd");
        $dumpvars(0, meta_decode_tb);
        
        // Initialize memory with test pattern
        for (i = 0; i < 256; i++) begin
            memory[i] = 32'hDEAD_0000 + i;
        end
        
        // Initialize
        rst_n = 0;
        req_valid = 0;
        req_addr = 0;
        meta_ready = 1;
        mem_rdata = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        $display("=== Meta Decode Unit Test ===");
        
        // Test 1: Single fetch (cache miss)
        $display("\nTest 1: Single fetch (cache miss)");
        req_valid = 1;
        req_addr = 32'd10;
        @(posedge clk);
        while (!req_ready) @(posedge clk);
        req_valid = 0;
        
        // Wait for output
        while (!meta_valid) @(posedge clk);
        $display("  Fetched addr 10: data = 0x%08X (expected 0xDEAD000A)", meta_rdata);
        @(posedge clk);
        
        // Test 2: Same address again (cache hit)
        $display("\nTest 2: Same address (cache hit)");
        req_valid = 1;
        req_addr = 32'd10;
        @(posedge clk);
        while (!req_ready) @(posedge clk);
        req_valid = 0;
        
        while (!meta_valid) @(posedge clk);
        $display("  Fetched addr 10: data = 0x%08X (should be cached)", meta_rdata);
        @(posedge clk);
        
        // Test 3: Multiple sequential fetches (miss then hit)
        $display("\nTest 3: Multiple sequential addresses");
        for (i = 0; i < 8; i++) begin
            req_valid = 1;
            req_addr = i * 4;
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            req_valid = 0;
            
            while (!meta_valid) @(posedge clk);
            $display("  addr %0d: data = 0x%08X", i * 4, meta_rdata);
            @(posedge clk);
        end
        
        // Test 4: Re-fetch cached addresses
        $display("\nTest 4: Re-fetch cached addresses (all hits)");
        for (i = 0; i < 8; i++) begin
            req_valid = 1;
            req_addr = i * 4;
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            req_valid = 0;
            
            while (!meta_valid) @(posedge clk);
            @(posedge clk);
        end
        $display("  All 8 re-fetches complete (cache hits)");
        
        // Test 5: Pipelined requests
        $display("\nTest 5: Pipelined requests");
        meta_ready = 1;
        for (i = 0; i < 4; i++) begin
            req_valid = 1;
            req_addr = 32'd20 + i;
            @(posedge clk);
            if (meta_valid) begin
                $display("  Pipelined output: 0x%08X", meta_rdata);
            end
        end
        req_valid = 0;
        
        // Drain remaining outputs
        repeat(10) begin
            @(posedge clk);
            if (meta_valid) begin
                $display("  Pipelined output: 0x%08X", meta_rdata);
            end
        end
        
        // Test 6: Stall output (meta_ready = 0)
        $display("\nTest 6: Output stall");
        req_valid = 1;
        req_addr = 32'd50;
        @(posedge clk);
        while (!req_ready) @(posedge clk);
        req_valid = 0;
        
        // Stall output acceptance
        meta_ready = 0;
        repeat(5) @(posedge clk);
        $display("  Stalled for 5 cycles, meta_valid = %b", meta_valid);
        meta_ready = 1;
        @(posedge clk);
        if (meta_valid) begin
            $display("  After unstall: data = 0x%08X", meta_rdata);
        end
        
        // Test 7: Reset during operation
        $display("\nTest 7: Reset during operation");
        req_valid = 1;
        req_addr = 32'd60;
        @(posedge clk);
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        req_valid = 0;
        repeat(3) @(posedge clk);
        $display("  Reset complete, decoder should be idle");
        
        // Test 8: High address (test cache index wrapping)
        $display("\nTest 8: High addresses (cache wrap)");
        req_valid = 1;
        req_addr = 32'd100;  // Will wrap in cache (64 entries)
        @(posedge clk);
        while (!req_ready) @(posedge clk);
        req_valid = 0;
        while (!meta_valid) @(posedge clk);
        $display("  addr 100: data = 0x%08X", meta_rdata);
        
        repeat(10) @(posedge clk);
        
        $display("\n=== Meta Decode Test Complete ===");
        $finish;
    end
    
    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end
    
endmodule
