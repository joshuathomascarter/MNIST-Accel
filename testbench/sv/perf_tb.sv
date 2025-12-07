// =============================================================================
// Performance Monitor Unit Testbench for Coverage
// Directly tests the perf monitor module
// =============================================================================

`timescale 1ns/1ps

module perf_tb;
    // Parameters
    localparam COUNTER_WIDTH = 32;
    
    // Signals
    reg clk;
    reg rst_n;
    
    // Control Inputs
    reg                       start_pulse;
    reg                       done_pulse;
    reg                       busy_signal;
    
    // Metadata Cache Inputs
    reg [COUNTER_WIDTH-1:0]   meta_cache_hits;
    reg [COUNTER_WIDTH-1:0]   meta_cache_misses;
    reg [COUNTER_WIDTH-1:0]   meta_decode_cycles;
    
    // Outputs
    wire [COUNTER_WIDTH-1:0]  total_cycles_count;
    wire [COUNTER_WIDTH-1:0]  active_cycles_count;
    wire [COUNTER_WIDTH-1:0]  idle_cycles_count;
    wire [COUNTER_WIDTH-1:0]  cache_hit_count;
    wire [COUNTER_WIDTH-1:0]  cache_miss_count;
    wire [COUNTER_WIDTH-1:0]  decode_count;
    wire                      measurement_done;
    
    // DUT
    perf #(
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) u_perf (
        .clk(clk),
        .rst_n(rst_n),
        .start_pulse(start_pulse),
        .done_pulse(done_pulse),
        .busy_signal(busy_signal),
        .meta_cache_hits(meta_cache_hits),
        .meta_cache_misses(meta_cache_misses),
        .meta_decode_cycles(meta_decode_cycles),
        .total_cycles_count(total_cycles_count),
        .active_cycles_count(active_cycles_count),
        .idle_cycles_count(idle_cycles_count),
        .cache_hit_count(cache_hit_count),
        .cache_miss_count(cache_miss_count),
        .decode_count(decode_count),
        .measurement_done(measurement_done)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        $dumpfile("perf_tb.vcd");
        $dumpvars(0, perf_tb);
        
        // Initialize
        rst_n = 0;
        start_pulse = 0;
        done_pulse = 0;
        busy_signal = 0;
        meta_cache_hits = 0;
        meta_cache_misses = 0;
        meta_decode_cycles = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        $display("=== Performance Monitor Unit Test ===");
        
        // Test 1: Basic measurement cycle
        $display("\nTest 1: Basic measurement (10 cycles, 6 active)");
        meta_cache_hits = 32'd100;
        meta_cache_misses = 32'd10;
        meta_decode_cycles = 32'd50;
        
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        // Simulate 10 cycles: 6 active, 4 idle
        busy_signal = 1;
        repeat(6) @(posedge clk);
        busy_signal = 0;
        repeat(4) @(posedge clk);
        
        done_pulse = 1;
        @(posedge clk);
        done_pulse = 0;
        @(posedge clk);
        
        // Check results
        $display("  total_cycles  = %0d (expected ~10)", total_cycles_count);
        $display("  active_cycles = %0d (expected 6)", active_cycles_count);
        $display("  idle_cycles   = %0d (expected 4)", idle_cycles_count);
        $display("  cache_hits    = %0d (expected 100)", cache_hit_count);
        $display("  measurement_done pulse received: %b", measurement_done);
        
        repeat(3) @(posedge clk);
        
        // Test 2: Multiple measurement cycles
        $display("\nTest 2: Second measurement (20 cycles, all active)");
        meta_cache_hits = 32'd200;
        meta_cache_misses = 32'd20;
        meta_decode_cycles = 32'd100;
        
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        // 20 fully active cycles
        busy_signal = 1;
        repeat(20) @(posedge clk);
        
        done_pulse = 1;
        @(posedge clk);
        done_pulse = 0;
        @(posedge clk);
        
        $display("  total_cycles  = %0d (expected ~20)", total_cycles_count);
        $display("  active_cycles = %0d (expected 20)", active_cycles_count);
        $display("  idle_cycles   = %0d (expected 0)", idle_cycles_count);
        
        repeat(3) @(posedge clk);
        
        // Test 3: Measurement with all idle cycles
        $display("\nTest 3: All idle cycles");
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        busy_signal = 0;
        repeat(15) @(posedge clk);
        
        done_pulse = 1;
        @(posedge clk);
        done_pulse = 0;
        @(posedge clk);
        
        $display("  total_cycles  = %0d (expected ~15)", total_cycles_count);
        $display("  active_cycles = %0d (expected 0)", active_cycles_count);
        $display("  idle_cycles   = %0d (expected 15)", idle_cycles_count);
        
        // Test 4: Alternating busy/idle
        $display("\nTest 4: Alternating busy/idle");
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        repeat(10) begin
            busy_signal = 1;
            @(posedge clk);
            busy_signal = 0;
            @(posedge clk);
        end
        
        done_pulse = 1;
        @(posedge clk);
        done_pulse = 0;
        @(posedge clk);
        
        $display("  total_cycles  = %0d (expected ~20)", total_cycles_count);
        $display("  active_cycles = %0d (expected 10)", active_cycles_count);
        $display("  idle_cycles   = %0d (expected 10)", idle_cycles_count);
        
        // Test 5: Reset during measurement
        $display("\nTest 5: Reset during measurement");
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        busy_signal = 1;
        repeat(5) @(posedge clk);
        
        // Reset mid-measurement
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        $display("  After reset: total_cycles = %0d (expected 0)", total_cycles_count);
        
        // Test 6: Start without done (edge case)
        $display("\nTest 6: Long measurement");
        meta_cache_hits = 32'd1000;
        meta_cache_misses = 32'd100;
        
        start_pulse = 1;
        @(posedge clk);
        start_pulse = 0;
        
        busy_signal = 1;
        repeat(100) @(posedge clk);
        busy_signal = 0;
        repeat(50) @(posedge clk);
        
        done_pulse = 1;
        @(posedge clk);
        done_pulse = 0;
        @(posedge clk);
        
        $display("  total_cycles  = %0d (expected ~150)", total_cycles_count);
        $display("  active_cycles = %0d (expected 100)", active_cycles_count);
        $display("  idle_cycles   = %0d (expected 50)", idle_cycles_count);
        $display("  cache_hits    = %0d (expected 1000)", cache_hit_count);
        
        repeat(10) @(posedge clk);
        
        $display("\n=== Performance Monitor Test Complete ===");
        $finish;
    end
    
    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end
    
endmodule
