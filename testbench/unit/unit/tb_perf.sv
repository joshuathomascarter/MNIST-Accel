// tb/unit/tb_perf.sv
//
// Testbench for the 'perf' performance monitor module.
//
// Verifies three scenarios:
//  1. 100% Utilization: 'busy_signal' is always high during measurement.
//  2. 50% Utilization: 'busy_signal' toggles every cycle.
//  3. Back-to-Back Operations: Ensures counters reset correctly for a new measurement.

`default_nettype none

module tb_perf;

    // Parameters
    localparam CLK_PERIOD = 10; // 10ns = 100MHz clock
    localparam COUNTER_WIDTH = 32;

    // Testbench signals
    logic clk;
    logic rst_n;
    logic start_pulse;
    logic done_pulse;
    logic busy_signal;
    logic [COUNTER_WIDTH-1:0] meta_cache_hits;
    logic [COUNTER_WIDTH-1:0] meta_cache_misses;
    logic [COUNTER_WIDTH-1:0] meta_decode_cycles;

    // DUT outputs
    wire [COUNTER_WIDTH-1:0] total_cycles_count;
    wire [COUNTER_WIDTH-1:0] active_cycles_count;
    wire [COUNTER_WIDTH-1:0] idle_cycles_count;
    wire [COUNTER_WIDTH-1:0] cache_hit_count;
    wire [COUNTER_WIDTH-1:0] cache_miss_count;
    wire [COUNTER_WIDTH-1:0] decode_count;
    wire                     measurement_done;

    // Instantiate the DUT (Device Under Test)
    perf #(
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) dut (
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

    // Clock generation
    always #(CLK_PERIOD / 2) clk = ~clk;

    // Main test sequence
    initial begin
        // Initialization
        clk = 0;
        rst_n = 0;
        start_pulse = 0;
        done_pulse = 0;
        busy_signal = 0;
        meta_cache_hits = 0;
        meta_cache_misses = 0;
        meta_decode_cycles = 0;
        $display("==================================================");
        $display("Starting Testbench for perf.v");
        $display("==================================================");

        // Apply reset
        #CLK_PERIOD;
        rst_n = 1;
        #CLK_PERIOD;

        // --- Scenario 1: 100% Utilization ---
        $display("\n--- SCENARIO 1: 100%% Utilization (100 cycles) ---");
        busy_signal = 1;
        start_pulse = 1;
        #CLK_PERIOD;
        start_pulse = 0;
        repeat (99) #CLK_PERIOD;
        done_pulse = 1;
        #CLK_PERIOD;
        done_pulse = 0;
        #CLK_PERIOD; // Wait for outputs to be valid

        // Check results
        if (total_cycles_count == 100 && active_cycles_count == 100 && idle_cycles_count == 0) begin
            $display("  [PASS] Scenario 1 Correct!");
        end else begin
            $error("  [FAIL] Scenario 1 Incorrect!");
            $display("    Expected: total=100, active=100, idle=0");
            $display("    Got:      total=%0d, active=%0d, idle=%0d", total_cycles_count, active_cycles_count, idle_cycles_count);
        end
        #CLK_PERIOD;

        // --- Scenario 2: 50% Utilization ---
        $display("\n--- SCENARIO 2: 50%% Utilization (50 cycles) ---");
        busy_signal = 0;
        start_pulse = 1;
        #CLK_PERIOD;
        start_pulse = 0;
        repeat (49) begin
            busy_signal = ~busy_signal;
            #CLK_PERIOD;
        end
        done_pulse = 1;
        #CLK_PERIOD;
        done_pulse = 0;
        #CLK_PERIOD;

        // Check results
        if (total_cycles_count == 50 && active_cycles_count == 26 && idle_cycles_count == 24) begin
            $display("  [PASS] Scenario 2 Correct!");
        end else begin
            $error("  [FAIL] Scenario 2 Incorrect!");
            $display("    Expected: total=50, active=26, idle=24");
            $display("    Got:      total=%0d, active=%0d, idle=%0d", total_cycles_count, active_cycles_count, idle_cycles_count);
        end
        #CLK_PERIOD;

        // --- Scenario 3: Back-to-Back Operations ---
        $display("\n--- SCENARIO 3: Back-to-Back Test (10 cycles) ---");
        busy_signal = 0;
        start_pulse = 1;
        #CLK_PERIOD;
        start_pulse = 0;
        repeat (9) #CLK_PERIOD;
        done_pulse = 1;
        #CLK_PERIOD;
        done_pulse = 0;
        #CLK_PERIOD;

        // Check results
        if (total_cycles_count == 10 && active_cycles_count == 0 && idle_cycles_count == 10) begin
            $display("  [PASS] Scenario 3 Correct!");
        end else begin
            $error("  [FAIL] Scenario 3 Incorrect!");
            $display("    Expected: total=10, active=0, idle=10");
            $display("    Got:      total=%0d, active=%0d, idle=%0d", total_cycles_count, active_cycles_count, idle_cycles_count);
        end
        #CLK_PERIOD;

        // --- Scenario 4: Metadata Cache Counters ---
        $display("\n--- SCENARIO 4: Metadata Cache Performance (20 cycles) ---");
        busy_signal = 1;
        meta_cache_hits = 150;
        meta_cache_misses = 10;
        meta_decode_cycles = 50;
        start_pulse = 1;
        #CLK_PERIOD;
        start_pulse = 0;
        repeat (19) #CLK_PERIOD;
        done_pulse = 1;
        #CLK_PERIOD;
        done_pulse = 0;
        #CLK_PERIOD;

        // Check results
        if (cache_hit_count == 150 && cache_miss_count == 10 && decode_count == 50) begin
            $display("  [PASS] Scenario 4 Metadata Cache Counters Correct!");
        end else begin
            $error("  [FAIL] Scenario 4 Incorrect!");
            $display("    Expected: hits=150, misses=10, decode=50");
            $display("    Got:      hits=%0d, misses=%0d, decode=%0d", cache_hit_count, cache_miss_count, decode_count);
        end
        #CLK_PERIOD;

        $display("\n==================================================");
        $display("Testbench Finished.");
        $display("==================================================");
        $finish;
    end

endmodule
`default_nettype wire