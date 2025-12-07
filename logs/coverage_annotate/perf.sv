//      // verilator_coverage annotation
        // rtl/monitors/perf.v
        //
        // ACCEL-v1 Performance Monitor
        //
        // Description:
        // A non-intrusive, synthesizable performance monitor that measures execution time
        // and utilization of a hardware accelerator. It operates by observing control
        // signals (start, done, busy) and counting clock cycles.
        //
        // FSM States:
        //  - S_IDLE:      Waiting for a 'start_pulse' to begin measurement.
        //  - S_MEASURING: Actively counting total, active, and idle cycles until a
        //                 'done_pulse' is received.
        //
        // Usage:
        // Instantiate this module within a top-level design (e.g., accel_top.v) and
        // connect its inputs to the core's control signals. The counter outputs should
        // be mapped to read-only CSRs for software access.
        
        `default_nettype none
        
        module perf #(
            parameter COUNTER_WIDTH = 32
        )(
            // System Inputs
 012713     input  wire clk,
%000007     input  wire rst_n,
        
            // Control Inputs (from accelerator core)
%000005     input  wire start_pulse,       // Single-cycle pulse to start measurement
%000004     input  wire done_pulse,        // Single-cycle pulse to stop measurement
%000006     input  wire busy_signal,       // High when the core is doing useful work
        
            // Metadata Cache Inputs (from meta_decode)
%000000     input  wire [COUNTER_WIDTH-1:0] meta_cache_hits,    // Cache hit count from meta_decode
%000000     input  wire [COUNTER_WIDTH-1:0] meta_cache_misses,  // Cache miss count from meta_decode
%000000     input  wire [COUNTER_WIDTH-1:0] meta_decode_cycles, // Decode cycle count from meta_decode
        
            // Status Outputs (to be mapped to CSRs)
%000003     output reg [COUNTER_WIDTH-1:0] total_cycles_count,  // Total cycles from start to done
%000004     output reg [COUNTER_WIDTH-1:0] active_cycles_count, // Cycles where busy_signal was high
%000004     output reg [COUNTER_WIDTH-1:0] idle_cycles_count,   // Cycles where busy_signal was low
%000000     output reg [COUNTER_WIDTH-1:0] cache_hit_count,     // Total metadata cache hits
%000000     output reg [COUNTER_WIDTH-1:0] cache_miss_count,    // Total metadata cache misses
%000000     output reg [COUNTER_WIDTH-1:0] decode_count,        // Total metadata decode operations
%000004     output reg                     measurement_done     // Single-cycle pulse when measurement is complete
        );
        
            // FSM State Definitions
            localparam S_IDLE      = 1'b0;
            localparam S_MEASURING = 1'b1;
        
%000004     reg state_reg, state_next;
%000004     reg prev_state;
        
            // Internal counters
~000023     reg [COUNTER_WIDTH-1:0] total_counter;
~000023     reg [COUNTER_WIDTH-1:0] active_counter;
%000004     reg [COUNTER_WIDTH-1:0] idle_counter;
        
            // State Register
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             state_reg  <= S_IDLE;
 000069             prev_state <= S_IDLE;
 012644         end else begin
 012644             prev_state <= state_reg;
 012644             state_reg  <= state_next;
                end
            end
        
            // FSM Next State Logic
 025523     always @(*) begin
 025523         state_next = state_reg;
 025523         case (state_reg)
 025423             S_IDLE: begin
~025419                 if (start_pulse) begin
%000004                     state_next = S_MEASURING;
                        end
                    end
 000100             S_MEASURING: begin
 000088                 if (done_pulse) begin
 000012                     state_next = S_IDLE;
                        end
                    end
                endcase
            end
        
            // Counter Logic - Manages the internal counters during measurement
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             total_counter  <= {COUNTER_WIDTH{1'b0}};
 000069             active_counter <= {COUNTER_WIDTH{1'b0}};
 000069             idle_counter   <= {COUNTER_WIDTH{1'b0}};
 012644         end else begin
                    // Reset counters on the first cycle of measurement
~012598             if (state_reg == S_IDLE && state_next == S_MEASURING) begin
%000004                 total_counter  <= {COUNTER_WIDTH{1'b0}};
%000004                 active_counter <= {COUNTER_WIDTH{1'b0}};
%000004                 idle_counter   <= {COUNTER_WIDTH{1'b0}};
                    end
                    // Increment counters while measuring
 012594             else if (state_reg == S_MEASURING) begin
 000046                 total_counter <= total_counter + 1;
~000042                 if (busy_signal) begin
 000042                     active_counter <= active_counter + 1;
%000004                 end else begin
%000004                     idle_counter <= idle_counter + 1;
                        end
                    end
                end
            end
        
            // Output Logic - Latches the final values to the output ports
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             total_cycles_count  <= {COUNTER_WIDTH{1'b0}};
 000069             active_cycles_count <= {COUNTER_WIDTH{1'b0}};
 000069             idle_cycles_count   <= {COUNTER_WIDTH{1'b0}};
 000069             cache_hit_count     <= {COUNTER_WIDTH{1'b0}};
 000069             cache_miss_count    <= {COUNTER_WIDTH{1'b0}};
 000069             decode_count        <= {COUNTER_WIDTH{1'b0}};
 000069             measurement_done    <= 1'b0;
 012644         end else begin
                    // Latch final counts when transitioning from MEASURING to IDLE
~012640             if (prev_state == S_MEASURING && state_reg == S_IDLE) begin
%000004                 total_cycles_count  <= total_counter;
%000004                 active_cycles_count <= active_counter;
%000004                 idle_cycles_count   <= idle_counter;
%000004                 cache_hit_count     <= meta_cache_hits;
%000004                 cache_miss_count    <= meta_cache_misses;
%000004                 decode_count        <= meta_decode_cycles;
%000004                 measurement_done    <= 1'b1;
 012640             end else begin
 012640                 measurement_done <= 1'b0;
                    end
                end
            end
        
        endmodule
        `default_nettype wire
        
