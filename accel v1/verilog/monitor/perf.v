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
    input  wire clk,
    input  wire rst_n,

    // Control Inputs (from accelerator core)
    input  wire start_pulse,       // Single-cycle pulse to start measurement
    input  wire done_pulse,        // Single-cycle pulse to stop measurement
    input  wire busy_signal,       // High when the core is doing useful work

    // Status Outputs (to be mapped to CSRs)
    output reg [COUNTER_WIDTH-1:0] total_cycles_count,  // Total cycles from start to done
    output reg [COUNTER_WIDTH-1:0] active_cycles_count, // Cycles where busy_signal was high
    output reg [COUNTER_WIDTH-1:0] idle_cycles_count,   // Cycles where busy_signal was low
    output reg                     measurement_done     // Single-cycle pulse when measurement is complete
);

    // FSM State Definitions
    localparam S_IDLE      = 1'b0;
    localparam S_MEASURING = 1'b1;

    reg state_reg, state_next;

    // Internal counters
    reg [COUNTER_WIDTH-1:0] total_counter;
    reg [COUNTER_WIDTH-1:0] active_counter;
    reg [COUNTER_WIDTH-1:0] idle_counter;

    // FSM State Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= S_IDLE;
        end else begin
            state_reg <= state_next;
        end
    end

    // FSM Next State Logic
    always @(*) begin
        state_next = state_reg;
        case (state_reg)
            S_IDLE: begin
                if (start_pulse) begin
                    state_next = S_MEASURING;
                end
            end
            S_MEASURING: begin
                if (done_pulse) begin
                    state_next = S_IDLE;
                end
            end
        endcase
    end

    // Counter Logic - Manages the internal counters during measurement
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_counter  <= {COUNTER_WIDTH{1'b0}};
            active_counter <= {COUNTER_WIDTH{1'b0}};
            idle_counter   <= {COUNTER_WIDTH{1'b0}};
        end else begin
            // Reset counters on the first cycle of measurement
            if (state_reg == S_IDLE && state_next == S_MEASURING) begin
                total_counter  <= {COUNTER_WIDTH{1'b0}};
                active_counter <= {COUNTER_WIDTH{1'b0}};
                idle_counter   <= {COUNTER_WIDTH{1'b0}};
            end
            // Increment counters while measuring
            else if (state_reg == S_MEASURING) begin
                total_counter <= total_counter + 1;
                if (busy_signal) begin
                    active_counter <= active_counter + 1;
                end else begin
                    idle_counter <= idle_counter + 1;
                end
            end
        end
    end

    // Output Logic - Latches the final values to the output ports
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_cycles_count  <= {COUNTER_WIDTH{1'b0}};
            active_cycles_count <= {COUNTER_WIDTH{1'b0}};
            idle_cycles_count   <= {COUNTER_WIDTH{1'b0}};
            measurement_done    <= 1'b0;
        end else begin
            // Latch final counts and pulse 'measurement_done' when operation finishes
            if (state_reg == S_MEASURING && done_pulse) begin
                total_cycles_count  <= total_counter + 1; // Add 1 to include the final cycle
                if (busy_signal) begin
                    active_cycles_count <= active_counter + 1;
                    idle_cycles_count   <= idle_counter;
                end else begin
                    active_cycles_count <= active_counter;
                    idle_cycles_count   <= idle_counter + 1;
                end
                measurement_done <= 1'b1;
            end else begin
                measurement_done <= 1'b0;
            end
        end
    end

endmodule
`default_nettype wire