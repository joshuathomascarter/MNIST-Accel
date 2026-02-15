// perf.sv — Non-intrusive Performance Monitor for MNIST BSR Accelerator
// Counts total/active/idle cycles, DMA bytes, blocks processed, stall cycles.
// 2-state FSM: IDLE → MEASURING → latch on done.

`timescale 1ns/1ps
`default_nettype none

module perf #(
    parameter COUNTER_WIDTH = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Control
    input  wire                      start_pulse,       // Begin measurement
    input  wire                      done_pulse,        // End measurement
    input  wire                      busy_signal,       // Active computation

    // Real-time inputs for hardware counters
    input  wire                      pe_en_signal,      // PE active this cycle
    input  wire                      sched_busy_signal,  // Scheduler busy this cycle
    input  wire                      dma_beat_valid,     // AXI beat accepted (rvalid & rready)
    input  wire                      block_done_pulse,   // BSR block completion pulse

    // Outputs (to CSR)
    output reg  [COUNTER_WIDTH-1:0]  total_cycles_count,
    output reg  [COUNTER_WIDTH-1:0]  active_cycles_count,
    output reg  [COUNTER_WIDTH-1:0]  idle_cycles_count,
    output reg  [31:0]               dma_bytes_count,
    output reg  [31:0]               blocks_processed_count,
    output reg  [31:0]               stall_cycles_count,
    output reg                       measurement_done
);

    localparam S_IDLE      = 1'b0;
    localparam S_MEASURING = 1'b1;

    reg state_reg;
    reg prev_state;

    // Running counters (cleared on start, latched to outputs on done)
    reg [COUNTER_WIDTH-1:0] run_total;
    reg [COUNTER_WIDTH-1:0] run_active;
    reg [COUNTER_WIDTH-1:0] run_idle;
    reg [31:0]              run_dma_bytes;
    reg [31:0]              run_blocks;
    reg [31:0]              run_stalls;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg              <= S_IDLE;
            prev_state             <= S_IDLE;
            run_total              <= {COUNTER_WIDTH{1'b0}};
            run_active             <= {COUNTER_WIDTH{1'b0}};
            run_idle               <= {COUNTER_WIDTH{1'b0}};
            run_dma_bytes          <= 32'd0;
            run_blocks             <= 32'd0;
            run_stalls             <= 32'd0;
            total_cycles_count     <= {COUNTER_WIDTH{1'b0}};
            active_cycles_count    <= {COUNTER_WIDTH{1'b0}};
            idle_cycles_count      <= {COUNTER_WIDTH{1'b0}};
            dma_bytes_count        <= 32'd0;
            blocks_processed_count <= 32'd0;
            stall_cycles_count     <= 32'd0;
            measurement_done       <= 1'b0;
        end else begin
            prev_state <= state_reg;
            measurement_done <= 1'b0;

            case (state_reg)
                S_IDLE: begin
                    if (start_pulse) begin
                        state_reg     <= S_MEASURING;
                        run_total     <= {COUNTER_WIDTH{1'b0}};
                        run_active    <= {COUNTER_WIDTH{1'b0}};
                        run_idle      <= {COUNTER_WIDTH{1'b0}};
                        run_dma_bytes <= 32'd0;
                        run_blocks    <= 32'd0;
                        run_stalls    <= 32'd0;
                    end
                end

                S_MEASURING: begin
                    // Total cycles
                    run_total <= run_total + 1;

                    // Active vs idle (based on top-level busy)
                    if (busy_signal)
                        run_active <= run_active + 1;
                    else
                        run_idle <= run_idle + 1;

                    // DMA bytes: each accepted AXI beat = 8 bytes (64-bit data bus)
                    if (dma_beat_valid)
                        run_dma_bytes <= run_dma_bytes + 32'd8;

                    // Blocks processed: count block completion pulses
                    if (block_done_pulse)
                        run_blocks <= run_blocks + 1;

                    // Stall detection: scheduler is busy but PEs are idle
                    // This indicates metadata fetch latency or BRAM read stalls
                    if (sched_busy_signal && !pe_en_signal)
                        run_stalls <= run_stalls + 1;

                    if (done_pulse)
                        state_reg <= S_IDLE;
                end

                default: state_reg <= S_IDLE;
            endcase

            // Latch outputs on transition from MEASURING → IDLE
            if (prev_state == S_MEASURING && state_reg == S_IDLE) begin
                total_cycles_count     <= run_total;
                active_cycles_count    <= run_active;
                idle_cycles_count      <= run_idle;
                dma_bytes_count        <= run_dma_bytes;
                blocks_processed_count <= run_blocks;
                stall_cycles_count     <= run_stalls;
                measurement_done       <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
