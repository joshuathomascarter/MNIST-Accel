// uart_rx.sv — UART Receiver with 16× Oversampling
// =============================================================================
//
// Receives serial data on RX pin, outputs parallel byte.
// 16× oversampling for mid-bit sampling with 3-sample majority vote
// for noise rejection.
//
// Frame format: idle (high) → start (low) → 8 data (LSB first) → stop (high)
//
// Resource Usage: ~40 LUTs, 0 BRAM, 0 DSPs
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module uart_rx #(
    parameter CLK_FREQ     = 50_000_000,
    parameter DEFAULT_BAUD = 115_200
)(
    input  wire        clk,
    input  wire        rst_n,

    // Baud configuration
    input  wire [15:0] baud_div,     // Clock cycles per bit (0 = use default)

    // Serial input
    input  wire        rx_i,         // UART RX pin

    // Data interface
    output reg  [7:0]  rx_data,      // Received byte
    output reg         rx_valid,     // Pulse: new byte available
    output reg         rx_error      // Framing error (bad stop bit)
);

    localparam [15:0] DEFAULT_DIV = CLK_FREQ / DEFAULT_BAUD;

    // FSM states
    typedef enum logic [1:0] {
        S_IDLE,
        S_START,
        S_DATA,
        S_STOP
    } state_t;

    state_t state;

    // 2-FF synchronizer for metastability
    reg rx_sync1, rx_sync2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync1 <= 1'b1;
            rx_sync2 <= 1'b1;
        end else begin
            rx_sync1 <= rx_i;
            rx_sync2 <= rx_sync1;
        end
    end

    wire rx_pin = rx_sync2;

    // 16× oversampling counter
    wire [15:0] divisor = (baud_div != 16'd0) ? baud_div : DEFAULT_DIV;
    wire [15:0] oversample_div = {4'b0000, divisor[15:4]};  // divisor / 16

    reg [15:0] sample_cnt;
    reg [3:0]  sample_idx;      // 0-15 within a bit period
    reg [2:0]  bit_idx;         // Data bit index 0-7
    reg [7:0]  shift_reg;

    // 3-sample majority vote (samples at positions 7, 8, 9)
    reg [2:0]  vote_bits;
    wire       voted_bit = (vote_bits[0] + vote_bits[1] + vote_bits[2]) >= 2'd2;

    wire sample_tick = (sample_cnt == 16'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            rx_data    <= 8'd0;
            rx_valid   <= 1'b0;
            rx_error   <= 1'b0;
            sample_cnt <= 16'd0;
            sample_idx <= 4'd0;
            bit_idx    <= 3'd0;
            shift_reg  <= 8'd0;
            vote_bits  <= 3'd0;
        end else begin
            rx_valid <= 1'b0;
            rx_error <= 1'b0;

            // Oversample counter
            if (sample_tick && state != S_IDLE)
                sample_cnt <= oversample_div;
            else if (!sample_tick && state != S_IDLE)
                sample_cnt <= sample_cnt - 16'd1;

            case (state)
                S_IDLE: begin
                    if (rx_pin == 1'b0) begin
                        // Falling edge detected — possible start bit
                        state      <= S_START;
                        sample_cnt <= oversample_div;
                        sample_idx <= 4'd0;
                        vote_bits  <= 3'd0;
                    end
                end

                S_START: begin
                    if (sample_tick) begin
                        // Collect votes at mid-bit (samples 7, 8, 9)
                        if (sample_idx == 4'd7 || sample_idx == 4'd8 || sample_idx == 4'd9)
                            vote_bits <= {vote_bits[1:0], rx_pin};

                        if (sample_idx == 4'd15) begin
                            // End of start bit period — verify it's still low
                            if (!voted_bit) begin
                                // Valid start bit
                                state      <= S_DATA;
                                bit_idx    <= 3'd0;
                                sample_idx <= 4'd0;
                                vote_bits  <= 3'd0;
                            end else begin
                                // False start — go back to idle
                                state <= S_IDLE;
                            end
                        end else begin
                            sample_idx <= sample_idx + 4'd1;
                        end
                        sample_cnt <= oversample_div;
                    end
                end

                S_DATA: begin
                    if (sample_tick) begin
                        if (sample_idx == 4'd7 || sample_idx == 4'd8 || sample_idx == 4'd9)
                            vote_bits <= {vote_bits[1:0], rx_pin};

                        if (sample_idx == 4'd15) begin
                            // Latch voted bit
                            shift_reg <= {voted_bit, shift_reg[7:1]};
                            vote_bits <= 3'd0;
                            sample_idx <= 4'd0;

                            if (bit_idx == 3'd7) begin
                                state <= S_STOP;
                            end else begin
                                bit_idx <= bit_idx + 3'd1;
                            end
                        end else begin
                            sample_idx <= sample_idx + 4'd1;
                        end
                        sample_cnt <= oversample_div;
                    end
                end

                S_STOP: begin
                    if (sample_tick) begin
                        if (sample_idx == 4'd7 || sample_idx == 4'd8 || sample_idx == 4'd9)
                            vote_bits <= {vote_bits[1:0], rx_pin};

                        if (sample_idx == 4'd15) begin
                            if (voted_bit) begin
                                // Valid stop bit
                                rx_data  <= shift_reg;
                                rx_valid <= 1'b1;
                            end else begin
                                // Framing error — stop bit was low
                                rx_error <= 1'b1;
                            end
                            state <= S_IDLE;
                        end else begin
                            sample_idx <= sample_idx + 4'd1;
                        end
                        sample_cnt <= oversample_div;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
