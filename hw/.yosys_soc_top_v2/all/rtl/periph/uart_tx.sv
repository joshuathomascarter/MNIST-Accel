// uart_tx.sv — UART Transmitter Shift Register
// =============================================================================
//
// Generates serial TX output from parallel byte input.
// Baud rate set by clock divisor: baud_period = CLK_FREQ / BAUD_RATE.
//
// Frame format: idle (high) → start (low) → 8 data (LSB first) → stop (high)
//
// Resource Usage: ~30 LUTs, 0 BRAM, 0 DSPs
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module uart_tx #(
    parameter CLK_FREQ     = 50_000_000,
    parameter DEFAULT_BAUD = 115_200
)(
    input  wire        clk,
    input  wire        rst_n,

    // Baud configuration
    input  wire [15:0] baud_div,     // Clock cycles per bit (0 = use default)

    // Data interface
    input  wire [7:0]  tx_data,      // Byte to transmit
    input  wire        tx_valid,     // Pulse high to start transmission
    output reg         tx_ready,     // High when idle (can accept new data)

    // Serial output
    output reg         tx_o          // UART TX pin
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

    reg [15:0] baud_cnt;        // Baud rate counter
    reg [2:0]  bit_idx;         // Current data bit index (0-7)
    reg [7:0]  shift_reg;       // TX shift register

    wire [15:0] divisor = (baud_div != 16'd0) ? baud_div : DEFAULT_DIV;

    // Baud tick: count down from divisor, tick when reaches 0
    wire baud_tick = (baud_cnt == 16'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            tx_o      <= 1'b1;       // Idle high
            tx_ready  <= 1'b1;
            baud_cnt  <= 16'd0;
            bit_idx   <= 3'd0;
            shift_reg <= 8'd0;
        end else begin
            // Baud counter
            if (baud_tick && state != S_IDLE)
                baud_cnt <= divisor - 16'd1;
            else if (!baud_tick)
                baud_cnt <= baud_cnt - 16'd1;

            case (state)
                S_IDLE: begin
                    tx_o     <= 1'b1;
                    tx_ready <= 1'b1;
                    if (tx_valid) begin
                        shift_reg <= tx_data;
                        state     <= S_START;
                        tx_ready  <= 1'b0;
                        baud_cnt  <= divisor - 16'd1;
                    end
                end

                S_START: begin
                    tx_o <= 1'b0;  // Start bit
                    if (baud_tick) begin
                        state   <= S_DATA;
                        bit_idx <= 3'd0;
                        baud_cnt <= divisor - 16'd1;
                    end
                end

                S_DATA: begin
                    tx_o <= shift_reg[0];  // LSB first
                    if (baud_tick) begin
                        shift_reg <= {1'b0, shift_reg[7:1]};
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                        baud_cnt <= divisor - 16'd1;
                    end
                end

                S_STOP: begin
                    tx_o <= 1'b1;  // Stop bit
                    if (baud_tick) begin
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
