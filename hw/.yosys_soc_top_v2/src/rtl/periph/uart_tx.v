`default_nettype none
module uart_tx (
	clk,
	rst_n,
	baud_div,
	tx_data,
	tx_valid,
	tx_ready,
	tx_o
);
	parameter CLK_FREQ = 50000000;
	parameter DEFAULT_BAUD = 115200;
	input wire clk;
	input wire rst_n;
	input wire [15:0] baud_div;
	input wire [7:0] tx_data;
	input wire tx_valid;
	output reg tx_ready;
	output reg tx_o;
	localparam [15:0] DEFAULT_DIV = CLK_FREQ / DEFAULT_BAUD;
	reg [1:0] state;
	reg [15:0] baud_cnt;
	reg [2:0] bit_idx;
	reg [7:0] shift_reg;
	wire [15:0] divisor = (baud_div != 16'd0 ? baud_div : DEFAULT_DIV);
	wire baud_tick = baud_cnt == 16'd0;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			tx_o <= 1'b1;
			tx_ready <= 1'b1;
			baud_cnt <= 16'd0;
			bit_idx <= 3'd0;
			shift_reg <= 8'd0;
		end
		else begin
			if (baud_tick && (state != 2'd0))
				baud_cnt <= divisor - 16'd1;
			else if (!baud_tick)
				baud_cnt <= baud_cnt - 16'd1;
			case (state)
				2'd0: begin
					tx_o <= 1'b1;
					tx_ready <= 1'b1;
					if (tx_valid) begin
						shift_reg <= tx_data;
						state <= 2'd1;
						tx_ready <= 1'b0;
						baud_cnt <= divisor - 16'd1;
					end
				end
				2'd1: begin
					tx_o <= 1'b0;
					if (baud_tick) begin
						state <= 2'd2;
						bit_idx <= 3'd0;
						baud_cnt <= divisor - 16'd1;
					end
				end
				2'd2: begin
					tx_o <= shift_reg[0];
					if (baud_tick) begin
						shift_reg <= {1'b0, shift_reg[7:1]};
						if (bit_idx == 3'd7)
							state <= 2'd3;
						else
							bit_idx <= bit_idx + 3'd1;
						baud_cnt <= divisor - 16'd1;
					end
				end
				2'd3: begin
					tx_o <= 1'b1;
					if (baud_tick)
						state <= 2'd0;
				end
			endcase
		end
endmodule
`default_nettype wire
