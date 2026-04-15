`default_nettype none
module uart_rx (
	clk,
	rst_n,
	baud_div,
	rx_i,
	rx_data,
	rx_valid,
	rx_error
);
	parameter CLK_FREQ = 50000000;
	parameter DEFAULT_BAUD = 115200;
	input wire clk;
	input wire rst_n;
	input wire [15:0] baud_div;
	input wire rx_i;
	output reg [7:0] rx_data;
	output reg rx_valid;
	output reg rx_error;
	localparam [15:0] DEFAULT_DIV = CLK_FREQ / DEFAULT_BAUD;
	reg [1:0] state;
	reg rx_sync1;
	reg rx_sync2;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			rx_sync1 <= 1'b1;
			rx_sync2 <= 1'b1;
		end
		else begin
			rx_sync1 <= rx_i;
			rx_sync2 <= rx_sync1;
		end
	wire rx_pin = rx_sync2;
	wire [15:0] divisor = (baud_div != 16'd0 ? baud_div : DEFAULT_DIV);
	wire [15:0] oversample_div = {4'b0000, divisor[15:4]};
	reg [15:0] sample_cnt;
	reg [3:0] sample_idx;
	reg [2:0] bit_idx;
	reg [7:0] shift_reg;
	reg [2:0] vote_bits;
	wire voted_bit = ((vote_bits[0] + vote_bits[1]) + vote_bits[2]) >= 2'd2;
	wire sample_tick = sample_cnt == 16'd0;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			rx_data <= 8'd0;
			rx_valid <= 1'b0;
			rx_error <= 1'b0;
			sample_cnt <= 16'd0;
			sample_idx <= 4'd0;
			bit_idx <= 3'd0;
			shift_reg <= 8'd0;
			vote_bits <= 3'd0;
		end
		else begin
			rx_valid <= 1'b0;
			rx_error <= 1'b0;
			if (sample_tick && (state != 2'd0))
				sample_cnt <= oversample_div;
			else if (!sample_tick && (state != 2'd0))
				sample_cnt <= sample_cnt - 16'd1;
			case (state)
				2'd0:
					if (rx_pin == 1'b0) begin
						state <= 2'd1;
						sample_cnt <= oversample_div;
						sample_idx <= 4'd0;
						vote_bits <= 3'd0;
					end
				2'd1:
					if (sample_tick) begin
						if (((sample_idx == 4'd7) || (sample_idx == 4'd8)) || (sample_idx == 4'd9))
							vote_bits <= {vote_bits[1:0], rx_pin};
						if (sample_idx == 4'd15) begin
							if (!voted_bit) begin
								state <= 2'd2;
								bit_idx <= 3'd0;
								sample_idx <= 4'd0;
								vote_bits <= 3'd0;
							end
							else
								state <= 2'd0;
						end
						else
							sample_idx <= sample_idx + 4'd1;
						sample_cnt <= oversample_div;
					end
				2'd2:
					if (sample_tick) begin
						if (((sample_idx == 4'd7) || (sample_idx == 4'd8)) || (sample_idx == 4'd9))
							vote_bits <= {vote_bits[1:0], rx_pin};
						if (sample_idx == 4'd15) begin
							shift_reg <= {voted_bit, shift_reg[7:1]};
							vote_bits <= 3'd0;
							sample_idx <= 4'd0;
							if (bit_idx == 3'd7)
								state <= 2'd3;
							else
								bit_idx <= bit_idx + 3'd1;
						end
						else
							sample_idx <= sample_idx + 4'd1;
						sample_cnt <= oversample_div;
					end
				2'd3:
					if (sample_tick) begin
						if (((sample_idx == 4'd7) || (sample_idx == 4'd8)) || (sample_idx == 4'd9))
							vote_bits <= {vote_bits[1:0], rx_pin};
						if (sample_idx == 4'd15) begin
							if (voted_bit) begin
								rx_data <= shift_reg;
								rx_valid <= 1'b1;
							end
							else
								rx_error <= 1'b1;
							state <= 2'd0;
						end
						else
							sample_idx <= sample_idx + 4'd1;
						sample_cnt <= oversample_div;
					end
			endcase
		end
endmodule
`default_nettype wire
