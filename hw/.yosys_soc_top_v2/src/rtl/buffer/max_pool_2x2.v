`default_nettype none
module max_pool_2x2 (
	clk,
	rst_n,
	bypass,
	feat_h,
	feat_w,
	in_valid,
	in_data,
	in_ready,
	out_valid,
	out_data,
	out_ready,
	pool_active
);
	parameter DATA_W = 8;
	parameter MAX_W = 128;
	parameter PACK_W = 64;
	input wire clk;
	input wire rst_n;
	input wire bypass;
	input wire [15:0] feat_h;
	input wire [15:0] feat_w;
	input wire in_valid;
	input wire [PACK_W - 1:0] in_data;
	output wire in_ready;
	output reg out_valid;
	output reg [PACK_W - 1:0] out_data;
	input wire out_ready;
	output wire pool_active;
	wire bypass_valid = bypass & in_valid;
	wire bypass_ready = bypass & out_ready;
	wire pool_ready_int;
	assign in_ready = (bypass ? out_ready : pool_ready_int);
	reg [15:0] col_cnt;
	reg [15:0] row_cnt;
	assign pool_active = !bypass && ((row_cnt != 0) || (col_cnt != 0));
	localparam LINE_BUF_DEPTH = MAX_W / (PACK_W / DATA_W);
	localparam LB_ADDR_W = $clog2(LINE_BUF_DEPTH);
	reg [PACK_W - 1:0] line_buf [0:LINE_BUF_DEPTH - 1];
	reg [LB_ADDR_W - 1:0] lb_wr_addr;
	reg [LB_ADDR_W - 1:0] lb_rd_addr;
	wire [15:0] words_per_row = {3'b000, feat_w[15:3]};
	reg pool_out_valid_r;
	wire is_even_row = ~row_cnt[0];
	wire is_odd_row = row_cnt[0];
	assign pool_ready_int = !pool_out_valid_r || out_ready;
	function automatic [DATA_W - 1:0] smax;
		input [DATA_W - 1:0] a;
		input [DATA_W - 1:0] b;
		smax = ($signed(a) > $signed(b) ? a : b);
	endfunction
	reg [PACK_W - 1:0] line_buf_data;
	reg [PACK_W - 1:0] pool_result;
	integer i;
	reg [DATA_W - 1:0] cur_pair_max [0:3];
	reg [DATA_W - 1:0] buf_pair_max [0:3];
	reg [DATA_W - 1:0] final_max [0:3];
	always @(*) begin
		for (i = 0; i < 4; i = i + 1)
			begin
				cur_pair_max[i] = smax(in_data[(2 * i) * DATA_W+:DATA_W], in_data[((2 * i) + 1) * DATA_W+:DATA_W]);
				buf_pair_max[i] = smax(line_buf_data[(2 * i) * DATA_W+:DATA_W], line_buf_data[((2 * i) + 1) * DATA_W+:DATA_W]);
				final_max[i] = smax(cur_pair_max[i], buf_pair_max[i]);
			end
		pool_result = {{PACK_W - (4 * DATA_W) {1'b0}}, final_max[3], final_max[2], final_max[1], final_max[0]};
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			row_cnt <= 16'd0;
			col_cnt <= 16'd0;
			lb_wr_addr <= {LB_ADDR_W {1'b0}};
			lb_rd_addr <= {LB_ADDR_W {1'b0}};
			line_buf_data <= {PACK_W {1'b0}};
			out_valid <= 1'b0;
			out_data <= {PACK_W {1'b0}};
			pool_out_valid_r <= 1'b0;
		end
		else if (bypass) begin
			out_valid <= in_valid;
			out_data <= in_data;
			pool_out_valid_r <= 1'b0;
			row_cnt <= 16'd0;
			col_cnt <= 16'd0;
		end
		else begin
			if (out_ready) begin
				out_valid <= 1'b0;
				pool_out_valid_r <= 1'b0;
			end
			if (in_valid && pool_ready_int) begin
				if (is_even_row) begin
					line_buf[lb_wr_addr] <= in_data;
					lb_wr_addr <= lb_wr_addr + {{LB_ADDR_W - 1 {1'b0}}, 1'b1};
				end
				else begin
					line_buf_data <= line_buf[lb_rd_addr];
					lb_rd_addr <= lb_rd_addr + {{LB_ADDR_W - 1 {1'b0}}, 1'b1};
					out_valid <= 1'b1;
					out_data <= pool_result;
					pool_out_valid_r <= 1'b1;
				end
				if (col_cnt == (words_per_row - 16'd1)) begin
					col_cnt <= 16'd0;
					lb_wr_addr <= {LB_ADDR_W {1'b0}};
					lb_rd_addr <= {LB_ADDR_W {1'b0}};
					if (row_cnt == (feat_h - 16'd1))
						row_cnt <= 16'd0;
					else
						row_cnt <= row_cnt + 16'd1;
				end
				else
					col_cnt <= col_cnt + 16'd1;
			end
		end
endmodule
`default_nettype wire
