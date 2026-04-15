`default_nettype none
module systolic_array_sparse (
	clk,
	rst_n,
	clk_en,
	block_valid,
	load_weight,
	clr,
	a_in_flat,
	b_in_flat,
	c_out_flat
);
	parameter N_ROWS = 16;
	parameter N_COLS = 16;
	parameter DATA_W = 8;
	parameter ACC_W = 32;
	input wire clk;
	input wire rst_n;
	input wire clk_en;
	input wire block_valid;
	input wire load_weight;
	input wire clr;
	input wire [(N_ROWS * DATA_W) - 1:0] a_in_flat;
	input wire [(N_COLS * DATA_W) - 1:0] b_in_flat;
	output wire [((N_ROWS * N_COLS) * ACC_W) - 1:0] c_out_flat;
	wire signed [DATA_W - 1:0] a_in_raw [0:N_ROWS - 1];
	wire signed [DATA_W - 1:0] b_in [0:N_COLS - 1];
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < N_ROWS; _gv_i_1 = _gv_i_1 + 1) begin : genblk1
			localparam i = _gv_i_1;
			assign a_in_raw[i] = a_in_flat[i * DATA_W+:DATA_W];
		end
		for (_gv_i_1 = 0; _gv_i_1 < N_COLS; _gv_i_1 = _gv_i_1 + 1) begin : genblk2
			localparam i = _gv_i_1;
			assign b_in[i] = b_in_flat[i * DATA_W+:DATA_W];
		end
	endgenerate
	wire signed [DATA_W - 1:0] a_in [0:N_ROWS - 1];
	generate
		for (_gv_i_1 = 0; _gv_i_1 < N_ROWS; _gv_i_1 = _gv_i_1 + 1) begin : SKEW
			localparam i = _gv_i_1;
			if (i == 0) begin : NO_DELAY
				assign a_in[0] = a_in_raw[0];
			end
			else begin : DELAY
				reg signed [DATA_W - 1:0] skew_sr [0:i - 1];
				integer j;
				always @(posedge clk or negedge rst_n)
					if (!rst_n)
						for (j = 0; j < i; j = j + 1)
							skew_sr[j] <= 0;
					else if (clk_en) begin
						skew_sr[0] <= a_in_raw[i];
						for (j = 1; j < i; j = j + 1)
							skew_sr[j] <= skew_sr[j - 1];
					end
				assign a_in[i] = skew_sr[i - 1];
			end
		end
	endgenerate
	reg [$clog2(N_ROWS) - 1:0] load_ptr;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			load_ptr <= 0;
		else if (clk_en) begin
			if (load_weight)
				load_ptr <= load_ptr + 1;
			else
				load_ptr <= 0;
		end
	wire signed [DATA_W - 1:0] a_fwd [0:N_ROWS - 1][0:N_COLS - 1];
	genvar _gv_r_1;
	genvar _gv_c_1;
	generate
		for (_gv_r_1 = 0; _gv_r_1 < N_ROWS; _gv_r_1 = _gv_r_1 + 1) begin : ROW
			localparam r = _gv_r_1;
			for (_gv_c_1 = 0; _gv_c_1 < N_COLS; _gv_c_1 = _gv_c_1 + 1) begin : COL
				localparam c = _gv_c_1;
				wire signed [DATA_W - 1:0] a_src;
				if (c == 0) begin : gen_a_input
					assign a_src = a_in[r];
				end
				else begin : gen_a_chain
					assign a_src = a_fwd[r][c - 1];
				end
				wire signed [DATA_W - 1:0] b_src = b_in[c];
				wire load_weight_src = load_weight && (load_ptr == r[$clog2(N_ROWS) - 1:0]);
				wire _unused_lw_out;
				pe #(.PIPE(1)) u_pe(
					.clk(clk),
					.rst_n(rst_n),
					.clk_en(clk_en),
					.en(block_valid),
					.clr(clr),
					.load_weight(load_weight_src),
					.a_in(a_src),
					.b_in(b_src),
					.a_out(a_fwd[r][c]),
					.load_weight_out(_unused_lw_out),
					.acc(c_out_flat[((r * N_COLS) + c) * ACC_W+:ACC_W])
				);
			end
		end
	endgenerate
endmodule
`default_nettype wire
