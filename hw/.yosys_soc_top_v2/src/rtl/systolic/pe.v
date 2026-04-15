`default_nettype none
module pe (
	clk,
	rst_n,
	clk_en,
	a_in,
	b_in,
	en,
	clr,
	load_weight,
	a_out,
	load_weight_out,
	acc
);
	reg _sv2v_0;
	parameter PIPE = 1;
	input wire clk;
	input wire rst_n;
	input wire clk_en;
	input wire signed [7:0] a_in;
	input wire signed [7:0] b_in;
	input wire en;
	input wire clr;
	input wire load_weight;
	output reg signed [7:0] a_out;
	output reg load_weight_out;
	output wire signed [31:0] acc;
	reg signed [7:0] weight_reg;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			weight_reg <= 8'sd0;
			load_weight_out <= 1'b0;
		end
		else if (clk_en) begin
			load_weight_out <= load_weight;
			if (load_weight)
				weight_reg <= b_in;
		end
	generate
		if (PIPE) begin : gen_pipe
			always @(posedge clk or negedge rst_n)
				if (!rst_n)
					a_out <= 8'sd0;
				else if (clk_en)
					a_out <= a_in;
		end
		else begin : gen_comb
			always @(*) begin
				if (_sv2v_0)
					;
				a_out = a_in;
			end
		end
	endgenerate
	mac8 u_mac(
		.clk(clk),
		.rst_n(rst_n),
		.clk_en(clk_en),
		.a(a_in),
		.b(weight_reg),
		.en(en),
		.clr(clr),
		.acc(acc)
	);
	initial _sv2v_0 = 0;
endmodule
`default_nettype wire
