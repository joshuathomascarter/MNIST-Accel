`default_nettype none
module mac8 (
	clk,
	rst_n,
	clk_en,
	a,
	b,
	en,
	clr,
	acc
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire clk_en;
	input wire signed [7:0] a;
	input wire signed [7:0] b;
	input wire en;
	input wire clr;
	output wire signed [31:0] acc;
	(* use_dsp = "yes" *) reg signed [15:0] prod;
	reg signed [31:0] sum_comb;
	reg signed [31:0] acc_reg;
	always @(*) begin
		if (_sv2v_0)
			;
		prod = a * b;
	end
	always @(*) begin
		if (_sv2v_0)
			;
		sum_comb = acc_reg + {{16 {prod[15]}}, prod};
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			acc_reg <= 32'sd0;
		else if (clk_en) begin
			if (clr)
				acc_reg <= 32'sd0;
			else if (en)
				acc_reg <= sum_comb;
		end
	assign acc = acc_reg;
	initial _sv2v_0 = 0;
endmodule
