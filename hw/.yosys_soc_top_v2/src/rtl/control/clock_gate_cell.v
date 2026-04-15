`default_nettype none
module clock_gate_cell (
	clk_i,
	en_i,
	test_en_i,
	clk_o
);
	reg _sv2v_0;
	input wire clk_i;
	input wire en_i;
	input wire test_en_i;
	output wire clk_o;
	reg gate_en_latched;
	always @(*) begin
		if (_sv2v_0)
			;
		if (!clk_i)
			gate_en_latched = en_i | test_en_i;
	end
	assign clk_o = clk_i & gate_en_latched;
	initial _sv2v_0 = 0;
endmodule
`default_nettype wire
