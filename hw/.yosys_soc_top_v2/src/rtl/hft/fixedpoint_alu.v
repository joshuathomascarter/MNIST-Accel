module fixedpoint_alu (
	clk,
	rst_n,
	a,
	b,
	op,
	valid_in,
	valid_out,
	result,
	overflow
);
	reg _sv2v_0;
	parameter signed [31:0] WIDTH = 32;
	parameter signed [31:0] FRAC = 16;
	parameter [0:0] REG_OUT = 1'b1;
	input wire clk;
	input wire rst_n;
	input wire [WIDTH - 1:0] a;
	input wire [WIDTH - 1:0] b;
	input wire [2:0] op;
	input wire valid_in;
	output reg valid_out;
	output reg [WIDTH - 1:0] result;
	output reg overflow;
	wire signed [WIDTH - 1:0] sa;
	wire signed [WIDTH - 1:0] sb;
	(* use_dsp = "yes" *) reg signed [(2 * WIDTH) - 1:0] mul_full;
	reg [WIDTH - 1:0] res_comb;
	reg ovf_comb;
	reg valid_comb;
	assign sa = $signed(a);
	assign sb = $signed(b);
	always @(*) begin
		if (_sv2v_0)
			;
		res_comb = '0;
		ovf_comb = 1'b0;
		valid_comb = valid_in;
		mul_full = sa * sb;
		case (op)
			3'b000: begin : sv2v_autoblock_1
				reg signed [WIDTH:0] sum;
				sum = {sa[WIDTH - 1], sa} + {sb[WIDTH - 1], sb};
				res_comb = sum[WIDTH - 1:0];
				ovf_comb = sum[WIDTH] != sum[WIDTH - 1];
			end
			3'b001: begin : sv2v_autoblock_2
				reg signed [WIDTH:0] diff;
				diff = {sa[WIDTH - 1], sa} - {sb[WIDTH - 1], sb};
				res_comb = diff[WIDTH - 1:0];
				ovf_comb = diff[WIDTH] != diff[WIDTH - 1];
			end
			3'b010: begin
				res_comb = mul_full[(WIDTH + FRAC) - 1:FRAC];
				ovf_comb = mul_full[(2 * WIDTH) - 1:WIDTH + FRAC] != {WIDTH - FRAC {mul_full[(WIDTH + FRAC) - 1]}};
			end
			3'b011: res_comb = (sa < 0 ? ~a + 1 : a);
			3'b100: res_comb = {{WIDTH - 2 {1'b0}}, (sa > sb ? 1'b1 : 1'b0), (sa == sb ? 1'b1 : 1'b0)};
			default: res_comb = '0;
		endcase
	end
	generate
		if (REG_OUT) begin : gen_reg
			always @(posedge clk or negedge rst_n)
				if (!rst_n) begin
					result <= '0;
					overflow <= 1'b0;
					valid_out <= 1'b0;
				end
				else begin
					result <= res_comb;
					overflow <= ovf_comb;
					valid_out <= valid_comb;
				end
		end
		else begin : gen_comb
			wire [WIDTH:1] sv2v_tmp_A8A1B;
			assign sv2v_tmp_A8A1B = res_comb;
			always @(*) result = sv2v_tmp_A8A1B;
			wire [1:1] sv2v_tmp_9E26F;
			assign sv2v_tmp_9E26F = ovf_comb;
			always @(*) overflow = sv2v_tmp_9E26F;
			wire [1:1] sv2v_tmp_7C0C9;
			assign sv2v_tmp_7C0C9 = valid_comb;
			always @(*) valid_out = sv2v_tmp_7C0C9;
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
