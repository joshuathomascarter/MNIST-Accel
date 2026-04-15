module l1_lru (
	clk,
	rst_n,
	access_valid,
	access_set,
	access_way,
	query_set,
	victim_way
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_SETS = 16;
	parameter signed [31:0] NUM_WAYS = 4;
	input wire clk;
	input wire rst_n;
	input wire access_valid;
	input wire [$clog2(NUM_SETS) - 1:0] access_set;
	input wire [$clog2(NUM_WAYS) - 1:0] access_way;
	input wire [$clog2(NUM_SETS) - 1:0] query_set;
	output reg [$clog2(NUM_WAYS) - 1:0] victim_way;
	reg [2:0] lru_bits [0:NUM_SETS - 1];
	always @(*) begin
		if (_sv2v_0)
			;
		case (lru_bits[query_set])
			3'b000: victim_way = 2'd0;
			3'b001: victim_way = 2'd2;
			3'b010: victim_way = 2'd1;
			3'b011: victim_way = 2'd3;
			3'b100: victim_way = 2'd0;
			3'b101: victim_way = 2'd2;
			3'b110: victim_way = 2'd1;
			3'b111: victim_way = 2'd3;
			default: victim_way = 2'd0;
		endcase
		if (!lru_bits[query_set][2])
			victim_way = (lru_bits[query_set][1] ? 2'd1 : 2'd0);
		else
			victim_way = (lru_bits[query_set][0] ? 2'd3 : 2'd2);
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_SETS; i = i + 1)
				lru_bits[i] <= 3'b000;
		end
		else if (access_valid)
			case (access_way)
				2'd0: begin
					lru_bits[access_set][2] <= 1'b1;
					lru_bits[access_set][1] <= 1'b1;
				end
				2'd1: begin
					lru_bits[access_set][2] <= 1'b1;
					lru_bits[access_set][1] <= 1'b0;
				end
				2'd2: begin
					lru_bits[access_set][2] <= 1'b0;
					lru_bits[access_set][0] <= 1'b1;
				end
				2'd3: begin
					lru_bits[access_set][2] <= 1'b0;
					lru_bits[access_set][0] <= 1'b0;
				end
				default:
					;
			endcase
	initial _sv2v_0 = 0;
endmodule
