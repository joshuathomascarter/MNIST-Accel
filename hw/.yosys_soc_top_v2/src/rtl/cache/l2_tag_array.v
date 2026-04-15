module l2_tag_array (
	clk,
	rst_n,
	lookup_valid,
	lookup_set,
	lookup_tag,
	lookup_hit,
	lookup_way,
	lookup_dirty,
	write_valid,
	write_set,
	write_way,
	write_tag,
	write_dirty,
	inv_valid,
	inv_set,
	inv_way,
	dirty_check_valid,
	dirty_check_set,
	dirty_check_way,
	dirty_check_is_dirty,
	dirty_check_tag
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] NUM_SETS = 256;
	parameter signed [31:0] NUM_WAYS = 8;
	parameter signed [31:0] LINE_BYTES = 64;
	localparam signed [31:0] OFFSET_BITS = $clog2(LINE_BYTES);
	localparam signed [31:0] INDEX_BITS = $clog2(NUM_SETS);
	localparam signed [31:0] TAG_WIDTH = (ADDR_WIDTH - INDEX_BITS) - OFFSET_BITS;
	localparam signed [31:0] WAY_BITS = $clog2(NUM_WAYS);
	input wire clk;
	input wire rst_n;
	input wire lookup_valid;
	input wire [INDEX_BITS - 1:0] lookup_set;
	input wire [TAG_WIDTH - 1:0] lookup_tag;
	output reg lookup_hit;
	output reg [WAY_BITS - 1:0] lookup_way;
	output reg lookup_dirty;
	input wire write_valid;
	input wire [INDEX_BITS - 1:0] write_set;
	input wire [WAY_BITS - 1:0] write_way;
	input wire [TAG_WIDTH - 1:0] write_tag;
	input wire write_dirty;
	input wire inv_valid;
	input wire [INDEX_BITS - 1:0] inv_set;
	input wire [WAY_BITS - 1:0] inv_way;
	input wire dirty_check_valid;
	input wire [INDEX_BITS - 1:0] dirty_check_set;
	input wire [WAY_BITS - 1:0] dirty_check_way;
	output reg dirty_check_is_dirty;
	output reg [TAG_WIDTH - 1:0] dirty_check_tag;
	reg valid_bits [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	reg dirty_bits [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	reg [TAG_WIDTH - 1:0] tag_bits [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	always @(*) begin
		if (_sv2v_0)
			;
		lookup_hit = 1'b0;
		lookup_way = '0;
		lookup_dirty = 1'b0;
		if (lookup_valid) begin : sv2v_autoblock_1
			reg signed [31:0] w;
			for (w = 0; w < NUM_WAYS; w = w + 1)
				if (valid_bits[lookup_set][w] && (tag_bits[lookup_set][w] == lookup_tag)) begin
					lookup_hit = 1'b1;
					lookup_way = w[$clog2(NUM_WAYS) - 1:0];
					lookup_dirty = dirty_bits[lookup_set][w];
				end
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		dirty_check_is_dirty = 1'b0;
		dirty_check_tag = '0;
		if (dirty_check_valid) begin
			dirty_check_is_dirty = dirty_bits[dirty_check_set][dirty_check_way];
			dirty_check_tag = tag_bits[dirty_check_set][dirty_check_way];
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_2
			reg signed [31:0] s;
			for (s = 0; s < NUM_SETS; s = s + 1)
				begin : sv2v_autoblock_3
					reg signed [31:0] w;
					for (w = 0; w < NUM_WAYS; w = w + 1)
						begin
							valid_bits[s][w] <= 1'b0;
							dirty_bits[s][w] <= 1'b0;
							tag_bits[s][w] <= '0;
						end
				end
		end
		else begin
			if (inv_valid)
				valid_bits[inv_set][inv_way] <= 1'b0;
			if (write_valid) begin
				valid_bits[write_set][write_way] <= 1'b1;
				dirty_bits[write_set][write_way] <= write_dirty;
				tag_bits[write_set][write_way] <= write_tag;
			end
		end
	initial _sv2v_0 = 0;
endmodule
