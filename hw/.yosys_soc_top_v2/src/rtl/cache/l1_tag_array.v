module l1_tag_array (
	clk,
	rst_n,
	lookup_addr,
	lookup_hit,
	lookup_way,
	lookup_dirty,
	write_en,
	write_set,
	write_way,
	write_tag,
	write_valid,
	write_dirty,
	inv_en,
	inv_set,
	inv_way,
	rb_set,
	rb_way,
	rb_tag,
	rb_dirty,
	rb_valid
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] NUM_SETS = 16;
	parameter signed [31:0] NUM_WAYS = 4;
	parameter signed [31:0] LINE_BYTES = 64;
	input wire clk;
	input wire rst_n;
	input wire [ADDR_WIDTH - 1:0] lookup_addr;
	output reg lookup_hit;
	output reg [$clog2(NUM_WAYS) - 1:0] lookup_way;
	output reg lookup_dirty;
	input wire write_en;
	input wire [$clog2(NUM_SETS) - 1:0] write_set;
	input wire [$clog2(NUM_WAYS) - 1:0] write_way;
	localparam signed [31:0] OFFSET_BITS = $clog2(LINE_BYTES);
	localparam signed [31:0] SET_BITS = $clog2(NUM_SETS);
	localparam signed [31:0] TAG_BITS = (ADDR_WIDTH - SET_BITS) - OFFSET_BITS;
	input wire [TAG_BITS - 1:0] write_tag;
	input wire write_valid;
	input wire write_dirty;
	input wire inv_en;
	input wire [$clog2(NUM_SETS) - 1:0] inv_set;
	input wire [$clog2(NUM_WAYS) - 1:0] inv_way;
	input wire [$clog2(NUM_SETS) - 1:0] rb_set;
	input wire [$clog2(NUM_WAYS) - 1:0] rb_way;
	output wire [TAG_BITS - 1:0] rb_tag;
	output wire rb_dirty;
	output wire rb_valid;
	wire [TAG_BITS - 1:0] addr_tag;
	wire [SET_BITS - 1:0] addr_set;
	assign addr_tag = lookup_addr[ADDR_WIDTH - 1-:TAG_BITS];
	assign addr_set = lookup_addr[OFFSET_BITS+:SET_BITS];
	reg valid [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	reg dirty [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	reg [TAG_BITS - 1:0] tags [0:NUM_SETS - 1][0:NUM_WAYS - 1];
	always @(*) begin
		if (_sv2v_0)
			;
		lookup_hit = 1'b0;
		lookup_way = '0;
		lookup_dirty = 1'b0;
		begin : sv2v_autoblock_1
			reg signed [31:0] w;
			for (w = 0; w < NUM_WAYS; w = w + 1)
				if (valid[addr_set][w] && (tags[addr_set][w] == addr_tag)) begin
					lookup_hit = 1'b1;
					lookup_way = w[$clog2(NUM_WAYS) - 1:0];
					lookup_dirty = dirty[addr_set][w];
				end
		end
	end
	assign rb_tag = tags[rb_set][rb_way];
	assign rb_dirty = dirty[rb_set][rb_way];
	assign rb_valid = valid[rb_set][rb_way];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_2
			reg signed [31:0] s;
			for (s = 0; s < NUM_SETS; s = s + 1)
				begin : sv2v_autoblock_3
					reg signed [31:0] w;
					for (w = 0; w < NUM_WAYS; w = w + 1)
						begin
							valid[s][w] <= 1'b0;
							dirty[s][w] <= 1'b0;
							tags[s][w] <= '0;
						end
				end
		end
		else begin
			if (write_en) begin
				valid[write_set][write_way] <= write_valid;
				dirty[write_set][write_way] <= write_dirty;
				tags[write_set][write_way] <= write_tag;
			end
			if (inv_en)
				valid[inv_set][inv_way] <= 1'b0;
		end
	initial _sv2v_0 = 0;
endmodule
