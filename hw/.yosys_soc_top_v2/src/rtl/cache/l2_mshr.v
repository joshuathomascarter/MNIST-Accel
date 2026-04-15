module l2_mshr (
	clk,
	rst_n,
	alloc_valid,
	alloc_ready,
	alloc_addr,
	alloc_id,
	alloc_is_write,
	alloc_idx,
	lookup_valid,
	lookup_addr,
	lookup_hit,
	lookup_idx,
	complete_valid,
	complete_idx,
	complete_addr,
	complete_id,
	complete_is_write,
	full,
	empty,
	count
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] NUM_ENTRIES = 4;
	parameter signed [31:0] ID_WIDTH = 4;
	parameter signed [31:0] LINE_BYTES = 64;
	input wire clk;
	input wire rst_n;
	input wire alloc_valid;
	output wire alloc_ready;
	input wire [ADDR_WIDTH - 1:0] alloc_addr;
	input wire [ID_WIDTH - 1:0] alloc_id;
	input wire alloc_is_write;
	output reg [$clog2(NUM_ENTRIES) - 1:0] alloc_idx;
	input wire lookup_valid;
	input wire [ADDR_WIDTH - 1:0] lookup_addr;
	output reg lookup_hit;
	output reg [$clog2(NUM_ENTRIES) - 1:0] lookup_idx;
	input wire complete_valid;
	input wire [$clog2(NUM_ENTRIES) - 1:0] complete_idx;
	output wire [ADDR_WIDTH - 1:0] complete_addr;
	output wire [ID_WIDTH - 1:0] complete_id;
	output wire complete_is_write;
	output wire full;
	output wire empty;
	output reg [$clog2(NUM_ENTRIES):0] count;
	localparam signed [31:0] LINE_ADDR_WIDTH = ADDR_WIDTH - $clog2(LINE_BYTES);
	reg entry_valid [0:NUM_ENTRIES - 1];
	reg [LINE_ADDR_WIDTH - 1:0] entry_line_addr [0:NUM_ENTRIES - 1];
	reg [ID_WIDTH - 1:0] entry_req_id [0:NUM_ENTRIES - 1];
	reg entry_is_write [0:NUM_ENTRIES - 1];
	reg [NUM_ENTRIES - 1:0] active_bits;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				active_bits[i] = entry_valid[i];
		end
	end
	assign full = &active_bits;
	assign empty = ~|active_bits;
	always @(*) begin
		if (_sv2v_0)
			;
		count = '0;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				count = count + {{$clog2(NUM_ENTRIES) {1'b0}}, active_bits[i]};
		end
	end
	reg has_free;
	always @(*) begin
		if (_sv2v_0)
			;
		has_free = 1'b0;
		alloc_idx = '0;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = NUM_ENTRIES - 1; i >= 0; i = i - 1)
				if (!entry_valid[i]) begin
					has_free = 1'b1;
					alloc_idx = i[$clog2(NUM_ENTRIES) - 1:0];
				end
		end
	end
	assign alloc_ready = has_free;
	wire [LINE_ADDR_WIDTH - 1:0] lookup_line_addr;
	assign lookup_line_addr = lookup_addr[ADDR_WIDTH - 1:$clog2(LINE_BYTES)];
	always @(*) begin
		if (_sv2v_0)
			;
		lookup_hit = 1'b0;
		lookup_idx = '0;
		if (lookup_valid) begin : sv2v_autoblock_4
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				if (entry_valid[i] && (entry_line_addr[i] == lookup_line_addr)) begin
					lookup_hit = 1'b1;
					lookup_idx = i[$clog2(NUM_ENTRIES) - 1:0];
				end
		end
	end
	assign complete_addr = {entry_line_addr[complete_idx], {$clog2(LINE_BYTES) {1'b0}}};
	assign complete_id = entry_req_id[complete_idx];
	assign complete_is_write = entry_is_write[complete_idx];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_5
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				begin
					entry_valid[i] <= 1'b0;
					entry_line_addr[i] <= '0;
					entry_req_id[i] <= '0;
					entry_is_write[i] <= 1'b0;
				end
		end
		else begin
			if (complete_valid)
				entry_valid[complete_idx] <= 1'b0;
			if (alloc_valid && alloc_ready) begin
				entry_valid[alloc_idx] <= 1'b1;
				entry_line_addr[alloc_idx] <= alloc_addr[ADDR_WIDTH - 1:$clog2(LINE_BYTES)];
				entry_req_id[alloc_idx] <= alloc_id;
				entry_is_write[alloc_idx] <= alloc_is_write;
			end
		end
	initial _sv2v_0 = 0;
endmodule
