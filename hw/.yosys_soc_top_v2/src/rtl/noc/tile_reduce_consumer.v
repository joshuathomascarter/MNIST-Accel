module tile_reduce_consumer (
	clk,
	rst_n,
	enable,
	flit_in,
	valid_in,
	credit_out,
	commit_valid,
	commit_id,
	commit_value,
	packets_consumed,
	groups_completed
);
	reg _sv2v_0;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	localparam signed [31:0] noc_pkg_INNET_SP_DEPTH = 8;
	parameter signed [31:0] ENTRY_DEPTH = noc_pkg_INNET_SP_DEPTH;
	input wire clk;
	input wire rst_n;
	input wire enable;
	input wire [63:0] flit_in;
	input wire valid_in;
	output reg [NUM_VCS - 1:0] credit_out;
	output reg commit_valid;
	output reg [7:0] commit_id;
	output reg [31:0] commit_value;
	output reg [15:0] packets_consumed;
	output reg [15:0] groups_completed;
	localparam signed [31:0] HIT_IDX_W = $clog2(ENTRY_DEPTH + 1);
	reg [56:0] entries [0:ENTRY_DEPTH - 1];
	wire is_reduce_single;
	wire [7:0] in_reduce_id;
	wire [3:0] in_expected;
	wire [3:0] in_contrib_count;
	wire [31:0] in_reduce_val;
	reg [HIT_IDX_W - 1:0] hit_idx;
	reg hit;
	reg [$clog2(ENTRY_DEPTH) - 1:0] free_idx;
	reg has_free;
	wire can_accept;
	wire [3:0] next_contrib;
	wire [31:0] next_value;
	assign is_reduce_single = ((enable && valid_in) && (flit_in[51-:4] == 4'h6)) && (flit_in[63-:2] == 2'b11);
	localparam signed [31:0] noc_pkg_REDUCE_ID_HI = 47;
	localparam signed [31:0] noc_pkg_REDUCE_ID_LO = 40;
	assign in_reduce_id = flit_in[47:40];
	localparam signed [31:0] noc_pkg_REDUCE_EXPECT_HI = 39;
	localparam signed [31:0] noc_pkg_REDUCE_EXPECT_LO = 36;
	assign in_expected = flit_in[39:36];
	localparam signed [31:0] noc_pkg_REDUCE_VAL_HI = 35;
	localparam signed [31:0] noc_pkg_REDUCE_VAL_LO = 4;
	assign in_reduce_val = flit_in[35:4];
	localparam signed [31:0] noc_pkg_REDUCE_COUNT_HI = 3;
	localparam signed [31:0] noc_pkg_REDUCE_COUNT_LO = 0;
	assign in_contrib_count = (flit_in[3:0] == 4'h0 ? 4'h1 : flit_in[3:0]);
	always @(*) begin
		if (_sv2v_0)
			;
		hit_idx = ENTRY_DEPTH;
		hit = 1'b0;
		begin : sv2v_autoblock_1
			reg signed [31:0] e;
			for (e = 0; e < ENTRY_DEPTH; e = e + 1)
				if (entries[e][56] && (entries[e][55-:8] == in_reduce_id)) begin
					hit_idx = e;
					hit = 1'b1;
				end
		end
	end
	function automatic signed [$clog2(ENTRY_DEPTH) - 1:0] sv2v_cast_14A97_signed;
		input reg signed [$clog2(ENTRY_DEPTH) - 1:0] inp;
		sv2v_cast_14A97_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		free_idx = '0;
		has_free = 1'b0;
		begin : sv2v_autoblock_2
			reg signed [31:0] e;
			for (e = ENTRY_DEPTH - 1; e >= 0; e = e - 1)
				if (!entries[e][56]) begin
					free_idx = sv2v_cast_14A97_signed(e);
					has_free = 1'b1;
				end
		end
	end
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	assign next_contrib = (hit ? entries[sv2v_cast_32_signed(hit_idx)][43-:4] + in_contrib_count : in_contrib_count);
	assign next_value = (hit ? entries[sv2v_cast_32_signed(hit_idx)][39-:32] + in_reduce_val : in_reduce_val);
	assign can_accept = is_reduce_single && (((hit || has_free) || (in_contrib_count >= in_expected)) || (in_expected <= 4'd1));
	always @(*) begin
		if (_sv2v_0)
			;
		credit_out = '0;
		if (can_accept)
			credit_out[flit_in[53-:2]] = 1'b1;
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			packets_consumed <= '0;
			groups_completed <= '0;
			commit_valid <= 1'b0;
			commit_id <= '0;
			commit_value <= '0;
			begin : sv2v_autoblock_3
				reg signed [31:0] e;
				for (e = 0; e < ENTRY_DEPTH; e = e + 1)
					begin
						entries[e][56] <= 1'b0;
						entries[e][55-:8] <= '0;
						entries[e][47-:4] <= '0;
						entries[e][43-:4] <= '0;
						entries[e][39-:32] <= '0;
						entries[e][7-:8] <= '0;
					end
			end
		end
		else begin
			commit_valid <= 1'b0;
			begin : sv2v_autoblock_4
				reg signed [31:0] e;
				for (e = 0; e < ENTRY_DEPTH; e = e + 1)
					if (entries[e][56]) begin
						if (entries[e][7-:8] == 8'hff) begin
							entries[e][56] <= 1'b0;
							entries[e][7-:8] <= '0;
						end
						else
							entries[e][7-:8] <= entries[e][7-:8] + 8'h01;
					end
			end
			if (can_accept) begin
				packets_consumed <= packets_consumed + 16'd1;
				if (next_contrib >= in_expected) begin
					commit_valid <= 1'b1;
					commit_id <= in_reduce_id;
					commit_value <= next_value;
					groups_completed <= groups_completed + 16'd1;
					if (hit)
						entries[sv2v_cast_32_signed(hit_idx)][56] <= 1'b0;
				end
				else if (hit) begin
					entries[sv2v_cast_32_signed(hit_idx)][43-:4] <= next_contrib;
					entries[sv2v_cast_32_signed(hit_idx)][39-:32] <= next_value;
					entries[sv2v_cast_32_signed(hit_idx)][7-:8] <= '0;
				end
				else begin
					entries[free_idx][56] <= 1'b1;
					entries[free_idx][55-:8] <= in_reduce_id;
					entries[free_idx][47-:4] <= in_expected;
					entries[free_idx][43-:4] <= in_contrib_count;
					entries[free_idx][39-:32] <= in_reduce_val;
					entries[free_idx][7-:8] <= '0;
				end
			end
		end
	initial _sv2v_0 = 0;
endmodule
