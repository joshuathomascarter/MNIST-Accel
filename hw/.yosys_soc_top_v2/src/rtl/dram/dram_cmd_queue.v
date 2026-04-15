module dram_cmd_queue (
	clk,
	rst_n,
	enq_valid,
	enq_ready,
	enq_rw,
	enq_addr,
	enq_id,
	enq_blen,
	deq_valid,
	deq_idx,
	deq_ready,
	count,
	empty,
	full,
	entry_valid,
	entry_rw,
	entry_addr,
	entry_id,
	entry_blen,
	entry_age
);
	reg _sv2v_0;
	parameter signed [31:0] DEPTH = 16;
	parameter signed [31:0] ADDR_W = 28;
	parameter signed [31:0] ID_W = 4;
	parameter signed [31:0] BLEN_W = 4;
	input wire clk;
	input wire rst_n;
	input wire enq_valid;
	output wire enq_ready;
	input wire enq_rw;
	input wire [ADDR_W - 1:0] enq_addr;
	input wire [ID_W - 1:0] enq_id;
	input wire [BLEN_W - 1:0] enq_blen;
	input wire deq_valid;
	input wire [$clog2(DEPTH) - 1:0] deq_idx;
	output wire deq_ready;
	output wire [$clog2(DEPTH):0] count;
	output wire empty;
	output wire full;
	output wire [DEPTH - 1:0] entry_valid;
	output wire [DEPTH - 1:0] entry_rw;
	output wire [(DEPTH * ADDR_W) - 1:0] entry_addr;
	output wire [(DEPTH * ID_W) - 1:0] entry_id;
	output wire [(DEPTH * BLEN_W) - 1:0] entry_blen;
	output wire [(DEPTH * 8) - 1:0] entry_age;
	localparam signed [31:0] IDX_W = $clog2(DEPTH);
	reg [DEPTH - 1:0] valid_r;
	reg [DEPTH - 1:0] rw_r;
	reg [(DEPTH * ADDR_W) - 1:0] addr_r;
	reg [(DEPTH * ID_W) - 1:0] id_r;
	reg [(DEPTH * BLEN_W) - 1:0] blen_r;
	reg [(DEPTH * 8) - 1:0] age_r;
	reg [IDX_W:0] cnt_r;
	assign entry_valid = valid_r;
	assign entry_rw = rw_r;
	assign entry_addr = addr_r;
	assign entry_id = id_r;
	assign entry_blen = blen_r;
	assign entry_age = age_r;
	assign count = cnt_r;
	assign empty = cnt_r == 0;
	assign full = cnt_r == DEPTH[IDX_W:0];
	assign enq_ready = !full;
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	assign deq_ready = (deq_valid && (sv2v_cast_32_signed(deq_idx) < DEPTH)) && valid_r[deq_idx];
	reg [IDX_W - 1:0] free_slot;
	reg has_free;
	always @(*) begin
		if (_sv2v_0)
			;
		free_slot = '0;
		has_free = 1'b0;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < DEPTH; i = i + 1)
				if (!valid_r[i] && !has_free) begin
					free_slot = i[IDX_W - 1:0];
					has_free = 1'b1;
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			valid_r <= '0;
			rw_r <= '0;
			age_r <= '0;
			cnt_r <= '0;
		end
		else begin
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < DEPTH; i = i + 1)
					if (valid_r[i] && (age_r[i * 8+:8] < 8'hff))
						age_r[i * 8+:8] <= age_r[i * 8+:8] + 1;
			end
			if (enq_valid && enq_ready) begin
				valid_r[free_slot] <= 1'b1;
				rw_r[free_slot] <= enq_rw;
				addr_r[free_slot * ADDR_W+:ADDR_W] <= enq_addr;
				id_r[free_slot * ID_W+:ID_W] <= enq_id;
				blen_r[free_slot * BLEN_W+:BLEN_W] <= enq_blen;
				age_r[free_slot * 8+:8] <= '0;
				cnt_r <= cnt_r + 1;
			end
			if (deq_valid && valid_r[deq_idx]) begin
				valid_r[deq_idx] <= 1'b0;
				cnt_r <= cnt_r - 1;
			end
			if (((enq_valid && enq_ready) && deq_valid) && valid_r[deq_idx])
				cnt_r <= cnt_r;
		end
	initial _sv2v_0 = 0;
endmodule
