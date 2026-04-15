module dram_write_buffer (
	clk,
	rst_n,
	wr_valid,
	wr_ready,
	wr_data,
	wr_strb,
	wr_id,
	drain_valid,
	drain_idx,
	drain_ready,
	drain_data,
	drain_strb,
	count,
	empty,
	full
);
	reg _sv2v_0;
	parameter signed [31:0] DEPTH = 16;
	parameter signed [31:0] DATA_W = 32;
	parameter signed [31:0] STRB_W = DATA_W / 8;
	parameter signed [31:0] ID_W = 4;
	input wire clk;
	input wire rst_n;
	input wire wr_valid;
	output wire wr_ready;
	input wire [DATA_W - 1:0] wr_data;
	input wire [STRB_W - 1:0] wr_strb;
	input wire [ID_W - 1:0] wr_id;
	input wire drain_valid;
	input wire [$clog2(DEPTH) - 1:0] drain_idx;
	output wire drain_ready;
	output wire [DATA_W - 1:0] drain_data;
	output wire [STRB_W - 1:0] drain_strb;
	output wire [$clog2(DEPTH):0] count;
	output wire empty;
	output wire full;
	localparam signed [31:0] IDX_W = $clog2(DEPTH);
	reg [DEPTH - 1:0] valid_r;
	reg [(DEPTH * DATA_W) - 1:0] data_r;
	reg [(DEPTH * STRB_W) - 1:0] strb_r;
	reg [(DEPTH * ID_W) - 1:0] id_r;
	reg [IDX_W:0] cnt_r;
	assign count = cnt_r;
	assign empty = cnt_r == 0;
	assign full = cnt_r == DEPTH[IDX_W:0];
	assign wr_ready = !full;
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	assign drain_ready = (drain_valid && (sv2v_cast_32_signed(drain_idx) < DEPTH)) && valid_r[drain_idx];
	assign drain_data = data_r[drain_idx * DATA_W+:DATA_W];
	assign drain_strb = strb_r[drain_idx * STRB_W+:STRB_W];
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
			cnt_r <= '0;
		end
		else begin
			if (wr_valid && wr_ready) begin
				valid_r[free_slot] <= 1'b1;
				data_r[free_slot * DATA_W+:DATA_W] <= wr_data;
				strb_r[free_slot * STRB_W+:STRB_W] <= wr_strb;
				id_r[free_slot * ID_W+:ID_W] <= wr_id;
				cnt_r <= cnt_r + 1;
			end
			if (drain_valid && valid_r[drain_idx]) begin
				valid_r[drain_idx] <= 1'b0;
				cnt_r <= cnt_r - 1;
			end
			if (((wr_valid && wr_ready) && drain_valid) && valid_r[drain_idx])
				cnt_r <= cnt_r;
		end
	initial _sv2v_0 = 0;
endmodule
