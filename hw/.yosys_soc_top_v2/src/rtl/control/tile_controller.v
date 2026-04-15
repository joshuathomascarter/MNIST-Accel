module tile_controller (
	clk,
	rst_n,
	cmd_valid,
	cmd_ready,
	cmd_opcode,
	cmd_arg0,
	cmd_arg1,
	cmd_arg2,
	busy,
	done,
	error,
	sparse_hint,
	sp_a_en,
	sp_a_we,
	sp_a_addr,
	sp_a_wdata,
	sp_a_rdata,
	sa_start,
	sa_load_weight,
	sa_pe_en,
	sa_accum_en,
	sa_done,
	dma_req_valid,
	dma_req_ready,
	dma_req_write,
	dma_req_addr,
	dma_req_len,
	dma_data_valid,
	dma_data_ready,
	dma_data_in,
	dma_wdata_valid,
	dma_wdata_ready,
	dma_wdata_out,
	dma_wdata_last,
	dma_done_valid,
	barrier_req,
	barrier_done,
	reduce_inj_valid,
	reduce_inj_id,
	reduce_inj_expect,
	reduce_inj_dst,
	reduce_inj_val,
	reduce_inj_ready
);
	parameter signed [31:0] TILE_ID = 0;
	parameter signed [31:0] SP_ADDR_W = 12;
	parameter signed [31:0] SP_DATA_W = 64;
	parameter signed [31:0] N_ROWS = 16;
	parameter signed [31:0] N_COLS = 16;
	parameter signed [31:0] DMA_MAX_BURST = 16;
	input wire clk;
	input wire rst_n;
	input wire cmd_valid;
	output wire cmd_ready;
	input wire [7:0] cmd_opcode;
	input wire [31:0] cmd_arg0;
	input wire [31:0] cmd_arg1;
	input wire [31:0] cmd_arg2;
	output wire busy;
	output wire done;
	output wire error;
	output reg sparse_hint;
	output wire sp_a_en;
	output wire sp_a_we;
	output wire [SP_ADDR_W - 1:0] sp_a_addr;
	output wire [SP_DATA_W - 1:0] sp_a_wdata;
	input wire [SP_DATA_W - 1:0] sp_a_rdata;
	output wire sa_start;
	output wire sa_load_weight;
	output wire sa_pe_en;
	output wire sa_accum_en;
	input wire sa_done;
	output wire dma_req_valid;
	input wire dma_req_ready;
	output wire dma_req_write;
	output wire [31:0] dma_req_addr;
	output wire [15:0] dma_req_len;
	input wire dma_data_valid;
	output wire dma_data_ready;
	input wire [SP_DATA_W - 1:0] dma_data_in;
	output wire dma_wdata_valid;
	input wire dma_wdata_ready;
	output wire [SP_DATA_W - 1:0] dma_wdata_out;
	output wire dma_wdata_last;
	input wire dma_done_valid;
	output wire barrier_req;
	input wire barrier_done;
	output wire reduce_inj_valid;
	output wire [7:0] reduce_inj_id;
	output wire [3:0] reduce_inj_expect;
	output wire [3:0] reduce_inj_dst;
	output wire [31:0] reduce_inj_val;
	input wire reduce_inj_ready;
	localparam [7:0] OP_NOP = 8'h00;
	localparam [7:0] OP_LOAD = 8'h01;
	localparam [7:0] OP_STORE = 8'h02;
	localparam [7:0] OP_COMPUTE = 8'h03;
	localparam [7:0] OP_BARRIER = 8'h04;
	localparam [7:0] OP_SPARSE = 8'h05;
	localparam [7:0] OP_REDUCE = 8'h06;
	localparam signed [31:0] BYTES_PER_WORD = SP_DATA_W / 8;
	function automatic signed [15:0] sv2v_cast_16_signed;
		input reg signed [15:0] inp;
		sv2v_cast_16_signed = inp;
	endfunction
	function automatic [15:0] burst_word_count;
		input reg [15:0] words_remaining;
		if (words_remaining > sv2v_cast_16_signed(DMA_MAX_BURST))
			burst_word_count = sv2v_cast_16_signed(DMA_MAX_BURST);
		else
			burst_word_count = words_remaining;
	endfunction
	reg [4:0] state;
	reg [SP_ADDR_W - 1:0] sp_ptr;
	reg [15:0] xfer_remaining;
	reg [15:0] burst_cnt;
	reg [15:0] burst_words;
	reg [31:0] saved_addr;
	reg [3:0] reduce_row_idx;
	reg [7:0] reduce_id_base;
	reg [3:0] reduce_expect_r;
	reg [3:0] reduce_dst_r;
	reg [SP_ADDR_W - 1:0] reduce_sp_base;
	reg [31:0] reduce_val_latch;
	function automatic [SP_ADDR_W - 1:0] sv2v_cast_EA26C;
		input reg [SP_ADDR_W - 1:0] inp;
		sv2v_cast_EA26C = inp;
	endfunction
	function automatic signed [3:0] sv2v_cast_4_signed;
		input reg signed [3:0] inp;
		sv2v_cast_4_signed = inp;
	endfunction
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 5'd0;
			sp_ptr <= '0;
			xfer_remaining <= '0;
			burst_cnt <= '0;
			burst_words <= '0;
			saved_addr <= '0;
			sparse_hint <= 1'b0;
		end
		else
			case (state)
				5'd0:
					if (cmd_valid)
						case (cmd_opcode)
							OP_LOAD:
								if (cmd_arg1[15:0] == '0)
									state <= 5'd12;
								else begin
									saved_addr <= cmd_arg0;
									xfer_remaining <= cmd_arg1[15:0];
									burst_words <= burst_word_count(cmd_arg1[15:0]);
									sp_ptr <= sv2v_cast_EA26C(cmd_arg2);
									burst_cnt <= '0;
									state <= 5'd1;
								end
							OP_STORE:
								if (cmd_arg1[15:0] == '0)
									state <= 5'd12;
								else begin
									saved_addr <= cmd_arg0;
									xfer_remaining <= cmd_arg1[15:0];
									burst_words <= burst_word_count(cmd_arg1[15:0]);
									sp_ptr <= sv2v_cast_EA26C(cmd_arg2);
									burst_cnt <= '0;
									state <= 5'd3;
								end
							OP_COMPUTE: state <= 5'd7;
							OP_BARRIER: state <= 5'd9;
							OP_SPARSE: begin
								sparse_hint <= cmd_arg0[0];
								state <= 5'd12;
							end
							OP_REDUCE: begin
								reduce_sp_base <= sv2v_cast_EA26C(cmd_arg0);
								reduce_dst_r <= cmd_arg1[3:0];
								reduce_id_base <= cmd_arg1[11:4];
								reduce_expect_r <= cmd_arg1[15:12];
								reduce_row_idx <= '0;
								state <= 5'd10;
							end
							default: state <= 5'd12;
						endcase
				5'd1:
					if (dma_req_ready) begin
						burst_cnt <= '0;
						state <= 5'd2;
					end
				5'd2:
					if (dma_data_valid && dma_data_ready) begin
						saved_addr <= saved_addr + BYTES_PER_WORD;
						sp_ptr <= sp_ptr + 1'b1;
						xfer_remaining <= xfer_remaining - 1'b1;
						if (burst_cnt == (burst_words - 1'b1)) begin
							burst_cnt <= '0;
							if (xfer_remaining == 16'd1) begin
								burst_words <= '0;
								state <= 5'd12;
							end
							else begin
								burst_words <= burst_word_count(xfer_remaining - 16'd1);
								state <= 5'd1;
							end
						end
						else
							burst_cnt <= burst_cnt + 1'b1;
					end
				5'd3:
					if (dma_req_ready) begin
						burst_cnt <= '0;
						state <= 5'd4;
					end
				5'd4: state <= 5'd5;
				5'd5:
					if (dma_wdata_valid && dma_wdata_ready) begin
						saved_addr <= saved_addr + BYTES_PER_WORD;
						sp_ptr <= sp_ptr + 1'b1;
						xfer_remaining <= xfer_remaining - 1'b1;
						if (burst_cnt == (burst_words - 1'b1)) begin
							burst_cnt <= '0;
							if (xfer_remaining == 16'd1)
								burst_words <= '0;
							else
								burst_words <= burst_word_count(xfer_remaining - 16'd1);
							state <= 5'd6;
						end
						else begin
							burst_cnt <= burst_cnt + 1'b1;
							state <= 5'd4;
						end
					end
				5'd6:
					if (dma_done_valid) begin
						if (xfer_remaining == '0)
							state <= 5'd12;
						else
							state <= 5'd3;
					end
				5'd7: state <= 5'd8;
				5'd8:
					if (sa_done)
						state <= 5'd12;
				5'd9:
					if (barrier_done)
						state <= 5'd12;
				5'd10: state <= 5'd11;
				5'd11: begin
					reduce_val_latch <= sp_a_rdata;
					if (reduce_inj_ready) begin
						if (reduce_row_idx == sv2v_cast_4_signed(N_ROWS - 1))
							state <= 5'd12;
						else begin
							reduce_row_idx <= reduce_row_idx + 1'b1;
							state <= 5'd10;
						end
					end
				end
				5'd12: state <= 5'd0;
				default: state <= 5'd0;
			endcase
	assign busy = state != 5'd0;
	assign done = state == 5'd12;
	assign error = 1'b0;
	assign cmd_ready = state == 5'd0;
	assign sp_a_en = (((state == 5'd2) && dma_data_valid) || (state == 5'd4)) || (state == 5'd10);
	assign sp_a_we = (state == 5'd2) && dma_data_valid;
	assign sp_a_addr = ((state == 5'd10) || (state == 5'd11) ? sv2v_cast_EA26C(reduce_sp_base + sv2v_cast_EA26C(reduce_row_idx)) : sp_ptr);
	assign sp_a_wdata = dma_data_in;
	assign dma_data_ready = state == 5'd2;
	assign dma_wdata_valid = state == 5'd5;
	assign dma_wdata_out = sp_a_rdata;
	assign dma_wdata_last = (state == 5'd5) && (burst_cnt == (burst_words - 1'b1));
	assign dma_req_valid = (state == 5'd1) || (state == 5'd3);
	assign dma_req_write = state == 5'd3;
	assign dma_req_addr = saved_addr;
	assign dma_req_len = (burst_words == '0 ? '0 : burst_words - 1'b1);
	assign sa_start = state == 5'd7;
	assign sa_load_weight = 1'b0;
	assign sa_pe_en = state == 5'd8;
	assign sa_accum_en = state == 5'd8;
	assign barrier_req = state == 5'd9;
	assign reduce_inj_valid = state == 5'd11;
	assign reduce_inj_id = reduce_id_base + {4'h0, reduce_row_idx};
	assign reduce_inj_expect = reduce_expect_r;
	assign reduce_inj_dst = reduce_dst_r;
	assign reduce_inj_val = (state == 5'd11 ? sp_a_rdata : reduce_val_latch);
endmodule
