`default_nettype none
module bsr_scheduler (
	clk,
	rst_n,
	start,
	abort,
	busy,
	done,
	MT,
	KT,
	row_ptr_rd_en,
	row_ptr_rd_addr,
	row_ptr_rd_data,
	col_idx_rd_en,
	col_idx_rd_addr,
	col_idx_rd_data,
	wgt_rd_en,
	wgt_addr,
	act_rd_en,
	act_addr,
	load_weight,
	pe_en,
	accum_en,
	pe_clr
);
	parameter M_W = 10;
	parameter K_W = 12;
	parameter ADDR_W = 32;
	parameter BRAM_ADDR_W = 10;
	parameter BLOCK_SIZE = 16;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire abort;
	output reg busy;
	output reg done;
	input wire [M_W - 1:0] MT;
	input wire [K_W - 1:0] KT;
	output reg row_ptr_rd_en;
	output reg [BRAM_ADDR_W - 1:0] row_ptr_rd_addr;
	input wire [31:0] row_ptr_rd_data;
	output reg col_idx_rd_en;
	output reg [BRAM_ADDR_W - 1:0] col_idx_rd_addr;
	input wire [15:0] col_idx_rd_data;
	output reg wgt_rd_en;
	output reg [ADDR_W - 1:0] wgt_addr;
	output reg act_rd_en;
	output reg [ADDR_W - 1:0] act_addr;
	output wire load_weight;
	output wire pe_en;
	output reg accum_en;
	output reg pe_clr;
	localparam [9:0] S_IDLE = 10'b0000000001;
	localparam [9:0] S_FETCH_PTR1 = 10'b0000000010;
	localparam [9:0] S_FETCH_PTR2 = 10'b0000000100;
	localparam [9:0] S_CALC_LEN = 10'b0000001000;
	localparam [9:0] S_FETCH_COL = 10'b0000010000;
	localparam [9:0] S_LOAD_WGT = 10'b0000100000;
	localparam [9:0] S_WAIT_WGT = 10'b0001000000;
	localparam [9:0] S_STREAM_ACT = 10'b0010000000;
	localparam [9:0] S_NEXT_BLK = 10'b0100000000;
	localparam [9:0] S_NEXT_K = 10'b1000000000;
	(* fsm_encoding = "one_hot" *) reg [9:0] state;
	(* fsm_encoding = "one_hot" *) reg [9:0] state_n;
	wire _unused_KT = &{1'b0, KT};
	reg [K_W - 1:0] k_idx;
	reg [M_W - 1:0] m_idx;
	reg [$clog2(3 * BLOCK_SIZE):0] load_cnt;
	reg [$clog2(3 * BLOCK_SIZE):0] wait_cnt;
	reg [$clog2(3 * BLOCK_SIZE):0] stream_cnt;
	localparam COUNT_W = $clog2(3 * BLOCK_SIZE) + 1;
	localparam LOAD_CNT_MAX = BLOCK_SIZE - 1;
	localparam BLOCK_SHIFT = $clog2(BLOCK_SIZE);
	localparam STREAM_CNT_MAX = (3 * BLOCK_SIZE) - 3;
	reg [31:0] blk_ptr;
	reg [31:0] blk_end;
	reg load_weight_r;
	reg load_weight_d;
	assign load_weight = load_weight_d;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			load_weight_d <= 1'b0;
		else
			load_weight_d <= load_weight_r;
	reg pe_en_r;
	reg pe_en_d;
	assign pe_en = pe_en_d;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			pe_en_d <= 1'b0;
		else
			pe_en_d <= pe_en_r;
	reg [31:0] ptr_start_reg;
	wire [K_W - 1:0] mt_last = {{K_W - M_W {1'b0}}, MT} - {{K_W - 1 {1'b0}}, 1'b1};
	wire [ADDR_W - 1:0] load_cnt_ext = {{ADDR_W - COUNT_W {1'b0}}, load_cnt};
	wire [ADDR_W - 1:0] stream_cnt_ext = {{ADDR_W - COUNT_W {1'b0}}, stream_cnt};
	wire [(ADDR_W - BLOCK_SHIFT) - 1:0] blk_base_index = blk_ptr[(ADDR_W - BLOCK_SHIFT) - 1:0];
	wire [ADDR_W - 1:0] blk_base_addr = {blk_base_index, {BLOCK_SHIFT {1'b0}}};
	reg [15:0] col_idx_reg;
	wire [(ADDR_W - BLOCK_SHIFT) - 1:0] act_base_index = {{(ADDR_W - BLOCK_SHIFT) - 16 {1'b0}}, col_idx_reg};
	wire [ADDR_W - 1:0] act_base_addr = {act_base_index, {BLOCK_SHIFT {1'b0}}};
	reg fetch_wait;
	always @(*) begin
		state_n = 10'b0000000000;
		casez (state)
			S_IDLE: state_n = (start && (MT != 0) ? S_FETCH_PTR1 : S_IDLE);
			S_FETCH_PTR1: state_n = (fetch_wait ? S_FETCH_PTR2 : S_FETCH_PTR1);
			S_FETCH_PTR2: state_n = (fetch_wait ? S_CALC_LEN : S_FETCH_PTR2);
			S_CALC_LEN: state_n = (ptr_start_reg == blk_end ? S_NEXT_K : S_FETCH_COL);
			S_FETCH_COL: state_n = (fetch_wait ? S_LOAD_WGT : S_FETCH_COL);
			S_LOAD_WGT: state_n = (load_cnt == BLOCK_SIZE ? S_WAIT_WGT : S_LOAD_WGT);
			S_WAIT_WGT: state_n = (wait_cnt == LOAD_CNT_MAX ? S_STREAM_ACT : S_WAIT_WGT);
			S_STREAM_ACT: state_n = (stream_cnt == STREAM_CNT_MAX ? S_NEXT_BLK : S_STREAM_ACT);
			S_NEXT_BLK: state_n = ((blk_ptr + 32'd1) < blk_end ? S_FETCH_COL : S_NEXT_K);
			S_NEXT_K: state_n = (k_idx < mt_last ? S_FETCH_PTR1 : S_IDLE);
			default: state_n = S_IDLE;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= S_IDLE;
			k_idx <= 0;
			m_idx <= 0;
			load_cnt <= 0;
			wait_cnt <= 0;
			stream_cnt <= 0;
			blk_ptr <= 0;
			blk_end <= 0;
			wgt_rd_en <= 0;
			act_rd_en <= 0;
			load_weight_r <= 0;
			pe_en_r <= 0;
			pe_clr <= 0;
			busy <= 0;
			done <= 0;
			fetch_wait <= 0;
			row_ptr_rd_en <= 0;
			row_ptr_rd_addr <= 0;
			col_idx_rd_en <= 0;
			col_idx_rd_addr <= 0;
			ptr_start_reg <= 0;
			col_idx_reg <= 0;
		end
		else begin
			state <= state_n;
			if (state != state_n)
				fetch_wait <= 0;
			if (state != state_n)
;
			wgt_rd_en <= 0;
			act_rd_en <= 0;
			pe_en_r <= 0;
			accum_en <= 0;
			pe_clr <= 0;
			done <= 0;
			row_ptr_rd_en <= 0;
			col_idx_rd_en <= 0;
			case (state)
				S_IDLE: begin
					busy <= 0;
					k_idx <= 0;
					load_cnt <= 0;
					wait_cnt <= 0;
					stream_cnt <= 0;
					load_weight_r <= 0;
					if (start && (MT != 0))
						busy <= 1;
					if (start && (MT == 0))
						done <= 1;
				end
				S_FETCH_PTR1: begin
					row_ptr_rd_addr <= k_idx[BRAM_ADDR_W - 1:0];
					if (fetch_wait) begin
						ptr_start_reg <= row_ptr_rd_data;
;
					end
					else begin
						row_ptr_rd_en <= 1'b1;
						fetch_wait <= 1;
					end
				end
				S_FETCH_PTR2: begin
					row_ptr_rd_addr <= k_idx[BRAM_ADDR_W - 1:0] + {{BRAM_ADDR_W - 1 {1'b0}}, 1'b1};
					if (fetch_wait) begin
						blk_end <= row_ptr_rd_data;
;
					end
					else begin
						row_ptr_rd_en <= 1'b1;
						fetch_wait <= 1;
					end
				end
				S_CALC_LEN: blk_ptr <= ptr_start_reg;
				S_FETCH_COL: begin
					col_idx_rd_addr <= blk_ptr[BRAM_ADDR_W - 1:0];
					if (!fetch_wait) begin
						col_idx_rd_en <= 1'b1;
						fetch_wait <= 1;
					end
					else
						col_idx_reg <= col_idx_rd_data;
				end
				S_LOAD_WGT: begin
					wgt_rd_en <= 1;
					if (load_cnt <= LOAD_CNT_MAX)
						wgt_addr <= blk_base_addr + load_cnt_ext;
					load_weight_r <= load_cnt < BLOCK_SIZE;
					load_cnt <= load_cnt + 1;
					if (load_cnt == BLOCK_SIZE) begin
						m_idx <= 0;
						load_cnt <= 0;
						wait_cnt <= 0;
					end
				end
				S_WAIT_WGT: begin
					load_weight_r <= 0;
					wait_cnt <= wait_cnt + 1;
					if (wait_cnt == LOAD_CNT_MAX) begin
						wait_cnt <= 0;
						stream_cnt <= 0;
					end
				end
				S_STREAM_ACT: begin
					pe_en_r <= 1;
					accum_en <= 1;
					stream_cnt <= stream_cnt + 1;
					if (stream_cnt < BLOCK_SIZE) begin
						act_rd_en <= 1;
						act_addr <= act_base_addr + stream_cnt_ext;
					end
					if (stream_cnt == STREAM_CNT_MAX)
						stream_cnt <= 0;
				end
				S_NEXT_BLK: blk_ptr <= blk_ptr + 1;
				S_NEXT_K: begin
					k_idx <= k_idx + 1;
					if (k_idx == mt_last) begin
						done <= 1;
						busy <= 0;
					end
				end
				default: busy <= 0;
			endcase
			if (abort) begin
				state <= S_IDLE;
				busy <= 0;
				pe_clr <= 1;
			end
		end
endmodule
`default_nettype wire
