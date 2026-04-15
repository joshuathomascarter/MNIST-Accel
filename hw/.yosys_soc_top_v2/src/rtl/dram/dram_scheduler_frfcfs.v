module dram_scheduler_frfcfs (
	clk,
	rst_n,
	entry_valid,
	entry_rw,
	entry_addr,
	entry_id,
	entry_blen,
	entry_age,
	deq_valid,
	deq_idx,
	bank_state,
	bank_open_row,
	bank_row_open,
	bank_cmd_valid,
	bank_cmd_op,
	bank_cmd_row,
	bank_cmd_col,
	ref_req,
	ref_ack,
	ref_busy,
	data_rd_valid,
	data_wr_valid,
	data_id,
	sched_busy
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_BANKS = 8;
	parameter signed [31:0] QUEUE_DEPTH = 16;
	parameter signed [31:0] ADDR_W = 28;
	parameter signed [31:0] ROW_BITS = 14;
	parameter signed [31:0] COL_BITS = 10;
	parameter signed [31:0] BANK_BITS = 3;
	parameter signed [31:0] ID_W = 4;
	parameter signed [31:0] BLEN_W = 4;
	input wire clk;
	input wire rst_n;
	input wire [QUEUE_DEPTH - 1:0] entry_valid;
	input wire [QUEUE_DEPTH - 1:0] entry_rw;
	input wire [(QUEUE_DEPTH * ADDR_W) - 1:0] entry_addr;
	input wire [(QUEUE_DEPTH * ID_W) - 1:0] entry_id;
	input wire [(QUEUE_DEPTH * BLEN_W) - 1:0] entry_blen;
	input wire [(QUEUE_DEPTH * 8) - 1:0] entry_age;
	output reg deq_valid;
	output reg [$clog2(QUEUE_DEPTH) - 1:0] deq_idx;
	input wire [(NUM_BANKS * 3) - 1:0] bank_state;
	input wire [(NUM_BANKS * ROW_BITS) - 1:0] bank_open_row;
	input wire [NUM_BANKS - 1:0] bank_row_open;
	output reg [NUM_BANKS - 1:0] bank_cmd_valid;
	output reg [(NUM_BANKS * 3) - 1:0] bank_cmd_op;
	output reg [(NUM_BANKS * ROW_BITS) - 1:0] bank_cmd_row;
	output reg [(NUM_BANKS * COL_BITS) - 1:0] bank_cmd_col;
	input wire ref_req;
	output reg ref_ack;
	input wire ref_busy;
	output reg data_rd_valid;
	output reg data_wr_valid;
	output reg [ID_W - 1:0] data_id;
	output reg sched_busy;
	localparam signed [31:0] QIX_W = $clog2(QUEUE_DEPTH);
	localparam [2:0] BS_IDLE = 3'd0;
	localparam [2:0] BS_ACTIVE = 3'd2;
	localparam [2:0] OP_ACT = 3'b001;
	localparam [2:0] OP_READ = 3'b010;
	localparam [2:0] OP_WRITE = 3'b011;
	localparam [2:0] OP_PRE = 3'b100;
	wire [(QUEUE_DEPTH * BANK_BITS) - 1:0] q_bank;
	wire [(QUEUE_DEPTH * ROW_BITS) - 1:0] q_row;
	wire [(QUEUE_DEPTH * COL_BITS) - 1:0] q_col;
	localparam signed [31:0] BYTE_OFF = 2;
	genvar _gv_gi_1;
	generate
		for (_gv_gi_1 = 0; _gv_gi_1 < QUEUE_DEPTH; _gv_gi_1 = _gv_gi_1 + 1) begin : gen_decode
			localparam gi = _gv_gi_1;
			assign q_col[gi * COL_BITS+:COL_BITS] = entry_addr[(gi * ADDR_W) + BYTE_OFF+:COL_BITS];
			assign q_bank[gi * BANK_BITS+:BANK_BITS] = entry_addr[(gi * ADDR_W) + (BYTE_OFF + COL_BITS)+:BANK_BITS];
			assign q_row[gi * ROW_BITS+:ROW_BITS] = entry_addr[(gi * ADDR_W) + ((BYTE_OFF + COL_BITS) + BANK_BITS)+:ROW_BITS];
		end
	endgenerate
	reg [QUEUE_DEPTH - 1:0] is_row_hit;
	reg [QUEUE_DEPTH - 1:0] is_bank_ready;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < QUEUE_DEPTH; i = i + 1)
				begin : sv2v_autoblock_2
					reg [BANK_BITS - 1:0] b;
					b = q_bank[i * BANK_BITS+:BANK_BITS];
					is_bank_ready[i] = entry_valid[i] && ((bank_state[b * 3+:3] == BS_IDLE) || (bank_state[b * 3+:3] == BS_ACTIVE));
					is_row_hit[i] = ((entry_valid[i] && bank_row_open[b]) && (bank_open_row[b * ROW_BITS+:ROW_BITS] == q_row[i * ROW_BITS+:ROW_BITS])) && (bank_state[b * 3+:3] == BS_ACTIVE);
				end
		end
	end
	reg [QUEUE_DEPTH - 1:0] is_row_hit_r;
	reg [QUEUE_DEPTH - 1:0] is_bank_ready_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			is_row_hit_r <= '0;
			is_bank_ready_r <= '0;
		end
		else begin
			is_row_hit_r <= is_row_hit;
			is_bank_ready_r <= is_bank_ready;
		end
	reg found_hit;
	reg found_ready;
	reg [QIX_W - 1:0] best_idx;
	reg [7:0] best_age;
	always @(*) begin
		if (_sv2v_0)
			;
		found_hit = 1'b0;
		found_ready = 1'b0;
		best_idx = '0;
		best_age = '0;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < QUEUE_DEPTH; i = i + 1)
				if ((is_row_hit_r[i] && entry_valid[i]) && (entry_age[i * 8+:8] >= best_age)) begin
					found_hit = 1'b1;
					best_idx = i[QIX_W - 1:0];
					best_age = entry_age[i * 8+:8];
				end
		end
		if (!found_hit) begin
			best_age = '0;
			begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < QUEUE_DEPTH; i = i + 1)
					if ((is_bank_ready_r[i] && entry_valid[i]) && (entry_age[i * 8+:8] >= best_age)) begin
						found_ready = 1'b1;
						best_idx = i[QIX_W - 1:0];
						best_age = entry_age[i * 8+:8];
					end
			end
		end
	end
	wire has_candidate;
	assign has_candidate = found_hit || found_ready;
	(* fsm_encoding = "one_hot" *) reg [4:0] sch_state;
	(* fsm_encoding = "one_hot" *) reg [4:0] sch_state_next;
	reg [QIX_W - 1:0] sel_idx_r;
	reg [BANK_BITS - 1:0] sel_bank_r;
	reg [ROW_BITS - 1:0] sel_row_r;
	reg [COL_BITS - 1:0] sel_col_r;
	reg sel_rw_r;
	reg [ID_W - 1:0] sel_id_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			sch_state <= 5'b00001;
			sel_idx_r <= '0;
			sel_bank_r <= '0;
			sel_row_r <= '0;
			sel_col_r <= '0;
			sel_rw_r <= 1'b0;
			sel_id_r <= '0;
		end
		else begin
			sch_state <= sch_state_next;
			if (((sch_state == 5'b00001) && has_candidate) && !ref_req) begin
				sel_idx_r <= best_idx;
				sel_bank_r <= q_bank[best_idx * BANK_BITS+:BANK_BITS];
				sel_row_r <= q_row[best_idx * ROW_BITS+:ROW_BITS];
				sel_col_r <= q_col[best_idx * COL_BITS+:COL_BITS];
				sel_rw_r <= entry_rw[best_idx];
				sel_id_r <= entry_id[best_idx * ID_W+:ID_W];
			end
		end
	always @(*) begin
		if (_sv2v_0)
			;
		sch_state_next = sch_state;
		deq_valid = 1'b0;
		deq_idx = '0;
		ref_ack = 1'b0;
		data_rd_valid = 1'b0;
		data_wr_valid = 1'b0;
		data_id = '0;
		sched_busy = 1'b1;
		begin : sv2v_autoblock_5
			reg signed [31:0] b;
			for (b = 0; b < NUM_BANKS; b = b + 1)
				begin
					bank_cmd_valid[b] = 1'b0;
					bank_cmd_op[b * 3+:3] = 3'b000;
					bank_cmd_row[b * ROW_BITS+:ROW_BITS] = '0;
					bank_cmd_col[b * COL_BITS+:COL_BITS] = '0;
				end
		end
		case (sch_state)
			5'b00001: begin
				sched_busy = 1'b0;
				if (ref_req && !ref_busy) begin
					ref_ack = 1'b1;
					sch_state_next = 5'b10000;
				end
				else if (has_candidate) begin
					if (is_row_hit_r[best_idx])
						sch_state_next = 5'b01000;
					else if (bank_row_open[q_bank[best_idx * BANK_BITS+:BANK_BITS]])
						sch_state_next = 5'b00010;
					else
						sch_state_next = 5'b00100;
				end
			end
			5'b00010: begin
				bank_cmd_valid[sel_bank_r] = 1'b1;
				bank_cmd_op[sel_bank_r * 3+:3] = OP_PRE;
				sch_state_next = 5'b00100;
			end
			5'b00100:
				if (bank_state[sel_bank_r * 3+:3] == BS_IDLE) begin
					bank_cmd_valid[sel_bank_r] = 1'b1;
					bank_cmd_op[sel_bank_r * 3+:3] = OP_ACT;
					bank_cmd_row[sel_bank_r * ROW_BITS+:ROW_BITS] = sel_row_r;
					sch_state_next = 5'b01000;
				end
			5'b01000:
				if (bank_state[sel_bank_r * 3+:3] == BS_ACTIVE) begin
					bank_cmd_valid[sel_bank_r] = 1'b1;
					bank_cmd_op[sel_bank_r * 3+:3] = (sel_rw_r ? OP_WRITE : OP_READ);
					bank_cmd_col[sel_bank_r * COL_BITS+:COL_BITS] = sel_col_r;
					data_rd_valid = !sel_rw_r;
					data_wr_valid = sel_rw_r;
					data_id = sel_id_r;
					deq_valid = 1'b1;
					deq_idx = sel_idx_r;
					sch_state_next = 5'b00001;
				end
			5'b10000:
				if (!ref_busy)
					sch_state_next = 5'b00001;
			default: sch_state_next = 5'b00001;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
