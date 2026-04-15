module accel_tile (
	clk,
	rst_n,
	noc_flit_out,
	noc_valid_out,
	noc_credit_in,
	noc_flit_in,
	noc_valid_in,
	noc_credit_out,
	csr_wdata,
	csr_addr,
	csr_wen,
	csr_rdata,
	barrier_req,
	barrier_done,
	tile_busy,
	tile_done
);
	reg _sv2v_0;
	parameter signed [31:0] TILE_ID = 0;
	parameter signed [31:0] N_ROWS = 16;
	parameter signed [31:0] N_COLS = 16;
	parameter signed [31:0] DATA_W = 8;
	parameter signed [31:0] ACC_W = 32;
	parameter signed [31:0] SP_DEPTH = 4096;
	parameter signed [31:0] SP_DATA_W = 32;
	parameter signed [31:0] SP_ADDR_W = $clog2(SP_DEPTH);
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	input wire clk;
	input wire rst_n;
	output wire [63:0] noc_flit_out;
	output wire noc_valid_out;
	input wire [NUM_VCS - 1:0] noc_credit_in;
	input wire [63:0] noc_flit_in;
	input wire noc_valid_in;
	output wire [NUM_VCS - 1:0] noc_credit_out;
	input wire [31:0] csr_wdata;
	input wire [7:0] csr_addr;
	input wire csr_wen;
	output reg [31:0] csr_rdata;
	output wire barrier_req;
	input wire barrier_done;
	output wire tile_busy;
	output wire tile_done;
	wire sp_a_en;
	wire sp_a_we;
	wire [SP_ADDR_W - 1:0] sp_a_addr;
	wire [SP_DATA_W - 1:0] sp_a_wdata;
	wire [SP_DATA_W - 1:0] sp_a_rdata;
	wire sp_a_mux_en;
	wire sp_a_mux_we;
	wire [SP_ADDR_W - 1:0] sp_a_mux_addr;
	wire [SP_DATA_W - 1:0] sp_a_mux_wdata;
	reg sp_b_en;
	reg [SP_ADDR_W - 1:0] sp_b_addr;
	wire [SP_DATA_W - 1:0] sp_b_rdata;
	wire sa_start;
	wire ctrl_sa_load_weight;
	wire ctrl_sa_pe_en;
	wire ctrl_sa_accum_en;
	wire sa_done;
	wire sparse_hint;
	wire [NUM_VCS - 1:0] ni_credit_out_int;
	wire [NUM_VCS - 1:0] reduce_credit_out;
	wire reduce_local_valid;
	wire reduce_commit_valid;
	wire [7:0] reduce_commit_id;
	wire [31:0] reduce_commit_value;
	wire [15:0] reduce_packets_consumed;
	wire [15:0] reduce_groups_completed;
	reg [7:0] reduce_last_id;
	reg [31:0] reduce_last_value;
	reg [31:0] reduce_results [0:255];
	reg [255:0] reduce_result_valid;
	wire dma_req_valid;
	wire dma_req_ready;
	wire dma_req_write;
	wire dma_req_aw_ready;
	wire dma_req_ar_ready;
	wire [31:0] dma_req_addr;
	wire [15:0] dma_req_len;
	wire dma_data_valid;
	wire dma_data_ready;
	wire [SP_DATA_W - 1:0] dma_data_in;
	wire [31:0] dma_data_word;
	wire dma_wdata_valid;
	wire dma_wdata_ready;
	wire dma_wdata_last;
	wire [SP_DATA_W - 1:0] dma_wdata_out;
	wire dma_store_done;
	wire reduce_inj_valid;
	wire reduce_inj_ready;
	wire [7:0] reduce_inj_id;
	wire [3:0] reduce_inj_expect;
	wire [3:0] reduce_inj_dst;
	wire [31:0] reduce_inj_val;
	wire compute_clk_en;
	wire scratchpad_clk_en;
	wire ni_clk_en;
	wire compute_clk;
	wire cmd_valid;
	wire cmd_ready;
	reg [7:0] cmd_opcode;
	wire [31:0] cmd_arg0;
	wire [31:0] cmd_arg1;
	wire [31:0] cmd_arg2;
	reg cmd_issue;
	localparam signed [31:0] ACT_VEC_W = N_ROWS * DATA_W;
	localparam signed [31:0] WGT_VEC_W = N_COLS * DATA_W;
	localparam signed [31:0] ACT_VEC_WORDS = ACT_VEC_W / SP_DATA_W;
	localparam signed [31:0] WGT_VEC_WORDS = WGT_VEC_W / SP_DATA_W;
	localparam signed [31:0] RESULT_WORDS = N_ROWS * N_COLS;
	localparam signed [31:0] ROW_IDX_W = (N_ROWS <= 1 ? 1 : $clog2(N_ROWS));
	localparam signed [31:0] ACT_WORD_IDX_W = (ACT_VEC_WORDS <= 1 ? 1 : $clog2(ACT_VEC_WORDS));
	localparam signed [31:0] WGT_WORD_IDX_W = (WGT_VEC_WORDS <= 1 ? 1 : $clog2(WGT_VEC_WORDS));
	localparam signed [31:0] RESULT_WORD_IDX_W = (RESULT_WORDS <= 1 ? 1 : $clog2(RESULT_WORDS));
	localparam signed [31:0] WEIGHT_SETTLE_CYCLES = (N_ROWS > 0 ? N_ROWS - 1 : 0);
	localparam signed [31:0] WEIGHT_WAIT_W = (WEIGHT_SETTLE_CYCLES <= 1 ? 1 : $clog2(WEIGHT_SETTLE_CYCLES + 1));
	localparam signed [31:0] DRAIN_CYCLES = (2 * N_COLS) - 2;
	localparam signed [31:0] DRAIN_CNT_W = (DRAIN_CYCLES <= 1 ? 1 : $clog2(DRAIN_CYCLES + 1));
	reg [3:0] compute_state;
	reg [SP_ADDR_W - 1:0] compute_act_base;
	reg [SP_ADDR_W - 1:0] compute_wgt_base;
	reg [SP_ADDR_W - 1:0] compute_out_base;
	reg [ROW_IDX_W - 1:0] wgt_row_idx;
	reg [ROW_IDX_W - 1:0] act_vec_idx;
	reg [WGT_WORD_IDX_W - 1:0] wgt_word_idx;
	reg [ACT_WORD_IDX_W - 1:0] act_word_idx;
	reg [WEIGHT_WAIT_W - 1:0] weight_wait_ctr;
	reg [DRAIN_CNT_W - 1:0] drain_ctr;
	reg [RESULT_WORD_IDX_W - 1:0] store_word_idx;
	reg [WGT_VEC_W - 1:0] wgt_block_buf [0:N_ROWS - 1];
	reg [ACT_VEC_W - 1:0] act_block_buf [0:N_ROWS - 1];
	reg compute_sa_clr;
	reg compute_sa_load_weight;
	reg compute_sa_block_valid;
	reg [ACT_VEC_W - 1:0] compute_sa_a_vec;
	reg [WGT_VEC_W - 1:0] compute_sa_b_vec;
	wire [((N_ROWS * N_COLS) * ACC_W) - 1:0] systolic_out_flat;
	reg compute_sp_a_en;
	reg compute_sp_a_we;
	reg [SP_ADDR_W - 1:0] compute_sp_a_addr;
	reg [SP_DATA_W - 1:0] compute_sp_a_wdata;
	reg [31:0] arg0_reg;
	reg [31:0] arg1_reg;
	reg [31:0] arg2_reg;
	reg cmd_pending;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			arg0_reg <= '0;
			arg1_reg <= '0;
			arg2_reg <= '0;
			cmd_pending <= 1'b0;
			cmd_issue <= 1'b0;
			cmd_opcode <= '0;
		end
		else begin
			cmd_issue <= 1'b0;
			if (cmd_pending && cmd_ready) begin
				cmd_issue <= 1'b1;
				cmd_pending <= 1'b0;
			end
			if (csr_wen)
				case (csr_addr)
					8'h04: arg0_reg <= csr_wdata;
					8'h08: arg1_reg <= csr_wdata;
					8'h0c: arg2_reg <= csr_wdata;
					8'h00: begin
						cmd_opcode <= csr_wdata[7:0];
						cmd_pending <= 1'b1;
					end
					default:
						;
				endcase
		end
	assign cmd_valid = cmd_issue;
	assign cmd_arg0 = arg0_reg;
	assign cmd_arg1 = arg1_reg;
	assign cmd_arg2 = arg2_reg;
	always @(*) begin
		if (_sv2v_0)
			;
		case (csr_addr)
			8'h00: csr_rdata = {24'h000000, cmd_opcode};
			8'h04: csr_rdata = arg0_reg;
			8'h08: csr_rdata = arg1_reg;
			8'h0c: csr_rdata = arg2_reg;
			8'h10: csr_rdata = {30'h00000000, tile_done, tile_busy};
			8'h20: csr_rdata = {16'h0000, reduce_groups_completed};
			8'h24: csr_rdata = {16'h0000, reduce_packets_consumed};
			8'h28: csr_rdata = {24'h000000, reduce_last_id};
			8'h2c: csr_rdata = reduce_last_value;
			8'h30: csr_rdata = systolic_out_flat[0+:32];
			8'h34: csr_rdata = systolic_out_flat[1 * ACC_W+:32];
			8'h38: csr_rdata = systolic_out_flat[2 * ACC_W+:32];
			8'h3c: csr_rdata = systolic_out_flat[3 * ACC_W+:32];
			default: csr_rdata = '0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			reduce_last_id <= '0;
			reduce_last_value <= '0;
			reduce_result_valid <= '0;
			begin : sv2v_autoblock_1
				reg signed [31:0] rid;
				for (rid = 0; rid < 256; rid = rid + 1)
					reduce_results[rid] <= '0;
			end
		end
		else if (reduce_commit_valid) begin
			reduce_results[reduce_commit_id] <= reduce_commit_value;
			reduce_result_valid[reduce_commit_id] <= 1'b1;
			reduce_last_id <= reduce_commit_id;
			reduce_last_value <= reduce_commit_value;
		end
	tile_controller #(
		.TILE_ID(TILE_ID),
		.SP_ADDR_W(SP_ADDR_W),
		.SP_DATA_W(SP_DATA_W),
		.N_ROWS(N_ROWS),
		.N_COLS(N_COLS)
	) u_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.cmd_valid(cmd_valid),
		.cmd_ready(cmd_ready),
		.cmd_opcode(cmd_opcode),
		.cmd_arg0(cmd_arg0),
		.cmd_arg1(cmd_arg1),
		.cmd_arg2(cmd_arg2),
		.busy(tile_busy),
		.done(tile_done),
		.error(),
		.sparse_hint(sparse_hint),
		.sp_a_en(sp_a_en),
		.sp_a_we(sp_a_we),
		.sp_a_addr(sp_a_addr),
		.sp_a_wdata(sp_a_wdata),
		.sp_a_rdata(sp_a_rdata),
		.sa_start(sa_start),
		.sa_load_weight(ctrl_sa_load_weight),
		.sa_pe_en(ctrl_sa_pe_en),
		.sa_accum_en(ctrl_sa_accum_en),
		.sa_done(sa_done),
		.dma_req_valid(dma_req_valid),
		.dma_req_ready(dma_req_ready),
		.dma_req_write(dma_req_write),
		.dma_req_addr(dma_req_addr),
		.dma_req_len(dma_req_len),
		.dma_data_valid(dma_data_valid),
		.dma_data_ready(dma_data_ready),
		.dma_data_in(dma_data_in),
		.dma_wdata_valid(dma_wdata_valid),
		.dma_wdata_ready(dma_wdata_ready),
		.dma_wdata_out(dma_wdata_out),
		.dma_wdata_last(dma_wdata_last),
		.dma_done_valid(dma_store_done),
		.barrier_req(barrier_req),
		.barrier_done(barrier_done),
		.reduce_inj_valid(reduce_inj_valid),
		.reduce_inj_id(reduce_inj_id),
		.reduce_inj_expect(reduce_inj_expect),
		.reduce_inj_dst(reduce_inj_dst),
		.reduce_inj_val(reduce_inj_val),
		.reduce_inj_ready(reduce_inj_ready)
	);
	assign compute_clk_en = sa_start || (compute_state != 4'd0);
	assign scratchpad_clk_en = sp_a_mux_en || sp_b_en;
	assign ni_clk_en = (tile_busy || noc_valid_in) || noc_valid_out;
	clock_gate_cell u_compute_clk_gate(
		.clk_i(clk),
		.en_i(compute_clk_en),
		.test_en_i(1'b0),
		.clk_o(compute_clk)
	);
	accel_scratchpad #(
		.DEPTH(SP_DEPTH),
		.DATA_WIDTH(SP_DATA_W),
		.ADDR_WIDTH(SP_ADDR_W)
	) u_sp(
		.clk(clk),
		.rst_n(rst_n),
		.clk_en(scratchpad_clk_en),
		.a_en(sp_a_mux_en),
		.a_we(sp_a_mux_we),
		.a_addr(sp_a_mux_addr),
		.a_wdata(sp_a_mux_wdata),
		.a_rdata(sp_a_rdata),
		.b_en(sp_b_en),
		.b_addr(sp_b_addr),
		.b_rdata(sp_b_rdata)
	);
	assign sp_a_mux_en = (compute_sp_a_en ? 1'b1 : sp_a_en);
	assign sp_a_mux_we = (compute_sp_a_en ? 1'b1 : sp_a_we);
	assign sp_a_mux_addr = (compute_sp_a_en ? compute_sp_a_addr : sp_a_addr);
	assign sp_a_mux_wdata = (compute_sp_a_en ? compute_sp_a_wdata : sp_a_wdata);
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			compute_state <= 4'd0;
			compute_act_base <= '0;
			compute_wgt_base <= '0;
			compute_out_base <= '0;
			wgt_row_idx <= '0;
			act_vec_idx <= '0;
			wgt_word_idx <= '0;
			act_word_idx <= '0;
			weight_wait_ctr <= '0;
			drain_ctr <= '0;
			store_word_idx <= '0;
			begin : sv2v_autoblock_2
				reg signed [31:0] row;
				for (row = 0; row < N_ROWS; row = row + 1)
					begin
						wgt_block_buf[row] <= '0;
						act_block_buf[row] <= '0;
					end
			end
		end
		else if (compute_clk_en)
			case (compute_state)
				4'd0:
					if (sa_start) begin
						compute_act_base <= cmd_arg0[SP_ADDR_W - 1:0];
						compute_wgt_base <= cmd_arg1[SP_ADDR_W - 1:0];
						compute_out_base <= cmd_arg2[SP_ADDR_W - 1:0];
						wgt_row_idx <= '0;
						act_vec_idx <= '0;
						wgt_word_idx <= '0;
						act_word_idx <= '0;
						weight_wait_ctr <= '0;
						drain_ctr <= '0;
						store_word_idx <= '0;
						compute_state <= 4'd1;
					end
				4'd1: compute_state <= 4'd2;
				4'd2: begin
					wgt_block_buf[wgt_row_idx][wgt_word_idx * SP_DATA_W+:SP_DATA_W] <= sp_b_rdata;
					if (wgt_word_idx == (WGT_VEC_WORDS - 1)) begin
						wgt_word_idx <= '0;
						if (wgt_row_idx == (N_ROWS - 1)) begin
							wgt_row_idx <= '0;
							compute_state <= 4'd3;
						end
						else begin
							wgt_row_idx <= wgt_row_idx + 1'b1;
							compute_state <= 4'd1;
						end
					end
					else begin
						wgt_word_idx <= wgt_word_idx + 1'b1;
						compute_state <= 4'd1;
					end
				end
				4'd3: compute_state <= 4'd4;
				4'd4: begin
					act_block_buf[act_vec_idx][act_word_idx * SP_DATA_W+:SP_DATA_W] <= sp_b_rdata;
					if (act_word_idx == (ACT_VEC_WORDS - 1)) begin
						act_word_idx <= '0;
						if (act_vec_idx == (N_ROWS - 1)) begin
							act_vec_idx <= '0;
							weight_wait_ctr <= '0;
							compute_state <= 4'd5;
						end
						else begin
							act_vec_idx <= act_vec_idx + 1'b1;
							compute_state <= 4'd3;
						end
					end
					else begin
						act_word_idx <= act_word_idx + 1'b1;
						compute_state <= 4'd3;
					end
				end
				4'd5: begin
					wgt_row_idx <= '0;
					compute_state <= 4'd6;
				end
				4'd6:
					if (wgt_row_idx == (N_ROWS - 1)) begin
						wgt_row_idx <= '0;
						if (WEIGHT_SETTLE_CYCLES == 0) begin
							act_vec_idx <= '0;
							compute_state <= 4'd8;
						end
						else begin
							weight_wait_ctr <= '0;
							compute_state <= 4'd7;
						end
					end
					else
						wgt_row_idx <= wgt_row_idx + 1'b1;
				4'd7:
					if (weight_wait_ctr == (WEIGHT_SETTLE_CYCLES - 1)) begin
						weight_wait_ctr <= '0;
						act_vec_idx <= '0;
						compute_state <= 4'd8;
					end
					else
						weight_wait_ctr <= weight_wait_ctr + 1'b1;
				4'd8:
					if (act_vec_idx == (N_ROWS - 1)) begin
						act_vec_idx <= '0;
						if (DRAIN_CYCLES == 0) begin
							store_word_idx <= '0;
							compute_state <= 4'd10;
						end
						else begin
							drain_ctr <= '0;
							compute_state <= 4'd9;
						end
					end
					else
						act_vec_idx <= act_vec_idx + 1'b1;
				4'd9:
					if (drain_ctr == (DRAIN_CYCLES - 1)) begin
						drain_ctr <= '0;
						store_word_idx <= '0;
						compute_state <= 4'd10;
					end
					else
						drain_ctr <= drain_ctr + 1'b1;
				4'd10:
					if (store_word_idx == (RESULT_WORDS - 1))
						compute_state <= 4'd11;
					else
						store_word_idx <= store_word_idx + 1'b1;
				4'd11: compute_state <= 4'd0;
				default: compute_state <= 4'd0;
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		compute_sa_clr = 1'b0;
		compute_sa_load_weight = 1'b0;
		compute_sa_block_valid = 1'b0;
		compute_sa_a_vec = '0;
		compute_sa_b_vec = '0;
		compute_sp_a_en = 1'b0;
		compute_sp_a_we = 1'b0;
		compute_sp_a_addr = '0;
		compute_sp_a_wdata = '0;
		sp_b_en = 1'b0;
		sp_b_addr = '0;
		case (compute_state)
			4'd5: compute_sa_clr = 1'b1;
			4'd1: begin
				sp_b_en = 1'b1;
				sp_b_addr = (compute_wgt_base + (wgt_row_idx * WGT_VEC_WORDS)) + wgt_word_idx;
			end
			4'd3: begin
				sp_b_en = 1'b1;
				sp_b_addr = (compute_act_base + (act_vec_idx * ACT_VEC_WORDS)) + act_word_idx;
			end
			4'd6: begin
				compute_sa_load_weight = 1'b1;
				compute_sa_b_vec = wgt_block_buf[wgt_row_idx];
			end
			4'd8: begin
				compute_sa_block_valid = ctrl_sa_pe_en && ctrl_sa_accum_en;
				compute_sa_a_vec = act_block_buf[act_vec_idx];
			end
			4'd9: compute_sa_block_valid = ctrl_sa_pe_en && ctrl_sa_accum_en;
			4'd10: begin
				compute_sp_a_en = 1'b1;
				compute_sp_a_we = 1'b1;
				compute_sp_a_addr = compute_out_base + store_word_idx;
				compute_sp_a_wdata = systolic_out_flat[store_word_idx * ACC_W+:SP_DATA_W];
			end
			default:
				;
		endcase
	end
	always @(posedge clk)
		if (compute_sp_a_en && sp_a_en)
;
	systolic_array_sparse #(
		.N_ROWS(N_ROWS),
		.N_COLS(N_COLS),
		.DATA_W(DATA_W),
		.ACC_W(ACC_W)
	) u_systolic(
		.clk(compute_clk),
		.rst_n(rst_n),
		.clk_en(1'b1),
		.block_valid(compute_sa_block_valid),
		.load_weight(compute_sa_load_weight),
		.clr(compute_sa_clr),
		.a_in_flat(compute_sa_a_vec),
		.b_in_flat(compute_sa_b_vec),
		.c_out_flat(systolic_out_flat)
	);
	assign sa_done = compute_state == 4'd11;
	assign dma_req_ready = (dma_req_write ? dma_req_aw_ready : dma_req_ar_ready);
	assign dma_data_in = dma_data_word;
	wire _unused_ctrl_compute = &{1'b0, ctrl_sa_load_weight};
	assign reduce_local_valid = (noc_valid_in && (noc_flit_in[51-:4] == 4'h6)) && (noc_flit_in[63-:2] == 2'b11);
	localparam signed [31:0] noc_pkg_INNET_SP_DEPTH = 8;
	tile_reduce_consumer #(
		.NUM_VCS(NUM_VCS),
		.ENTRY_DEPTH(noc_pkg_INNET_SP_DEPTH)
	) u_reduce_sink(
		.clk(clk),
		.rst_n(rst_n),
		.enable(ni_clk_en),
		.flit_in(noc_flit_in),
		.valid_in(reduce_local_valid),
		.credit_out(reduce_credit_out),
		.commit_valid(reduce_commit_valid),
		.commit_id(reduce_commit_id),
		.commit_value(reduce_commit_value),
		.packets_consumed(reduce_packets_consumed),
		.groups_completed(reduce_groups_completed)
	);
	function automatic signed [3:0] sv2v_cast_4_signed;
		input reg signed [3:0] inp;
		sv2v_cast_4_signed = inp;
	endfunction
	noc_network_interface #(
		.NODE_ID(TILE_ID),
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.NUM_VCS(NUM_VCS)
	) u_ni(
		.clk(clk),
		.rst_n(rst_n),
		.clk_en(ni_clk_en),
		.aw_valid(dma_req_valid && dma_req_write),
		.aw_ready(dma_req_aw_ready),
		.aw_addr(dma_req_addr),
		.aw_len(dma_req_len[7:0]),
		.aw_id(sv2v_cast_4_signed(TILE_ID)),
		.w_valid(dma_wdata_valid),
		.w_ready(dma_wdata_ready),
		.w_data(dma_wdata_out[31:0]),
		.w_last(dma_wdata_last),
		.b_valid(dma_store_done),
		.b_ready(1'b1),
		.b_id(),
		.b_resp(),
		.ar_valid(dma_req_valid && !dma_req_write),
		.ar_ready(dma_req_ar_ready),
		.ar_addr(dma_req_addr),
		.ar_id(sv2v_cast_4_signed(TILE_ID)),
		.ar_len(dma_req_len[7:0]),
		.r_valid(dma_data_valid),
		.r_ready(dma_data_ready),
		.r_data(dma_data_word),
		.r_id(),
		.r_resp(),
		.r_last(),
		.sparse_hint(sparse_hint),
		.reduce_inj_valid(reduce_inj_valid),
		.reduce_inj_ready(reduce_inj_ready),
		.reduce_inj_id(reduce_inj_id),
		.reduce_inj_expect(reduce_inj_expect),
		.reduce_inj_dst(reduce_inj_dst),
		.reduce_inj_val(reduce_inj_val),
		.noc_flit_out(noc_flit_out),
		.noc_valid_out(noc_valid_out),
		.noc_credit_in(noc_credit_in),
		.noc_flit_in(noc_flit_in),
		.noc_valid_in(noc_valid_in && !reduce_local_valid),
		.noc_credit_out(ni_credit_out_int)
	);
	assign noc_credit_out = ni_credit_out_int | reduce_credit_out;
	initial _sv2v_0 = 0;
endmodule
