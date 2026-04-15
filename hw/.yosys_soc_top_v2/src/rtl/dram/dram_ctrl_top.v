module dram_ctrl_top (
	clk,
	rst_n,
	s_axi_awvalid,
	s_axi_awready,
	s_axi_awaddr,
	s_axi_awid,
	s_axi_awlen,
	s_axi_awsize,
	s_axi_wvalid,
	s_axi_wready,
	s_axi_wdata,
	s_axi_wstrb,
	s_axi_wlast,
	s_axi_bvalid,
	s_axi_bready,
	s_axi_bresp,
	s_axi_bid,
	s_axi_arvalid,
	s_axi_arready,
	s_axi_araddr,
	s_axi_arid,
	s_axi_arlen,
	s_axi_arsize,
	s_axi_rvalid,
	s_axi_rready,
	s_axi_rdata,
	s_axi_rresp,
	s_axi_rid,
	s_axi_rlast,
	dram_phy_act,
	dram_phy_read,
	dram_phy_write,
	dram_phy_pre,
	dram_phy_row,
	dram_phy_col,
	dram_phy_ref,
	dram_phy_wdata,
	dram_phy_wstrb,
	dram_phy_rdata,
	dram_phy_rdata_valid,
	ctrl_busy
);
	reg _sv2v_0;
	parameter signed [31:0] AXI_ADDR_W = 32;
	parameter signed [31:0] AXI_DATA_W = 32;
	parameter signed [31:0] AXI_ID_W = 4;
	parameter signed [31:0] NUM_BANKS = 8;
	parameter signed [31:0] ROW_BITS = 14;
	parameter signed [31:0] COL_BITS = 10;
	parameter signed [31:0] BANK_BITS = 3;
	parameter signed [31:0] QUEUE_DEPTH = 16;
	parameter signed [31:0] ADDR_MODE = 0;
	parameter signed [31:0] T_RCD = 3;
	parameter signed [31:0] T_RP = 3;
	parameter signed [31:0] T_RAS = 7;
	parameter signed [31:0] T_RC = 10;
	parameter signed [31:0] T_RTP = 2;
	parameter signed [31:0] T_WR = 3;
	parameter signed [31:0] T_CAS = 3;
	parameter signed [31:0] T_REFI = 1560;
	parameter signed [31:0] T_RFC = 52;
	input wire clk;
	input wire rst_n;
	input wire s_axi_awvalid;
	output reg s_axi_awready;
	input wire [AXI_ADDR_W - 1:0] s_axi_awaddr;
	input wire [AXI_ID_W - 1:0] s_axi_awid;
	input wire [7:0] s_axi_awlen;
	input wire [2:0] s_axi_awsize;
	input wire s_axi_wvalid;
	output reg s_axi_wready;
	input wire [AXI_DATA_W - 1:0] s_axi_wdata;
	input wire [(AXI_DATA_W / 8) - 1:0] s_axi_wstrb;
	input wire s_axi_wlast;
	output reg s_axi_bvalid;
	input wire s_axi_bready;
	output reg [1:0] s_axi_bresp;
	output reg [AXI_ID_W - 1:0] s_axi_bid;
	input wire s_axi_arvalid;
	output reg s_axi_arready;
	input wire [AXI_ADDR_W - 1:0] s_axi_araddr;
	input wire [AXI_ID_W - 1:0] s_axi_arid;
	input wire [7:0] s_axi_arlen;
	input wire [2:0] s_axi_arsize;
	output wire s_axi_rvalid;
	input wire s_axi_rready;
	output wire [AXI_DATA_W - 1:0] s_axi_rdata;
	output wire [1:0] s_axi_rresp;
	output wire [AXI_ID_W - 1:0] s_axi_rid;
	output wire s_axi_rlast;
	output wire [NUM_BANKS - 1:0] dram_phy_act;
	output wire [NUM_BANKS - 1:0] dram_phy_read;
	output wire [NUM_BANKS - 1:0] dram_phy_write;
	output wire [NUM_BANKS - 1:0] dram_phy_pre;
	output reg [ROW_BITS - 1:0] dram_phy_row;
	output reg [COL_BITS - 1:0] dram_phy_col;
	output wire dram_phy_ref;
	output wire [AXI_DATA_W - 1:0] dram_phy_wdata;
	output wire [(AXI_DATA_W / 8) - 1:0] dram_phy_wstrb;
	input wire [AXI_DATA_W - 1:0] dram_phy_rdata;
	input wire dram_phy_rdata_valid;
	output wire ctrl_busy;
	localparam signed [31:0] DRAM_ADDR_W = ((BANK_BITS + ROW_BITS) + COL_BITS) + 2;
	localparam signed [31:0] BLEN_W = 4;
	localparam signed [31:0] QIX_W = $clog2(QUEUE_DEPTH);
	wire [BANK_BITS - 1:0] dec_bank;
	wire [ROW_BITS - 1:0] dec_row;
	wire [COL_BITS - 1:0] dec_col;
	dram_addr_decoder #(
		.AXI_ADDR_W(AXI_ADDR_W),
		.BANK_BITS(BANK_BITS),
		.ROW_BITS(ROW_BITS),
		.COL_BITS(COL_BITS),
		.BUS_BYTES(AXI_DATA_W / 8),
		.MODE(ADDR_MODE)
	) u_addr_dec(
		.axi_addr((s_axi_arvalid ? s_axi_araddr : s_axi_awaddr)),
		.bank(dec_bank),
		.row(dec_row),
		.col(dec_col)
	);
	reg enq_valid;
	wire enq_ready;
	reg enq_rw;
	reg [DRAM_ADDR_W - 1:0] enq_addr;
	reg [AXI_ID_W - 1:0] enq_id;
	reg [3:0] enq_blen;
	wire deq_valid;
	wire [QIX_W - 1:0] deq_idx;
	wire deq_ready;
	wire [QUEUE_DEPTH - 1:0] q_entry_valid;
	wire [QUEUE_DEPTH - 1:0] q_entry_rw;
	wire [(QUEUE_DEPTH * DRAM_ADDR_W) - 1:0] q_entry_addr;
	wire [(QUEUE_DEPTH * AXI_ID_W) - 1:0] q_entry_id;
	wire [(QUEUE_DEPTH * BLEN_W) - 1:0] q_entry_blen;
	wire [(QUEUE_DEPTH * 8) - 1:0] q_entry_age;
	wire [QIX_W:0] q_count;
	wire q_empty;
	wire q_full;
	dram_cmd_queue #(
		.DEPTH(QUEUE_DEPTH),
		.ADDR_W(DRAM_ADDR_W),
		.ID_W(AXI_ID_W),
		.BLEN_W(BLEN_W)
	) u_cmd_queue(
		.clk(clk),
		.rst_n(rst_n),
		.enq_valid(enq_valid),
		.enq_ready(enq_ready),
		.enq_rw(enq_rw),
		.enq_addr(enq_addr),
		.enq_id(enq_id),
		.enq_blen(enq_blen),
		.deq_valid(deq_valid),
		.deq_idx(deq_idx),
		.deq_ready(deq_ready),
		.count(q_count),
		.empty(q_empty),
		.full(q_full),
		.entry_valid(q_entry_valid),
		.entry_rw(q_entry_rw),
		.entry_addr(q_entry_addr),
		.entry_id(q_entry_id),
		.entry_blen(q_entry_blen),
		.entry_age(q_entry_age)
	);
	reg wb_wr_valid;
	wire wb_wr_ready;
	reg [AXI_DATA_W - 1:0] wb_wr_data;
	reg [(AXI_DATA_W / 8) - 1:0] wb_wr_strb;
	wire wb_drain_valid;
	wire wb_drain_ready;
	wire [QIX_W - 1:0] wb_drain_idx;
	wire [AXI_DATA_W - 1:0] wb_drain_data;
	wire [(AXI_DATA_W / 8) - 1:0] wb_drain_strb;
	dram_write_buffer #(
		.DEPTH(QUEUE_DEPTH),
		.DATA_W(AXI_DATA_W),
		.ID_W(AXI_ID_W)
	) u_write_buf(
		.clk(clk),
		.rst_n(rst_n),
		.wr_valid(wb_wr_valid),
		.wr_ready(wb_wr_ready),
		.wr_data(wb_wr_data),
		.wr_strb(wb_wr_strb),
		.wr_id(s_axi_awid),
		.drain_valid(wb_drain_valid),
		.drain_idx(wb_drain_idx),
		.drain_ready(wb_drain_ready),
		.drain_data(wb_drain_data),
		.drain_strb(wb_drain_strb),
		.count(),
		.empty(),
		.full()
	);
	wire [(NUM_BANKS * 3) - 1:0] bk_state;
	wire [(NUM_BANKS * ROW_BITS) - 1:0] bk_open_row;
	wire [NUM_BANKS - 1:0] bk_row_open;
	wire [NUM_BANKS - 1:0] bk_cmd_valid;
	wire [(NUM_BANKS * 3) - 1:0] bk_cmd_op;
	wire [(NUM_BANKS * ROW_BITS) - 1:0] bk_cmd_row;
	wire [(NUM_BANKS * COL_BITS) - 1:0] bk_cmd_col;
	genvar _gv_bi_1;
	generate
		for (_gv_bi_1 = 0; _gv_bi_1 < NUM_BANKS; _gv_bi_1 = _gv_bi_1 + 1) begin : gen_bank
			localparam bi = _gv_bi_1;
			dram_bank_fsm #(
				.ROW_BITS(ROW_BITS),
				.COL_BITS(COL_BITS),
				.T_RCD(T_RCD),
				.T_RP(T_RP),
				.T_RAS(T_RAS),
				.T_RC(T_RC),
				.T_RTP(T_RTP),
				.T_WR(T_WR),
				.T_CAS(T_CAS)
			) u_bank(
				.clk(clk),
				.rst_n(rst_n),
				.cmd_valid(bk_cmd_valid[bi]),
				.cmd_op(bk_cmd_op[bi * 3+:3]),
				.cmd_row(bk_cmd_row[bi * ROW_BITS+:ROW_BITS]),
				.cmd_col(bk_cmd_col[bi * COL_BITS+:COL_BITS]),
				.cmd_ready(),
				.bank_state(bk_state[bi * 3+:3]),
				.open_row(bk_open_row[bi * ROW_BITS+:ROW_BITS]),
				.row_open(bk_row_open[bi]),
				.row_hit(),
				.phy_act(dram_phy_act[bi]),
				.phy_read(dram_phy_read[bi]),
				.phy_write(dram_phy_write[bi]),
				.phy_pre(dram_phy_pre[bi]),
				.phy_row(),
				.phy_col()
			);
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		dram_phy_row = '0;
		dram_phy_col = '0;
		begin : sv2v_autoblock_1
			reg signed [31:0] b;
			for (b = 0; b < NUM_BANKS; b = b + 1)
				begin
					if (dram_phy_act[b])
						dram_phy_row = bk_cmd_row[b * ROW_BITS+:ROW_BITS];
					if (dram_phy_read[b] || dram_phy_write[b])
						dram_phy_col = bk_cmd_col[b * COL_BITS+:COL_BITS];
				end
		end
	end
	wire ref_req;
	wire ref_ack;
	wire ref_cmd;
	wire ref_busy;
	dram_refresh_ctrl #(
		.T_REFI(T_REFI),
		.T_RFC(T_RFC)
	) u_refresh(
		.clk(clk),
		.rst_n(rst_n),
		.ref_req(ref_req),
		.ref_ack(ref_ack),
		.ref_cmd(ref_cmd),
		.ref_busy(ref_busy)
	);
	assign dram_phy_ref = ref_cmd;
	wire sched_data_rd;
	wire sched_data_wr;
	wire [AXI_ID_W - 1:0] sched_data_id;
	wire sched_busy;
	dram_scheduler_frfcfs #(
		.NUM_BANKS(NUM_BANKS),
		.QUEUE_DEPTH(QUEUE_DEPTH),
		.ADDR_W(DRAM_ADDR_W),
		.ROW_BITS(ROW_BITS),
		.COL_BITS(COL_BITS),
		.BANK_BITS(BANK_BITS),
		.ID_W(AXI_ID_W),
		.BLEN_W(BLEN_W)
	) u_scheduler(
		.clk(clk),
		.rst_n(rst_n),
		.entry_valid(q_entry_valid),
		.entry_rw(q_entry_rw),
		.entry_addr(q_entry_addr),
		.entry_id(q_entry_id),
		.entry_blen(q_entry_blen),
		.entry_age(q_entry_age),
		.deq_valid(deq_valid),
		.deq_idx(deq_idx),
		.bank_state(bk_state),
		.bank_open_row(bk_open_row),
		.bank_row_open(bk_row_open),
		.bank_cmd_valid(bk_cmd_valid),
		.bank_cmd_op(bk_cmd_op),
		.bank_cmd_row(bk_cmd_row),
		.bank_cmd_col(bk_cmd_col),
		.ref_req(ref_req),
		.ref_ack(ref_ack),
		.ref_busy(ref_busy),
		.data_rd_valid(sched_data_rd),
		.data_wr_valid(sched_data_wr),
		.data_id(sched_data_id),
		.sched_busy(sched_busy)
	);
	reg [1:0] axi_st;
	reg [1:0] axi_st_next;
	reg [AXI_ID_W - 1:0] aw_id_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			axi_st <= 2'd0;
			aw_id_r <= '0;
		end
		else begin
			axi_st <= axi_st_next;
			if (s_axi_awvalid && s_axi_awready)
				aw_id_r <= s_axi_awid;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		axi_st_next = axi_st;
		s_axi_awready = 1'b0;
		s_axi_arready = 1'b0;
		s_axi_wready = 1'b0;
		s_axi_bvalid = 1'b0;
		s_axi_bresp = 2'b00;
		s_axi_bid = '0;
		enq_valid = 1'b0;
		enq_rw = 1'b0;
		enq_addr = '0;
		enq_id = '0;
		enq_blen = '0;
		wb_wr_valid = 1'b0;
		wb_wr_data = '0;
		wb_wr_strb = '0;
		case (axi_st)
			2'd0:
				if (s_axi_arvalid && enq_ready) begin
					s_axi_arready = 1'b1;
					enq_valid = 1'b1;
					enq_rw = 1'b0;
					enq_addr = s_axi_araddr[DRAM_ADDR_W - 1:0];
					enq_id = s_axi_arid;
					enq_blen = s_axi_arlen[3:0];
				end
				else if (s_axi_awvalid && enq_ready) begin
					s_axi_awready = 1'b1;
					enq_valid = 1'b1;
					enq_rw = 1'b1;
					enq_addr = s_axi_awaddr[DRAM_ADDR_W - 1:0];
					enq_id = s_axi_awid;
					enq_blen = s_axi_awlen[3:0];
					axi_st_next = 2'd1;
				end
			2'd1: begin
				s_axi_wready = wb_wr_ready;
				wb_wr_valid = s_axi_wvalid;
				wb_wr_data = s_axi_wdata;
				wb_wr_strb = s_axi_wstrb;
				if ((s_axi_wvalid && s_axi_wlast) && wb_wr_ready)
					axi_st_next = 2'd2;
			end
			2'd2:
				if (sched_data_wr && wb_drain_ready)
					axi_st_next = 2'd3;
			2'd3: begin
				s_axi_bvalid = 1'b1;
				s_axi_bresp = 2'b00;
				s_axi_bid = aw_id_r;
				if (s_axi_bready)
					axi_st_next = 2'd0;
			end
			default: axi_st_next = 2'd0;
		endcase
	end
	assign s_axi_rvalid = dram_phy_rdata_valid;
	assign s_axi_rdata = dram_phy_rdata;
	assign s_axi_rresp = 2'b00;
	assign s_axi_rid = sched_data_id;
	assign s_axi_rlast = dram_phy_rdata_valid;
	assign wb_drain_valid = sched_data_wr;
	assign wb_drain_idx = deq_idx;
	assign dram_phy_wdata = wb_drain_data;
	assign dram_phy_wstrb = wb_drain_strb;
	assign ctrl_busy = sched_busy || !q_empty;
	initial _sv2v_0 = 0;
endmodule
