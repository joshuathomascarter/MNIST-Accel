module accel_tile_array (
	clk,
	rst_n,
	inr_meta_cfg_valid,
	inr_meta_cfg_reduce_id,
	inr_meta_cfg_target,
	inr_meta_cfg_enable,
	s_axi_awaddr,
	s_axi_awvalid,
	s_axi_awready,
	s_axi_wdata,
	s_axi_wstrb,
	s_axi_wvalid,
	s_axi_wready,
	s_axi_bresp,
	s_axi_bvalid,
	s_axi_bready,
	s_axi_araddr,
	s_axi_arvalid,
	s_axi_arready,
	s_axi_rdata,
	s_axi_rresp,
	s_axi_rvalid,
	s_axi_rready,
	m_axi_arid,
	m_axi_araddr,
	m_axi_arlen,
	m_axi_arsize,
	m_axi_arburst,
	m_axi_arvalid,
	m_axi_arready,
	m_axi_rid,
	m_axi_rdata,
	m_axi_rresp,
	m_axi_rlast,
	m_axi_rvalid,
	m_axi_rready,
	m_axi_awid,
	m_axi_awaddr,
	m_axi_awlen,
	m_axi_awsize,
	m_axi_awburst,
	m_axi_awvalid,
	m_axi_awready,
	m_axi_wdata,
	m_axi_wstrb,
	m_axi_wlast,
	m_axi_wvalid,
	m_axi_wready,
	m_axi_bid,
	m_axi_bresp,
	m_axi_bvalid,
	m_axi_bready,
	tile_busy_o,
	tile_done_o
);
	reg _sv2v_0;
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	parameter signed [31:0] MESH_ROWS = noc_pkg_MESH_ROWS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	parameter signed [31:0] MESH_COLS = noc_pkg_MESH_COLS;
	parameter signed [31:0] N_ROWS = 16;
	parameter signed [31:0] N_COLS = 16;
	parameter signed [31:0] DATA_W = 8;
	parameter signed [31:0] ACC_W = 32;
	parameter signed [31:0] SP_DEPTH = 4096;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	parameter [0:0] SPARSE_VC_ALLOC = 1'b0;
	parameter [0:0] INNET_REDUCE = 1'b0;
	parameter signed [31:0] AXI_ADDR_W = 32;
	parameter signed [31:0] AXI_DATA_W = 32;
	parameter signed [31:0] AXI_ID_W = 4;
	input wire clk;
	input wire rst_n;
	input wire [0:(MESH_ROWS * MESH_COLS) - 1] inr_meta_cfg_valid;
	input wire [((MESH_ROWS * MESH_COLS) * 8) - 1:0] inr_meta_cfg_reduce_id;
	input wire [((MESH_ROWS * MESH_COLS) * 4) - 1:0] inr_meta_cfg_target;
	input wire [0:(MESH_ROWS * MESH_COLS) - 1] inr_meta_cfg_enable;
	input wire [AXI_ADDR_W - 1:0] s_axi_awaddr;
	input wire s_axi_awvalid;
	output wire s_axi_awready;
	input wire [AXI_DATA_W - 1:0] s_axi_wdata;
	input wire [3:0] s_axi_wstrb;
	input wire s_axi_wvalid;
	output wire s_axi_wready;
	output wire [1:0] s_axi_bresp;
	output wire s_axi_bvalid;
	input wire s_axi_bready;
	input wire [AXI_ADDR_W - 1:0] s_axi_araddr;
	input wire s_axi_arvalid;
	output wire s_axi_arready;
	output wire [AXI_DATA_W - 1:0] s_axi_rdata;
	output wire [1:0] s_axi_rresp;
	output wire s_axi_rvalid;
	input wire s_axi_rready;
	output wire [AXI_ID_W - 1:0] m_axi_arid;
	output wire [AXI_ADDR_W - 1:0] m_axi_araddr;
	output wire [7:0] m_axi_arlen;
	output wire [2:0] m_axi_arsize;
	output wire [1:0] m_axi_arburst;
	output wire m_axi_arvalid;
	input wire m_axi_arready;
	input wire [AXI_ID_W - 1:0] m_axi_rid;
	input wire [AXI_DATA_W - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire m_axi_rlast;
	input wire m_axi_rvalid;
	output wire m_axi_rready;
	output wire [AXI_ID_W - 1:0] m_axi_awid;
	output wire [AXI_ADDR_W - 1:0] m_axi_awaddr;
	output wire [7:0] m_axi_awlen;
	output wire [2:0] m_axi_awsize;
	output wire [1:0] m_axi_awburst;
	output wire m_axi_awvalid;
	input wire m_axi_awready;
	output wire [AXI_DATA_W - 1:0] m_axi_wdata;
	output wire [(AXI_DATA_W / 8) - 1:0] m_axi_wstrb;
	output wire m_axi_wlast;
	output wire m_axi_wvalid;
	input wire m_axi_wready;
	input wire [AXI_ID_W - 1:0] m_axi_bid;
	input wire [1:0] m_axi_bresp;
	input wire m_axi_bvalid;
	output wire m_axi_bready;
	output wire [(MESH_ROWS * MESH_COLS) - 1:0] tile_busy_o;
	output wire [(MESH_ROWS * MESH_COLS) - 1:0] tile_done_o;
	localparam signed [31:0] NUM_TILES = MESH_ROWS * MESH_COLS;
	wire [63:0] tile_flit_out [0:NUM_TILES - 1];
	wire tile_valid_out [0:NUM_TILES - 1];
	wire [(NUM_TILES * NUM_VCS) - 1:0] tile_credit_in;
	wire [(NUM_TILES * 64) - 1:0] tile_flit_in;
	wire [0:NUM_TILES - 1] tile_valid_in;
	wire [NUM_VCS - 1:0] tile_credit_out [0:NUM_TILES - 1];
	wire [NUM_TILES - 1:0] tile_barrier_req;
	wire barrier_release;
	wire [3:0] csr_tile_sel;
	wire csr_is_array_reg;
	wire [31:0] tile_csr_wdata;
	wire [7:0] tile_csr_addr;
	reg tile_csr_wen [0:NUM_TILES - 1];
	wire [31:0] tile_csr_rdata [0:NUM_TILES - 1];
	reg [31:0] array_csr_rdata;
	reg [3:0] noc_meta_router_reg;
	reg [7:0] noc_meta_reduce_id_reg;
	reg [3:0] noc_meta_target_reg;
	reg noc_meta_enable_reg;
	reg noc_meta_apply_pulse;
	reg [0:NUM_TILES - 1] mesh_inr_meta_cfg_valid;
	reg [(NUM_TILES * 8) - 1:0] mesh_inr_meta_cfg_reduce_id;
	reg [(NUM_TILES * 4) - 1:0] mesh_inr_meta_cfg_target;
	reg [0:NUM_TILES - 1] mesh_inr_meta_cfg_enable;
	localparam [7:0] NOC_META_ROUTER_OFF = 8'h80;
	localparam [7:0] NOC_META_REDUCE_ID_OFF = 8'h84;
	localparam [7:0] NOC_META_TARGET_OFF = 8'h88;
	localparam [7:0] NOC_META_CTRL_OFF = 8'h8c;
	localparam [7:0] NOC_META_STATUS_OFF = 8'h90;
	reg [7:0] csr_addr_r;
	reg [31:0] csr_wdata_r;
	reg csr_write_pulse;
	assign csr_tile_sel = (s_axi_awvalid ? s_axi_awaddr[15:12] : (s_axi_arvalid ? s_axi_araddr[15:12] : '0));
	assign tile_csr_addr = csr_addr_r;
	assign csr_is_array_reg = csr_addr_r[7];
	wire [NUM_TILES - 1:0] tile_busy;
	wire [NUM_TILES - 1:0] tile_done;
	reg [1:0] axi_s_state;
	reg [31:0] axi_s_rdata_reg;
	reg axi_s_write_done;
	reg [3:0] axi_s_tile;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			axi_s_state <= 2'd0;
			axi_s_rdata_reg <= '0;
			axi_s_write_done <= 1'b0;
			axi_s_tile <= '0;
			csr_addr_r <= '0;
			csr_wdata_r <= '0;
			csr_write_pulse <= 1'b0;
			noc_meta_router_reg <= '0;
			noc_meta_reduce_id_reg <= '0;
			noc_meta_target_reg <= '0;
			noc_meta_enable_reg <= 1'b0;
			noc_meta_apply_pulse <= 1'b0;
		end
		else begin
			csr_write_pulse <= 1'b0;
			noc_meta_apply_pulse <= 1'b0;
			case (axi_s_state)
				2'd0:
					if (s_axi_awvalid && s_axi_wvalid) begin
						axi_s_tile <= csr_tile_sel;
						csr_addr_r <= s_axi_awaddr[7:0];
						csr_wdata_r <= s_axi_wdata;
						csr_write_pulse <= 1'b1;
						axi_s_state <= 2'd1;
					end
					else if (s_axi_arvalid) begin
						axi_s_tile <= csr_tile_sel;
						csr_addr_r <= s_axi_araddr[7:0];
						axi_s_state <= 2'd2;
					end
				2'd1: begin
					if (csr_is_array_reg)
						case (csr_addr_r)
							NOC_META_ROUTER_OFF: noc_meta_router_reg <= csr_wdata_r[3:0];
							NOC_META_REDUCE_ID_OFF: noc_meta_reduce_id_reg <= csr_wdata_r[7:0];
							NOC_META_TARGET_OFF: noc_meta_target_reg <= csr_wdata_r[3:0];
							NOC_META_CTRL_OFF: begin
								noc_meta_enable_reg <= csr_wdata_r[0];
								noc_meta_apply_pulse <= csr_wdata_r[1];
							end
							default:
								;
						endcase
					axi_s_write_done <= 1'b1;
					axi_s_state <= 2'd3;
				end
				2'd2: begin
					axi_s_rdata_reg <= (csr_is_array_reg ? array_csr_rdata : tile_csr_rdata[axi_s_tile]);
					axi_s_state <= 2'd3;
				end
				2'd3:
					if ((axi_s_write_done && s_axi_bready) || (!axi_s_write_done && s_axi_rready)) begin
						axi_s_state <= 2'd0;
						axi_s_write_done <= 1'b0;
					end
			endcase
		end
	assign s_axi_awready = (axi_s_state == 2'd0) && s_axi_awvalid;
	assign s_axi_wready = (axi_s_state == 2'd0) && s_axi_awvalid;
	assign s_axi_arready = ((axi_s_state == 2'd0) && !s_axi_awvalid) && s_axi_arvalid;
	assign s_axi_bresp = 2'b00;
	assign s_axi_bvalid = (axi_s_state == 2'd3) && axi_s_write_done;
	assign s_axi_rdata = axi_s_rdata_reg;
	assign s_axi_rresp = 2'b00;
	assign s_axi_rvalid = (axi_s_state == 2'd3) && !axi_s_write_done;
	assign tile_csr_wdata = csr_wdata_r;
	function automatic signed [3:0] sv2v_cast_4_signed;
		input reg signed [3:0] inp;
		sv2v_cast_4_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] t;
			for (t = 0; t < NUM_TILES; t = t + 1)
				tile_csr_wen[t] = (csr_write_pulse && !csr_is_array_reg) && (axi_s_tile == sv2v_cast_4_signed(t));
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		case (csr_addr_r)
			NOC_META_ROUTER_OFF: array_csr_rdata = {28'h0000000, noc_meta_router_reg};
			NOC_META_REDUCE_ID_OFF: array_csr_rdata = {24'h000000, noc_meta_reduce_id_reg};
			NOC_META_TARGET_OFF: array_csr_rdata = {28'h0000000, noc_meta_target_reg};
			NOC_META_CTRL_OFF: array_csr_rdata = {31'h00000000, noc_meta_enable_reg};
			NOC_META_STATUS_OFF: array_csr_rdata = {7'h00, noc_meta_enable_reg, 4'h0, noc_meta_target_reg, 4'h0, noc_meta_reduce_id_reg, noc_meta_router_reg};
			default: array_csr_rdata = '0;
		endcase
	end
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_2
			reg signed [31:0] t;
			for (t = 0; t < NUM_TILES; t = t + 1)
				begin
					mesh_inr_meta_cfg_valid[t] = inr_meta_cfg_valid[t];
					mesh_inr_meta_cfg_reduce_id[((NUM_TILES - 1) - t) * 8+:8] = inr_meta_cfg_reduce_id[(((MESH_ROWS * MESH_COLS) - 1) - t) * 8+:8];
					mesh_inr_meta_cfg_target[((NUM_TILES - 1) - t) * 4+:4] = inr_meta_cfg_target[(((MESH_ROWS * MESH_COLS) - 1) - t) * 4+:4];
					mesh_inr_meta_cfg_enable[t] = inr_meta_cfg_enable[t];
					if (noc_meta_apply_pulse && (noc_meta_router_reg == sv2v_cast_4_signed(t))) begin
						mesh_inr_meta_cfg_valid[t] = 1'b1;
						mesh_inr_meta_cfg_reduce_id[((NUM_TILES - 1) - t) * 8+:8] = noc_meta_reduce_id_reg;
						mesh_inr_meta_cfg_target[((NUM_TILES - 1) - t) * 4+:4] = noc_meta_target_reg;
						mesh_inr_meta_cfg_enable[t] = noc_meta_enable_reg;
					end
				end
		end
	end
	genvar _gv_t_1;
	generate
		for (_gv_t_1 = 0; _gv_t_1 < NUM_TILES; _gv_t_1 = _gv_t_1 + 1) begin : gen_tile
			localparam t = _gv_t_1;
			accel_tile #(
				.TILE_ID(t),
				.N_ROWS(N_ROWS),
				.N_COLS(N_COLS),
				.DATA_W(DATA_W),
				.ACC_W(ACC_W),
				.SP_DEPTH(SP_DEPTH),
				.SP_DATA_W(32),
				.NUM_VCS(NUM_VCS)
			) u_tile(
				.clk(clk),
				.rst_n(rst_n),
				.noc_flit_out(tile_flit_out[t]),
				.noc_valid_out(tile_valid_out[t]),
				.noc_credit_in(tile_credit_in[((NUM_TILES - 1) - t) * NUM_VCS+:NUM_VCS]),
				.noc_flit_in(tile_flit_in[((NUM_TILES - 1) - t) * 64+:64]),
				.noc_valid_in(tile_valid_in[t]),
				.noc_credit_out(tile_credit_out[t]),
				.csr_wdata(tile_csr_wdata),
				.csr_addr(tile_csr_addr),
				.csr_wen(tile_csr_wen[t]),
				.csr_rdata(tile_csr_rdata[t]),
				.barrier_req(tile_barrier_req[t]),
				.barrier_done(barrier_release),
				.tile_busy(tile_busy[t]),
				.tile_done(tile_done[t])
			);
		end
	endgenerate
	reg [(NUM_TILES * NUM_VCS) - 1:0] mesh_credit_out;
	reg [(NUM_TILES * 64) - 1:0] mesh_flit_in;
	reg [0:NUM_TILES - 1] mesh_valid_in;
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	noc_mesh_4x4 #(
		.MESH_ROWS(MESH_ROWS),
		.MESH_COLS(MESH_COLS),
		.NUM_VCS(NUM_VCS),
		.BUF_DEPTH(noc_pkg_BUF_DEPTH),
		.SPARSE_VC_ALLOC(SPARSE_VC_ALLOC),
		.INNET_REDUCE(INNET_REDUCE)
	) u_mesh(
		.clk(clk),
		.rst_n(rst_n),
		.inr_meta_cfg_valid(mesh_inr_meta_cfg_valid),
		.inr_meta_cfg_reduce_id(mesh_inr_meta_cfg_reduce_id),
		.inr_meta_cfg_target(mesh_inr_meta_cfg_target),
		.inr_meta_cfg_enable(mesh_inr_meta_cfg_enable),
		.local_flit_in(mesh_flit_in),
		.local_valid_in(mesh_valid_in),
		.local_credit_out(tile_credit_in),
		.local_flit_out(tile_flit_in),
		.local_valid_out(tile_valid_in),
		.local_credit_in(mesh_credit_out)
	);
	barrier_sync #(.NUM_TILES(NUM_TILES)) u_barrier(
		.clk(clk),
		.rst_n(rst_n),
		.tile_barrier_req(tile_barrier_req),
		.participant_mask({NUM_TILES {1'b1}}),
		.barrier_release(barrier_release),
		.arrived_mask(),
		.barrier_active()
	);
	assign tile_busy_o = tile_busy;
	assign tile_done_o = tile_done;
	localparam signed [31:0] GW_NODE = NUM_TILES - 1;
	wire [63:0] gw_flit_in;
	wire [63:0] gw_flit_out;
	wire gw_valid_in;
	wire gw_valid_out;
	wire gw_credit_in;
	wire gw_credit_out;
	assign gw_flit_in = tile_flit_in[((NUM_TILES - 1) - GW_NODE) * 64+:64];
	assign gw_valid_in = tile_valid_in[GW_NODE];
	assign gw_credit_in = tile_credit_in[((NUM_TILES - 1) - GW_NODE) * NUM_VCS];
	reg [NUM_VCS - 1:0] gw_credit_vec;
	always @(*) begin
		if (_sv2v_0)
			;
		gw_credit_vec = '0;
		if (gw_credit_out)
			gw_credit_vec[gw_flit_in[53-:2]] = 1'b1;
	end
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_3
			reg signed [31:0] t;
			for (t = 0; t < NUM_TILES; t = t + 1)
				if (t == GW_NODE) begin
					mesh_flit_in[((NUM_TILES - 1) - t) * 64+:64] = (gw_valid_out ? gw_flit_out : tile_flit_out[t]);
					mesh_valid_in[t] = (gw_valid_out ? 1'b1 : tile_valid_out[t]);
					mesh_credit_out[((NUM_TILES - 1) - t) * NUM_VCS+:NUM_VCS] = tile_credit_out[t] | gw_credit_vec;
				end
				else begin
					mesh_flit_in[((NUM_TILES - 1) - t) * 64+:64] = tile_flit_out[t];
					mesh_valid_in[t] = tile_valid_out[t];
					mesh_credit_out[((NUM_TILES - 1) - t) * NUM_VCS+:NUM_VCS] = tile_credit_out[t];
				end
		end
	end
	tile_dma_gateway #(
		.AXI_ADDR_W(AXI_ADDR_W),
		.AXI_DATA_W(AXI_DATA_W),
		.AXI_ID_W(AXI_ID_W),
		.MAX_BURST(16)
	) u_dma_gw(
		.clk(clk),
		.rst_n(rst_n),
		.noc_flit_in(gw_flit_in),
		.noc_valid_in(gw_valid_in),
		.noc_credit_out(gw_credit_out),
		.noc_flit_out(gw_flit_out),
		.noc_valid_out(gw_valid_out),
		.noc_credit_in(gw_credit_in),
		.m_axi_arid(m_axi_arid),
		.m_axi_araddr(m_axi_araddr),
		.m_axi_arlen(m_axi_arlen),
		.m_axi_arsize(m_axi_arsize),
		.m_axi_arburst(m_axi_arburst),
		.m_axi_arvalid(m_axi_arvalid),
		.m_axi_arready(m_axi_arready),
		.m_axi_rid(m_axi_rid),
		.m_axi_rdata(m_axi_rdata),
		.m_axi_rresp(m_axi_rresp),
		.m_axi_rlast(m_axi_rlast),
		.m_axi_rvalid(m_axi_rvalid),
		.m_axi_rready(m_axi_rready),
		.m_axi_awid(m_axi_awid),
		.m_axi_awaddr(m_axi_awaddr),
		.m_axi_awlen(m_axi_awlen),
		.m_axi_awsize(m_axi_awsize),
		.m_axi_awburst(m_axi_awburst),
		.m_axi_awvalid(m_axi_awvalid),
		.m_axi_awready(m_axi_awready),
		.m_axi_wdata(m_axi_wdata),
		.m_axi_wstrb(m_axi_wstrb),
		.m_axi_wlast(m_axi_wlast),
		.m_axi_wvalid(m_axi_wvalid),
		.m_axi_wready(m_axi_wready),
		.m_axi_bid(m_axi_bid),
		.m_axi_bresp(m_axi_bresp),
		.m_axi_bvalid(m_axi_bvalid),
		.m_axi_bready(m_axi_bready)
	);
	initial _sv2v_0 = 0;
endmodule
