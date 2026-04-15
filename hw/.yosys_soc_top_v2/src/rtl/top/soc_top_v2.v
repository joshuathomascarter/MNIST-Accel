module soc_top_v2 (
	clk,
	rst_n,
	uart_rx,
	uart_tx,
	gpio_o,
	gpio_i,
	gpio_oe,
	irq_external,
	irq_timer,
	accel_busy,
	accel_done,
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
	dram_ctrl_busy
);
	reg _sv2v_0;
	parameter BOOT_ROM_FILE = "firmware.hex";
	parameter [31:0] CLK_FREQ = 50000000;
	parameter [31:0] UART_BAUD = 115200;
	parameter signed [31:0] MESH_ROWS = 4;
	parameter signed [31:0] MESH_COLS = 4;
	parameter [0:0] SPARSE_VC_ALLOC = 1'b0;
	parameter [0:0] INNET_REDUCE = 1'b0;
	input wire clk;
	input wire rst_n;
	input wire uart_rx;
	output wire uart_tx;
	output wire [7:0] gpio_o;
	input wire [7:0] gpio_i;
	output wire [7:0] gpio_oe;
	output wire irq_external;
	output wire irq_timer;
	output wire accel_busy;
	output wire accel_done;
	output wire [7:0] dram_phy_act;
	output wire [7:0] dram_phy_read;
	output wire [7:0] dram_phy_write;
	output wire [7:0] dram_phy_pre;
	output wire [13:0] dram_phy_row;
	output wire [9:0] dram_phy_col;
	output wire dram_phy_ref;
	output wire [31:0] dram_phy_wdata;
	output wire [3:0] dram_phy_wstrb;
	input wire [31:0] dram_phy_rdata;
	input wire dram_phy_rdata_valid;
	output wire dram_ctrl_busy;
	localparam signed [31:0] NUM_MASTERS = 3;
	localparam signed [31:0] NUM_SLAVES = 8;
	localparam signed [31:0] ID_WIDTH = 4;
	localparam signed [31:0] NUM_TILES = MESH_ROWS * MESH_COLS;
	reg [0:NUM_TILES - 1] inr_meta_cfg_valid;
	reg [(NUM_TILES * 8) - 1:0] inr_meta_cfg_reduce_id;
	reg [(NUM_TILES * 4) - 1:0] inr_meta_cfg_target;
	reg [0:NUM_TILES - 1] inr_meta_cfg_enable;
	reg [1:0] rst_sync_ff;
	wire clk_core;
	wire rst_core_n;
	assign clk_core = clk;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			rst_sync_ff <= 2'b00;
		else
			rst_sync_ff <= {rst_sync_ff[0], 1'b1};
	assign rst_core_n = rst_sync_ff[1];
	wire obi_req;
	wire obi_gnt;
	wire [31:0] obi_addr;
	wire obi_we;
	wire [3:0] obi_be;
	wire [31:0] obi_wdata;
	wire obi_rvalid;
	wire [31:0] obi_rdata;
	wire obi_err;
	assign obi_err = 1'b0;
	wire irq_timer_int;
	simple_cpu #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.ID_WIDTH(ID_WIDTH)
	) u_cpu(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.cpu_reset(~rst_core_n),
		.irq_external(irq_external),
		.irq_timer(irq_timer_int),
		.req(obi_req),
		.gnt(obi_gnt),
		.addr(obi_addr),
		.we(obi_we),
		.be(obi_be),
		.wdata(obi_wdata),
		.rvalid(obi_rvalid),
		.rdata(obi_rdata),
		.err(obi_err)
	);
	wire tlb_lookup_valid;
	wire tlb_lookup_ready;
	wire [31:0] tlb_va;
	wire tlb_hit;
	wire [21:0] tlb_ppn_out;
	wire [33:0] tlb_pa_out;
	assign tlb_ppn_out = tlb_pa_out[33:12];
	wire tlb_miss;
	wire ptw_fill_valid;
	wire [31:0] ptw_fill_va;
	wire [8:0] ptw_fill_asid;
	wire [21:0] ptw_fill_ppn;
	wire ptw_fill_superpage;
	wire ptw_fill_dirty;
	wire ptw_fill_accessed;
	wire ptw_fill_global;
	wire ptw_fill_user;
	wire ptw_fill_exec;
	wire ptw_fill_write;
	wire ptw_fill_read;
	wire [21:0] satp_ppn;
	wire satp_mode;
	assign satp_ppn = 22'h000000;
	assign satp_mode = 1'b0;
	tlb u_tlb(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.lookup_valid(obi_req),
		.lookup_va(obi_addr),
		.lookup_asid(9'b000000000),
		.lookup_hit(tlb_hit),
		.lookup_pa(tlb_pa_out),
		.lookup_fault(),
		.lookup_is_store(1'b0),
		.lookup_is_exec(1'b0),
		.fill_valid(ptw_fill_valid),
		.fill_va(ptw_fill_va),
		.fill_asid(ptw_fill_asid),
		.fill_ppn(ptw_fill_ppn),
		.fill_superpage(ptw_fill_superpage),
		.fill_dirty(ptw_fill_dirty),
		.fill_accessed(ptw_fill_accessed),
		.fill_global(ptw_fill_global),
		.fill_user(ptw_fill_user),
		.fill_exec(ptw_fill_exec),
		.fill_write(ptw_fill_write),
		.fill_read(ptw_fill_read),
		.sfence_valid(1'b0),
		.sfence_va(32'b00000000000000000000000000000000),
		.sfence_asid(9'b000000000),
		.sfence_all(1'b0)
	);
	wire ptw_mem_req_valid;
	wire ptw_mem_req_ready;
	wire [33:0] ptw_mem_req_addr;
	wire ptw_mem_resp_valid;
	wire [31:0] ptw_mem_resp_data;
	page_table_walker u_ptw(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.walk_req_valid((obi_req && !tlb_hit) && satp_mode),
		.walk_req_ready(),
		.walk_va(obi_addr),
		.walk_asid(9'b000000000),
		.walk_is_store(obi_we),
		.walk_is_exec(1'b0),
		.walk_done(ptw_fill_valid),
		.walk_fault(),
		.walk_fault_cause(),
		.walk_result_va(ptw_fill_va),
		.walk_result_asid(ptw_fill_asid),
		.walk_result_ppn(ptw_fill_ppn),
		.walk_result_superpage(ptw_fill_superpage),
		.walk_result_dirty(ptw_fill_dirty),
		.walk_result_accessed(ptw_fill_accessed),
		.walk_result_global(ptw_fill_global),
		.walk_result_user(ptw_fill_user),
		.walk_result_exec(ptw_fill_exec),
		.walk_result_write(ptw_fill_write),
		.walk_result_read(ptw_fill_read),
		.mem_req_valid(ptw_mem_req_valid),
		.mem_req_ready(ptw_mem_req_ready),
		.mem_req_addr(ptw_mem_req_addr),
		.mem_resp_valid(ptw_mem_resp_valid),
		.mem_resp_data(ptw_mem_resp_data),
		.satp_ppn(satp_ppn),
		.satp_mode(satp_mode)
	);
	assign ptw_mem_req_ready = 1'b1;
	assign ptw_mem_resp_valid = 1'b0;
	assign ptw_mem_resp_data = 32'b00000000000000000000000000000000;
	wire [31:0] cpu_phys_addr;
	assign cpu_phys_addr = (satp_mode ? {tlb_ppn_out[19:0], obi_addr[11:0]} : obi_addr);
	wire l1_m_axi_awvalid;
	wire l1_m_axi_awready;
	wire [31:0] l1_m_axi_awaddr;
	wire [3:0] l1_m_axi_awid;
	wire [7:0] l1_m_axi_awlen;
	wire [2:0] l1_m_axi_awsize;
	wire [1:0] l1_m_axi_awburst;
	wire l1_m_axi_wvalid;
	wire l1_m_axi_wready;
	wire [31:0] l1_m_axi_wdata;
	wire [3:0] l1_m_axi_wstrb;
	wire l1_m_axi_wlast;
	wire l1_m_axi_bvalid;
	wire [1:0] l1_m_axi_bresp;
	wire [3:0] l1_m_axi_bid;
	wire l1_m_axi_bready;
	wire l1_m_axi_arvalid;
	wire l1_m_axi_arready;
	wire [31:0] l1_m_axi_araddr;
	wire [3:0] l1_m_axi_arid;
	wire [7:0] l1_m_axi_arlen;
	wire [2:0] l1_m_axi_arsize;
	wire [1:0] l1_m_axi_arburst;
	wire l1_m_axi_rvalid;
	wire [31:0] l1_m_axi_rdata;
	wire [1:0] l1_m_axi_rresp;
	wire [3:0] l1_m_axi_rid;
	wire l1_m_axi_rlast;
	wire l1_m_axi_rready;
	wire is_io;
	wire [31:0] io_axi_addr;
	function automatic [31:0] translate_uncached_alias;
		input reg [31:0] addr;
		case (addr[31:28])
			4'h6: translate_uncached_alias = {4'h4, addr[27:0]};
			4'h7: translate_uncached_alias = {4'h1, addr[27:0]};
			default: translate_uncached_alias = addr;
		endcase
	endfunction
	assign is_io = ((((cpu_phys_addr[31:28] == 4'h2) || (cpu_phys_addr[31:28] == 4'h3)) || (cpu_phys_addr[31:28] == 4'h5)) || (cpu_phys_addr[31:28] == 4'h6)) || (cpu_phys_addr[31:28] == 4'h7);
	reg [31:0] io_addr_r;
	assign io_axi_addr = translate_uncached_alias(io_addr_r);
	wire dcache_req;
	wire dcache_gnt;
	wire dcache_rvalid;
	wire [31:0] dcache_rdata;
	wire io_req;
	wire io_gnt;
	wire io_rvalid;
	wire [31:0] io_rdata;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] tile;
			for (tile = 0; tile < NUM_TILES; tile = tile + 1)
				begin
					inr_meta_cfg_valid[tile] = 1'b0;
					inr_meta_cfg_reduce_id[((NUM_TILES - 1) - tile) * 8+:8] = '0;
					inr_meta_cfg_target[((NUM_TILES - 1) - tile) * 4+:4] = '0;
					inr_meta_cfg_enable[tile] = 1'b0;
				end
		end
	end
	assign dcache_req = obi_req && !is_io;
	assign io_req = obi_req && is_io;
	assign obi_gnt = (is_io ? io_gnt : dcache_gnt);
	assign obi_rvalid = dcache_rvalid | io_rvalid;
	assign obi_rdata = (io_rvalid ? io_rdata : dcache_rdata);
	l1_dcache_top #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.ID_WIDTH(ID_WIDTH),
		.NUM_SETS(16),
		.NUM_WAYS(4),
		.LINE_BYTES(64)
	) u_l1_dcache(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.cpu_req(dcache_req),
		.cpu_gnt(dcache_gnt),
		.cpu_addr(cpu_phys_addr),
		.cpu_we(obi_we),
		.cpu_be(obi_be),
		.cpu_wdata(obi_wdata),
		.cpu_rvalid(dcache_rvalid),
		.cpu_rdata(dcache_rdata),
		.m_axi_awvalid(l1_m_axi_awvalid),
		.m_axi_awready(l1_m_axi_awready),
		.m_axi_awaddr(l1_m_axi_awaddr),
		.m_axi_awid(l1_m_axi_awid),
		.m_axi_awlen(l1_m_axi_awlen),
		.m_axi_awsize(l1_m_axi_awsize),
		.m_axi_awburst(l1_m_axi_awburst),
		.m_axi_wvalid(l1_m_axi_wvalid),
		.m_axi_wready(l1_m_axi_wready),
		.m_axi_wdata(l1_m_axi_wdata),
		.m_axi_wstrb(l1_m_axi_wstrb),
		.m_axi_wlast(l1_m_axi_wlast),
		.m_axi_bvalid(l1_m_axi_bvalid),
		.m_axi_bready(l1_m_axi_bready),
		.m_axi_bresp(l1_m_axi_bresp),
		.m_axi_bid(l1_m_axi_bid),
		.m_axi_arvalid(l1_m_axi_arvalid),
		.m_axi_arready(l1_m_axi_arready),
		.m_axi_araddr(l1_m_axi_araddr),
		.m_axi_arid(l1_m_axi_arid),
		.m_axi_arlen(l1_m_axi_arlen),
		.m_axi_arsize(l1_m_axi_arsize),
		.m_axi_arburst(l1_m_axi_arburst),
		.m_axi_rvalid(l1_m_axi_rvalid),
		.m_axi_rready(l1_m_axi_rready),
		.m_axi_rdata(l1_m_axi_rdata),
		.m_axi_rresp(l1_m_axi_rresp),
		.m_axi_rid(l1_m_axi_rid),
		.m_axi_rlast(l1_m_axi_rlast),
		.cache_busy()
	);
	wire m0_awvalid;
	wire m0_awready;
	wire [31:0] m0_awaddr;
	wire [3:0] m0_awid;
	wire m0_wvalid;
	wire m0_wready;
	wire [31:0] m0_wdata;
	wire [3:0] m0_wstrb;
	wire m0_wlast;
	wire m0_bvalid;
	wire [1:0] m0_bresp;
	wire [3:0] m0_bid;
	wire m0_bready;
	wire m0_arvalid;
	wire m0_arready;
	wire [31:0] m0_araddr;
	wire [3:0] m0_arid;
	wire m0_rvalid;
	wire [31:0] m0_rdata;
	wire [1:0] m0_rresp;
	wire [3:0] m0_rid;
	wire m0_rlast;
	wire m0_rready;
	assign m0_awvalid = l1_m_axi_awvalid;
	assign l1_m_axi_awready = m0_awready;
	assign m0_awaddr = l1_m_axi_awaddr;
	assign m0_awid = l1_m_axi_awid;
	assign m0_wvalid = l1_m_axi_wvalid;
	assign l1_m_axi_wready = m0_wready;
	assign m0_wdata = l1_m_axi_wdata;
	assign m0_wstrb = l1_m_axi_wstrb;
	assign m0_wlast = l1_m_axi_wlast;
	assign l1_m_axi_bvalid = m0_bvalid;
	assign l1_m_axi_bresp = m0_bresp;
	assign l1_m_axi_bid = m0_bid;
	assign m0_bready = l1_m_axi_bready;
	assign m0_arvalid = l1_m_axi_arvalid;
	assign l1_m_axi_arready = m0_arready;
	assign m0_araddr = l1_m_axi_araddr;
	assign m0_arid = l1_m_axi_arid;
	assign l1_m_axi_rvalid = m0_rvalid;
	assign l1_m_axi_rdata = m0_rdata;
	assign l1_m_axi_rresp = m0_rresp;
	assign l1_m_axi_rid = m0_rid;
	assign l1_m_axi_rlast = m0_rlast;
	assign m0_rready = l1_m_axi_rready;
	wire m2_awvalid;
	wire m2_awready;
	wire [31:0] m2_awaddr;
	wire [3:0] m2_awid;
	wire m2_wvalid;
	wire m2_wready;
	wire [31:0] m2_wdata;
	wire [3:0] m2_wstrb;
	wire m2_wlast;
	wire m2_bvalid;
	wire [1:0] m2_bresp;
	wire [3:0] m2_bid;
	wire m2_bready;
	wire m2_arvalid;
	wire m2_arready;
	wire [31:0] m2_araddr;
	wire [3:0] m2_arid;
	wire m2_rvalid;
	wire [31:0] m2_rdata;
	wire [1:0] m2_rresp;
	wire [3:0] m2_rid;
	wire m2_rlast;
	wire m2_rready;
	reg [2:0] io_state;
	reg [31:0] io_wdata_r;
	reg [3:0] io_be_r;
	reg io_we_r;
	reg io_aw_done;
	reg io_w_done;
	always @(posedge clk_core or negedge rst_core_n)
		if (!rst_core_n) begin
			io_state <= 3'd0;
			io_addr_r <= '0;
			io_wdata_r <= '0;
			io_be_r <= '0;
			io_we_r <= 1'b0;
			io_aw_done <= 1'b0;
			io_w_done <= 1'b0;
		end
		else
			case (io_state)
				3'd0: begin
					io_aw_done <= 1'b0;
					io_w_done <= 1'b0;
					if (io_req) begin
						io_addr_r <= cpu_phys_addr;
						io_wdata_r <= obi_wdata;
						io_be_r <= obi_be;
						io_we_r <= obi_we;
						io_state <= (obi_we ? 3'd1 : 3'd3);
					end
				end
				3'd1: begin
					if (m2_awready)
						io_aw_done <= 1'b1;
					if (m2_wready)
						io_w_done <= 1'b1;
					if ((m2_awready || io_aw_done) && (m2_wready || io_w_done))
						io_state <= 3'd2;
				end
				3'd2:
					if (m2_bvalid)
						io_state <= 3'd0;
				3'd3:
					if (m2_arready)
						io_state <= 3'd4;
				3'd4:
					if (m2_rvalid)
						io_state <= 3'd0;
				default: io_state <= 3'd0;
			endcase
	assign io_gnt = (io_state == 3'd0) && io_req;
	assign io_rvalid = ((io_state == 3'd2) && m2_bvalid) || ((io_state == 3'd4) && m2_rvalid);
	assign io_rdata = m2_rdata;
	assign m2_awvalid = (io_state == 3'd1) && !io_aw_done;
	assign m2_awaddr = io_axi_addr;
	assign m2_awid = '0;
	assign m2_wvalid = (io_state == 3'd1) && !io_w_done;
	assign m2_wdata = io_wdata_r;
	assign m2_wstrb = io_be_r;
	assign m2_wlast = 1'b1;
	assign m2_bready = 1'b1;
	assign m2_arvalid = io_state == 3'd3;
	assign m2_araddr = io_axi_addr;
	assign m2_arid = '0;
	assign m2_rready = 1'b1;
	wire m1_awvalid;
	wire m1_awready;
	wire [31:0] m1_awaddr;
	wire [3:0] m1_awid;
	wire m1_wvalid;
	wire m1_wready;
	wire [31:0] m1_wdata;
	wire [3:0] m1_wstrb;
	wire m1_wlast;
	wire m1_bvalid;
	wire [1:0] m1_bresp;
	wire [3:0] m1_bid;
	wire m1_bready;
	wire m1_arvalid;
	wire m1_arready;
	wire [31:0] m1_araddr;
	wire [3:0] m1_arid;
	wire [7:0] m1_arlen;
	wire [7:0] m1_awlen;
	wire m1_rvalid;
	wire [31:0] m1_rdata;
	wire [1:0] m1_rresp;
	wire [3:0] m1_rid;
	wire m1_rlast;
	wire m1_rready;
	wire [7:0] s_awvalid;
	wire [7:0] s_awready;
	wire [255:0] s_awaddr;
	wire [(NUM_SLAVES * ID_WIDTH) - 1:0] s_awid;
	wire [7:0] s_wvalid;
	wire [7:0] s_wready;
	wire [255:0] s_wdata;
	wire [31:0] s_wstrb;
	wire [7:0] s_wlast;
	reg [7:0] s_bvalid;
	reg [15:0] s_bresp;
	reg [(NUM_SLAVES * ID_WIDTH) - 1:0] s_bid;
	wire [7:0] s_bready;
	wire [7:0] s_arvalid;
	wire [7:0] s_arready;
	wire [255:0] s_araddr;
	wire [(NUM_SLAVES * ID_WIDTH) - 1:0] s_arid;
	wire [63:0] s_arlen;
	wire [63:0] s_awlen;
	reg [7:0] s_rvalid;
	reg [255:0] s_rdata;
	reg [15:0] s_rresp;
	reg [(NUM_SLAVES * ID_WIDTH) - 1:0] s_rid;
	reg [7:0] s_rlast;
	wire [7:0] s_rready;
	wire [(MESH_ROWS * MESH_COLS) - 1:0] tile_busy_vec;
	wire [(MESH_ROWS * MESH_COLS) - 1:0] tile_done_vec;
	accel_tile_array #(
		.MESH_ROWS(MESH_ROWS),
		.MESH_COLS(MESH_COLS),
		.N_ROWS(16),
		.N_COLS(16),
		.DATA_W(8),
		.ACC_W(32),
		.SP_DEPTH(4096),
		.SPARSE_VC_ALLOC(SPARSE_VC_ALLOC),
		.INNET_REDUCE(INNET_REDUCE),
		.AXI_ADDR_W(32),
		.AXI_DATA_W(32),
		.AXI_ID_W(ID_WIDTH)
	) u_tile_array(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.inr_meta_cfg_valid(inr_meta_cfg_valid),
		.inr_meta_cfg_reduce_id(inr_meta_cfg_reduce_id),
		.inr_meta_cfg_target(inr_meta_cfg_target),
		.inr_meta_cfg_enable(inr_meta_cfg_enable),
		.s_axi_awaddr(s_awaddr[96+:32]),
		.s_axi_awvalid(s_awvalid[3]),
		.s_axi_awready(s_awready[3]),
		.s_axi_wdata(s_wdata[96+:32]),
		.s_axi_wstrb(s_wstrb[12+:4]),
		.s_axi_wvalid(s_wvalid[3]),
		.s_axi_wready(s_wready[3]),
		.s_axi_bresp(s_bresp[6+:2]),
		.s_axi_bvalid(s_bvalid[3]),
		.s_axi_bready(s_bready[3]),
		.s_axi_araddr(s_araddr[96+:32]),
		.s_axi_arvalid(s_arvalid[3]),
		.s_axi_arready(s_arready[3]),
		.s_axi_rdata(s_rdata[96+:32]),
		.s_axi_rresp(s_rresp[6+:2]),
		.s_axi_rvalid(s_rvalid[3]),
		.s_axi_rready(s_rready[3]),
		.m_axi_arid(m1_arid),
		.m_axi_araddr(m1_araddr),
		.m_axi_arlen(m1_arlen),
		.m_axi_arsize(),
		.m_axi_arburst(),
		.m_axi_arvalid(m1_arvalid),
		.m_axi_arready(m1_arready),
		.m_axi_rid(m1_rid),
		.m_axi_rdata(m1_rdata),
		.m_axi_rresp(m1_rresp),
		.m_axi_rlast(m1_rlast),
		.m_axi_rvalid(m1_rvalid),
		.m_axi_rready(m1_rready),
		.m_axi_awid(m1_awid),
		.m_axi_awaddr(m1_awaddr),
		.m_axi_awlen(m1_awlen),
		.m_axi_awsize(),
		.m_axi_awburst(),
		.m_axi_awvalid(m1_awvalid),
		.m_axi_awready(m1_awready),
		.m_axi_wdata(m1_wdata),
		.m_axi_wstrb(m1_wstrb),
		.m_axi_wlast(m1_wlast),
		.m_axi_wvalid(m1_wvalid),
		.m_axi_wready(m1_wready),
		.m_axi_bid(m1_bid),
		.m_axi_bresp(m1_bresp),
		.m_axi_bvalid(m1_bvalid),
		.m_axi_bready(m1_bready),
		.tile_busy_o(tile_busy_vec),
		.tile_done_o(tile_done_vec)
	);
	wire [4:1] sv2v_tmp_7E22C;
	assign sv2v_tmp_7E22C = '0;
	always @(*) s_bid[12+:ID_WIDTH] = sv2v_tmp_7E22C;
	wire [4:1] sv2v_tmp_37C7B;
	assign sv2v_tmp_37C7B = '0;
	always @(*) s_rid[12+:ID_WIDTH] = sv2v_tmp_37C7B;
	wire [1:1] sv2v_tmp_83802;
	assign sv2v_tmp_83802 = s_rvalid[3];
	always @(*) s_rlast[3] = sv2v_tmp_83802;
	assign accel_busy = |tile_busy_vec;
	reg [(MESH_ROWS * MESH_COLS) - 1:0] tile_done_sticky;
	always @(posedge clk_core or negedge rst_core_n)
		if (!rst_core_n)
			tile_done_sticky <= '0;
		else begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < (MESH_ROWS * MESH_COLS); i = i + 1)
				if (tile_done_vec[i])
					tile_done_sticky[i] <= 1'b1;
				else if (tile_busy_vec[i])
					tile_done_sticky[i] <= 1'b0;
		end
	assign accel_done = &tile_done_sticky;
	axi_crossbar #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.ID_WIDTH(ID_WIDTH),
		.NUM_MASTERS(NUM_MASTERS),
		.NUM_SLAVES(NUM_SLAVES)
	) u_crossbar(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.m_awvalid({m2_awvalid, m1_awvalid, m0_awvalid}),
		.m_awready({m2_awready, m1_awready, m0_awready}),
		.m_awaddr({m2_awaddr, m1_awaddr, m0_awaddr}),
		.m_awid({m2_awid, m1_awid, m0_awid}),
		.m_awlen({8'd0, m1_awlen, l1_m_axi_awlen}),
		.m_wvalid({m2_wvalid, m1_wvalid, m0_wvalid}),
		.m_wready({m2_wready, m1_wready, m0_wready}),
		.m_wdata({m2_wdata, m1_wdata, m0_wdata}),
		.m_wstrb({m2_wstrb, m1_wstrb, m0_wstrb}),
		.m_wlast({m2_wlast, m1_wlast, m0_wlast}),
		.m_bvalid({m2_bvalid, m1_bvalid, m0_bvalid}),
		.m_bready({m2_bready, m1_bready, m0_bready}),
		.m_bresp({m2_bresp, m1_bresp, m0_bresp}),
		.m_bid({m2_bid, m1_bid, m0_bid}),
		.m_arvalid({m2_arvalid, m1_arvalid, m0_arvalid}),
		.m_arready({m2_arready, m1_arready, m0_arready}),
		.m_araddr({m2_araddr, m1_araddr, m0_araddr}),
		.m_arid({m2_arid, m1_arid, m0_arid}),
		.m_arlen({8'd0, m1_arlen, l1_m_axi_arlen}),
		.m_rvalid({m2_rvalid, m1_rvalid, m0_rvalid}),
		.m_rready({m2_rready, m1_rready, m0_rready}),
		.m_rdata({m2_rdata, m1_rdata, m0_rdata}),
		.m_rresp({m2_rresp, m1_rresp, m0_rresp}),
		.m_rid({m2_rid, m1_rid, m0_rid}),
		.m_rlast({m2_rlast, m1_rlast, m0_rlast}),
		.s_awvalid(s_awvalid),
		.s_awready(s_awready),
		.s_awaddr(s_awaddr),
		.s_awid(s_awid),
		.s_awlen(s_awlen),
		.s_wvalid(s_wvalid),
		.s_wready(s_wready),
		.s_wdata(s_wdata),
		.s_wstrb(s_wstrb),
		.s_wlast(s_wlast),
		.s_bvalid(s_bvalid),
		.s_bready(s_bready),
		.s_bresp(s_bresp),
		.s_bid(s_bid),
		.s_arvalid(s_arvalid),
		.s_arready(s_arready),
		.s_araddr(s_araddr),
		.s_arid(s_arid),
		.s_arlen(s_arlen),
		.s_rvalid(s_rvalid),
		.s_rready(s_rready),
		.s_rdata(s_rdata),
		.s_rresp(s_rresp),
		.s_rid(s_rid),
		.s_rlast(s_rlast)
	);
	boot_rom #(
		.ADDR_WIDTH(13),
		.DATA_WIDTH(32),
		.INIT_FILE(BOOT_ROM_FILE)
	) u_boot_rom(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.awvalid(s_awvalid[0]),
		.awready(s_awready[0]),
		.awaddr(s_awaddr[0+:32]),
		.awsize(3'b010),
		.awburst(2'b00),
		.awid(s_awid[0+:ID_WIDTH]),
		.wvalid(s_wvalid[0]),
		.wready(s_wready[0]),
		.wdata(s_wdata[0+:32]),
		.wstrb(s_wstrb[0+:4]),
		.wlast(s_wlast[0]),
		.bvalid(s_bvalid[0]),
		.bready(s_bready[0]),
		.bresp(s_bresp[0+:2]),
		.bid(s_bid[0+:ID_WIDTH]),
		.arvalid(s_arvalid[0]),
		.arready(s_arready[0]),
		.araddr(s_araddr[0+:32]),
		.arsize(3'b010),
		.arburst(2'b01),
		.arid(s_arid[0+:ID_WIDTH]),
		.arlen(s_arlen[0+:8]),
		.rvalid(s_rvalid[0]),
		.rready(s_rready[0]),
		.rdata(s_rdata[0+:32]),
		.rresp(s_rresp[0+:2]),
		.rid(s_rid[0+:ID_WIDTH]),
		.rlast(s_rlast[0])
	);
	sram_ctrl #(
		.ADDR_WIDTH(15),
		.DATA_WIDTH(32)
	) u_sram(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.awvalid(s_awvalid[1]),
		.awready(s_awready[1]),
		.awaddr(s_awaddr[32+:32]),
		.awlen(s_awlen[8+:8]),
		.awsize(3'b010),
		.awburst(2'b01),
		.awid(s_awid[ID_WIDTH+:ID_WIDTH]),
		.wvalid(s_wvalid[1]),
		.wready(s_wready[1]),
		.wdata(s_wdata[32+:32]),
		.wstrb(s_wstrb[4+:4]),
		.wlast(s_wlast[1]),
		.bvalid(s_bvalid[1]),
		.bready(s_bready[1]),
		.bresp(s_bresp[2+:2]),
		.bid(s_bid[ID_WIDTH+:ID_WIDTH]),
		.arvalid(s_arvalid[1]),
		.arready(s_arready[1]),
		.araddr(s_araddr[32+:32]),
		.arsize(3'b010),
		.arburst(2'b01),
		.arid(s_arid[ID_WIDTH+:ID_WIDTH]),
		.arlen(s_arlen[8+:8]),
		.rvalid(s_rvalid[1]),
		.rready(s_rready[1]),
		.rdata(s_rdata[32+:32]),
		.rresp(s_rresp[2+:2]),
		.rid(s_rid[ID_WIDTH+:ID_WIDTH]),
		.rlast(s_rlast[1])
	);
	wire [7:0] periph_aw_sel;
	wire [7:0] periph_ar_sel;
	reg [7:0] periph_aw_sel_r;
	reg [7:0] periph_ar_sel_r;
	reg [31:0] periph_awaddr_r;
	reg [31:0] periph_araddr_r;
	reg [3:0] periph_awid_r;
	reg [3:0] periph_arid_r;
	reg periph_aw_active;
	reg periph_ar_active;
	wire [7:0] periph_aw_sel_cur;
	wire [7:0] periph_ar_sel_cur;
	wire [31:0] periph_awaddr_cur;
	wire [31:0] periph_araddr_cur;
	wire [3:0] periph_awid_cur;
	wire [3:0] periph_arid_cur;
	assign periph_aw_sel = s_awaddr[87-:8];
	assign periph_ar_sel = s_araddr[87-:8];
	assign periph_aw_sel_cur = (periph_aw_active ? periph_aw_sel_r : periph_aw_sel);
	assign periph_ar_sel_cur = (periph_ar_active ? periph_ar_sel_r : periph_ar_sel);
	assign periph_awaddr_cur = (periph_aw_active ? periph_awaddr_r : s_awaddr[64+:32]);
	assign periph_araddr_cur = (periph_ar_active ? periph_araddr_r : s_araddr[64+:32]);
	assign periph_awid_cur = (periph_aw_active ? periph_awid_r : s_awid[8+:ID_WIDTH]);
	assign periph_arid_cur = (periph_ar_active ? periph_arid_r : s_arid[8+:ID_WIDTH]);
	always @(posedge clk_core or negedge rst_core_n)
		if (!rst_core_n) begin
			periph_aw_sel_r <= '0;
			periph_ar_sel_r <= '0;
			periph_awaddr_r <= '0;
			periph_araddr_r <= '0;
			periph_awid_r <= '0;
			periph_arid_r <= '0;
			periph_aw_active <= 1'b0;
			periph_ar_active <= 1'b0;
		end
		else begin
			if (s_awvalid[2] && s_awready[2]) begin
				periph_aw_sel_r <= s_awaddr[87-:8];
				periph_awaddr_r <= s_awaddr[64+:32];
				periph_awid_r <= s_awid[8+:ID_WIDTH];
				periph_aw_active <= 1'b1;
			end
			else if (s_bvalid[2] && s_bready[2])
				periph_aw_active <= 1'b0;
			if (s_arvalid[2] && s_arready[2]) begin
				periph_ar_sel_r <= s_araddr[87-:8];
				periph_araddr_r <= s_araddr[64+:32];
				periph_arid_r <= s_arid[8+:ID_WIDTH];
				periph_ar_active <= 1'b1;
			end
			else if ((s_rvalid[2] && s_rready[2]) && s_rlast[2])
				periph_ar_active <= 1'b0;
		end
	wire irq_uart;
	wire irq_plic_ext;
	wire accel_irq;
	assign accel_irq = |tile_done_vec;
	wire uart_awready;
	wire uart_wready;
	wire uart_bvalid;
	wire uart_arready;
	wire uart_rvalid;
	wire [1:0] uart_bresp;
	wire [1:0] uart_rresp;
	wire [3:0] uart_bid;
	wire [3:0] uart_rid;
	wire [31:0] uart_rdata;
	wire uart_rlast;
	wire timer_awready;
	wire timer_wready;
	wire timer_bvalid;
	wire timer_arready;
	wire timer_rvalid;
	wire [1:0] timer_bresp;
	wire [1:0] timer_rresp;
	wire [3:0] timer_bid;
	wire [3:0] timer_rid;
	wire [31:0] timer_rdata;
	wire timer_rlast;
	wire gpio_awready;
	wire gpio_wready;
	wire gpio_bvalid;
	wire gpio_arready;
	wire gpio_rvalid;
	wire [1:0] gpio_bresp;
	wire [1:0] gpio_rresp;
	wire [3:0] gpio_bid;
	wire [3:0] gpio_rid;
	wire [31:0] gpio_rdata;
	wire gpio_rlast;
	wire plic_awready;
	wire plic_wready;
	wire plic_bvalid;
	wire plic_arready;
	wire plic_rvalid;
	wire [1:0] plic_bresp;
	wire [1:0] plic_rresp;
	wire [3:0] plic_bid;
	wire [3:0] plic_rid;
	wire [31:0] plic_rdata;
	wire plic_rlast;
	uart_ctrl #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.CLK_FREQ(CLK_FREQ),
		.DEFAULT_BAUD(UART_BAUD)
	) u_uart(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.rx(uart_rx),
		.tx(uart_tx),
		.awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h00)),
		.awready(uart_awready),
		.awaddr(periph_awaddr_cur),
		.awsize(3'b010),
		.awburst(2'b00),
		.awid(periph_awid_cur),
		.wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h00)),
		.wready(uart_wready),
		.wdata(s_wdata[64+:32]),
		.wstrb(s_wstrb[8+:4]),
		.wlast(s_wlast[2]),
		.bvalid(uart_bvalid),
		.bready(s_bready[2]),
		.bresp(uart_bresp),
		.bid(uart_bid),
		.arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h00)),
		.arready(uart_arready),
		.araddr(periph_araddr_cur),
		.arsize(3'b010),
		.arburst(2'b00),
		.arid(periph_arid_cur),
		.rvalid(uart_rvalid),
		.rready(s_rready[2]),
		.rdata(uart_rdata),
		.rresp(uart_rresp),
		.rid(uart_rid),
		.rlast(uart_rlast),
		.irq_o(irq_uart)
	);
	timer_ctrl #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32)
	) u_timer(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h01)),
		.awready(timer_awready),
		.awaddr(periph_awaddr_cur),
		.awsize(3'b010),
		.awburst(2'b00),
		.awid(periph_awid_cur),
		.wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h01)),
		.wready(timer_wready),
		.wdata(s_wdata[64+:32]),
		.wstrb(s_wstrb[8+:4]),
		.wlast(s_wlast[2]),
		.bvalid(timer_bvalid),
		.bready(s_bready[2]),
		.bresp(timer_bresp),
		.bid(timer_bid),
		.arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h01)),
		.arready(timer_arready),
		.araddr(periph_araddr_cur),
		.arsize(3'b010),
		.arburst(2'b00),
		.arid(periph_arid_cur),
		.rvalid(timer_rvalid),
		.rready(s_rready[2]),
		.rdata(timer_rdata),
		.rresp(timer_rresp),
		.rid(timer_rid),
		.rlast(timer_rlast),
		.irq_timer_o(irq_timer_int)
	);
	gpio_ctrl #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.GPIO_WIDTH(8)
	) u_gpio(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.gpio_o(gpio_o),
		.gpio_i(gpio_i),
		.gpio_oe(gpio_oe),
		.awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h02)),
		.awready(gpio_awready),
		.awaddr(periph_awaddr_cur),
		.awsize(3'b010),
		.awburst(2'b00),
		.awid(periph_awid_cur),
		.wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h02)),
		.wready(gpio_wready),
		.wdata(s_wdata[64+:32]),
		.wstrb(s_wstrb[8+:4]),
		.wlast(s_wlast[2]),
		.bvalid(gpio_bvalid),
		.bready(s_bready[2]),
		.bresp(gpio_bresp),
		.bid(gpio_bid),
		.arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h02)),
		.arready(gpio_arready),
		.araddr(periph_araddr_cur),
		.arsize(3'b010),
		.arburst(2'b00),
		.arid(periph_arid_cur),
		.rvalid(gpio_rvalid),
		.rready(s_rready[2]),
		.rdata(gpio_rdata),
		.rresp(gpio_rresp),
		.rid(gpio_rid),
		.rlast(gpio_rlast)
	);
	plic #(
		.ADDR_WIDTH(32),
		.DATA_WIDTH(32),
		.NUM_SOURCES(32),
		.NUM_TARGETS(1)
	) u_plic(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.irq_i({25'b0000000000000000000000000, accel_irq, 1'b0, irq_timer_int, irq_uart, 3'b000}),
		.irq_o(irq_plic_ext),
		.awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h03)),
		.awready(plic_awready),
		.awaddr(periph_awaddr_cur),
		.awsize(3'b010),
		.awburst(2'b00),
		.awid(periph_awid_cur),
		.wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h03)),
		.wready(plic_wready),
		.wdata(s_wdata[64+:32]),
		.wstrb(s_wstrb[8+:4]),
		.wlast(s_wlast[2]),
		.bvalid(plic_bvalid),
		.bready(s_bready[2]),
		.bresp(plic_bresp),
		.bid(plic_bid),
		.arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h03)),
		.arready(plic_arready),
		.araddr(periph_araddr_cur),
		.arsize(3'b010),
		.arburst(2'b00),
		.arid(periph_arid_cur),
		.rvalid(plic_rvalid),
		.rready(s_rready[2]),
		.rdata(plic_rdata),
		.rresp(plic_rresp),
		.rid(plic_rid),
		.rlast(plic_rlast)
	);
	always @(*) begin
		if (_sv2v_0)
			;
		if (timer_rvalid) begin
			s_rvalid[2] = 1'b1;
			s_rdata[64+:32] = timer_rdata;
			s_rresp[4+:2] = timer_rresp;
			s_rid[8+:ID_WIDTH] = timer_rid;
			s_rlast[2] = timer_rlast;
		end
		else if (gpio_rvalid) begin
			s_rvalid[2] = 1'b1;
			s_rdata[64+:32] = gpio_rdata;
			s_rresp[4+:2] = gpio_rresp;
			s_rid[8+:ID_WIDTH] = gpio_rid;
			s_rlast[2] = gpio_rlast;
		end
		else if (plic_rvalid) begin
			s_rvalid[2] = 1'b1;
			s_rdata[64+:32] = plic_rdata;
			s_rresp[4+:2] = plic_rresp;
			s_rid[8+:ID_WIDTH] = plic_rid;
			s_rlast[2] = plic_rlast;
		end
		else if (uart_rvalid) begin
			s_rvalid[2] = 1'b1;
			s_rdata[64+:32] = uart_rdata;
			s_rresp[4+:2] = uart_rresp;
			s_rid[8+:ID_WIDTH] = uart_rid;
			s_rlast[2] = uart_rlast;
		end
		else begin
			s_rvalid[2] = 1'b0;
			s_rdata[64+:32] = '0;
			s_rresp[4+:2] = 2'b00;
			s_rid[8+:ID_WIDTH] = '0;
			s_rlast[2] = 1'b0;
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (timer_bvalid) begin
			s_bvalid[2] = 1'b1;
			s_bresp[4+:2] = timer_bresp;
			s_bid[8+:ID_WIDTH] = timer_bid;
		end
		else if (gpio_bvalid) begin
			s_bvalid[2] = 1'b1;
			s_bresp[4+:2] = gpio_bresp;
			s_bid[8+:ID_WIDTH] = gpio_bid;
		end
		else if (plic_bvalid) begin
			s_bvalid[2] = 1'b1;
			s_bresp[4+:2] = plic_bresp;
			s_bid[8+:ID_WIDTH] = plic_bid;
		end
		else if (uart_bvalid) begin
			s_bvalid[2] = 1'b1;
			s_bresp[4+:2] = uart_bresp;
			s_bid[8+:ID_WIDTH] = uart_bid;
		end
		else begin
			s_bvalid[2] = 1'b0;
			s_bresp[4+:2] = 2'b00;
			s_bid[8+:ID_WIDTH] = '0;
		end
	end
	assign s_awready[2] = (periph_aw_sel_cur == 8'h00 ? uart_awready : (periph_aw_sel_cur == 8'h01 ? timer_awready : (periph_aw_sel_cur == 8'h02 ? gpio_awready : (periph_aw_sel_cur == 8'h03 ? plic_awready : 1'b1))));
	assign s_wready[2] = (periph_aw_sel_cur == 8'h00 ? uart_wready : (periph_aw_sel_cur == 8'h01 ? timer_wready : (periph_aw_sel_cur == 8'h02 ? gpio_wready : (periph_aw_sel_cur == 8'h03 ? plic_wready : 1'b1))));
	assign s_arready[2] = (periph_ar_sel_cur == 8'h00 ? uart_arready : (periph_ar_sel_cur == 8'h01 ? timer_arready : (periph_ar_sel_cur == 8'h02 ? gpio_arready : (periph_ar_sel_cur == 8'h03 ? plic_arready : 1'b1))));
	wire l2_m_axi_awvalid;
	wire l2_m_axi_awready;
	wire [31:0] l2_m_axi_awaddr;
	wire [3:0] l2_m_axi_awid;
	wire [7:0] l2_m_axi_awlen;
	wire [2:0] l2_m_axi_awsize;
	wire [1:0] l2_m_axi_awburst;
	wire l2_m_axi_wvalid;
	wire l2_m_axi_wready;
	wire [31:0] l2_m_axi_wdata;
	wire [3:0] l2_m_axi_wstrb;
	wire l2_m_axi_wlast;
	wire l2_m_axi_bvalid;
	wire [1:0] l2_m_axi_bresp;
	wire [3:0] l2_m_axi_bid;
	wire l2_m_axi_bready;
	wire l2_m_axi_arvalid;
	wire l2_m_axi_arready;
	wire [31:0] l2_m_axi_araddr;
	wire [3:0] l2_m_axi_arid;
	wire [7:0] l2_m_axi_arlen;
	wire [2:0] l2_m_axi_arsize;
	wire [1:0] l2_m_axi_arburst;
	wire l2_m_axi_rvalid;
	wire [31:0] l2_m_axi_rdata;
	wire [1:0] l2_m_axi_rresp;
	wire [3:0] l2_m_axi_rid;
	wire l2_m_axi_rlast;
	wire l2_m_axi_rready;
	assign l2_m_axi_arvalid = s_arvalid[4];
	assign s_arready[4] = l2_m_axi_arready;
	assign l2_m_axi_araddr = s_araddr[128+:32];
	assign l2_m_axi_arid = s_arid[16+:ID_WIDTH];
	assign l2_m_axi_arlen = s_arlen[32+:8];
	assign l2_m_axi_arsize = 3'b010;
	assign l2_m_axi_arburst = 2'b01;
	wire [1:1] sv2v_tmp_15E97;
	assign sv2v_tmp_15E97 = l2_m_axi_rvalid;
	always @(*) s_rvalid[4] = sv2v_tmp_15E97;
	assign l2_m_axi_rready = s_rready[4];
	wire [32:1] sv2v_tmp_3B73A;
	assign sv2v_tmp_3B73A = l2_m_axi_rdata;
	always @(*) s_rdata[128+:32] = sv2v_tmp_3B73A;
	wire [2:1] sv2v_tmp_22C52;
	assign sv2v_tmp_22C52 = l2_m_axi_rresp;
	always @(*) s_rresp[8+:2] = sv2v_tmp_22C52;
	wire [4:1] sv2v_tmp_3EF82;
	assign sv2v_tmp_3EF82 = l2_m_axi_rid;
	always @(*) s_rid[16+:ID_WIDTH] = sv2v_tmp_3EF82;
	wire [1:1] sv2v_tmp_66DAB;
	assign sv2v_tmp_66DAB = l2_m_axi_rlast;
	always @(*) s_rlast[4] = sv2v_tmp_66DAB;
	assign l2_m_axi_awvalid = s_awvalid[4];
	assign s_awready[4] = l2_m_axi_awready;
	assign l2_m_axi_awaddr = s_awaddr[128+:32];
	assign l2_m_axi_awid = s_awid[16+:ID_WIDTH];
	assign l2_m_axi_awlen = s_awlen[32+:8];
	assign l2_m_axi_awsize = 3'b010;
	assign l2_m_axi_awburst = 2'b01;
	assign l2_m_axi_wvalid = s_wvalid[4];
	assign s_wready[4] = l2_m_axi_wready;
	assign l2_m_axi_wdata = s_wdata[128+:32];
	assign l2_m_axi_wstrb = s_wstrb[16+:4];
	assign l2_m_axi_wlast = s_wlast[4];
	wire [1:1] sv2v_tmp_FC40B;
	assign sv2v_tmp_FC40B = l2_m_axi_bvalid;
	always @(*) s_bvalid[4] = sv2v_tmp_FC40B;
	assign l2_m_axi_bready = s_bready[4];
	wire [2:1] sv2v_tmp_D1E64;
	assign sv2v_tmp_D1E64 = l2_m_axi_bresp;
	always @(*) s_bresp[8+:2] = sv2v_tmp_D1E64;
	wire [4:1] sv2v_tmp_EE9B0;
	assign sv2v_tmp_EE9B0 = l2_m_axi_bid;
	always @(*) s_bid[16+:ID_WIDTH] = sv2v_tmp_EE9B0;
	dram_ctrl_top #(
		.AXI_ADDR_W(32),
		.AXI_DATA_W(32),
		.AXI_ID_W(ID_WIDTH),
		.NUM_BANKS(8),
		.ROW_BITS(14),
		.COL_BITS(10),
		.BANK_BITS(3),
		.QUEUE_DEPTH(16),
		.ADDR_MODE(0)
	) u_dram_ctrl(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.s_axi_awvalid(l2_m_axi_awvalid),
		.s_axi_awready(l2_m_axi_awready),
		.s_axi_awaddr(l2_m_axi_awaddr),
		.s_axi_awid(l2_m_axi_awid),
		.s_axi_awlen(l2_m_axi_awlen),
		.s_axi_awsize(l2_m_axi_awsize),
		.s_axi_wvalid(l2_m_axi_wvalid),
		.s_axi_wready(l2_m_axi_wready),
		.s_axi_wdata(l2_m_axi_wdata),
		.s_axi_wstrb(l2_m_axi_wstrb),
		.s_axi_wlast(l2_m_axi_wlast),
		.s_axi_bvalid(l2_m_axi_bvalid),
		.s_axi_bready(l2_m_axi_bready),
		.s_axi_bresp(l2_m_axi_bresp),
		.s_axi_bid(l2_m_axi_bid),
		.s_axi_arvalid(l2_m_axi_arvalid),
		.s_axi_arready(l2_m_axi_arready),
		.s_axi_araddr(l2_m_axi_araddr),
		.s_axi_arid(l2_m_axi_arid),
		.s_axi_arlen(l2_m_axi_arlen),
		.s_axi_arsize(l2_m_axi_arsize),
		.s_axi_rvalid(l2_m_axi_rvalid),
		.s_axi_rready(l2_m_axi_rready),
		.s_axi_rdata(l2_m_axi_rdata),
		.s_axi_rresp(l2_m_axi_rresp),
		.s_axi_rid(l2_m_axi_rid),
		.s_axi_rlast(l2_m_axi_rlast),
		.dram_phy_act(dram_phy_act),
		.dram_phy_read(dram_phy_read),
		.dram_phy_write(dram_phy_write),
		.dram_phy_pre(dram_phy_pre),
		.dram_phy_row(dram_phy_row),
		.dram_phy_col(dram_phy_col),
		.dram_phy_ref(dram_phy_ref),
		.dram_phy_wdata(dram_phy_wdata),
		.dram_phy_wstrb(dram_phy_wstrb),
		.dram_phy_rdata(dram_phy_rdata),
		.dram_phy_rdata_valid(dram_phy_rdata_valid),
		.ctrl_busy(dram_ctrl_busy)
	);
	perf_axi #(.NUM_COUNTERS(6)) u_perf(
		.clk(clk_core),
		.rst_n(rst_core_n),
		.event_valid({l2_m_axi_arvalid && l2_m_axi_arready, l2_m_axi_awvalid && l2_m_axi_awready, s_arvalid[4] && s_arready[4], s_awvalid[4] && s_awready[4], obi_req && obi_gnt, 1'b1}),
		.s_axi_awvalid(s_awvalid[5]),
		.s_axi_awready(s_awready[5]),
		.s_axi_awaddr(s_awaddr[167-:8]),
		.s_axi_wvalid(s_wvalid[5]),
		.s_axi_wready(s_wready[5]),
		.s_axi_wdata(s_wdata[160+:32]),
		.s_axi_wstrb(s_wstrb[20+:4]),
		.s_axi_bvalid(s_bvalid[5]),
		.s_axi_bready(s_bready[5]),
		.s_axi_bresp(s_bresp[10+:2]),
		.s_axi_arvalid(s_arvalid[5]),
		.s_axi_arready(s_arready[5]),
		.s_axi_araddr(s_araddr[167-:8]),
		.s_axi_rvalid(s_rvalid[5]),
		.s_axi_rready(s_rready[5]),
		.s_axi_rdata(s_rdata[160+:32]),
		.s_axi_rresp(s_rresp[10+:2])
	);
	wire [4:1] sv2v_tmp_8E4EA;
	assign sv2v_tmp_8E4EA = '0;
	always @(*) s_bid[20+:ID_WIDTH] = sv2v_tmp_8E4EA;
	wire [4:1] sv2v_tmp_32187;
	assign sv2v_tmp_32187 = '0;
	always @(*) s_rid[20+:ID_WIDTH] = sv2v_tmp_32187;
	wire [1:1] sv2v_tmp_71602;
	assign sv2v_tmp_71602 = s_rvalid[5];
	always @(*) s_rlast[5] = sv2v_tmp_71602;
	genvar _gv_si_1;
	generate
		for (_gv_si_1 = 6; _gv_si_1 < NUM_SLAVES; _gv_si_1 = _gv_si_1 + 1) begin : gen_dummy_slaves
			localparam si = _gv_si_1;
			assign s_awready[si] = 1'b1;
			assign s_wready[si] = 1'b1;
			assign s_arready[si] = 1'b1;
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 6; i < NUM_SLAVES; i = i + 1)
				begin
					s_bvalid[i] = 1'b0;
					s_bresp[i * 2+:2] = 2'b11;
					s_bid[i * ID_WIDTH+:ID_WIDTH] = '0;
					s_rvalid[i] = 1'b0;
					s_rdata[i * 32+:32] = 32'hdeaddead;
					s_rresp[i * 2+:2] = 2'b11;
					s_rid[i * ID_WIDTH+:ID_WIDTH] = '0;
					s_rlast[i] = 1'b0;
				end
		end
	end
	assign irq_external = irq_plic_ext;
	assign irq_timer = irq_timer_int;
	initial _sv2v_0 = 0;
endmodule
