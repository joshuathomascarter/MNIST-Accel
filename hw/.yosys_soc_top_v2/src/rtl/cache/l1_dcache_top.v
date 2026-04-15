module l1_dcache_top (
	clk,
	rst_n,
	cpu_req,
	cpu_gnt,
	cpu_addr,
	cpu_we,
	cpu_be,
	cpu_wdata,
	cpu_rvalid,
	cpu_rdata,
	m_axi_awvalid,
	m_axi_awready,
	m_axi_awaddr,
	m_axi_awid,
	m_axi_awlen,
	m_axi_awsize,
	m_axi_awburst,
	m_axi_wvalid,
	m_axi_wready,
	m_axi_wdata,
	m_axi_wstrb,
	m_axi_wlast,
	m_axi_bvalid,
	m_axi_bready,
	m_axi_bresp,
	m_axi_bid,
	m_axi_arvalid,
	m_axi_arready,
	m_axi_araddr,
	m_axi_arid,
	m_axi_arlen,
	m_axi_arsize,
	m_axi_arburst,
	m_axi_rvalid,
	m_axi_rready,
	m_axi_rdata,
	m_axi_rresp,
	m_axi_rid,
	m_axi_rlast,
	cache_busy
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] ID_WIDTH = 4;
	parameter signed [31:0] NUM_SETS = 16;
	parameter signed [31:0] NUM_WAYS = 4;
	parameter signed [31:0] LINE_BYTES = 64;
	input wire clk;
	input wire rst_n;
	input wire cpu_req;
	output wire cpu_gnt;
	input wire [ADDR_WIDTH - 1:0] cpu_addr;
	input wire cpu_we;
	input wire [(DATA_WIDTH / 8) - 1:0] cpu_be;
	input wire [DATA_WIDTH - 1:0] cpu_wdata;
	output wire cpu_rvalid;
	output wire [DATA_WIDTH - 1:0] cpu_rdata;
	output wire m_axi_awvalid;
	input wire m_axi_awready;
	output wire [ADDR_WIDTH - 1:0] m_axi_awaddr;
	output wire [ID_WIDTH - 1:0] m_axi_awid;
	output wire [7:0] m_axi_awlen;
	output wire [2:0] m_axi_awsize;
	output wire [1:0] m_axi_awburst;
	output wire m_axi_wvalid;
	input wire m_axi_wready;
	output wire [DATA_WIDTH - 1:0] m_axi_wdata;
	output wire [(DATA_WIDTH / 8) - 1:0] m_axi_wstrb;
	output wire m_axi_wlast;
	input wire m_axi_bvalid;
	output wire m_axi_bready;
	input wire [1:0] m_axi_bresp;
	input wire [ID_WIDTH - 1:0] m_axi_bid;
	output wire m_axi_arvalid;
	input wire m_axi_arready;
	output wire [ADDR_WIDTH - 1:0] m_axi_araddr;
	output wire [ID_WIDTH - 1:0] m_axi_arid;
	output wire [7:0] m_axi_arlen;
	output wire [2:0] m_axi_arsize;
	output wire [1:0] m_axi_arburst;
	input wire m_axi_rvalid;
	output wire m_axi_rready;
	input wire [DATA_WIDTH - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire [ID_WIDTH - 1:0] m_axi_rid;
	input wire m_axi_rlast;
	output wire cache_busy;
	wire ctrl_mem_req;
	wire ctrl_mem_gnt;
	wire [ADDR_WIDTH - 1:0] ctrl_mem_addr;
	wire ctrl_mem_we;
	wire [DATA_WIDTH - 1:0] ctrl_mem_wdata;
	wire ctrl_mem_rvalid;
	wire [DATA_WIDTH - 1:0] ctrl_mem_rdata;
	wire ctrl_mem_last;
	l1_cache_ctrl #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.DATA_WIDTH(DATA_WIDTH),
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS),
		.LINE_BYTES(LINE_BYTES)
	) u_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.cpu_req(cpu_req),
		.cpu_gnt(cpu_gnt),
		.cpu_addr(cpu_addr),
		.cpu_we(cpu_we),
		.cpu_be(cpu_be),
		.cpu_wdata(cpu_wdata),
		.cpu_rvalid(cpu_rvalid),
		.cpu_rdata(cpu_rdata),
		.mem_req(ctrl_mem_req),
		.mem_gnt(ctrl_mem_gnt),
		.mem_addr(ctrl_mem_addr),
		.mem_we(ctrl_mem_we),
		.mem_wdata(ctrl_mem_wdata),
		.mem_rvalid(ctrl_mem_rvalid),
		.mem_rdata(ctrl_mem_rdata),
		.mem_last(ctrl_mem_last),
		.cache_busy(cache_busy)
	);
	localparam signed [31:0] WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
	reg [2:0] axi_state;
	reg [2:0] axi_state_next;
	reg [7:0] axi_beat_cnt;
	reg [ADDR_WIDTH - 1:0] axi_base_addr;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			axi_state <= 3'd0;
			axi_beat_cnt <= '0;
			axi_base_addr <= '0;
		end
		else begin
			axi_state <= axi_state_next;
			case (axi_state)
				3'd0:
					if (ctrl_mem_req) begin
						axi_base_addr <= {ctrl_mem_addr[ADDR_WIDTH - 1:$clog2(LINE_BYTES)], {$clog2(LINE_BYTES) {1'b0}}};
						axi_beat_cnt <= '0;
					end
				3'd2:
					if (m_axi_rvalid && m_axi_rready)
						axi_beat_cnt <= axi_beat_cnt + 1;
				3'd4:
					if (m_axi_wvalid && m_axi_wready)
						axi_beat_cnt <= axi_beat_cnt + 1;
				default:
					;
			endcase
		end
	always @(*) begin
		if (_sv2v_0)
			;
		axi_state_next = axi_state;
		case (axi_state)
			3'd0:
				if (ctrl_mem_req && !ctrl_mem_we)
					axi_state_next = 3'd1;
				else if (ctrl_mem_req && ctrl_mem_we)
					axi_state_next = 3'd3;
			3'd1:
				if (m_axi_arready)
					axi_state_next = 3'd2;
			3'd2:
				if (m_axi_rvalid && m_axi_rlast)
					axi_state_next = 3'd0;
			3'd3:
				if (m_axi_awready)
					axi_state_next = 3'd4;
			3'd4:
				if ((m_axi_wvalid && m_axi_wready) && m_axi_wlast)
					axi_state_next = 3'd5;
			3'd5:
				if (m_axi_bvalid)
					axi_state_next = 3'd0;
			default: axi_state_next = 3'd0;
		endcase
	end
	assign m_axi_arvalid = axi_state == 3'd1;
	assign m_axi_araddr = axi_base_addr;
	assign m_axi_arid = '0;
	function automatic signed [7:0] sv2v_cast_8_signed;
		input reg signed [7:0] inp;
		sv2v_cast_8_signed = inp;
	endfunction
	assign m_axi_arlen = sv2v_cast_8_signed(WORDS_PER_LINE - 1);
	function automatic signed [2:0] sv2v_cast_3_signed;
		input reg signed [2:0] inp;
		sv2v_cast_3_signed = inp;
	endfunction
	assign m_axi_arsize = sv2v_cast_3_signed($clog2(DATA_WIDTH / 8));
	assign m_axi_arburst = 2'b01;
	assign m_axi_rready = axi_state == 3'd2;
	assign ctrl_mem_rvalid = (axi_state == 3'd2) && m_axi_rvalid;
	assign ctrl_mem_rdata = m_axi_rdata;
	assign m_axi_awvalid = axi_state == 3'd3;
	assign m_axi_awaddr = axi_base_addr;
	assign m_axi_awid = '0;
	assign m_axi_awlen = sv2v_cast_8_signed(WORDS_PER_LINE - 1);
	assign m_axi_awsize = sv2v_cast_3_signed($clog2(DATA_WIDTH / 8));
	assign m_axi_awburst = 2'b01;
	assign m_axi_wvalid = axi_state == 3'd4;
	assign m_axi_wdata = ctrl_mem_wdata;
	assign m_axi_wstrb = '1;
	assign m_axi_wlast = axi_beat_cnt == sv2v_cast_8_signed(WORDS_PER_LINE - 1);
	assign m_axi_bready = axi_state == 3'd5;
	assign ctrl_mem_gnt = ((axi_state == 3'd4) && m_axi_wready) || ((axi_state == 3'd2) && m_axi_rvalid);
	initial _sv2v_0 = 0;
endmodule
