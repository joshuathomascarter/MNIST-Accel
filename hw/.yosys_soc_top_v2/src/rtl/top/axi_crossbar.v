module axi_crossbar (
	clk,
	rst_n,
	m_awvalid,
	m_awready,
	m_awaddr,
	m_awid,
	m_awlen,
	m_wvalid,
	m_wready,
	m_wdata,
	m_wstrb,
	m_wlast,
	m_bvalid,
	m_bready,
	m_bresp,
	m_bid,
	m_arvalid,
	m_arready,
	m_araddr,
	m_arid,
	m_arlen,
	m_rvalid,
	m_rready,
	m_rdata,
	m_rresp,
	m_rid,
	m_rlast,
	s_awvalid,
	s_awready,
	s_awaddr,
	s_awid,
	s_awlen,
	s_wvalid,
	s_wready,
	s_wdata,
	s_wstrb,
	s_wlast,
	s_bvalid,
	s_bready,
	s_bresp,
	s_bid,
	s_arvalid,
	s_arready,
	s_araddr,
	s_arid,
	s_arlen,
	s_rvalid,
	s_rready,
	s_rdata,
	s_rresp,
	s_rid,
	s_rlast
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] ID_WIDTH = 4;
	parameter [31:0] NUM_MASTERS = 2;
	parameter [31:0] NUM_SLAVES = 8;
	input wire clk;
	input wire rst_n;
	input wire [NUM_MASTERS - 1:0] m_awvalid;
	output reg [NUM_MASTERS - 1:0] m_awready;
	input wire [(NUM_MASTERS * ADDR_WIDTH) - 1:0] m_awaddr;
	input wire [(NUM_MASTERS * ID_WIDTH) - 1:0] m_awid;
	input wire [(NUM_MASTERS * 8) - 1:0] m_awlen;
	input wire [NUM_MASTERS - 1:0] m_wvalid;
	output reg [NUM_MASTERS - 1:0] m_wready;
	input wire [(NUM_MASTERS * DATA_WIDTH) - 1:0] m_wdata;
	input wire [(NUM_MASTERS * (DATA_WIDTH / 8)) - 1:0] m_wstrb;
	input wire [NUM_MASTERS - 1:0] m_wlast;
	output reg [NUM_MASTERS - 1:0] m_bvalid;
	input wire [NUM_MASTERS - 1:0] m_bready;
	output reg [(NUM_MASTERS * 2) - 1:0] m_bresp;
	output reg [(NUM_MASTERS * ID_WIDTH) - 1:0] m_bid;
	input wire [NUM_MASTERS - 1:0] m_arvalid;
	output reg [NUM_MASTERS - 1:0] m_arready;
	input wire [(NUM_MASTERS * ADDR_WIDTH) - 1:0] m_araddr;
	input wire [(NUM_MASTERS * ID_WIDTH) - 1:0] m_arid;
	input wire [(NUM_MASTERS * 8) - 1:0] m_arlen;
	output reg [NUM_MASTERS - 1:0] m_rvalid;
	input wire [NUM_MASTERS - 1:0] m_rready;
	output reg [(NUM_MASTERS * DATA_WIDTH) - 1:0] m_rdata;
	output reg [(NUM_MASTERS * 2) - 1:0] m_rresp;
	output reg [(NUM_MASTERS * ID_WIDTH) - 1:0] m_rid;
	output reg [NUM_MASTERS - 1:0] m_rlast;
	output reg [NUM_SLAVES - 1:0] s_awvalid;
	input wire [NUM_SLAVES - 1:0] s_awready;
	output reg [(NUM_SLAVES * ADDR_WIDTH) - 1:0] s_awaddr;
	output reg [(NUM_SLAVES * ID_WIDTH) - 1:0] s_awid;
	output reg [(NUM_SLAVES * 8) - 1:0] s_awlen;
	output reg [NUM_SLAVES - 1:0] s_wvalid;
	input wire [NUM_SLAVES - 1:0] s_wready;
	output reg [(NUM_SLAVES * DATA_WIDTH) - 1:0] s_wdata;
	output reg [(NUM_SLAVES * (DATA_WIDTH / 8)) - 1:0] s_wstrb;
	output reg [NUM_SLAVES - 1:0] s_wlast;
	input wire [NUM_SLAVES - 1:0] s_bvalid;
	output reg [NUM_SLAVES - 1:0] s_bready;
	input wire [(NUM_SLAVES * 2) - 1:0] s_bresp;
	input wire [(NUM_SLAVES * ID_WIDTH) - 1:0] s_bid;
	output reg [NUM_SLAVES - 1:0] s_arvalid;
	input wire [NUM_SLAVES - 1:0] s_arready;
	output reg [(NUM_SLAVES * ADDR_WIDTH) - 1:0] s_araddr;
	output reg [(NUM_SLAVES * ID_WIDTH) - 1:0] s_arid;
	output reg [(NUM_SLAVES * 8) - 1:0] s_arlen;
	input wire [NUM_SLAVES - 1:0] s_rvalid;
	output reg [NUM_SLAVES - 1:0] s_rready;
	input wire [(NUM_SLAVES * DATA_WIDTH) - 1:0] s_rdata;
	input wire [(NUM_SLAVES * 2) - 1:0] s_rresp;
	input wire [(NUM_SLAVES * ID_WIDTH) - 1:0] s_rid;
	input wire [NUM_SLAVES - 1:0] s_rlast;
	localparam [31:0] MIDX_W = $clog2(NUM_MASTERS);
	function automatic signed [MIDX_W - 1:0] sv2v_cast_6A140_signed;
		input reg signed [MIDX_W - 1:0] inp;
		sv2v_cast_6A140_signed = inp;
	endfunction
	function automatic [MIDX_W - 1:0] onehot_to_idx;
		input [NUM_MASTERS - 1:0] oh;
		begin
			onehot_to_idx = '0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < NUM_MASTERS; i = i + 1)
					if (oh[i])
						onehot_to_idx = sv2v_cast_6A140_signed(i);
			end
		end
	endfunction
	wire [NUM_SLAVES - 1:0] m_aw_target [0:NUM_MASTERS - 1];
	wire [NUM_SLAVES - 1:0] m_ar_target [0:NUM_MASTERS - 1];
	genvar _gv_mi_1;
	generate
		for (_gv_mi_1 = 0; _gv_mi_1 < NUM_MASTERS; _gv_mi_1 = _gv_mi_1 + 1) begin : gen_decoders
			localparam mi = _gv_mi_1;
			axi_addr_decoder #(
				.ADDR_WIDTH(ADDR_WIDTH),
				.NUM_SLAVES(NUM_SLAVES)
			) u_aw_dec(
				.addr(m_awaddr[mi * ADDR_WIDTH+:ADDR_WIDTH]),
				.slave_sel(m_aw_target[mi]),
				.decode_error()
			);
			axi_addr_decoder #(
				.ADDR_WIDTH(ADDR_WIDTH),
				.NUM_SLAVES(NUM_SLAVES)
			) u_ar_dec(
				.addr(m_araddr[mi * ADDR_WIDTH+:ADDR_WIDTH]),
				.slave_sel(m_ar_target[mi]),
				.decode_error()
			);
		end
	endgenerate
	wire [NUM_MASTERS - 1:0] aw_grant [0:NUM_SLAVES - 1];
	wire [NUM_MASTERS - 1:0] ar_grant [0:NUM_SLAVES - 1];
	wire [MIDX_W - 1:0] aw_grant_idx [0:NUM_SLAVES - 1];
	wire [MIDX_W - 1:0] ar_grant_idx [0:NUM_SLAVES - 1];
	genvar _gv_si_1;
	generate
		for (_gv_si_1 = 0; _gv_si_1 < NUM_SLAVES; _gv_si_1 = _gv_si_1 + 1) begin : gen_arbiters
			localparam si = _gv_si_1;
			wire [NUM_MASTERS - 1:0] aw_req;
			wire [NUM_MASTERS - 1:0] ar_req;
			for (_gv_mi_1 = 0; _gv_mi_1 < NUM_MASTERS; _gv_mi_1 = _gv_mi_1 + 1) begin : gen_req
				localparam mi = _gv_mi_1;
				assign aw_req[mi] = m_aw_target[mi][si] && m_awvalid[mi];
				assign ar_req[mi] = m_ar_target[mi][si] && m_arvalid[mi];
			end
			axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_aw_arb(
				.clk(clk),
				.rst_n(rst_n),
				.req(aw_req),
				.handshake_done(s_awvalid[si] && s_awready[si]),
				.grant(aw_grant[si]),
				.grant_idx(aw_grant_idx[si])
			);
			axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_ar_arb(
				.clk(clk),
				.rst_n(rst_n),
				.req(ar_req),
				.handshake_done(s_arvalid[si] && s_arready[si]),
				.grant(ar_grant[si]),
				.grant_idx(ar_grant_idx[si])
			);
		end
	endgenerate
	reg ar_resp_active [0:NUM_SLAVES - 1];
	reg [NUM_MASTERS - 1:0] ar_resp_owner [0:NUM_SLAVES - 1];
	reg aw_resp_active [0:NUM_SLAVES - 1];
	reg [NUM_MASTERS - 1:0] aw_resp_owner [0:NUM_SLAVES - 1];
	generate
		for (_gv_si_1 = 0; _gv_si_1 < NUM_SLAVES; _gv_si_1 = _gv_si_1 + 1) begin : gen_slave_mux
			localparam si = _gv_si_1;
			always @(*) begin
				if (_sv2v_0)
					;
				s_awvalid[si] = |aw_grant[si];
				s_awaddr[si * ADDR_WIDTH+:ADDR_WIDTH] = m_awaddr[aw_grant_idx[si] * ADDR_WIDTH+:ADDR_WIDTH];
				s_awid[si * ID_WIDTH+:ID_WIDTH] = m_awid[aw_grant_idx[si] * ID_WIDTH+:ID_WIDTH];
				s_awlen[si * 8+:8] = m_awlen[aw_grant_idx[si] * 8+:8];
			end
			reg [NUM_MASTERS - 1:0] w_owner;
			reg w_owner_valid;
			reg [MIDX_W - 1:0] w_owner_idx;
			always @(*) begin
				if (_sv2v_0)
					;
				w_owner = (aw_grant[si] & {NUM_MASTERS {s_awready[si]}}) | (aw_resp_owner[si] & {NUM_MASTERS {aw_resp_active[si]}});
				w_owner_valid = |w_owner;
				w_owner_idx = onehot_to_idx(w_owner);
			end
			always @(*) begin
				if (_sv2v_0)
					;
				s_wvalid[si] = 1'b0;
				s_wdata[si * DATA_WIDTH+:DATA_WIDTH] = '0;
				s_wstrb[si * (DATA_WIDTH / 8)+:DATA_WIDTH / 8] = '0;
				s_wlast[si] = 1'b0;
				if (w_owner_valid) begin
					s_wvalid[si] = m_wvalid[w_owner_idx];
					s_wdata[si * DATA_WIDTH+:DATA_WIDTH] = m_wdata[w_owner_idx * DATA_WIDTH+:DATA_WIDTH];
					s_wstrb[si * (DATA_WIDTH / 8)+:DATA_WIDTH / 8] = m_wstrb[w_owner_idx * (DATA_WIDTH / 8)+:DATA_WIDTH / 8];
					s_wlast[si] = m_wlast[w_owner_idx];
				end
			end
			always @(*) begin
				if (_sv2v_0)
					;
				s_arvalid[si] = |ar_grant[si];
				s_araddr[si * ADDR_WIDTH+:ADDR_WIDTH] = m_araddr[ar_grant_idx[si] * ADDR_WIDTH+:ADDR_WIDTH];
				s_arid[si * ID_WIDTH+:ID_WIDTH] = m_arid[ar_grant_idx[si] * ID_WIDTH+:ID_WIDTH];
				s_arlen[si * 8+:8] = m_arlen[ar_grant_idx[si] * 8+:8];
			end
			always @(*) begin
				if (_sv2v_0)
					;
				s_bready[si] = 1'b0;
				s_rready[si] = 1'b0;
				begin : sv2v_autoblock_2
					reg signed [31:0] m;
					for (m = 0; m < NUM_MASTERS; m = m + 1)
						begin
							s_bready[si] = s_bready[si] | ((aw_resp_owner[si][m] & aw_resp_active[si]) & m_bready[m]);
							s_rready[si] = s_rready[si] | ((ar_resp_owner[si][m] & ar_resp_active[si]) & m_rready[m]);
						end
				end
			end
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		m_awready = '0;
		m_wready = '0;
		m_arready = '0;
		begin : sv2v_autoblock_3
			reg signed [31:0] m;
			for (m = 0; m < NUM_MASTERS; m = m + 1)
				begin : sv2v_autoblock_4
					reg signed [31:0] s;
					for (s = 0; s < NUM_SLAVES; s = s + 1)
						begin
							m_awready[m] = m_awready[m] | (m_aw_target[m][s] & aw_grant[s][m] ? s_awready[s] : 1'b0);
							m_wready[m] = m_wready[m] | ((aw_grant[s][m] & s_awready[s]) | (aw_resp_owner[s][m] & aw_resp_active[s]) ? s_wready[s] : 1'b0);
							m_arready[m] = m_arready[m] | (m_ar_target[m][s] & ar_grant[s][m] ? s_arready[s] : 1'b0);
						end
				end
		end
	end
	generate
		for (_gv_si_1 = 0; _gv_si_1 < NUM_SLAVES; _gv_si_1 = _gv_si_1 + 1) begin : gen_resp_track
			localparam si = _gv_si_1;
			always @(posedge clk or negedge rst_n)
				if (!rst_n) begin
					aw_resp_owner[si] <= '0;
					ar_resp_owner[si] <= '0;
					aw_resp_active[si] <= 1'b0;
					ar_resp_active[si] <= 1'b0;
				end
				else begin
					if (s_awvalid[si] && s_awready[si]) begin
						aw_resp_owner[si] <= aw_grant[si];
						aw_resp_active[si] <= 1'b1;
					end
					else if ((aw_resp_active[si] && s_bvalid[si]) && s_bready[si])
						aw_resp_active[si] <= 1'b0;
					if (s_arvalid[si] && s_arready[si]) begin
						ar_resp_owner[si] <= ar_grant[si];
						ar_resp_active[si] <= 1'b1;
					end
					else if (((ar_resp_active[si] && s_rvalid[si]) && s_rready[si]) && s_rlast[si])
						ar_resp_active[si] <= 1'b0;
				end
		end
		for (_gv_mi_1 = 0; _gv_mi_1 < NUM_MASTERS; _gv_mi_1 = _gv_mi_1 + 1) begin : gen_responses
			localparam mi = _gv_mi_1;
			always @(*) begin
				if (_sv2v_0)
					;
				m_bvalid[mi] = 1'b0;
				m_bresp[mi * 2+:2] = '0;
				m_bid[mi * ID_WIDTH+:ID_WIDTH] = '0;
				m_rvalid[mi] = 1'b0;
				m_rdata[mi * DATA_WIDTH+:DATA_WIDTH] = '0;
				m_rresp[mi * 2+:2] = '0;
				m_rid[mi * ID_WIDTH+:ID_WIDTH] = '0;
				m_rlast[mi] = 1'b0;
				begin : sv2v_autoblock_5
					reg signed [31:0] s;
					for (s = 0; s < NUM_SLAVES; s = s + 1)
						begin
							m_bvalid[mi] = m_bvalid[mi] | ((s_bvalid[s] & aw_resp_owner[s][mi]) & aw_resp_active[s]);
							m_bresp[mi * 2+:2] = m_bresp[mi * 2+:2] | (aw_resp_owner[s][mi] & aw_resp_active[s] ? s_bresp[s * 2+:2] : 2'b00);
							m_bid[mi * ID_WIDTH+:ID_WIDTH] = m_bid[mi * ID_WIDTH+:ID_WIDTH] | (aw_resp_owner[s][mi] & aw_resp_active[s] ? s_bid[s * ID_WIDTH+:ID_WIDTH] : '0);
							m_rvalid[mi] = m_rvalid[mi] | ((s_rvalid[s] & ar_resp_owner[s][mi]) & ar_resp_active[s]);
							m_rdata[mi * DATA_WIDTH+:DATA_WIDTH] = m_rdata[mi * DATA_WIDTH+:DATA_WIDTH] | (ar_resp_owner[s][mi] & ar_resp_active[s] ? s_rdata[s * DATA_WIDTH+:DATA_WIDTH] : '0);
							m_rresp[mi * 2+:2] = m_rresp[mi * 2+:2] | (ar_resp_owner[s][mi] & ar_resp_active[s] ? s_rresp[s * 2+:2] : 2'b00);
							m_rid[mi * ID_WIDTH+:ID_WIDTH] = m_rid[mi * ID_WIDTH+:ID_WIDTH] | (ar_resp_owner[s][mi] & ar_resp_active[s] ? s_rid[s * ID_WIDTH+:ID_WIDTH] : '0);
							m_rlast[mi] = m_rlast[mi] | ((s_rlast[s] & ar_resp_owner[s][mi]) & ar_resp_active[s]);
						end
				end
			end
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
