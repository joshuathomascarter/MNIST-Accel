module noc_router (
	clk,
	rst_n,
	inr_meta_cfg_valid,
	inr_meta_cfg_reduce_id,
	inr_meta_cfg_target,
	inr_meta_cfg_enable,
	link_flit_in,
	link_valid_in,
	link_credit_out,
	link_flit_out,
	link_valid_out,
	link_credit_in
);
	reg _sv2v_0;
	parameter signed [31:0] ROUTER_ID = 0;
	parameter signed [31:0] NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	parameter signed [31:0] BUF_DEPTH = noc_pkg_BUF_DEPTH;
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	parameter signed [31:0] MESH_ROWS = noc_pkg_MESH_ROWS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	parameter signed [31:0] MESH_COLS = noc_pkg_MESH_COLS;
	parameter [0:0] SPARSE_VC_ALLOC = 1'b0;
	parameter [0:0] INNET_REDUCE = 1'b0;
	input wire clk;
	input wire rst_n;
	input wire inr_meta_cfg_valid;
	input wire [7:0] inr_meta_cfg_reduce_id;
	input wire [3:0] inr_meta_cfg_target;
	input wire inr_meta_cfg_enable;
	input wire [(NUM_PORTS * 64) - 1:0] link_flit_in;
	input wire [0:NUM_PORTS - 1] link_valid_in;
	output wire [(NUM_PORTS * NUM_VCS) - 1:0] link_credit_out;
	output reg [(NUM_PORTS * 64) - 1:0] link_flit_out;
	output reg [0:NUM_PORTS - 1] link_valid_out;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] link_credit_in;
	localparam signed [31:0] noc_pkg_ROW_BITS = 2;
	localparam signed [31:0] noc_pkg_NUM_NODES = noc_pkg_MESH_ROWS * noc_pkg_MESH_COLS;
	localparam signed [31:0] noc_pkg_NODE_BITS = $clog2(noc_pkg_NUM_NODES);
	function automatic [1:0] noc_pkg_node_row;
		input [noc_pkg_NODE_BITS - 1:0] id;
		noc_pkg_node_row = id[noc_pkg_NODE_BITS - 1-:noc_pkg_ROW_BITS];
	endfunction
	localparam [1:0] CUR_ROW = noc_pkg_node_row(ROUTER_ID);
	localparam signed [31:0] noc_pkg_COL_BITS = 2;
	function automatic [1:0] noc_pkg_node_col;
		input [noc_pkg_NODE_BITS - 1:0] id;
		noc_pkg_node_col = id[1:0];
	endfunction
	localparam [1:0] CUR_COL = noc_pkg_node_col(ROUTER_ID);
	wire [NUM_VCS - 1:0] ip_vc_has_flit [0:NUM_PORTS - 1];
	wire [NUM_VCS - 1:0] ip_vc_has_head [0:NUM_PORTS - 1];
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	wire [(NUM_VCS * noc_pkg_PORT_BITS) - 1:0] ip_vc_route [0:NUM_PORTS - 1];
	wire [(NUM_VCS * 64) - 1:0] ip_vc_head_flit [0:NUM_PORTS - 1];
	reg [NUM_VCS - 1:0] ip_vc_read [0:NUM_PORTS - 1];
	wire [(NUM_PORTS * 64) - 1:0] ip_read_flit;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	wire [1:0] ip_read_vc [0:NUM_PORTS - 1];
	reg [(NUM_PORTS * NUM_VCS) - 1:0] vca_req;
	reg [((NUM_PORTS * NUM_VCS) * 3) - 1:0] vca_req_port;
	reg [((NUM_PORTS * NUM_VCS) * 4) - 1:0] vca_req_msg;
	wire [(NUM_PORTS * NUM_VCS) - 1:0] vca_grant;
	wire [((NUM_PORTS * NUM_VCS) * 2) - 1:0] vca_grant_vc;
	reg [(NUM_PORTS * NUM_VCS) - 1:0] vca_vc_busy;
	reg [(NUM_PORTS * NUM_VCS) - 1:0] vca_release;
	reg [((NUM_PORTS * NUM_VCS) * 2) - 1:0] vca_release_id;
	reg [NUM_VCS - 1:0] out_vc_busy [0:NUM_PORTS - 1];
	wire [1:0] out_vc_owner_vc [0:NUM_PORTS - 1][0:NUM_VCS - 1];
	reg [(NUM_PORTS * NUM_VCS) - 1:0] sa_req;
	reg [((NUM_PORTS * NUM_VCS) * 3) - 1:0] sa_target;
	wire [(NUM_PORTS * NUM_VCS) - 1:0] sa_grant;
	wire [(NUM_PORTS * noc_pkg_PORT_BITS) - 1:0] xbar_sel;
	wire [0:NUM_PORTS - 1] xbar_valid;
	wire [(NUM_PORTS * NUM_VCS) - 1:0] out_has_credit;
	wire [(NUM_PORTS * 64) - 1:0] xbar_out_flit;
	wire [0:NUM_PORTS - 1] xbar_out_valid;
	wire [0:NUM_PORTS - 1] inr_intercept;
	wire [63:0] inr_inject_flit;
	wire inr_inject_valid;
	wire inr_inject_ready;
	reg [2:0] inr_inject_port;
	reg [63:0] out_flit_final [0:NUM_PORTS - 1];
	reg out_valid_final [0:NUM_PORTS - 1];
	reg [((NUM_PORTS * NUM_VCS) * 2) - 1:0] alloc_out_vc;
	reg alloc_valid [0:NUM_PORTS - 1][0:NUM_VCS - 1];
	genvar _gv_p_1;
	generate
		for (_gv_p_1 = 0; _gv_p_1 < NUM_PORTS; _gv_p_1 = _gv_p_1 + 1) begin : gen_ip
			localparam p = _gv_p_1;
			noc_input_port #(
				.NUM_VCS(NUM_VCS),
				.BUF_DEPTH(BUF_DEPTH),
				.MESH_ROWS(MESH_ROWS),
				.MESH_COLS(MESH_COLS)
			) u_ip(
				.clk(clk),
				.rst_n(rst_n),
				.flit_in(link_flit_in[((NUM_PORTS - 1) - p) * 64+:64]),
				.flit_valid_in(link_valid_in[p]),
				.credit_out(link_credit_out[((NUM_PORTS - 1) - p) * NUM_VCS+:NUM_VCS]),
				.cur_row(CUR_ROW),
				.cur_col(CUR_COL),
				.vc_has_flit(ip_vc_has_flit[p]),
				.vc_has_head(ip_vc_has_head[p]),
				.vc_route(ip_vc_route[p]),
				.vc_head_flit(ip_vc_head_flit[p]),
				.vc_read(ip_vc_read[p]),
				.read_flit(ip_read_flit[((NUM_PORTS - 1) - p) * 64+:64]),
				.read_vc(ip_read_vc[p])
			);
		end
	endgenerate
	function automatic [3:0] sv2v_cast_4;
		input reg [3:0] inp;
		sv2v_cast_4 = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				begin : sv2v_autoblock_2
					reg signed [31:0] iv;
					for (iv = 0; iv < NUM_VCS; iv = iv + 1)
						begin
							vca_req[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] = ip_vc_has_head[ip][iv] && !alloc_valid[ip][iv];
							vca_req_port[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3] = ip_vc_route[ip][((NUM_VCS - 1) - iv) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS];
							vca_req_msg[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 4+:4] = sv2v_cast_4(ip_vc_head_flit[ip][(((NUM_VCS - 1) - iv) * 64) + 51-:4]);
						end
				end
		end
		begin : sv2v_autoblock_3
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				vca_vc_busy[((NUM_PORTS - 1) - op) * NUM_VCS+:NUM_VCS] = out_vc_busy[op];
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_4
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					vca_release[((NUM_PORTS - 1) - op) * NUM_VCS+:NUM_VCS] = '0;
					begin : sv2v_autoblock_5
						reg signed [31:0] ov;
						for (ov = 0; ov < NUM_VCS; ov = ov + 1)
							vca_release_id[((((NUM_PORTS - 1) - op) * NUM_VCS) + ((NUM_VCS - 1) - ov)) * 2+:2] = '0;
					end
				end
		end
		begin : sv2v_autoblock_6
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				if (xbar_valid[op] && xbar_out_valid[op]) begin
					if ((xbar_out_flit[(((NUM_PORTS - 1) - op) * 64) + 63-:2] == 2'b10) || (xbar_out_flit[(((NUM_PORTS - 1) - op) * 64) + 63-:2] == 2'b11)) begin
						vca_release[(((NUM_PORTS - 1) - op) * NUM_VCS) + xbar_out_flit[(((NUM_PORTS - 1) - op) * 64) + 53-:2]] = 1'b1;
						vca_release_id[((((NUM_PORTS - 1) - op) * NUM_VCS) + ((NUM_VCS - 1) - xbar_out_flit[(((NUM_PORTS - 1) - op) * 64) + 53-:2])) * 2+:2] = xbar_out_flit[(((NUM_PORTS - 1) - op) * 64) + 53-:2];
					end
				end
		end
	end
	generate
		if (SPARSE_VC_ALLOC) begin : gen_sparse_vca
			noc_vc_allocator_sparse #(
				.NUM_PORTS(NUM_PORTS),
				.NUM_VCS(NUM_VCS)
			) u_vca(
				.clk(clk),
				.rst_n(rst_n),
				.req(vca_req),
				.req_port(vca_req_port),
				.req_msg(vca_req_msg),
				.vc_busy(vca_vc_busy),
				.grant(vca_grant),
				.grant_vc(vca_grant_vc),
				.release_vc(vca_release),
				.release_id(vca_release_id)
			);
		end
		else begin : gen_baseline_vca
			noc_vc_allocator #(
				.NUM_PORTS(NUM_PORTS),
				.NUM_VCS(NUM_VCS)
			) u_vca(
				.clk(clk),
				.rst_n(rst_n),
				.req(vca_req),
				.req_port(vca_req_port),
				.vc_busy(vca_vc_busy),
				.grant(vca_grant),
				.grant_vc(vca_grant_vc),
				.release_vc(vca_release),
				.release_id(vca_release_id)
			);
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			begin : sv2v_autoblock_7
				reg signed [31:0] ip;
				for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
					begin : sv2v_autoblock_8
						reg signed [31:0] iv;
						for (iv = 0; iv < NUM_VCS; iv = iv + 1)
							begin
								alloc_valid[ip][iv] <= 1'b0;
								alloc_out_vc[((ip * NUM_VCS) + iv) * 2+:2] <= '0;
							end
					end
			end
			begin : sv2v_autoblock_9
				reg signed [31:0] op;
				for (op = 0; op < NUM_PORTS; op = op + 1)
					out_vc_busy[op] <= '0;
			end
		end
		else begin
			begin : sv2v_autoblock_10
				reg signed [31:0] ip;
				for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
					begin : sv2v_autoblock_11
						reg signed [31:0] iv;
						for (iv = 0; iv < NUM_VCS; iv = iv + 1)
							if (vca_grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv]) begin
								alloc_valid[ip][iv] <= 1'b1;
								alloc_out_vc[((ip * NUM_VCS) + iv) * 2+:2] <= vca_grant_vc[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 2+:2];
								out_vc_busy[vca_req_port[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3]][vca_grant_vc[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 2+:2]] <= 1'b1;
							end
					end
			end
			begin : sv2v_autoblock_12
				reg signed [31:0] op;
				for (op = 0; op < NUM_PORTS; op = op + 1)
					begin : sv2v_autoblock_13
						reg signed [31:0] ov;
						for (ov = 0; ov < NUM_VCS; ov = ov + 1)
							if (vca_release[(((NUM_PORTS - 1) - op) * NUM_VCS) + ov])
								out_vc_busy[op][ov] <= 1'b0;
					end
			end
			begin : sv2v_autoblock_14
				reg signed [31:0] ip;
				for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
					begin : sv2v_autoblock_15
						reg signed [31:0] iv;
						for (iv = 0; iv < NUM_VCS; iv = iv + 1)
							if (sa_grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] && alloc_valid[ip][iv]) begin : sv2v_autoblock_16
								reg [63:0] f;
								f = ip_vc_head_flit[ip][((NUM_VCS - 1) - iv) * 64+:64];
								if ((f[63-:2] == 2'b10) || (f[63-:2] == 2'b11))
									alloc_valid[ip][iv] <= 1'b0;
							end
					end
			end
		end
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_17
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				begin : sv2v_autoblock_18
					reg signed [31:0] iv;
					for (iv = 0; iv < NUM_VCS; iv = iv + 1)
						begin
							sa_req[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] = ip_vc_has_flit[ip][iv] && alloc_valid[ip][iv];
							sa_target[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3] = ip_vc_route[ip][((NUM_VCS - 1) - iv) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS];
						end
				end
		end
	end
	noc_switch_allocator #(
		.NUM_PORTS(NUM_PORTS),
		.NUM_VCS(NUM_VCS)
	) u_sa(
		.clk(clk),
		.rst_n(rst_n),
		.sa_req(sa_req),
		.sa_target(sa_target),
		.out_has_credit(out_has_credit),
		.alloc_vc(alloc_out_vc),
		.sa_grant(sa_grant),
		.xbar_sel(xbar_sel),
		.xbar_valid(xbar_valid)
	);
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_19
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				ip_vc_read[ip] = sa_grant[((NUM_PORTS - 1) - ip) * NUM_VCS+:NUM_VCS];
		end
	end
	reg [(NUM_PORTS * noc_pkg_VC_BITS) - 1:0] xbar_in_vc;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_20
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				xbar_in_vc[((NUM_PORTS - 1) - ip) * noc_pkg_VC_BITS+:noc_pkg_VC_BITS] = alloc_out_vc[((ip * NUM_VCS) + ip_read_vc[ip]) * 2+:2];
		end
	end
	noc_crossbar_5x5 #(.NUM_PORTS(NUM_PORTS)) u_xbar(
		.in_flit(ip_read_flit),
		.in_vc(xbar_in_vc),
		.xbar_sel(xbar_sel),
		.xbar_valid(xbar_valid),
		.out_flit(xbar_out_flit),
		.out_valid(xbar_out_valid)
	);
	genvar _gv_p_2;
	generate
		for (_gv_p_2 = 0; _gv_p_2 < NUM_PORTS; _gv_p_2 = _gv_p_2 + 1) begin : gen_credit
			localparam p = _gv_p_2;
			noc_credit_counter #(
				.BUF_DEPTH(BUF_DEPTH),
				.NUM_VCS(NUM_VCS)
			) u_cc(
				.clk(clk),
				.rst_n(rst_n),
				.credit_in(link_credit_in[((NUM_PORTS - 1) - p) * NUM_VCS+:NUM_VCS]),
				.flit_sent(out_valid_final[p]),
				.flit_vc(out_flit_final[p][53-:2]),
				.has_credit(out_has_credit[((NUM_PORTS - 1) - p) * NUM_VCS+:NUM_VCS])
			);
		end
	endgenerate
	localparam signed [31:0] noc_pkg_PORT_EAST = 1;
	localparam signed [31:0] noc_pkg_PORT_LOCAL = 4;
	localparam signed [31:0] noc_pkg_PORT_NORTH = 0;
	localparam signed [31:0] noc_pkg_PORT_SOUTH = 2;
	localparam signed [31:0] noc_pkg_PORT_WEST = 3;
	always @(*) begin : sv2v_autoblock_21
		reg [1:0] dst_row;
		reg [1:0] dst_col;
		reg signed [noc_pkg_ROW_BITS:0] dst_row_delta;
		reg signed [noc_pkg_COL_BITS:0] dst_col_delta;
		if (_sv2v_0)
			;
		dst_row = noc_pkg_node_row(inr_inject_flit[57-:4]);
		dst_col = noc_pkg_node_col(inr_inject_flit[57-:4]);
		dst_row_delta = $signed({1'b0, dst_row}) - $signed({1'b0, CUR_ROW});
		dst_col_delta = $signed({1'b0, dst_col}) - $signed({1'b0, CUR_COL});
		if (dst_col_delta > 0)
			inr_inject_port = noc_pkg_PORT_EAST;
		else if (dst_col_delta < 0)
			inr_inject_port = noc_pkg_PORT_WEST;
		else if (dst_row_delta > 0)
			inr_inject_port = noc_pkg_PORT_SOUTH;
		else if (dst_row_delta < 0)
			inr_inject_port = noc_pkg_PORT_NORTH;
		else
			inr_inject_port = noc_pkg_PORT_LOCAL;
	end
	localparam signed [31:0] noc_pkg_INNET_SP_DEPTH = 8;
	generate
		if (INNET_REDUCE) begin : gen_innet_reduce
			noc_innet_reduce #(
				.NODE_ID(ROUTER_ID),
				.NUM_PORTS(NUM_PORTS),
				.MESH_ROWS(MESH_ROWS),
				.MESH_COLS(MESH_COLS),
				.SP_DEPTH(noc_pkg_INNET_SP_DEPTH)
			) u_inr(
				.clk(clk),
				.rst_n(rst_n),
				.cfg_valid(inr_meta_cfg_valid),
				.cfg_reduce_id(inr_meta_cfg_reduce_id),
				.cfg_target(inr_meta_cfg_target),
				.cfg_enable(inr_meta_cfg_enable),
				.enable(1'b1),
				.flit_in(xbar_out_flit),
				.valid_in(xbar_out_valid),
				.src_port_in(xbar_sel),
				.intercept(inr_intercept),
				.inject_flit(inr_inject_flit),
				.inject_valid(inr_inject_valid),
				.inject_ready(inr_inject_ready)
			);
			assign inr_inject_ready = (!xbar_out_valid[inr_inject_port] || inr_intercept[inr_inject_port]) && out_has_credit[(((NUM_PORTS - 1) - inr_inject_port) * NUM_VCS) + inr_inject_flit[53-:2]];
			always @(*) begin
				if (_sv2v_0)
					;
				begin : sv2v_autoblock_22
					reg signed [31:0] p;
					for (p = 0; p < NUM_PORTS; p = p + 1)
						if (inr_intercept[p]) begin
							out_flit_final[p] = '0;
							out_valid_final[p] = 1'b0;
						end
						else begin
							out_flit_final[p] = xbar_out_flit[((NUM_PORTS - 1) - p) * 64+:64];
							out_valid_final[p] = xbar_out_valid[p];
						end
				end
				if (inr_inject_valid && inr_inject_ready) begin
					out_flit_final[inr_inject_port] = inr_inject_flit;
					out_valid_final[inr_inject_port] = 1'b1;
				end
			end
		end
		else begin : gen_passthru
			genvar _gv_p_3;
			for (_gv_p_3 = 0; _gv_p_3 < NUM_PORTS; _gv_p_3 = _gv_p_3 + 1) begin : gen_passthru_intercept
				localparam p = _gv_p_3;
				assign inr_intercept[p] = 1'b0;
			end
			assign inr_inject_flit = '0;
			assign inr_inject_valid = 1'b0;
			assign inr_inject_ready = 1'b0;
			genvar _gv_p_4;
			for (_gv_p_4 = 0; _gv_p_4 < NUM_PORTS; _gv_p_4 = _gv_p_4 + 1) begin : gen_passthru_output
				localparam p = _gv_p_4;
				wire [64:1] sv2v_tmp_8F3EF;
				assign sv2v_tmp_8F3EF = xbar_out_flit[((NUM_PORTS - 1) - p) * 64+:64];
				always @(*) out_flit_final[p] = sv2v_tmp_8F3EF;
				wire [1:1] sv2v_tmp_B00DF;
				assign sv2v_tmp_B00DF = xbar_out_valid[p];
				always @(*) out_valid_final[p] = sv2v_tmp_B00DF;
			end
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_23
			reg signed [31:0] p;
			for (p = 0; p < NUM_PORTS; p = p + 1)
				begin
					link_flit_out[((NUM_PORTS - 1) - p) * 64+:64] = out_flit_final[p];
					link_valid_out[p] = out_valid_final[p];
				end
		end
	end
	initial _sv2v_0 = 0;
endmodule
