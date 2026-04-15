module noc_mesh_4x4 (
	clk,
	rst_n,
	inr_meta_cfg_valid,
	inr_meta_cfg_reduce_id,
	inr_meta_cfg_target,
	inr_meta_cfg_enable,
	local_flit_in,
	local_valid_in,
	local_credit_out,
	local_flit_out,
	local_valid_out,
	local_credit_in
);
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	parameter signed [31:0] MESH_ROWS = noc_pkg_MESH_ROWS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	parameter signed [31:0] MESH_COLS = noc_pkg_MESH_COLS;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	parameter signed [31:0] BUF_DEPTH = noc_pkg_BUF_DEPTH;
	parameter [0:0] SPARSE_VC_ALLOC = 1'b0;
	parameter [0:0] INNET_REDUCE = 1'b0;
	input wire clk;
	input wire rst_n;
	input wire [0:(MESH_ROWS * MESH_COLS) - 1] inr_meta_cfg_valid;
	input wire [((MESH_ROWS * MESH_COLS) * 8) - 1:0] inr_meta_cfg_reduce_id;
	input wire [((MESH_ROWS * MESH_COLS) * 4) - 1:0] inr_meta_cfg_target;
	input wire [0:(MESH_ROWS * MESH_COLS) - 1] inr_meta_cfg_enable;
	input wire [((MESH_ROWS * MESH_COLS) * 64) - 1:0] local_flit_in;
	input wire [0:(MESH_ROWS * MESH_COLS) - 1] local_valid_in;
	output wire [((MESH_ROWS * MESH_COLS) * NUM_VCS) - 1:0] local_credit_out;
	output wire [((MESH_ROWS * MESH_COLS) * 64) - 1:0] local_flit_out;
	output wire [0:(MESH_ROWS * MESH_COLS) - 1] local_valid_out;
	input wire [((MESH_ROWS * MESH_COLS) * NUM_VCS) - 1:0] local_credit_in;
	localparam signed [31:0] NUM_NODES = MESH_ROWS * MESH_COLS;
	localparam signed [31:0] NP = 5;
	wire [319:0] r_flit_in [0:NUM_NODES - 1];
	wire [0:4] r_valid_in [0:NUM_NODES - 1];
	wire [(NP * NUM_VCS) - 1:0] r_credit_out [0:NUM_NODES - 1];
	wire [319:0] r_flit_out [0:NUM_NODES - 1];
	wire [0:4] r_valid_out [0:NUM_NODES - 1];
	wire [(NP * NUM_VCS) - 1:0] r_credit_in [0:NUM_NODES - 1];
	genvar _gv_r_1;
	generate
		for (_gv_r_1 = 0; _gv_r_1 < MESH_ROWS; _gv_r_1 = _gv_r_1 + 1) begin : gen_row
			localparam r = _gv_r_1;
			genvar _gv_c_1;
			for (_gv_c_1 = 0; _gv_c_1 < MESH_COLS; _gv_c_1 = _gv_c_1 + 1) begin : gen_col
				localparam c = _gv_c_1;
				localparam signed [31:0] ID = (r * MESH_COLS) + c;
				noc_router #(
					.ROUTER_ID(ID),
					.NUM_PORTS(NP),
					.NUM_VCS(NUM_VCS),
					.BUF_DEPTH(BUF_DEPTH),
					.MESH_ROWS(MESH_ROWS),
					.MESH_COLS(MESH_COLS),
					.SPARSE_VC_ALLOC(SPARSE_VC_ALLOC),
					.INNET_REDUCE(INNET_REDUCE)
				) u_router(
					.clk(clk),
					.rst_n(rst_n),
					.inr_meta_cfg_valid(inr_meta_cfg_valid[ID]),
					.inr_meta_cfg_reduce_id(inr_meta_cfg_reduce_id[(((MESH_ROWS * MESH_COLS) - 1) - ID) * 8+:8]),
					.inr_meta_cfg_target(inr_meta_cfg_target[(((MESH_ROWS * MESH_COLS) - 1) - ID) * 4+:4]),
					.inr_meta_cfg_enable(inr_meta_cfg_enable[ID]),
					.link_flit_in(r_flit_in[ID]),
					.link_valid_in(r_valid_in[ID]),
					.link_credit_out(r_credit_out[ID]),
					.link_flit_out(r_flit_out[ID]),
					.link_valid_out(r_valid_out[ID]),
					.link_credit_in(r_credit_in[ID])
				);
			end
		end
	endgenerate
	genvar _gv_r_2;
	localparam signed [31:0] noc_pkg_PORT_EAST = 1;
	localparam signed [31:0] noc_pkg_PORT_LOCAL = 4;
	localparam signed [31:0] noc_pkg_PORT_NORTH = 0;
	localparam signed [31:0] noc_pkg_PORT_SOUTH = 2;
	localparam signed [31:0] noc_pkg_PORT_WEST = 3;
	generate
		for (_gv_r_2 = 0; _gv_r_2 < MESH_ROWS; _gv_r_2 = _gv_r_2 + 1) begin : gen_wire_row
			localparam r = _gv_r_2;
			genvar _gv_c_2;
			for (_gv_c_2 = 0; _gv_c_2 < MESH_COLS; _gv_c_2 = _gv_c_2 + 1) begin : gen_wire_col
				localparam c = _gv_c_2;
				localparam signed [31:0] ID = (r * MESH_COLS) + c;
				if (r > 0) begin : north_link
					localparam signed [31:0] N_ID = ((r - 1) * MESH_COLS) + c;
					assign r_flit_in[ID][256+:64] = r_flit_out[N_ID][128+:64];
					assign r_valid_in[ID][noc_pkg_PORT_NORTH] = r_valid_out[N_ID][noc_pkg_PORT_SOUTH];
					assign r_credit_in[ID][4 * NUM_VCS+:NUM_VCS] = r_credit_out[N_ID][2 * NUM_VCS+:NUM_VCS];
				end
				else begin : north_tieoff
					assign r_flit_in[ID][256+:64] = '0;
					assign r_valid_in[ID][noc_pkg_PORT_NORTH] = 1'b0;
					assign r_credit_in[ID][4 * NUM_VCS+:NUM_VCS] = '0;
				end
				if (r < (MESH_ROWS - 1)) begin : south_link
					localparam signed [31:0] S_ID = ((r + 1) * MESH_COLS) + c;
					assign r_flit_in[ID][128+:64] = r_flit_out[S_ID][256+:64];
					assign r_valid_in[ID][noc_pkg_PORT_SOUTH] = r_valid_out[S_ID][noc_pkg_PORT_NORTH];
					assign r_credit_in[ID][2 * NUM_VCS+:NUM_VCS] = r_credit_out[S_ID][4 * NUM_VCS+:NUM_VCS];
				end
				else begin : south_tieoff
					assign r_flit_in[ID][128+:64] = '0;
					assign r_valid_in[ID][noc_pkg_PORT_SOUTH] = 1'b0;
					assign r_credit_in[ID][2 * NUM_VCS+:NUM_VCS] = '0;
				end
				if (c < (MESH_COLS - 1)) begin : east_link
					localparam signed [31:0] E_ID = (r * MESH_COLS) + (c + 1);
					assign r_flit_in[ID][192+:64] = r_flit_out[E_ID][64+:64];
					assign r_valid_in[ID][noc_pkg_PORT_EAST] = r_valid_out[E_ID][noc_pkg_PORT_WEST];
					assign r_credit_in[ID][3 * NUM_VCS+:NUM_VCS] = r_credit_out[E_ID][1 * NUM_VCS+:NUM_VCS];
				end
				else begin : east_tieoff
					assign r_flit_in[ID][192+:64] = '0;
					assign r_valid_in[ID][noc_pkg_PORT_EAST] = 1'b0;
					assign r_credit_in[ID][3 * NUM_VCS+:NUM_VCS] = '0;
				end
				if (c > 0) begin : west_link
					localparam signed [31:0] W_ID = (r * MESH_COLS) + (c - 1);
					assign r_flit_in[ID][64+:64] = r_flit_out[W_ID][192+:64];
					assign r_valid_in[ID][noc_pkg_PORT_WEST] = r_valid_out[W_ID][noc_pkg_PORT_EAST];
					assign r_credit_in[ID][1 * NUM_VCS+:NUM_VCS] = r_credit_out[W_ID][3 * NUM_VCS+:NUM_VCS];
				end
				else begin : west_tieoff
					assign r_flit_in[ID][64+:64] = '0;
					assign r_valid_in[ID][noc_pkg_PORT_WEST] = 1'b0;
					assign r_credit_in[ID][1 * NUM_VCS+:NUM_VCS] = '0;
				end
				assign r_flit_in[ID][0+:64] = local_flit_in[(((MESH_ROWS * MESH_COLS) - 1) - ID) * 64+:64];
				assign r_valid_in[ID][noc_pkg_PORT_LOCAL] = local_valid_in[ID];
				assign r_credit_in[ID][0+:NUM_VCS] = local_credit_in[(((MESH_ROWS * MESH_COLS) - 1) - ID) * NUM_VCS+:NUM_VCS];
				assign local_flit_out[(((MESH_ROWS * MESH_COLS) - 1) - ID) * 64+:64] = r_flit_out[ID][0+:64];
				assign local_valid_out[ID] = r_valid_out[ID][noc_pkg_PORT_LOCAL];
				assign local_credit_out[(((MESH_ROWS * MESH_COLS) - 1) - ID) * NUM_VCS+:NUM_VCS] = r_credit_out[ID][0+:NUM_VCS];
			end
		end
	endgenerate
endmodule
