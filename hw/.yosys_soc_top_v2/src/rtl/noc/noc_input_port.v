module noc_input_port (
	clk,
	rst_n,
	flit_in,
	flit_valid_in,
	credit_out,
	cur_row,
	cur_col,
	vc_has_flit,
	vc_has_head,
	vc_route,
	vc_head_flit,
	vc_read,
	read_flit,
	read_vc
);
	reg _sv2v_0;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	parameter signed [31:0] BUF_DEPTH = noc_pkg_BUF_DEPTH;
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	parameter signed [31:0] MESH_ROWS = noc_pkg_MESH_ROWS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	parameter signed [31:0] MESH_COLS = noc_pkg_MESH_COLS;
	input wire clk;
	input wire rst_n;
	input wire [63:0] flit_in;
	input wire flit_valid_in;
	output reg [NUM_VCS - 1:0] credit_out;
	localparam signed [31:0] noc_pkg_ROW_BITS = 2;
	input wire [1:0] cur_row;
	localparam signed [31:0] noc_pkg_COL_BITS = 2;
	input wire [1:0] cur_col;
	output wire [NUM_VCS - 1:0] vc_has_flit;
	output wire [NUM_VCS - 1:0] vc_has_head;
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	output wire [(NUM_VCS * noc_pkg_PORT_BITS) - 1:0] vc_route;
	output wire [(NUM_VCS * 64) - 1:0] vc_head_flit;
	input wire [NUM_VCS - 1:0] vc_read;
	output reg [63:0] read_flit;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	output reg [1:0] read_vc;
	reg [63:0] fifo_mem [0:NUM_VCS - 1][0:BUF_DEPTH - 1];
	reg [$clog2(BUF_DEPTH) - 1:0] wr_ptr [0:NUM_VCS - 1];
	reg [$clog2(BUF_DEPTH) - 1:0] rd_ptr [0:NUM_VCS - 1];
	reg [$clog2(BUF_DEPTH + 1) - 1:0] count [0:NUM_VCS - 1];
	wire [1:0] in_vc;
	assign in_vc = flit_in[53-:2];
	function automatic signed [$clog2(BUF_DEPTH) - 1:0] sv2v_cast_8B2D3_signed;
		input reg signed [$clog2(BUF_DEPTH) - 1:0] inp;
		sv2v_cast_8B2D3_signed = inp;
	endfunction
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				begin
					wr_ptr[v] <= '0;
					rd_ptr[v] <= '0;
					count[v] <= '0;
				end
		end
		else begin
			if (flit_valid_in) begin
				fifo_mem[in_vc][wr_ptr[in_vc]] <= flit_in;
				wr_ptr[in_vc] <= (wr_ptr[in_vc] == sv2v_cast_8B2D3_signed(BUF_DEPTH - 1) ? '0 : wr_ptr[in_vc] + 1);
			end
			begin : sv2v_autoblock_2
				reg signed [31:0] v;
				for (v = 0; v < NUM_VCS; v = v + 1)
					if (vc_read[v])
						rd_ptr[v] <= (rd_ptr[v] == sv2v_cast_8B2D3_signed(BUF_DEPTH - 1) ? '0 : rd_ptr[v] + 1);
			end
			begin : sv2v_autoblock_3
				reg signed [31:0] v;
				for (v = 0; v < NUM_VCS; v = v + 1)
					begin : sv2v_autoblock_4
						reg wr_this;
						reg rd_this;
						wr_this = flit_valid_in && (in_vc == v);
						rd_this = vc_read[v];
						case ({wr_this, rd_this})
							2'b10: count[v] <= count[v] + 1;
							2'b01: count[v] <= count[v] - 1;
							default: count[v] <= count[v];
						endcase
					end
			end
		end
	always @(*) begin
		if (_sv2v_0)
			;
		credit_out = '0;
		begin : sv2v_autoblock_5
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				credit_out[v] = vc_read[v];
		end
	end
	genvar _gv_v_1;
	localparam signed [31:0] noc_pkg_NUM_NODES = noc_pkg_MESH_ROWS * noc_pkg_MESH_COLS;
	localparam signed [31:0] noc_pkg_NODE_BITS = $clog2(noc_pkg_NUM_NODES);
	function automatic [1:0] noc_pkg_node_col;
		input [noc_pkg_NODE_BITS - 1:0] id;
		noc_pkg_node_col = id[1:0];
	endfunction
	function automatic [1:0] noc_pkg_node_row;
		input [noc_pkg_NODE_BITS - 1:0] id;
		noc_pkg_node_row = id[noc_pkg_NODE_BITS - 1-:noc_pkg_ROW_BITS];
	endfunction
	generate
		for (_gv_v_1 = 0; _gv_v_1 < NUM_VCS; _gv_v_1 = _gv_v_1 + 1) begin : gen_vc
			localparam v = _gv_v_1;
			assign vc_has_flit[v] = count[v] != '0;
			assign vc_head_flit[((NUM_VCS - 1) - v) * 64+:64] = fifo_mem[v][rd_ptr[v]];
			assign vc_has_head[v] = vc_has_flit[v] && ((fifo_mem[v][rd_ptr[v]][63-:2] == 2'b00) || (fifo_mem[v][rd_ptr[v]][63-:2] == 2'b11));
			wire [1:0] dst_row_v;
			wire [1:0] dst_col_v;
			wire [2:0] route_port_v;
			wire route_valid_v;
			assign dst_row_v = noc_pkg_node_row(fifo_mem[v][rd_ptr[v]][57-:4]);
			assign dst_col_v = noc_pkg_node_col(fifo_mem[v][rd_ptr[v]][57-:4]);
			noc_route_compute #(
				.MESH_ROWS(MESH_ROWS),
				.MESH_COLS(MESH_COLS)
			) u_rc(
				.cur_row(cur_row),
				.cur_col(cur_col),
				.dst_row(dst_row_v),
				.dst_col(dst_col_v),
				.out_port(route_port_v),
				.route_valid(route_valid_v)
			);
			assign vc_route[((NUM_VCS - 1) - v) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS] = route_port_v;
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		read_flit = '0;
		read_vc = '0;
		begin : sv2v_autoblock_6
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				if (vc_read[v]) begin
					read_flit = fifo_mem[v][rd_ptr[v]];
					read_vc = v;
				end
		end
	end
	initial _sv2v_0 = 0;
endmodule
