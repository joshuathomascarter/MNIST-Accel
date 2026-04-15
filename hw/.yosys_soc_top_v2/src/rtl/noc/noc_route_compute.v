module noc_route_compute (
	cur_row,
	cur_col,
	dst_row,
	dst_col,
	out_port,
	route_valid
);
	reg _sv2v_0;
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	parameter signed [31:0] MESH_ROWS = noc_pkg_MESH_ROWS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	parameter signed [31:0] MESH_COLS = noc_pkg_MESH_COLS;
	localparam signed [31:0] noc_pkg_ROW_BITS = 2;
	input wire [1:0] cur_row;
	localparam signed [31:0] noc_pkg_COL_BITS = 2;
	input wire [1:0] cur_col;
	input wire [1:0] dst_row;
	input wire [1:0] dst_col;
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	output reg [2:0] out_port;
	output reg route_valid;
	localparam signed [31:0] noc_pkg_PORT_EAST = 1;
	localparam signed [31:0] noc_pkg_PORT_LOCAL = 4;
	localparam signed [31:0] noc_pkg_PORT_NORTH = 0;
	localparam signed [31:0] noc_pkg_PORT_SOUTH = 2;
	localparam signed [31:0] noc_pkg_PORT_WEST = 3;
	always @(*) begin
		if (_sv2v_0)
			;
		route_valid = 1'b1;
		if (dst_col > cur_col)
			out_port = noc_pkg_PORT_EAST;
		else if (dst_col < cur_col)
			out_port = noc_pkg_PORT_WEST;
		else if (dst_row > cur_row)
			out_port = noc_pkg_PORT_SOUTH;
		else if (dst_row < cur_row)
			out_port = noc_pkg_PORT_NORTH;
		else begin
			out_port = noc_pkg_PORT_LOCAL;
			route_valid = 1'b1;
		end
	end
	initial _sv2v_0 = 0;
endmodule
