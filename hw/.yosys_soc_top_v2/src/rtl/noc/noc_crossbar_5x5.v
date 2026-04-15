module noc_crossbar_5x5 (
	in_flit,
	in_vc,
	xbar_sel,
	xbar_valid,
	out_flit,
	out_valid
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_PORTS = 5;
	input wire [(NUM_PORTS * 64) - 1:0] in_flit;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	input wire [(NUM_PORTS * noc_pkg_VC_BITS) - 1:0] in_vc;
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	input wire [(NUM_PORTS * noc_pkg_PORT_BITS) - 1:0] xbar_sel;
	input wire [0:NUM_PORTS - 1] xbar_valid;
	output reg [(NUM_PORTS * 64) - 1:0] out_flit;
	output reg [0:NUM_PORTS - 1] out_valid;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				if (xbar_valid[op]) begin
					out_flit[((NUM_PORTS - 1) - op) * 64+:64] = in_flit[((NUM_PORTS - 1) - xbar_sel[((NUM_PORTS - 1) - op) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS]) * 64+:64];
					out_flit[(((NUM_PORTS - 1) - op) * 64) + 53-:2] = in_vc[((NUM_PORTS - 1) - xbar_sel[((NUM_PORTS - 1) - op) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS]) * noc_pkg_VC_BITS+:noc_pkg_VC_BITS];
					out_valid[op] = 1'b1;
				end
				else begin
					out_flit[((NUM_PORTS - 1) - op) * 64+:64] = '0;
					out_valid[op] = 1'b0;
				end
		end
	end
	initial _sv2v_0 = 0;
endmodule
