module noc_vc_allocator (
	clk,
	rst_n,
	req,
	req_port,
	vc_busy,
	grant,
	grant_vc,
	release_vc,
	release_id
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	input wire clk;
	input wire rst_n;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] req;
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	input wire [((NUM_PORTS * NUM_VCS) * 3) - 1:0] req_port;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] vc_busy;
	output reg [(NUM_PORTS * NUM_VCS) - 1:0] grant;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	output reg [((NUM_PORTS * NUM_VCS) * 2) - 1:0] grant_vc;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] release_vc;
	input wire [((NUM_PORTS * NUM_VCS) * 2) - 1:0] release_id;
	reg [1:0] rr_ptr [0:NUM_PORTS - 1];
	reg [NUM_VCS - 1:0] vc_free [0:NUM_PORTS - 1];
	reg [1:0] free_vc_idx_c;
	reg found_free_c;
	reg granted_one_c;
	reg any_grant_c;
	integer rotated_c;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				vc_free[op] = ~vc_busy[((NUM_PORTS - 1) - op) * NUM_VCS+:NUM_VCS];
		end
	end
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_2
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				begin : sv2v_autoblock_3
					reg signed [31:0] iv;
					for (iv = 0; iv < NUM_VCS; iv = iv + 1)
						begin
							grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] = 1'b0;
							grant_vc[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 2+:2] = '0;
						end
				end
		end
		begin : sv2v_autoblock_4
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					found_free_c = 1'b0;
					free_vc_idx_c = '0;
					granted_one_c = 1'b0;
					begin : sv2v_autoblock_5
						reg signed [31:0] v;
						for (v = 0; v < NUM_VCS; v = v + 1)
							begin
								rotated_c = (sv2v_cast_32_signed(rr_ptr[op]) + v) % NUM_VCS;
								if (!found_free_c && vc_free[op][rotated_c]) begin
									found_free_c = 1'b1;
									free_vc_idx_c = rotated_c;
								end
							end
					end
					if (found_free_c) begin : sv2v_autoblock_6
						reg signed [31:0] ip;
						for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
							begin : sv2v_autoblock_7
								reg signed [31:0] iv;
								for (iv = 0; iv < NUM_VCS; iv = iv + 1)
									if ((!granted_one_c && req[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv]) && (req_port[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3] == op)) begin
										grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] = 1'b1;
										grant_vc[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 2+:2] = free_vc_idx_c;
										granted_one_c = 1'b1;
									end
							end
					end
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_8
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				rr_ptr[op] <= '0;
		end
		else begin : sv2v_autoblock_9
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					any_grant_c = 1'b0;
					begin : sv2v_autoblock_10
						reg signed [31:0] ip;
						for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
							begin : sv2v_autoblock_11
								reg signed [31:0] iv;
								for (iv = 0; iv < NUM_VCS; iv = iv + 1)
									if (grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] && (req_port[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3] == op))
										any_grant_c = 1'b1;
							end
					end
					if (any_grant_c)
						rr_ptr[op] <= (rr_ptr[op] == (NUM_VCS - 1) ? '0 : rr_ptr[op] + 1);
				end
		end
	initial _sv2v_0 = 0;
endmodule
