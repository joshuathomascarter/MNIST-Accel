module noc_switch_allocator (
	clk,
	rst_n,
	sa_req,
	sa_target,
	out_has_credit,
	alloc_vc,
	sa_grant,
	xbar_sel,
	xbar_valid
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	input wire clk;
	input wire rst_n;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] sa_req;
	localparam signed [31:0] noc_pkg_NUM_PORTS = 5;
	localparam signed [31:0] noc_pkg_PORT_BITS = 3;
	input wire [((NUM_PORTS * NUM_VCS) * 3) - 1:0] sa_target;
	input wire [(NUM_PORTS * NUM_VCS) - 1:0] out_has_credit;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	input wire [((NUM_PORTS * NUM_VCS) * 2) - 1:0] alloc_vc;
	output reg [(NUM_PORTS * NUM_VCS) - 1:0] sa_grant;
	output reg [(NUM_PORTS * noc_pkg_PORT_BITS) - 1:0] xbar_sel;
	output reg [0:NUM_PORTS - 1] xbar_valid;
	reg [(NUM_PORTS * NUM_VCS) - 1:0] op_req_flat [0:NUM_PORTS - 1];
	reg [$clog2(NUM_PORTS * NUM_VCS) - 1:0] op_rr [0:NUM_PORTS - 1];
	localparam signed [31:0] TOTAL = NUM_PORTS * NUM_VCS;
	reg [$clog2(TOTAL) - 1:0] flat_idx_c;
	integer rotated_c;
	reg done_c;
	reg [$clog2(NUM_PORTS) - 1:0] selected_op_c;
	integer winner_c;
	reg p1_grant_valid [0:NUM_PORTS - 1];
	reg [2:0] p1_grant_ip [0:NUM_PORTS - 1];
	reg [1:0] p1_grant_iv [0:NUM_PORTS - 1];
	function automatic signed [$clog2(TOTAL) - 1:0] sv2v_cast_8D092_signed;
		input reg signed [$clog2(TOTAL) - 1:0] inp;
		sv2v_cast_8D092_signed = inp;
	endfunction
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					op_req_flat[op] = '0;
					begin : sv2v_autoblock_2
						reg signed [31:0] ip;
						for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
							begin : sv2v_autoblock_3
								reg signed [31:0] iv;
								for (iv = 0; iv < NUM_VCS; iv = iv + 1)
									begin
										flat_idx_c = sv2v_cast_8D092_signed((ip * NUM_VCS) + iv);
										op_req_flat[op][flat_idx_c] = (sa_req[(((NUM_PORTS - 1) - ip) * NUM_VCS) + iv] && (sa_target[((((NUM_PORTS - 1) - ip) * NUM_VCS) + ((NUM_VCS - 1) - iv)) * 3+:3] == op)) && out_has_credit[(((NUM_PORTS - 1) - op) * NUM_VCS) + alloc_vc[((ip * NUM_VCS) + iv) * 2+:2]];
									end
							end
					end
				end
		end
		begin : sv2v_autoblock_4
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					p1_grant_valid[op] = 1'b0;
					p1_grant_ip[op] = '0;
					p1_grant_iv[op] = '0;
					begin : sv2v_autoblock_5
						reg signed [31:0] t;
						for (t = 0; t < TOTAL; t = t + 1)
							begin
								rotated_c = (sv2v_cast_32_signed(op_rr[op]) + t) % TOTAL;
								if (!p1_grant_valid[op] && op_req_flat[op][rotated_c]) begin
									p1_grant_valid[op] = 1'b1;
									p1_grant_ip[op] = rotated_c / NUM_VCS;
									p1_grant_iv[op] = rotated_c % NUM_VCS;
								end
							end
					end
				end
		end
	end
	reg [$clog2(NUM_PORTS) - 1:0] ip_rr [0:NUM_PORTS - 1];
	function automatic signed [$clog2(NUM_PORTS) - 1:0] sv2v_cast_05882_signed;
		input reg signed [$clog2(NUM_PORTS) - 1:0] inp;
		sv2v_cast_05882_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_6
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				sa_grant[((NUM_PORTS - 1) - ip) * NUM_VCS+:NUM_VCS] = '0;
		end
		begin : sv2v_autoblock_7
			reg signed [31:0] op;
			for (op = 0; op < NUM_PORTS; op = op + 1)
				begin
					xbar_sel[((NUM_PORTS - 1) - op) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS] = '0;
					xbar_valid[op] = 1'b0;
				end
		end
		begin : sv2v_autoblock_8
			reg signed [31:0] ip;
			for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
				begin
					done_c = 1'b0;
					begin : sv2v_autoblock_9
						reg signed [31:0] r;
						for (r = 0; r < NUM_PORTS; r = r + 1)
							begin
								selected_op_c = sv2v_cast_05882_signed((sv2v_cast_32_signed(ip_rr[ip]) + r) % NUM_PORTS);
								if ((!done_c && p1_grant_valid[selected_op_c]) && (p1_grant_ip[selected_op_c] == ip)) begin
									sa_grant[(((NUM_PORTS - 1) - ip) * NUM_VCS) + p1_grant_iv[selected_op_c]] = 1'b1;
									xbar_sel[((NUM_PORTS - 1) - selected_op_c) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS] = ip;
									xbar_valid[selected_op_c] = 1'b1;
									done_c = 1'b1;
								end
							end
					end
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			begin : sv2v_autoblock_10
				reg signed [31:0] op;
				for (op = 0; op < NUM_PORTS; op = op + 1)
					op_rr[op] <= '0;
			end
			begin : sv2v_autoblock_11
				reg signed [31:0] ip;
				for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
					ip_rr[ip] <= '0;
			end
		end
		else begin
			begin : sv2v_autoblock_12
				reg signed [31:0] op;
				for (op = 0; op < NUM_PORTS; op = op + 1)
					if (xbar_valid[op]) begin
						winner_c = (sv2v_cast_32_signed(p1_grant_ip[op]) * NUM_VCS) + sv2v_cast_32_signed(p1_grant_iv[op]);
						op_rr[op] <= sv2v_cast_8D092_signed((winner_c + 1) % TOTAL);
					end
			end
			begin : sv2v_autoblock_13
				reg signed [31:0] ip;
				for (ip = 0; ip < NUM_PORTS; ip = ip + 1)
					if (sa_grant[((NUM_PORTS - 1) - ip) * NUM_VCS+:NUM_VCS] != '0) begin : sv2v_autoblock_14
						reg signed [31:0] op;
						for (op = 0; op < NUM_PORTS; op = op + 1)
							if (xbar_valid[op] && (xbar_sel[((NUM_PORTS - 1) - op) * noc_pkg_PORT_BITS+:noc_pkg_PORT_BITS] == ip))
								ip_rr[ip] <= sv2v_cast_05882_signed((op + 1) % NUM_PORTS);
					end
			end
		end
	initial _sv2v_0 = 0;
endmodule
