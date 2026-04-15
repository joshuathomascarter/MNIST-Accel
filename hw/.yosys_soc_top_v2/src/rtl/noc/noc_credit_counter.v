module noc_credit_counter (
	clk,
	rst_n,
	credit_in,
	flit_sent,
	flit_vc,
	has_credit
);
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	parameter signed [31:0] BUF_DEPTH = noc_pkg_BUF_DEPTH;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	input wire clk;
	input wire rst_n;
	input wire [NUM_VCS - 1:0] credit_in;
	input wire flit_sent;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	input wire [1:0] flit_vc;
	output wire [NUM_VCS - 1:0] has_credit;
	localparam signed [31:0] CNT_BITS = $clog2(BUF_DEPTH + 1);
	reg [CNT_BITS - 1:0] count [0:NUM_VCS - 1];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				count[v] <= BUF_DEPTH;
		end
		else begin : sv2v_autoblock_2
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				begin : sv2v_autoblock_3
					reg inc;
					reg dec;
					inc = credit_in[v];
					dec = flit_sent && (flit_vc == v);
					case ({inc, dec})
						2'b10: count[v] <= count[v] + 1;
						2'b01: count[v] <= count[v] - 1;
						default: count[v] <= count[v];
					endcase
				end
		end
	genvar _gv_v_1;
	generate
		for (_gv_v_1 = 0; _gv_v_1 < NUM_VCS; _gv_v_1 = _gv_v_1 + 1) begin : gen_has_credit
			localparam v = _gv_v_1;
			assign has_credit[v] = count[v] != '0;
		end
	endgenerate
endmodule
