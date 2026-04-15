module sram_ctrl (
	clk,
	rst_n,
	awvalid,
	awready,
	awaddr,
	awlen,
	awsize,
	awburst,
	awid,
	wvalid,
	wready,
	wdata,
	wstrb,
	wlast,
	bvalid,
	bready,
	bresp,
	bid,
	arvalid,
	arready,
	araddr,
	arsize,
	arburst,
	arid,
	arlen,
	rvalid,
	rready,
	rdata,
	rresp,
	rid,
	rlast
);
	parameter [31:0] ADDR_WIDTH = 15;
	parameter [31:0] DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire awvalid;
	output wire awready;
	input wire [31:0] awaddr;
	input wire [7:0] awlen;
	input wire [2:0] awsize;
	input wire [1:0] awburst;
	input wire [3:0] awid;
	input wire wvalid;
	output wire wready;
	input wire [DATA_WIDTH - 1:0] wdata;
	input wire [(DATA_WIDTH / 8) - 1:0] wstrb;
	input wire wlast;
	output reg bvalid;
	input wire bready;
	output wire [1:0] bresp;
	output reg [3:0] bid;
	input wire arvalid;
	output wire arready;
	input wire [31:0] araddr;
	input wire [2:0] arsize;
	input wire [1:0] arburst;
	input wire [3:0] arid;
	input wire [7:0] arlen;
	output wire rvalid;
	input wire rready;
	output wire [DATA_WIDTH - 1:0] rdata;
	output wire [1:0] rresp;
	output wire [3:0] rid;
	output wire rlast;
	localparam [31:0] DEPTH = 2 ** (ADDR_WIDTH - 2);
	reg [DATA_WIDTH - 1:0] sram [0:DEPTH - 1];
	reg [ADDR_WIDTH - 3:0] aw_addr_cur;
	reg [3:0] aw_id;
	reg aw_valid;
	wire w_valid;
	reg [ADDR_WIDTH - 3:0] ar_addr;
	reg [3:0] ar_id;
	reg ar_valid;
	reg [7:0] ar_len;
	reg [7:0] ar_beat_cnt;
	initial begin : sv2v_autoblock_1
		reg signed [31:0] i;
		for (i = 0; i < DEPTH; i = i + 1)
			sram[i] = '0;
	end
	assign awready = !aw_valid;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			aw_valid <= 1'b0;
			aw_addr_cur <= '0;
			aw_id <= '0;
		end
		else if (awvalid && awready) begin
			aw_valid <= 1'b1;
			aw_addr_cur <= awaddr[ADDR_WIDTH - 1:2];
			aw_id <= awid;
		end
		else if (wvalid && wready) begin
			aw_addr_cur <= aw_addr_cur + 1;
			if (wlast)
				aw_valid <= 1'b0;
		end
	assign wready = aw_valid;
	always @(posedge clk)
		if (wvalid && wready) begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < (DATA_WIDTH / 8); i = i + 1)
				if (wstrb[i])
					sram[aw_addr_cur][i * 8+:8] <= wdata[i * 8+:8];
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			bvalid <= 1'b0;
			bid <= '0;
		end
		else if ((wvalid && wready) && wlast) begin
			bvalid <= 1'b1;
			bid <= aw_id;
		end
		else if (bready)
			bvalid <= 1'b0;
	localparam [1:0] soc_pkg_RESP_OKAY = 2'b00;
	assign bresp = soc_pkg_RESP_OKAY;
	assign arready = !ar_valid;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			ar_valid <= 1'b0;
			ar_addr <= '0;
			ar_id <= '0;
			ar_len <= '0;
			ar_beat_cnt <= '0;
		end
		else if (arvalid && arready) begin
			ar_valid <= 1'b1;
			ar_addr <= araddr[ADDR_WIDTH - 1:2];
			ar_id <= arid;
			ar_len <= arlen;
			ar_beat_cnt <= '0;
		end
		else if (ar_valid && rready) begin
			if (ar_beat_cnt == ar_len)
				ar_valid <= 1'b0;
			else begin
				ar_beat_cnt <= ar_beat_cnt + 1;
				ar_addr <= ar_addr + 1;
			end
		end
	assign rvalid = ar_valid;
	assign rdata = sram[ar_addr];
	assign rresp = soc_pkg_RESP_OKAY;
	assign rid = ar_id;
	assign rlast = ar_beat_cnt == ar_len;
endmodule
