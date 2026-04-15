module boot_rom (
	clk,
	rst_n,
	awvalid,
	awready,
	awaddr,
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
	arlen,
	arsize,
	arburst,
	arid,
	rvalid,
	rready,
	rdata,
	rresp,
	rid,
	rlast
);
	parameter [31:0] ADDR_WIDTH = 12;
	parameter [31:0] DATA_WIDTH = 32;
	parameter INIT_FILE = "firmware.hex";
	input wire clk;
	input wire rst_n;
	input wire awvalid;
	output wire awready;
	input wire [31:0] awaddr;
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
	input wire [7:0] arlen;
	input wire [2:0] arsize;
	input wire [1:0] arburst;
	input wire [3:0] arid;
	output wire rvalid;
	input wire rready;
	output wire [DATA_WIDTH - 1:0] rdata;
	output wire [1:0] rresp;
	output wire [3:0] rid;
	output wire rlast;
	localparam [31:0] BYTE_DEPTH = 2 ** ADDR_WIDTH;
	reg [7:0] rom_array [0:BYTE_DEPTH - 1];
	reg [3:0] aw_id;
	reg aw_valid;
	initial if (INIT_FILE != "")
		$readmemh(INIT_FILE, rom_array);
	assign awready = 1'b1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			aw_valid <= 1'b0;
			aw_id <= '0;
		end
		else if (awvalid && awready) begin
			aw_valid <= 1'b1;
			aw_id <= awid;
		end
		else if (bvalid && bready)
			aw_valid <= 1'b0;
	assign wready = wvalid & aw_valid;
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
	localparam [1:0] soc_pkg_RESP_SLVERR = 2'b10;
	assign bresp = soc_pkg_RESP_SLVERR;
	reg [ADDR_WIDTH - 1:0] ar_addr;
	reg [3:0] ar_id;
	reg ar_active;
	reg [7:0] ar_len;
	reg [7:0] ar_beat_cnt;
	assign arready = !ar_active;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			ar_active <= 1'b0;
			ar_addr <= '0;
			ar_id <= '0;
			ar_len <= '0;
			ar_beat_cnt <= '0;
		end
		else if (!ar_active && arvalid) begin
			ar_active <= 1'b1;
			ar_addr <= araddr[ADDR_WIDTH - 1:0];
			ar_id <= arid;
			ar_len <= arlen;
			ar_beat_cnt <= '0;
		end
		else if ((ar_active && rvalid) && rready) begin
			if (ar_beat_cnt == ar_len)
				ar_active <= 1'b0;
			else begin
				ar_beat_cnt <= ar_beat_cnt + 1;
				ar_addr <= ar_addr + (DATA_WIDTH / 8);
			end
		end
	assign rvalid = ar_active;
	assign rdata = {rom_array[{ar_addr[ADDR_WIDTH - 1:2], 2'd3}], rom_array[{ar_addr[ADDR_WIDTH - 1:2], 2'd2}], rom_array[{ar_addr[ADDR_WIDTH - 1:2], 2'd1}], rom_array[{ar_addr[ADDR_WIDTH - 1:2], 2'd0}]};
	localparam [1:0] soc_pkg_RESP_OKAY = 2'b00;
	assign rresp = soc_pkg_RESP_OKAY;
	assign rid = ar_id;
	assign rlast = ar_beat_cnt == ar_len;
endmodule
