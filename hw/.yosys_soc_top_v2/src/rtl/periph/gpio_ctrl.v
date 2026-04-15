module gpio_ctrl (
	clk,
	rst_n,
	gpio_o,
	gpio_i,
	gpio_oe,
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
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] GPIO_WIDTH = 8;
	input wire clk;
	input wire rst_n;
	output wire [GPIO_WIDTH - 1:0] gpio_o;
	input wire [GPIO_WIDTH - 1:0] gpio_i;
	output wire [GPIO_WIDTH - 1:0] gpio_oe;
	input wire awvalid;
	output wire awready;
	input wire [ADDR_WIDTH - 1:0] awaddr;
	input wire [2:0] awsize;
	input wire [1:0] awburst;
	input wire [3:0] awid;
	input wire wvalid;
	output wire wready;
	input wire [DATA_WIDTH - 1:0] wdata;
	input wire [(DATA_WIDTH / 8) - 1:0] wstrb;
	input wire wlast;
	output wire bvalid;
	input wire bready;
	output wire [1:0] bresp;
	output wire [3:0] bid;
	input wire arvalid;
	output wire arready;
	input wire [ADDR_WIDTH - 1:0] araddr;
	input wire [2:0] arsize;
	input wire [1:0] arburst;
	input wire [3:0] arid;
	output wire rvalid;
	input wire rready;
	output reg [DATA_WIDTH - 1:0] rdata;
	output wire [1:0] rresp;
	output wire [3:0] rid;
	output wire rlast;
	localparam [7:0] DIR = 8'h00;
	localparam [7:0] OUT = 8'h04;
	localparam [7:0] IN = 8'h08;
	reg [GPIO_WIDTH - 1:0] dir_reg;
	reg [GPIO_WIDTH - 1:0] out_reg;
	reg [GPIO_WIDTH - 1:0] in_sync;
	reg [GPIO_WIDTH - 1:0] in_sync_r;
	reg [3:0] aw_id;
	reg [3:0] ar_id;
	reg [7:0] aw_addr_r;
	reg [7:0] ar_addr_r;
	reg [DATA_WIDTH - 1:0] wdata_r;
	reg [(DATA_WIDTH / 8) - 1:0] wstrb_r;
	reg b_pending;
	reg ar_valid;
	reg aw_seen;
	reg w_seen;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			in_sync_r <= '0;
			in_sync <= '0;
		end
		else begin
			in_sync_r <= gpio_i;
			in_sync <= in_sync_r;
		end
	assign gpio_o = out_reg;
	assign gpio_oe = dir_reg;
	assign awready = !b_pending && !aw_seen;
	assign wready = !b_pending && !w_seen;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			b_pending <= 1'b0;
			aw_id <= '0;
			aw_addr_r <= '0;
			wdata_r <= '0;
			wstrb_r <= '0;
			aw_seen <= 1'b0;
			w_seen <= 1'b0;
			dir_reg <= '0;
			out_reg <= '0;
		end
		else begin
			if (b_pending && bready)
				b_pending <= 1'b0;
			if (awvalid && awready) begin
				aw_seen <= 1'b1;
				aw_id <= awid;
				aw_addr_r <= awaddr[7:0];
			end
			if (wvalid && wready) begin
				w_seen <= 1'b1;
				wdata_r <= wdata;
				wstrb_r <= wstrb;
			end
			if ((!b_pending && aw_seen) && w_seen) begin
				b_pending <= 1'b1;
				aw_seen <= 1'b0;
				w_seen <= 1'b0;
				case (aw_addr_r)
					DIR:
						if (wstrb_r[0])
							dir_reg <= wdata_r[GPIO_WIDTH - 1:0];
					OUT:
						if (wstrb_r[0])
							out_reg <= wdata_r[GPIO_WIDTH - 1:0];
					default:
						;
				endcase
			end
		end
	assign bvalid = b_pending;
	assign bresp = 2'b00;
	assign bid = aw_id;
	assign arready = 1'b1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			ar_valid <= 1'b0;
			ar_id <= '0;
			ar_addr_r <= '0;
		end
		else if (arvalid && arready) begin
			ar_valid <= 1'b1;
			ar_id <= arid;
			ar_addr_r <= araddr[7:0];
		end
		else if (rvalid && rready)
			ar_valid <= 1'b0;
	assign rvalid = ar_valid;
	assign rid = ar_id;
	assign rresp = 2'b00;
	assign rlast = 1'b1;
	always @(*) begin
		if (_sv2v_0)
			;
		case (ar_addr_r)
			DIR: rdata = {24'b000000000000000000000000, dir_reg};
			OUT: rdata = {24'b000000000000000000000000, out_reg};
			IN: rdata = {24'b000000000000000000000000, in_sync};
			default: rdata = '0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
