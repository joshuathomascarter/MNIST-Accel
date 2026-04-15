module timer_ctrl (
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
	arsize,
	arburst,
	arid,
	rvalid,
	rready,
	rdata,
	rresp,
	rid,
	rlast,
	irq_timer_o
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
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
	output wire irq_timer_o;
	localparam [7:0] MTIME_LO = 8'h00;
	localparam [7:0] MTIME_HI = 8'h04;
	localparam [7:0] MTIMECMP_LO = 8'h08;
	localparam [7:0] MTIMECMP_HI = 8'h0c;
	reg [63:0] mtime;
	reg [63:0] mtimecmp;
	reg [3:0] aw_id;
	reg [3:0] ar_id;
	reg [7:0] ar_addr_r;
	reg b_pending;
	reg ar_valid;
	assign irq_timer_o = mtime >= mtimecmp;
	assign awready = !b_pending;
	assign wready = awvalid && !b_pending;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			b_pending <= 1'b0;
			aw_id <= '0;
			mtime <= 64'b0000000000000000000000000000000000000000000000000000000000000000;
			mtimecmp <= 64'hffffffffffffffff;
		end
		else begin
			if (b_pending && bready)
				b_pending <= 1'b0;
			mtime <= mtime + 64'b0000000000000000000000000000000000000000000000000000000000000001;
			if ((awvalid && wvalid) && !b_pending) begin
				b_pending <= 1'b1;
				aw_id <= awid;
				case (awaddr[7:0])
					MTIME_LO: begin
						if (wstrb[0])
							mtime[7:0] <= wdata[7:0];
						if (wstrb[1])
							mtime[15:8] <= wdata[15:8];
						if (wstrb[2])
							mtime[23:16] <= wdata[23:16];
						if (wstrb[3])
							mtime[31:24] <= wdata[31:24];
					end
					MTIME_HI: begin
						if (wstrb[0])
							mtime[39:32] <= wdata[7:0];
						if (wstrb[1])
							mtime[47:40] <= wdata[15:8];
						if (wstrb[2])
							mtime[55:48] <= wdata[23:16];
						if (wstrb[3])
							mtime[63:56] <= wdata[31:24];
					end
					MTIMECMP_LO: begin
						if (wstrb[0])
							mtimecmp[7:0] <= wdata[7:0];
						if (wstrb[1])
							mtimecmp[15:8] <= wdata[15:8];
						if (wstrb[2])
							mtimecmp[23:16] <= wdata[23:16];
						if (wstrb[3])
							mtimecmp[31:24] <= wdata[31:24];
					end
					MTIMECMP_HI: begin
						if (wstrb[0])
							mtimecmp[39:32] <= wdata[7:0];
						if (wstrb[1])
							mtimecmp[47:40] <= wdata[15:8];
						if (wstrb[2])
							mtimecmp[55:48] <= wdata[23:16];
						if (wstrb[3])
							mtimecmp[63:56] <= wdata[31:24];
					end
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
			MTIME_LO: rdata = mtime[31:0];
			MTIME_HI: rdata = mtime[63:32];
			MTIMECMP_LO: rdata = mtimecmp[31:0];
			MTIMECMP_HI: rdata = mtimecmp[63:32];
			default: rdata = '0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
