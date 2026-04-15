module axi_lite_slave (
	clk,
	rst_n,
	s_axi_awvalid,
	s_axi_arvalid,
	s_axi_wvalid,
	s_axi_awready,
	s_axi_arready,
	s_axi_wready,
	s_axi_awaddr,
	s_axi_araddr,
	s_axi_awprot,
	s_axi_arprot,
	s_axi_wdata,
	s_axi_wstrb,
	s_axi_bvalid,
	s_axi_rvalid,
	s_axi_bready,
	s_axi_rready,
	s_axi_bresp,
	s_axi_rresp,
	s_axi_rdata,
	csr_wen,
	csr_ren,
	csr_addr,
	csr_wdata,
	csr_rdata,
	axi_error
);
	parameter CSR_ADDR_WIDTH = 8;
	parameter CSR_DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire s_axi_awvalid;
	input wire s_axi_arvalid;
	input wire s_axi_wvalid;
	output wire s_axi_awready;
	output wire s_axi_arready;
	output wire s_axi_wready;
	input wire [CSR_ADDR_WIDTH - 1:0] s_axi_awaddr;
	input wire [CSR_ADDR_WIDTH - 1:0] s_axi_araddr;
	input wire [2:0] s_axi_awprot;
	input wire [2:0] s_axi_arprot;
	input wire [CSR_DATA_WIDTH - 1:0] s_axi_wdata;
	input wire [(CSR_DATA_WIDTH / 8) - 1:0] s_axi_wstrb;
	output reg s_axi_bvalid;
	output reg s_axi_rvalid;
	input wire s_axi_bready;
	input wire s_axi_rready;
	output reg [1:0] s_axi_bresp;
	output reg [1:0] s_axi_rresp;
	output reg [CSR_DATA_WIDTH - 1:0] s_axi_rdata;
	output wire csr_wen;
	output wire csr_ren;
	output wire [CSR_ADDR_WIDTH - 1:0] csr_addr;
	output wire [CSR_DATA_WIDTH - 1:0] csr_wdata;
	input wire [CSR_DATA_WIDTH - 1:0] csr_rdata;
	output wire axi_error;
	reg [CSR_ADDR_WIDTH - 1:0] waddr_latch;
	reg [CSR_DATA_WIDTH - 1:0] wdata_latch;
	reg got_waddr;
	reg got_wdata;
	assign axi_error = 1'b0;
	wire _unused_axi_proto = &{1'b0, s_axi_awprot, s_axi_arprot, s_axi_wstrb};
	assign s_axi_awready = !got_waddr && !s_axi_bvalid;
	assign s_axi_wready = !got_wdata && !s_axi_bvalid;
	wire aw_handshake = s_axi_awvalid && s_axi_awready;
	wire w_handshake = s_axi_wvalid && s_axi_wready;
	wire both_complete = (got_waddr || aw_handshake) && (got_wdata || w_handshake);
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			got_waddr <= 1'b0;
			got_wdata <= 1'b0;
			waddr_latch <= '0;
			wdata_latch <= '0;
			s_axi_bvalid <= 1'b0;
			s_axi_bresp <= 2'b00;
		end
		else if (s_axi_bvalid && s_axi_bready) begin
			s_axi_bvalid <= 1'b0;
			got_waddr <= 1'b0;
			got_wdata <= 1'b0;
		end
		else begin
			if (aw_handshake) begin
				waddr_latch <= s_axi_awaddr;
				got_waddr <= 1'b1;
			end
			if (w_handshake) begin
				wdata_latch <= s_axi_wdata;
				got_wdata <= 1'b1;
			end
			if (both_complete && !s_axi_bvalid) begin
				s_axi_bvalid <= 1'b1;
				s_axi_bresp <= 2'b00;
			end
		end
	assign s_axi_arready = !s_axi_rvalid;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			s_axi_rvalid <= 1'b0;
			s_axi_rdata <= '0;
			s_axi_rresp <= 2'b00;
		end
		else begin
			if (s_axi_arvalid && s_axi_arready) begin
				s_axi_rvalid <= 1'b1;
				s_axi_rresp <= 2'b00;
				s_axi_rdata <= csr_rdata;
			end
			if (s_axi_rvalid && s_axi_rready)
				s_axi_rvalid <= 1'b0;
		end
	assign csr_wen = both_complete && !s_axi_bvalid;
	assign csr_addr = (s_axi_arvalid ? s_axi_araddr : (aw_handshake ? s_axi_awaddr : waddr_latch));
	assign csr_wdata = (w_handshake ? s_axi_wdata : wdata_latch);
	assign csr_ren = s_axi_arvalid && s_axi_arready;
endmodule
