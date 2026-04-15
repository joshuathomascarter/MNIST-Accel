`default_nettype none
module perf_axi (
	clk,
	rst_n,
	event_valid,
	s_axi_awvalid,
	s_axi_awready,
	s_axi_awaddr,
	s_axi_wvalid,
	s_axi_wready,
	s_axi_wdata,
	s_axi_wstrb,
	s_axi_bvalid,
	s_axi_bready,
	s_axi_bresp,
	s_axi_arvalid,
	s_axi_arready,
	s_axi_araddr,
	s_axi_rvalid,
	s_axi_rready,
	s_axi_rdata,
	s_axi_rresp
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_COUNTERS = 6;
	parameter signed [31:0] COUNTER_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire [NUM_COUNTERS - 1:0] event_valid;
	input wire s_axi_awvalid;
	output reg s_axi_awready;
	input wire [7:0] s_axi_awaddr;
	input wire s_axi_wvalid;
	output reg s_axi_wready;
	input wire [31:0] s_axi_wdata;
	input wire [3:0] s_axi_wstrb;
	output reg s_axi_bvalid;
	input wire s_axi_bready;
	output reg [1:0] s_axi_bresp;
	input wire s_axi_arvalid;
	output reg s_axi_arready;
	input wire [7:0] s_axi_araddr;
	output reg s_axi_rvalid;
	input wire s_axi_rready;
	output reg [31:0] s_axi_rdata;
	output reg [1:0] s_axi_rresp;
	reg [COUNTER_WIDTH - 1:0] counters [0:NUM_COUNTERS - 1];
	reg signed [31:0] axi_widx;
	reg signed [31:0] axi_ridx;
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		axi_widx = sv2v_cast_32_signed(s_axi_awaddr[7:2]);
		axi_ridx = sv2v_cast_32_signed(s_axi_araddr[7:2]);
	end
	reg [NUM_COUNTERS - 1:0] clear_en;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			clear_en <= '0;
			s_axi_awready <= 1'b0;
			s_axi_wready <= 1'b0;
			s_axi_bvalid <= 1'b0;
			s_axi_bresp <= 2'b00;
			s_axi_arready <= 1'b0;
			s_axi_rvalid <= 1'b0;
			s_axi_rdata <= '0;
			s_axi_rresp <= 2'b00;
		end
		else begin
			clear_en <= '0;
			if (s_axi_awvalid && !s_axi_bvalid) begin
				s_axi_awready <= 1'b1;
				s_axi_wready <= 1'b1;
				if (s_axi_wvalid) begin
					if ((axi_widx >= 0) && (axi_widx < NUM_COUNTERS))
						clear_en[axi_widx] <= 1'b1;
					s_axi_bvalid <= 1'b1;
					s_axi_bresp <= 2'b00;
				end
			end
			else begin
				s_axi_awready <= 1'b0;
				s_axi_wready <= 1'b0;
			end
			if (s_axi_bvalid && s_axi_bready)
				s_axi_bvalid <= 1'b0;
			if (s_axi_arvalid && !s_axi_rvalid) begin
				s_axi_arready <= 1'b1;
				s_axi_rvalid <= 1'b1;
				s_axi_rresp <= 2'b00;
				s_axi_rdata <= ((axi_ridx >= 0) && (axi_ridx < NUM_COUNTERS) ? counters[axi_ridx][31:0] : 32'b00000000000000000000000000000000);
			end
			else
				s_axi_arready <= 1'b0;
			if (s_axi_rvalid && s_axi_rready)
				s_axi_rvalid <= 1'b0;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_COUNTERS; i = i + 1)
				counters[i] <= '0;
		end
		else begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < NUM_COUNTERS; i = i + 1)
				if (clear_en[i])
					counters[i] <= '0;
				else if (event_valid[i])
					counters[i] <= counters[i] + 1;
		end
	initial _sv2v_0 = 0;
endmodule
`default_nettype wire
