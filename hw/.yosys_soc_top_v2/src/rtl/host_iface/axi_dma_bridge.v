`default_nettype none
module axi_dma_bridge (
	clk,
	rst_n,
	s0_arid,
	s0_araddr,
	s0_arlen,
	s0_arsize,
	s0_arburst,
	s0_arvalid,
	s0_arready,
	s1_arid,
	s1_araddr,
	s1_arlen,
	s1_arsize,
	s1_arburst,
	s1_arvalid,
	s1_arready,
	m_arid,
	m_araddr,
	m_arlen,
	m_arsize,
	m_arburst,
	m_arvalid,
	m_arready,
	m_rid,
	m_rdata,
	m_rresp,
	m_rlast,
	m_rvalid,
	m_rready,
	s0_rid,
	s0_rdata,
	s0_rresp,
	s0_rlast,
	s0_rvalid,
	s0_rready,
	s1_rid,
	s1_rdata,
	s1_rresp,
	s1_rlast,
	s1_rvalid,
	s1_rready
);
	parameter DATA_WIDTH = 64;
	parameter ADDR_WIDTH = 32;
	parameter ID_WIDTH = 4;
	input wire clk;
	input wire rst_n;
	input wire [ID_WIDTH - 1:0] s0_arid;
	input wire [ADDR_WIDTH - 1:0] s0_araddr;
	input wire [7:0] s0_arlen;
	input wire [2:0] s0_arsize;
	input wire [1:0] s0_arburst;
	input wire s0_arvalid;
	output reg s0_arready;
	input wire [ID_WIDTH - 1:0] s1_arid;
	input wire [ADDR_WIDTH - 1:0] s1_araddr;
	input wire [7:0] s1_arlen;
	input wire [2:0] s1_arsize;
	input wire [1:0] s1_arburst;
	input wire s1_arvalid;
	output reg s1_arready;
	output reg [ID_WIDTH - 1:0] m_arid;
	output reg [ADDR_WIDTH - 1:0] m_araddr;
	output reg [7:0] m_arlen;
	output reg [2:0] m_arsize;
	output reg [1:0] m_arburst;
	output reg m_arvalid;
	input wire m_arready;
	input wire [ID_WIDTH - 1:0] m_rid;
	input wire [DATA_WIDTH - 1:0] m_rdata;
	input wire [1:0] m_rresp;
	input wire m_rlast;
	input wire m_rvalid;
	output reg m_rready;
	output reg [ID_WIDTH - 1:0] s0_rid;
	output reg [DATA_WIDTH - 1:0] s0_rdata;
	output reg [1:0] s0_rresp;
	output reg s0_rlast;
	output reg s0_rvalid;
	input wire s0_rready;
	output reg [ID_WIDTH - 1:0] s1_rid;
	output reg [DATA_WIDTH - 1:0] s1_rdata;
	output reg [1:0] s1_rresp;
	output reg s1_rlast;
	output reg s1_rvalid;
	input wire s1_rready;
	reg [1:0] state;
	reg current_master;
	reg last_master;
	reg [9:0] watchdog_timer;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			current_master <= 1'b0;
			last_master <= 1'b0;
			watchdog_timer <= 10'd0;
		end
		else begin
			if (state == 2'd2)
				watchdog_timer <= watchdog_timer + 1'b1;
			else
				watchdog_timer <= 10'd0;
			case (state)
				2'd0:
					if (s0_arvalid && (!s1_arvalid || (last_master == 1'b1))) begin
						current_master <= 1'b0;
						state <= 2'd1;
					end
					else if (s1_arvalid) begin
						current_master <= 1'b1;
						state <= 2'd1;
					end
				2'd1:
					if (m_arready && m_arvalid)
						state <= 2'd2;
				2'd2:
					if ((m_rvalid && m_rready) && m_rlast) begin
						last_master <= current_master;
						state <= 2'd0;
					end
					else if (watchdog_timer == 10'h3ff)
						state <= 2'd0;
				default: state <= 2'd0;
			endcase
		end
	always @(*) begin
		m_arid = {ID_WIDTH {1'b0}};
		m_araddr = {ADDR_WIDTH {1'b0}};
		m_arlen = 8'd0;
		m_arsize = 3'd0;
		m_arburst = 2'd0;
		m_arvalid = 1'b0;
		s0_arready = 1'b0;
		s1_arready = 1'b0;
		if (state == 2'd1) begin
			if (current_master == 1'b0) begin
				m_arid = s0_arid;
				m_araddr = s0_araddr;
				m_arlen = s0_arlen;
				m_arsize = s0_arsize;
				m_arburst = s0_arburst;
				m_arvalid = s0_arvalid;
				s0_arready = m_arready;
			end
			else begin
				m_arid = s1_arid;
				m_araddr = s1_araddr;
				m_arlen = s1_arlen;
				m_arsize = s1_arsize;
				m_arburst = s1_arburst;
				m_arvalid = s1_arvalid;
				s1_arready = m_arready;
			end
		end
	end
	always @(*) begin
		s0_rid = m_rid;
		s0_rdata = m_rdata;
		s0_rresp = m_rresp;
		s0_rlast = m_rlast;
		s0_rvalid = 1'b0;
		s1_rid = m_rid;
		s1_rdata = m_rdata;
		s1_rresp = m_rresp;
		s1_rlast = m_rlast;
		s1_rvalid = 1'b0;
		m_rready = 1'b0;
		if (state == 2'd2) begin
			if (current_master == 1'b0) begin
				s0_rvalid = m_rvalid;
				m_rready = s0_rready;
			end
			else begin
				s1_rvalid = m_rvalid;
				m_rready = s1_rready;
			end
		end
	end
	always @(posedge clk)
		if ((state == 2'd2) && (watchdog_timer == 10'h3fe)) begin
;
;
		end
	always @(posedge clk)
		if ((state == 2'd0) && m_rvalid) begin
;
;
		end
	always @(posedge clk)
		if ((state != 2'd0) && (current_master === 1'bx)) begin
;
;
;
		end
endmodule
`default_nettype wire
