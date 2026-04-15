`default_nettype none
module act_dma (
	clk,
	rst_n,
	start,
	src_addr,
	transfer_length,
	done,
	busy,
	error,
	m_axi_arid,
	m_axi_araddr,
	m_axi_arlen,
	m_axi_arsize,
	m_axi_arburst,
	m_axi_arvalid,
	m_axi_arready,
	m_axi_rid,
	m_axi_rdata,
	m_axi_rresp,
	m_axi_rlast,
	m_axi_rvalid,
	m_axi_rready,
	act_we,
	act_addr,
	act_wdata
);
	parameter AXI_ADDR_W = 32;
	parameter AXI_DATA_W = 64;
	parameter AXI_ID_W = 4;
	parameter STREAM_ID = 1;
	parameter BURST_LEN = 8'd15;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [AXI_ADDR_W - 1:0] src_addr;
	input wire [31:0] transfer_length;
	output reg done;
	output reg busy;
	output reg error;
	output wire [AXI_ID_W - 1:0] m_axi_arid;
	output reg [AXI_ADDR_W - 1:0] m_axi_araddr;
	output reg [7:0] m_axi_arlen;
	output reg [2:0] m_axi_arsize;
	output reg [1:0] m_axi_arburst;
	output reg m_axi_arvalid;
	input wire m_axi_arready;
	input wire [AXI_ID_W - 1:0] m_axi_rid;
	input wire [AXI_DATA_W - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire m_axi_rlast;
	input wire m_axi_rvalid;
	output reg m_axi_rready;
	output reg act_we;
	output reg [AXI_ADDR_W - 1:0] act_addr;
	output reg [AXI_DATA_W - 1:0] act_wdata;
	reg [1:0] state;
	reg [AXI_ADDR_W - 1:0] current_axi_addr;
	reg [31:0] bytes_remaining;
	localparam [2:0] AXI_SIZE_64 = 3'b011;
	localparam [1:0] AXI_BURST_INCR = 2'b01;
	assign m_axi_arid = STREAM_ID[AXI_ID_W - 1:0];
	wire _unused_act_axi_rid = &{1'b0, m_axi_rid};
	wire [12:0] page_byte_rem = 13'h1000 - {1'b0, current_axi_addr[11:0]};
	wire [9:0] page_max_beats = page_byte_rem[12:3];
	wire _unused_page_byte_lo = &{1'b0, page_byte_rem[2:0]};
	wire [31:0] max_burst_bytes = 32'd8 * (32'd1 + {24'd0, BURST_LEN});
	wire [31:0] data_arlen_w = (bytes_remaining > max_burst_bytes ? {24'd0, BURST_LEN} : ((bytes_remaining + 32'd7) >> 3) - 32'd1);
	wire [7:0] data_arlen = data_arlen_w[7:0];
	wire _unused_arlen_hi = &{1'b0, data_arlen_w[31:8]};
	wire [7:0] safe_arlen = (({2'b00, data_arlen} + 10'd1) > page_max_beats ? page_max_beats[7:0] - 8'd1 : data_arlen);
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			busy <= 1'b0;
			done <= 1'b0;
			error <= 1'b0;
			m_axi_arvalid <= 1'b0;
			m_axi_rready <= 1'b0;
			act_we <= 1'b0;
			act_addr <= 0;
			current_axi_addr <= 0;
			bytes_remaining <= 0;
		end
		else begin
			act_we <= 1'b0;
			done <= 1'b0;
			case (state)
				2'd0:
					if (start) begin
						busy <= 1'b1;
						error <= 1'b0;
						current_axi_addr <= src_addr;
						bytes_remaining <= transfer_length;
						act_addr <= 0;
						state <= 2'd1;
					end
				2'd1: begin
					m_axi_araddr <= current_axi_addr;
					m_axi_arsize <= AXI_SIZE_64;
					m_axi_arburst <= AXI_BURST_INCR;
					m_axi_arvalid <= 1'b1;
					m_axi_arlen <= safe_arlen;
					if (m_axi_arready && m_axi_arvalid) begin
						m_axi_arvalid <= 1'b0;
						m_axi_rready <= 1'b1;
						state <= 2'd2;
					end
				end
				2'd2:
					if (m_axi_rvalid) begin
						if (m_axi_rresp != 2'b00) begin
							error <= 1'b1;
							busy <= 1'b0;
							done <= 1'b1;
							state <= 2'd0;
						end
						else begin
							act_we <= 1'b1;
							act_wdata <= m_axi_rdata;
							act_addr <= act_addr + 1;
							if (bytes_remaining >= 8)
								bytes_remaining <= bytes_remaining - 8;
							else
								bytes_remaining <= 0;
							if (m_axi_rlast) begin
								m_axi_rready <= 1'b0;
								current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
								if (bytes_remaining <= 8)
									state <= 2'd3;
								else
									state <= 2'd1;
							end
						end
					end
				2'd3: begin
					busy <= 1'b0;
					done <= 1'b1;
					state <= 2'd0;
				end
				default: state <= 2'd0;
			endcase
		end
	always @(posedge clk) begin
		if (start && (transfer_length == 0))
;
		if (start && (src_addr[2:0] != 3'b000))
;
	end
	always @(posedge clk)
		if ((state == 2'd1) && m_axi_arvalid) begin
			if ((m_axi_araddr[11:0] + (({24'd0, m_axi_arlen} + 32'd1) * 8)) > 32'h00000fff)
;
		end
endmodule
