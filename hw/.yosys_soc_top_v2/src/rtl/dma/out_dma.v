`default_nettype none
module out_dma (
	clk,
	rst_n,
	start,
	dst_addr,
	done,
	busy,
	accum_rd_en,
	accum_rd_addr,
	accum_rd_data,
	accum_ready,
	m_axi_awid,
	m_axi_awaddr,
	m_axi_awlen,
	m_axi_awsize,
	m_axi_awburst,
	m_axi_awvalid,
	m_axi_awready,
	m_axi_wdata,
	m_axi_wstrb,
	m_axi_wlast,
	m_axi_wvalid,
	m_axi_wready,
	m_axi_bid,
	m_axi_bresp,
	m_axi_bvalid,
	m_axi_bready
);
	parameter AXI_ADDR_W = 32;
	parameter AXI_DATA_W = 64;
	parameter AXI_ID_W = 4;
	parameter BRAM_ADDR_W = 10;
	parameter NUM_ACCS = 256;
	parameter STREAM_ID = 2;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [AXI_ADDR_W - 1:0] dst_addr;
	output reg done;
	output reg busy;
	output reg accum_rd_en;
	output reg [BRAM_ADDR_W - 1:0] accum_rd_addr;
	input wire [63:0] accum_rd_data;
	input wire accum_ready;
	output wire [AXI_ID_W - 1:0] m_axi_awid;
	output reg [AXI_ADDR_W - 1:0] m_axi_awaddr;
	output reg [7:0] m_axi_awlen;
	output wire [2:0] m_axi_awsize;
	output wire [1:0] m_axi_awburst;
	output reg m_axi_awvalid;
	input wire m_axi_awready;
	output reg [AXI_DATA_W - 1:0] m_axi_wdata;
	output wire [(AXI_DATA_W / 8) - 1:0] m_axi_wstrb;
	output reg m_axi_wlast;
	output reg m_axi_wvalid;
	input wire m_axi_wready;
	input wire [AXI_ID_W - 1:0] m_axi_bid;
	input wire [1:0] m_axi_bresp;
	input wire m_axi_bvalid;
	output reg m_axi_bready;
	localparam NUM_WORDS = (NUM_ACCS + 7) / 8;
	reg [2:0] state;
	reg [63:0] rd_buf [0:NUM_WORDS - 1];
	reg [5:0] rd_cnt;
	reg [5:0] cap_cnt;
	reg [5:0] wr_cnt;
	reg pipe_d1;
	reg pipe_d2;
	assign m_axi_awid = STREAM_ID[AXI_ID_W - 1:0];
	assign m_axi_awsize = 3'b011;
	assign m_axi_awburst = 2'b01;
	assign m_axi_wstrb = {AXI_DATA_W / 8 {1'b1}};
	wire _unused_bresp = &{1'b0, m_axi_bid, m_axi_bresp};
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			done <= 1'b0;
			busy <= 1'b0;
			accum_rd_en <= 1'b0;
			accum_rd_addr <= {BRAM_ADDR_W {1'b0}};
			m_axi_awaddr <= {AXI_ADDR_W {1'b0}};
			m_axi_awlen <= 8'd0;
			m_axi_awvalid <= 1'b0;
			m_axi_wdata <= {AXI_DATA_W {1'b0}};
			m_axi_wlast <= 1'b0;
			m_axi_wvalid <= 1'b0;
			m_axi_bready <= 1'b0;
			rd_cnt <= 6'd0;
			cap_cnt <= 6'd0;
			wr_cnt <= 6'd0;
			pipe_d1 <= 1'b0;
			pipe_d2 <= 1'b0;
		end
		else begin
			done <= 1'b0;
			case (state)
				3'd0:
					if (start) begin
						if (dst_addr == {AXI_ADDR_W {1'b0}}) begin
							done <= 1'b1;
;
						end
						else begin
							busy <= 1'b1;
							state <= 3'd1;
;
						end
					end
				3'd1:
					if (accum_ready) begin
						rd_cnt <= 6'd0;
						cap_cnt <= 6'd0;
						pipe_d1 <= 1'b0;
						pipe_d2 <= 1'b0;
						state <= 3'd2;
;
					end
				3'd2: begin
					if (rd_cnt < NUM_WORDS[5:0]) begin
						accum_rd_en <= 1'b1;
						accum_rd_addr <= {{BRAM_ADDR_W - 6 {1'b0}}, rd_cnt};
						rd_cnt <= rd_cnt + 6'd1;
					end
					else
						accum_rd_en <= 1'b0;
					pipe_d1 <= accum_rd_en;
					pipe_d2 <= pipe_d1;
					if (pipe_d1) begin
						rd_buf[cap_cnt] <= accum_rd_data;
						if (cap_cnt == (NUM_WORDS[5:0] - 6'd1)) begin
							state <= 3'd3;
;
						end
						cap_cnt <= cap_cnt + 6'd1;
					end
				end
				3'd3: begin
					m_axi_awaddr <= dst_addr;
					m_axi_awlen <= NUM_WORDS[7:0] - 8'd1;
					m_axi_awvalid <= 1'b1;
					if (m_axi_awvalid && m_axi_awready) begin
						m_axi_awvalid <= 1'b0;
						wr_cnt <= 6'd0;
						state <= 3'd4;
					end
				end
				3'd4: begin
					m_axi_wvalid <= 1'b1;
					m_axi_wdata <= rd_buf[wr_cnt];
					m_axi_wlast <= wr_cnt == (NUM_WORDS[5:0] - 6'd1);
					if (m_axi_wvalid && m_axi_wready) begin
						if (wr_cnt == (NUM_WORDS[5:0] - 6'd1)) begin
							m_axi_wvalid <= 1'b0;
							m_axi_wlast <= 1'b0;
							m_axi_bready <= 1'b1;
							state <= 3'd5;
						end
						else
							wr_cnt <= wr_cnt + 6'd1;
					end
				end
				3'd5:
					if (m_axi_bvalid && m_axi_bready) begin
						m_axi_bready <= 1'b0;
						done <= 1'b1;
						busy <= 1'b0;
						state <= 3'd0;
;
					end
				default: state <= 3'd0;
			endcase
		end
endmodule
`default_nettype wire
