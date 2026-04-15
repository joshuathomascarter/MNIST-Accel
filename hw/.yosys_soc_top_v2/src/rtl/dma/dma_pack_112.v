`default_nettype none
module dma_pack_112 (
	clk,
	rst_n,
	dma_we,
	dma_wdata,
	buf_we,
	buf_waddr,
	buf_wdata
);
	parameter OUT_W = 128;
	parameter ADDR_W = 10;
	parameter ROWS_PER_BLOCK = 16;
	input wire clk;
	input wire rst_n;
	input wire dma_we;
	input wire [63:0] dma_wdata;
	output reg buf_we;
	output reg [ADDR_W - 1:0] buf_waddr;
	output reg [OUT_W - 1:0] buf_wdata;
	reg [63:0] sr;
	reg phase;
	wire _unused_rpb = &{1'b0, ROWS_PER_BLOCK[0]};
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			sr <= 64'd0;
			phase <= 1'b0;
			buf_we <= 1'b0;
			buf_waddr <= {ADDR_W {1'b1}};
			buf_wdata <= {OUT_W {1'b0}};
		end
		else begin
			buf_we <= 1'b0;
			if (dma_we) begin
				if (!phase) begin
					sr <= dma_wdata;
					phase <= 1'b1;
				end
				else begin
					buf_we <= 1'b1;
					buf_waddr <= buf_waddr + {{ADDR_W - 1 {1'b0}}, 1'b1};
					buf_wdata <= {dma_wdata, sr};
					phase <= 1'b0;
				end
			end
		end
endmodule
`default_nettype wire
