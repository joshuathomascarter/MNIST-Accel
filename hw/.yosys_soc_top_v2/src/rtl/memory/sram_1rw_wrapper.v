`default_nettype none
module sram_1rw_wrapper (
	clk,
	rst_n,
	en,
	we,
	addr,
	wdata,
	rdata
);
	parameter DATA_W = 32;
	parameter ADDR_W = 10;
	parameter DEPTH = 1 << ADDR_W;
	input wire clk;
	input wire rst_n;
	input wire en;
	input wire we;
	input wire [ADDR_W - 1:0] addr;
	input wire [DATA_W - 1:0] wdata;
	output reg [DATA_W - 1:0] rdata;
endmodule
`default_nettype wire
