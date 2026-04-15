module accel_scratchpad (
	clk,
	rst_n,
	clk_en,
	a_en,
	a_we,
	a_addr,
	a_wdata,
	a_rdata,
	b_en,
	b_addr,
	b_rdata
);
	parameter signed [31:0] DEPTH = 4096;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] ADDR_WIDTH = $clog2(DEPTH);
	input wire clk;
	input wire rst_n;
	input wire clk_en;
	input wire a_en;
	input wire a_we;
	input wire [ADDR_WIDTH - 1:0] a_addr;
	input wire [DATA_WIDTH - 1:0] a_wdata;
	output reg [DATA_WIDTH - 1:0] a_rdata;
	input wire b_en;
	input wire [ADDR_WIDTH - 1:0] b_addr;
	output reg [DATA_WIDTH - 1:0] b_rdata;
	reg [DATA_WIDTH - 1:0] mem [0:DEPTH - 1];
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			a_rdata <= '0;
		else if (clk_en && a_en) begin
			if (a_we)
				mem[a_addr] <= a_wdata;
			a_rdata <= mem[a_addr];
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			b_rdata <= '0;
		else if (clk_en && b_en)
			b_rdata <= mem[b_addr];
endmodule
