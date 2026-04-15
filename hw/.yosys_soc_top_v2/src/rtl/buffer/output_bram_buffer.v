`default_nettype none
module output_bram_buffer (
	clk,
	rst_n,
	bank_sel,
	wr_en,
	wr_addr,
	wr_data,
	rd_en,
	rd_addr,
	rd_data,
	rd_valid
);
	parameter DATA_W = 64;
	parameter ADDR_W = 10;
	parameter DEPTH = 1024;
	input wire clk;
	input wire rst_n;
	input wire bank_sel;
	input wire wr_en;
	input wire [ADDR_W - 1:0] wr_addr;
	input wire [DATA_W - 1:0] wr_data;
	input wire rd_en;
	input wire [ADDR_W - 1:0] rd_addr;
	output wire [DATA_W - 1:0] rd_data;
	output wire rd_valid;
	wire bank0_en;
	wire bank0_we;
	wire [ADDR_W - 1:0] bank0_addr;
	wire [DATA_W - 1:0] bank0_rdata;
	wire bank1_en;
	wire bank1_we;
	wire [ADDR_W - 1:0] bank1_addr;
	wire [DATA_W - 1:0] bank1_rdata;
	assign bank0_we = wr_en && !bank_sel;
	assign bank1_we = wr_en && bank_sel;
	assign bank0_en = bank0_we || (rd_en && bank_sel);
	assign bank1_en = bank1_we || (rd_en && !bank_sel);
	assign bank0_addr = (bank0_we ? wr_addr : rd_addr);
	assign bank1_addr = (bank1_we ? wr_addr : rd_addr);
	reg rd_valid_r;
	assign rd_valid = rd_valid_r;
	assign rd_data = (bank_sel ? bank0_rdata : bank1_rdata);
	sram_1rw_wrapper #(
		.DATA_W(DATA_W),
		.ADDR_W(ADDR_W),
		.DEPTH(DEPTH)
	) u_bank0_mem(
		.clk(clk),
		.rst_n(rst_n),
		.en(bank0_en),
		.we(bank0_we),
		.addr(bank0_addr),
		.wdata(wr_data),
		.rdata(bank0_rdata)
	);
	sram_1rw_wrapper #(
		.DATA_W(DATA_W),
		.ADDR_W(ADDR_W),
		.DEPTH(DEPTH)
	) u_bank1_mem(
		.clk(clk),
		.rst_n(rst_n),
		.en(bank1_en),
		.we(bank1_we),
		.addr(bank1_addr),
		.wdata(wr_data),
		.rdata(bank1_rdata)
	);
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			rd_valid_r <= 1'b0;
		else
			rd_valid_r <= rd_en;
endmodule
`default_nettype wire
