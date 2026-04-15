`default_nettype none
module act_buffer (
	clk,
	rst_n,
	we,
	waddr,
	wdata,
	bank_sel_wr,
	rd_en,
	k_idx,
	bank_sel_rd,
	a_vec
);
	parameter TM = 16;
	parameter ADDR_WIDTH = 7;
	parameter ENABLE_CLOCK_GATING = 1;
	input wire clk;
	input wire rst_n;
	input wire we;
	input wire [ADDR_WIDTH - 1:0] waddr;
	input wire [(TM * 8) - 1:0] wdata;
	input wire bank_sel_wr;
	input wire rd_en;
	input wire [ADDR_WIDTH - 1:0] k_idx;
	input wire bank_sel_rd;
	output wire [(TM * 8) - 1:0] a_vec;
	wire buf_clk_en;
	wire buf_gated_clk;
	assign buf_clk_en = we | rd_en;
	generate
		if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
			assign buf_gated_clk = clk;
		end
		else begin : gen_no_gate
			assign buf_gated_clk = clk;
		end
	endgenerate
	wire bank0_en;
	wire bank0_we;
	wire [ADDR_WIDTH - 1:0] bank0_addr;
	wire [(TM * 8) - 1:0] bank0_rdata;
	wire bank1_en;
	wire bank1_we;
	wire [ADDR_WIDTH - 1:0] bank1_addr;
	wire [(TM * 8) - 1:0] bank1_rdata;
	assign bank0_we = we && (bank_sel_wr == 1'b0);
	assign bank1_we = we && (bank_sel_wr == 1'b1);
	assign bank0_en = bank0_we || (rd_en && (bank_sel_rd == 1'b0));
	assign bank1_en = bank1_we || (rd_en && (bank_sel_rd == 1'b1));
	assign bank0_addr = (bank0_we ? waddr : k_idx);
	assign bank1_addr = (bank1_we ? waddr : k_idx);
	sram_1rw_wrapper #(
		.DATA_W(TM * 8),
		.ADDR_W(ADDR_WIDTH),
		.DEPTH(1 << ADDR_WIDTH)
	) u_bank0_mem(
		.clk(buf_gated_clk),
		.rst_n(rst_n),
		.en(bank0_en),
		.we(bank0_we),
		.addr(bank0_addr),
		.wdata(wdata),
		.rdata(bank0_rdata)
	);
	sram_1rw_wrapper #(
		.DATA_W(TM * 8),
		.ADDR_W(ADDR_WIDTH),
		.DEPTH(1 << ADDR_WIDTH)
	) u_bank1_mem(
		.clk(buf_gated_clk),
		.rst_n(rst_n),
		.en(bank1_en),
		.we(bank1_we),
		.addr(bank1_addr),
		.wdata(wdata),
		.rdata(bank1_rdata)
	);
	assign a_vec = (rd_en ? (bank_sel_rd == 1'b0 ? bank0_rdata : bank1_rdata) : {TM * 8 {1'b0}});
endmodule
`default_nettype wire
