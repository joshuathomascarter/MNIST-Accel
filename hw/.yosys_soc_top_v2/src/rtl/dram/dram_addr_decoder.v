module dram_addr_decoder (
	axi_addr,
	bank,
	row,
	col
);
	parameter signed [31:0] AXI_ADDR_W = 32;
	parameter signed [31:0] BANK_BITS = 3;
	parameter signed [31:0] ROW_BITS = 14;
	parameter signed [31:0] COL_BITS = 10;
	parameter signed [31:0] BUS_BYTES = 2;
	parameter signed [31:0] MODE = 0;
	input wire [AXI_ADDR_W - 1:0] axi_addr;
	output wire [BANK_BITS - 1:0] bank;
	output wire [ROW_BITS - 1:0] row;
	output wire [COL_BITS - 1:0] col;
	localparam signed [31:0] BYTE_OFF = $clog2(BUS_BYTES);
	generate
		if (MODE == 0) begin : gen_rbc
			assign col = axi_addr[BYTE_OFF+:COL_BITS];
			assign bank = axi_addr[BYTE_OFF + COL_BITS+:BANK_BITS];
			assign row = axi_addr[(BYTE_OFF + COL_BITS) + BANK_BITS+:ROW_BITS];
		end
		else begin : gen_brc
			assign col = axi_addr[BYTE_OFF+:COL_BITS];
			assign row = axi_addr[BYTE_OFF + COL_BITS+:ROW_BITS];
			assign bank = axi_addr[(BYTE_OFF + COL_BITS) + ROW_BITS+:BANK_BITS];
		end
	endgenerate
endmodule
