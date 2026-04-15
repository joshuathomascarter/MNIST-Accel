module l2_data_array (
	clk,
	rd_en,
	rd_set,
	rd_way,
	rd_word,
	rd_data,
	wr_en,
	wr_set,
	wr_way,
	wr_word,
	wr_data,
	wr_be,
	line_rd_en,
	line_rd_set,
	line_rd_way,
	line_rd_word,
	line_rd_data,
	line_wr_en,
	line_wr_set,
	line_wr_way,
	line_wr_word,
	line_wr_data
);
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] NUM_SETS = 256;
	parameter signed [31:0] NUM_WAYS = 8;
	parameter signed [31:0] LINE_BYTES = 64;
	localparam signed [31:0] WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
	localparam signed [31:0] SET_BITS = $clog2(NUM_SETS);
	localparam signed [31:0] WAY_BITS = $clog2(NUM_WAYS);
	localparam signed [31:0] WORD_BITS = $clog2(WORDS_PER_LINE);
	input wire clk;
	input wire rd_en;
	input wire [SET_BITS - 1:0] rd_set;
	input wire [WAY_BITS - 1:0] rd_way;
	input wire [WORD_BITS - 1:0] rd_word;
	output reg [DATA_WIDTH - 1:0] rd_data;
	input wire wr_en;
	input wire [SET_BITS - 1:0] wr_set;
	input wire [WAY_BITS - 1:0] wr_way;
	input wire [WORD_BITS - 1:0] wr_word;
	input wire [DATA_WIDTH - 1:0] wr_data;
	input wire [(DATA_WIDTH / 8) - 1:0] wr_be;
	input wire line_rd_en;
	input wire [SET_BITS - 1:0] line_rd_set;
	input wire [WAY_BITS - 1:0] line_rd_way;
	input wire [WORD_BITS - 1:0] line_rd_word;
	output reg [DATA_WIDTH - 1:0] line_rd_data;
	input wire line_wr_en;
	input wire [SET_BITS - 1:0] line_wr_set;
	input wire [WAY_BITS - 1:0] line_wr_way;
	input wire [WORD_BITS - 1:0] line_wr_word;
	input wire [DATA_WIDTH - 1:0] line_wr_data;
	reg [DATA_WIDTH - 1:0] data [0:NUM_SETS - 1][0:NUM_WAYS - 1][0:WORDS_PER_LINE - 1];
	always @(posedge clk)
		if (rd_en)
			rd_data <= data[rd_set][rd_way][rd_word];
	always @(posedge clk)
		if (wr_en) begin : sv2v_autoblock_1
			reg signed [31:0] b;
			for (b = 0; b < (DATA_WIDTH / 8); b = b + 1)
				if (wr_be[b])
					data[wr_set][wr_way][wr_word][b * 8+:8] <= wr_data[b * 8+:8];
		end
	always @(posedge clk)
		if (line_rd_en)
			line_rd_data <= data[line_rd_set][line_rd_way][line_rd_word];
	always @(posedge clk)
		if (line_wr_en)
			data[line_wr_set][line_wr_way][line_wr_word] <= line_wr_data;
endmodule
