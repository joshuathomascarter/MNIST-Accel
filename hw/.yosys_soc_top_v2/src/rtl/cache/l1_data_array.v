module l1_data_array (
	clk,
	word_en,
	word_we,
	word_set,
	word_way,
	word_offset,
	word_be,
	word_wdata,
	word_rdata,
	line_en,
	line_we,
	line_set,
	line_way,
	line_wdata,
	line_rdata
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_SETS = 16;
	parameter signed [31:0] NUM_WAYS = 4;
	parameter signed [31:0] LINE_BYTES = 64;
	parameter signed [31:0] WORD_WIDTH = 32;
	input wire clk;
	input wire word_en;
	input wire word_we;
	input wire [$clog2(NUM_SETS) - 1:0] word_set;
	input wire [$clog2(NUM_WAYS) - 1:0] word_way;
	input wire [$clog2(LINE_BYTES) - 1:0] word_offset;
	input wire [(WORD_WIDTH / 8) - 1:0] word_be;
	input wire [WORD_WIDTH - 1:0] word_wdata;
	output reg [WORD_WIDTH - 1:0] word_rdata;
	input wire line_en;
	input wire line_we;
	input wire [$clog2(NUM_SETS) - 1:0] line_set;
	input wire [$clog2(NUM_WAYS) - 1:0] line_way;
	input wire [(LINE_BYTES * 8) - 1:0] line_wdata;
	output reg [(LINE_BYTES * 8) - 1:0] line_rdata;
	localparam signed [31:0] OFFSET_W = $clog2(LINE_BYTES);
	reg [7:0] data [0:NUM_SETS - 1][0:NUM_WAYS - 1][0:LINE_BYTES - 1];
	wire [OFFSET_W - 1:0] word_base;
	assign word_base = {word_offset[OFFSET_W - 1:2], 2'b00};
	function automatic signed [OFFSET_W - 1:0] sv2v_cast_B686B_signed;
		input reg signed [OFFSET_W - 1:0] inp;
		sv2v_cast_B686B_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		word_rdata = '0;
		begin : sv2v_autoblock_1
			reg signed [31:0] b;
			for (b = 0; b < (WORD_WIDTH / 8); b = b + 1)
				word_rdata[b * 8+:8] = data[word_set][word_way][word_base + sv2v_cast_B686B_signed(b)];
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		line_rdata = '0;
		begin : sv2v_autoblock_2
			reg signed [31:0] b;
			for (b = 0; b < LINE_BYTES; b = b + 1)
				line_rdata[b * 8+:8] = data[line_set][line_way][b];
		end
	end
	always @(posedge clk) begin
		if (word_en && word_we) begin : sv2v_autoblock_3
			reg signed [31:0] b;
			for (b = 0; b < (WORD_WIDTH / 8); b = b + 1)
				if (word_be[b])
					data[word_set][word_way][word_base + sv2v_cast_B686B_signed(b)] <= word_wdata[b * 8+:8];
		end
		if (line_en && line_we) begin : sv2v_autoblock_4
			reg signed [31:0] b;
			for (b = 0; b < LINE_BYTES; b = b + 1)
				data[line_set][line_way][b] <= line_wdata[b * 8+:8];
		end
	end
	initial _sv2v_0 = 0;
endmodule
