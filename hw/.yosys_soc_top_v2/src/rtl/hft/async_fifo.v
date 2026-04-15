module async_fifo (
	wr_clk,
	wr_rst_n,
	wr_data,
	wr_en,
	full,
	rd_clk,
	rd_rst_n,
	rd_data,
	rd_en,
	empty
);
	parameter signed [31:0] DEPTH = 16;
	parameter signed [31:0] WIDTH = 64;
	input wire wr_clk;
	input wire wr_rst_n;
	input wire [WIDTH - 1:0] wr_data;
	input wire wr_en;
	output wire full;
	input wire rd_clk;
	input wire rd_rst_n;
	output wire [WIDTH - 1:0] rd_data;
	input wire rd_en;
	output wire empty;
	localparam signed [31:0] ADDR_W = $clog2(DEPTH);
	reg [WIDTH - 1:0] mem [0:DEPTH - 1];
	reg [ADDR_W:0] wr_ptr_bin;
	reg [ADDR_W:0] wr_ptr_gray;
	wire [ADDR_W:0] wr_ptr_gray_next;
	wire [ADDR_W:0] rd_ptr_gray_sync;
	reg [ADDR_W:0] rd_ptr_bin;
	reg [ADDR_W:0] rd_ptr_gray;
	wire [ADDR_W:0] rd_ptr_gray_next;
	wire [ADDR_W:0] wr_ptr_gray_sync;
	function [ADDR_W:0] bin2gray;
		input [ADDR_W:0] b;
		bin2gray = b ^ (b >> 1);
	endfunction
	assign wr_ptr_gray_next = bin2gray(wr_ptr_bin + 1);
	always @(posedge wr_clk or negedge wr_rst_n)
		if (!wr_rst_n) begin
			wr_ptr_bin <= '0;
			wr_ptr_gray <= '0;
		end
		else if (wr_en && !full) begin
			mem[wr_ptr_bin[ADDR_W - 1:0]] <= wr_data;
			wr_ptr_bin <= wr_ptr_bin + 1;
			wr_ptr_gray <= wr_ptr_gray_next;
		end
	assign rd_ptr_gray_next = bin2gray(rd_ptr_bin + 1);
	always @(posedge rd_clk or negedge rd_rst_n)
		if (!rd_rst_n) begin
			rd_ptr_bin <= '0;
			rd_ptr_gray <= '0;
		end
		else if (rd_en && !empty) begin
			rd_ptr_bin <= rd_ptr_bin + 1;
			rd_ptr_gray <= rd_ptr_gray_next;
		end
	assign rd_data = mem[rd_ptr_bin[ADDR_W - 1:0]];
	reg [ADDR_W:0] rd_gray_meta;
	reg [ADDR_W:0] rd_gray_sync;
	always @(posedge wr_clk or negedge wr_rst_n)
		if (!wr_rst_n) begin
			rd_gray_meta <= '0;
			rd_gray_sync <= '0;
		end
		else begin
			rd_gray_meta <= rd_ptr_gray;
			rd_gray_sync <= rd_gray_meta;
		end
	assign rd_ptr_gray_sync = rd_gray_sync;
	reg [ADDR_W:0] wr_gray_meta;
	reg [ADDR_W:0] wr_gray_sync;
	always @(posedge rd_clk or negedge rd_rst_n)
		if (!rd_rst_n) begin
			wr_gray_meta <= '0;
			wr_gray_sync <= '0;
		end
		else begin
			wr_gray_meta <= wr_ptr_gray;
			wr_gray_sync <= wr_gray_meta;
		end
	assign wr_ptr_gray_sync = wr_gray_sync;
	assign full = ((wr_ptr_gray_next == {~rd_ptr_gray_sync[ADDR_W:ADDR_W - 1], rd_ptr_gray_sync[ADDR_W - 2:0]}) && wr_en ? 1'b1 : wr_ptr_gray == {~rd_ptr_gray_sync[ADDR_W:ADDR_W - 1], rd_ptr_gray_sync[ADDR_W - 2:0]});
	assign empty = rd_ptr_gray == wr_ptr_gray_sync;
	async_fifo_sva #(
		.DEPTH(DEPTH),
		.WIDTH(WIDTH)
	) u_sva(
		.wr_clk(wr_clk),
		.wr_rst_n(wr_rst_n),
		.wr_en(wr_en),
		.full(full),
		.wr_ptr_gray(wr_ptr_gray),
		.rd_clk(rd_clk),
		.rd_rst_n(rd_rst_n),
		.rd_en(rd_en),
		.empty(empty),
		.rd_ptr_gray(rd_ptr_gray)
	);
endmodule
