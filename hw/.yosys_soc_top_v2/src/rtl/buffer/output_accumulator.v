`default_nettype none
module output_accumulator (
	clk,
	rst_n,
	acc_valid,
	acc_clear,
	tile_done,
	relu_en,
	scale_factor,
	systolic_out,
	dma_rd_en,
	dma_rd_addr,
	dma_rd_data,
	dma_ready,
	busy,
	bank_sel,
	acc_debug
);
	parameter N_ROWS = 16;
	parameter N_COLS = 16;
	parameter ACC_W = 32;
	parameter OUT_W = 8;
	parameter ADDR_W = 10;
	input wire clk;
	input wire rst_n;
	input wire acc_valid;
	input wire acc_clear;
	input wire tile_done;
	input wire relu_en;
	input wire [31:0] scale_factor;
	input wire [((N_ROWS * N_COLS) * ACC_W) - 1:0] systolic_out;
	input wire dma_rd_en;
	input wire [ADDR_W - 1:0] dma_rd_addr;
	output reg [63:0] dma_rd_data;
	output wire dma_ready;
	output reg busy;
	output reg bank_sel;
	output wire [31:0] acc_debug;
	localparam NUM_ACCS = N_ROWS * N_COLS;
	localparam BANK_DEPTH = ((NUM_ACCS + 7) / 8) * 8;
	reg signed [ACC_W - 1:0] acc_bank0 [0:BANK_DEPTH - 1];
	reg signed [ACC_W - 1:0] acc_bank1 [0:BANK_DEPTH - 1];
	reg bank0_ready;
	reg bank1_ready;
	assign dma_ready = (bank_sel ? bank0_ready : bank1_ready);
	wire signed [ACC_W - 1:0] sys_out [0:NUM_ACCS - 1];
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < NUM_ACCS; _gv_i_1 = _gv_i_1 + 1) begin : UNPACK
			localparam i = _gv_i_1;
			assign sys_out[i] = systolic_out[i * ACC_W+:ACC_W];
		end
	endgenerate
	integer j;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			bank_sel <= 1'b0;
			bank0_ready <= 1'b0;
			bank1_ready <= 1'b0;
			busy <= 1'b0;
			for (j = 0; j < BANK_DEPTH; j = j + 1)
				begin
					acc_bank0[j] <= 32'sd0;
					acc_bank1[j] <= 32'sd0;
				end
		end
		else begin
			if (acc_clear) begin
				if (bank_sel == 1'b0)
					for (j = 0; j < BANK_DEPTH; j = j + 1)
						acc_bank0[j] <= 32'sd0;
				else
					for (j = 0; j < BANK_DEPTH; j = j + 1)
						acc_bank1[j] <= 32'sd0;
				busy <= 1'b1;
			end
			if (acc_valid) begin
				if (bank_sel == 1'b0)
					for (j = 0; j < BANK_DEPTH; j = j + 1)
						acc_bank0[j] <= acc_bank0[j] + sys_out[j];
				else
					for (j = 0; j < BANK_DEPTH; j = j + 1)
						acc_bank1[j] <= acc_bank1[j] + sys_out[j];
			end
			if (tile_done) begin
				bank_sel <= ~bank_sel;
				busy <= 1'b0;
				if (bank_sel == 1'b0)
					bank0_ready <= 1'b1;
				else
					bank1_ready <= 1'b1;
;
			end
			if (dma_rd_en) begin
				if (bank_sel == 1'b1)
					bank0_ready <= 1'b0;
				else
					bank1_ready <= 1'b0;
			end
		end
	reg signed [ACC_W - 1:0] rd_acc_0;
	reg signed [ACC_W - 1:0] rd_acc_1;
	reg signed [ACC_W - 1:0] rd_acc_2;
	reg signed [ACC_W - 1:0] rd_acc_3;
	reg signed [ACC_W - 1:0] rd_acc_4;
	reg signed [ACC_W - 1:0] rd_acc_5;
	reg signed [ACC_W - 1:0] rd_acc_6;
	reg signed [ACC_W - 1:0] rd_acc_7;
	reg rd_valid_d1;
	wire [7:0] rd_base_idx = dma_rd_addr[7:0];
	wire _unused_dma_rd_addr_hi = &{1'b0, dma_rd_addr[ADDR_W - 1:8]};
	always @(posedge clk) begin
		rd_valid_d1 <= dma_rd_en;
		if (dma_rd_en) begin
			if (bank_sel == 1'b1) begin
				rd_acc_0 <= acc_bank0[rd_base_idx + 8'd0];
				rd_acc_1 <= acc_bank0[rd_base_idx + 8'd1];
				rd_acc_2 <= acc_bank0[rd_base_idx + 8'd2];
				rd_acc_3 <= acc_bank0[rd_base_idx + 8'd3];
				rd_acc_4 <= acc_bank0[rd_base_idx + 8'd4];
				rd_acc_5 <= acc_bank0[rd_base_idx + 8'd5];
				rd_acc_6 <= acc_bank0[rd_base_idx + 8'd6];
				rd_acc_7 <= acc_bank0[rd_base_idx + 8'd7];
			end
			else begin
				rd_acc_0 <= acc_bank1[rd_base_idx + 8'd0];
				rd_acc_1 <= acc_bank1[rd_base_idx + 8'd1];
				rd_acc_2 <= acc_bank1[rd_base_idx + 8'd2];
				rd_acc_3 <= acc_bank1[rd_base_idx + 8'd3];
				rd_acc_4 <= acc_bank1[rd_base_idx + 8'd4];
				rd_acc_5 <= acc_bank1[rd_base_idx + 8'd5];
				rd_acc_6 <= acc_bank1[rd_base_idx + 8'd6];
				rd_acc_7 <= acc_bank1[rd_base_idx + 8'd7];
			end
		end
	end
	function automatic [OUT_W - 1:0] quantize_relu;
		input signed [ACC_W - 1:0] acc_val;
		input [31:0] scale;
		input relu;
		reg signed [49:0] scaled;
		reg signed [ACC_W - 1:0] relu_val;
		reg signed [OUT_W - 1:0] quant_val;
		reg signed [17:0] scale_narrow;
		begin
			if (relu && (acc_val < 0))
				relu_val = 32'sd0;
			else
				relu_val = acc_val;
			scale_narrow = $signed({1'b0, scale[16:0]});
			scaled = (relu_val * scale_narrow) >>> 16;
			if (scaled > 127)
				quant_val = 127;
			else if (scaled < -128)
				quant_val = -128;
			else
				quant_val = scaled[OUT_W - 1:0];
			quantize_relu = quant_val;
		end
	endfunction
	always @(posedge clk)
		if (rd_valid_d1)
			dma_rd_data <= {quantize_relu(rd_acc_7, scale_factor, relu_en), quantize_relu(rd_acc_6, scale_factor, relu_en), quantize_relu(rd_acc_5, scale_factor, relu_en), quantize_relu(rd_acc_4, scale_factor, relu_en), quantize_relu(rd_acc_3, scale_factor, relu_en), quantize_relu(rd_acc_2, scale_factor, relu_en), quantize_relu(rd_acc_1, scale_factor, relu_en), quantize_relu(rd_acc_0, scale_factor, relu_en)};
	assign acc_debug = (bank_sel ? acc_bank1[0] : acc_bank0[0]);
endmodule
`default_nettype wire
