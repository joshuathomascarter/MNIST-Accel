`default_nettype none
module output_bram_ctrl (
	clk,
	rst_n,
	layer_total,
	layer_current,
	pool_en,
	output_h,
	output_w,
	accum_rd_en,
	accum_rd_addr,
	accum_rd_data,
	accum_ready,
	bram_wr_en,
	bram_wr_addr,
	bram_wr_data,
	bram_bank_sel,
	bram_rd_en,
	bram_rd_addr,
	bram_rd_data,
	bram_rd_valid,
	fb_act_we,
	fb_act_waddr,
	fb_act_wdata,
	sched_done,
	start,
	out_dma_trigger,
	out_dma_done,
	layer_done,
	last_layer_done,
	feedback_busy,
	busy
);
	parameter DATA_W = 64;
	parameter ADDR_W = 10;
	parameter ACT_ADDR_W = 10;
	parameter ACT_DATA_W = 128;
	parameter NUM_ACCS = 256;
	input wire clk;
	input wire rst_n;
	input wire [7:0] layer_total;
	input wire [7:0] layer_current;
	input wire pool_en;
	input wire [15:0] output_h;
	input wire [15:0] output_w;
	output reg accum_rd_en;
	output reg [ADDR_W - 1:0] accum_rd_addr;
	input wire [DATA_W - 1:0] accum_rd_data;
	input wire accum_ready;
	output reg bram_wr_en;
	output reg [ADDR_W - 1:0] bram_wr_addr;
	output reg [DATA_W - 1:0] bram_wr_data;
	output wire bram_bank_sel;
	output reg bram_rd_en;
	output reg [ADDR_W - 1:0] bram_rd_addr;
	input wire [DATA_W - 1:0] bram_rd_data;
	input wire bram_rd_valid;
	output reg fb_act_we;
	output reg [ACT_ADDR_W - 1:0] fb_act_waddr;
	output reg [ACT_DATA_W - 1:0] fb_act_wdata;
	input wire sched_done;
	input wire start;
	output wire out_dma_trigger;
	input wire out_dma_done;
	output reg layer_done;
	output reg last_layer_done;
	output reg feedback_busy;
	output reg busy;
	localparam NUM_WORDS = (NUM_ACCS + 7) / 8;
	reg [3:0] state;
	reg [ADDR_W - 1:0] cap_cnt;
	reg [ADDR_W - 1:0] fb_rd_cnt;
	reg [ADDR_W - 1:0] fb_wr_cnt;
	reg bank_sel_r;
	reg is_last_layer;
	reg trigger_dma_r;
	reg pipe_d1;
	reg pipe_d2;
	reg fb_pipe_d1;
	reg [ADDR_W - 1:0] num_words_to_capture;
	assign bram_bank_sel = bank_sel_r;
	assign out_dma_trigger = trigger_dma_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 4'd0;
			bank_sel_r <= 1'b0;
			busy <= 1'b0;
			feedback_busy <= 1'b0;
			layer_done <= 1'b0;
			last_layer_done <= 1'b0;
			accum_rd_en <= 1'b0;
			accum_rd_addr <= {ADDR_W {1'b0}};
			bram_wr_en <= 1'b0;
			bram_wr_addr <= {ADDR_W {1'b0}};
			bram_wr_data <= {DATA_W {1'b0}};
			bram_rd_en <= 1'b0;
			bram_rd_addr <= {ADDR_W {1'b0}};
			fb_act_we <= 1'b0;
			fb_act_waddr <= {ACT_ADDR_W {1'b0}};
			fb_act_wdata <= {ACT_DATA_W {1'b0}};
			cap_cnt <= {ADDR_W {1'b0}};
			fb_rd_cnt <= {ADDR_W {1'b0}};
			fb_wr_cnt <= {ADDR_W {1'b0}};
			pipe_d1 <= 1'b0;
			pipe_d2 <= 1'b0;
			fb_pipe_d1 <= 1'b0;
			is_last_layer <= 1'b0;
			trigger_dma_r <= 1'b0;
			num_words_to_capture <= NUM_WORDS[ADDR_W - 1:0];
		end
		else begin
			layer_done <= 1'b0;
			last_layer_done <= 1'b0;
			trigger_dma_r <= 1'b0;
			case (state)
				4'd0:
					if (sched_done) begin
						busy <= 1'b1;
						is_last_layer <= layer_current >= (layer_total - 8'd1);
						num_words_to_capture <= NUM_WORDS[ADDR_W - 1:0];
						state <= 4'd1;
;
					end
				4'd1:
					if (accum_ready) begin
						cap_cnt <= {ADDR_W {1'b0}};
						pipe_d1 <= 1'b0;
						pipe_d2 <= 1'b0;
						state <= 4'd2;
;
					end
				4'd2: begin
					if (cap_cnt < num_words_to_capture) begin
						accum_rd_en <= 1'b1;
						accum_rd_addr <= cap_cnt[ADDR_W - 4:0] << 3;
						cap_cnt <= cap_cnt + {{ADDR_W - 1 {1'b0}}, 1'b1};
					end
					else
						accum_rd_en <= 1'b0;
					pipe_d1 <= accum_rd_en;
					pipe_d2 <= pipe_d1;
					if (pipe_d2) begin
						bram_wr_en <= 1'b1;
						bram_wr_addr <= bram_wr_addr + {{ADDR_W - 1 {1'b0}}, 1'b1};
						bram_wr_data <= accum_rd_data;
					end
					else
						bram_wr_en <= 1'b0;
					if ((((cap_cnt >= num_words_to_capture) && !pipe_d1) && !pipe_d2) && !accum_rd_en) begin
						bram_wr_en <= 1'b0;
						bram_wr_addr <= {ADDR_W {1'b0}};
						state <= 4'd3;
					end
				end
				4'd3: begin
					bram_wr_en <= 1'b0;
					bank_sel_r <= ~bank_sel_r;
					state <= 4'd4;
;
				end
				4'd4:
					if (is_last_layer) begin
						trigger_dma_r <= 1'b1;
						state <= 4'd8;
;
					end
					else begin
						feedback_busy <= 1'b1;
						fb_rd_cnt <= {ADDR_W {1'b0}};
						fb_wr_cnt <= {ADDR_W {1'b0}};
						fb_pipe_d1 <= 1'b0;
						state <= 4'd5;
					end
				4'd5: begin
					fb_rd_cnt <= {ADDR_W {1'b0}};
					state <= 4'd6;
				end
				4'd6: begin
					if (fb_rd_cnt < num_words_to_capture) begin
						bram_rd_en <= 1'b1;
						bram_rd_addr <= fb_rd_cnt;
						fb_rd_cnt <= fb_rd_cnt + {{ADDR_W - 1 {1'b0}}, 1'b1};
					end
					else
						bram_rd_en <= 1'b0;
					fb_pipe_d1 <= bram_rd_en;
					if (bram_rd_valid) begin
						fb_act_we <= 1'b1;
						fb_act_waddr <= fb_wr_cnt[ACT_ADDR_W - 1:0];
						fb_act_wdata <= {{ACT_DATA_W - DATA_W {1'b0}}, bram_rd_data};
						fb_wr_cnt <= fb_wr_cnt + {{ADDR_W - 1 {1'b0}}, 1'b1};
					end
					else
						fb_act_we <= 1'b0;
					if (((fb_rd_cnt >= num_words_to_capture) && !bram_rd_en) && !fb_pipe_d1)
						state <= 4'd7;
				end
				4'd7: begin
					fb_act_we <= 1'b0;
					feedback_busy <= 1'b0;
					layer_done <= 1'b1;
					state <= 4'd9;
				end
				4'd8:
					if (out_dma_done) begin
						last_layer_done <= 1'b1;
						state <= 4'd9;
;
					end
				4'd9: begin
					busy <= 1'b0;
					state <= 4'd0;
				end
				default: state <= 4'd0;
			endcase
		end
endmodule
`default_nettype wire
