module dram_refresh_ctrl (
	clk,
	rst_n,
	ref_req,
	ref_ack,
	ref_cmd,
	ref_busy
);
	reg _sv2v_0;
	parameter signed [31:0] T_REFI = 1560;
	parameter signed [31:0] T_RFC = 52;
	parameter signed [31:0] CTR_W = 11;
	input wire clk;
	input wire rst_n;
	output reg ref_req;
	input wire ref_ack;
	output reg ref_cmd;
	output reg ref_busy;
	reg [1:0] state;
	reg [1:0] state_next;
	reg [CTR_W - 1:0] cnt;
	reg [CTR_W - 1:0] cnt_next;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			cnt <= T_REFI[CTR_W - 1:0] - 1;
		end
		else begin
			state <= state_next;
			cnt <= cnt_next;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		cnt_next = cnt;
		ref_req = 1'b0;
		ref_cmd = 1'b0;
		ref_busy = 1'b0;
		case (state)
			2'd0:
				if (cnt == 0)
					state_next = 2'd1;
				else
					cnt_next = cnt - 1;
			2'd1: begin
				ref_req = 1'b1;
				if (ref_ack) begin
					ref_cmd = 1'b1;
					cnt_next = T_RFC[CTR_W - 1:0] - 1;
					state_next = 2'd2;
				end
			end
			2'd2: begin
				ref_busy = 1'b1;
				if (cnt == 0) begin
					cnt_next = T_REFI[CTR_W - 1:0] - 1;
					state_next = 2'd0;
				end
				else
					cnt_next = cnt - 1;
			end
			default: state_next = 2'd0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
