module dram_power_model (
	clk,
	rst_n,
	ctrl_busy,
	bank_row_open,
	cke,
	cnt_active_cycles,
	cnt_idle_cycles,
	cnt_pd_cycles,
	power_state
);
	reg _sv2v_0;
	parameter signed [31:0] IDLE_THRESHOLD = 64;
	parameter signed [31:0] T_XP = 3;
	parameter signed [31:0] NUM_BANKS = 8;
	input wire clk;
	input wire rst_n;
	input wire ctrl_busy;
	input wire [NUM_BANKS - 1:0] bank_row_open;
	output reg cke;
	output reg [31:0] cnt_active_cycles;
	output reg [31:0] cnt_idle_cycles;
	output reg [31:0] cnt_pd_cycles;
	output wire [1:0] power_state;
	reg [1:0] state;
	reg [1:0] state_next;
	reg [$clog2(IDLE_THRESHOLD + 1) - 1:0] idle_cnt;
	reg [$clog2(T_XP + 1) - 1:0] wakeup_cnt;
	wire all_precharged;
	assign all_precharged = bank_row_open == '0;
	assign power_state = state;
	function automatic signed [$clog2(T_XP + 1) - 1:0] sv2v_cast_130EB_signed;
		input reg signed [$clog2(T_XP + 1) - 1:0] inp;
		sv2v_cast_130EB_signed = inp;
	endfunction
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'b00;
			idle_cnt <= '0;
			wakeup_cnt <= '0;
			cnt_active_cycles <= '0;
			cnt_idle_cycles <= '0;
			cnt_pd_cycles <= '0;
		end
		else begin
			state <= state_next;
			case (state)
				2'b00: cnt_active_cycles <= cnt_active_cycles + 1;
				2'b01: cnt_idle_cycles <= cnt_idle_cycles + 1;
				2'b10: cnt_pd_cycles <= cnt_pd_cycles + 1;
				2'b11: cnt_pd_cycles <= cnt_pd_cycles + 1;
			endcase
			if (ctrl_busy || (state == 2'b00))
				idle_cnt <= '0;
			else if (state == 2'b01)
				idle_cnt <= idle_cnt + 1;
			if ((state == 2'b11) && ctrl_busy)
				wakeup_cnt <= sv2v_cast_130EB_signed(T_XP);
			else if (wakeup_cnt != 0)
				wakeup_cnt <= wakeup_cnt - 1;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		cke = 1'b1;
		case (state)
			2'b00:
				if (!ctrl_busy)
					state_next = 2'b01;
			2'b01:
				if (ctrl_busy)
					state_next = 2'b00;
				else if (idle_cnt >= IDLE_THRESHOLD[$clog2(IDLE_THRESHOLD + 1) - 1:0])
					state_next = 2'b10;
			2'b10: begin
				cke = 1'b0;
				state_next = 2'b11;
			end
			2'b11: begin
				cke = 1'b0;
				if (ctrl_busy)
					state_next = 2'b00;
			end
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
