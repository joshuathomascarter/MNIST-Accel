module dram_bank_fsm (
	clk,
	rst_n,
	cmd_valid,
	cmd_op,
	cmd_row,
	cmd_col,
	cmd_ready,
	bank_state,
	open_row,
	row_open,
	row_hit,
	phy_act,
	phy_read,
	phy_write,
	phy_pre,
	phy_row,
	phy_col
);
	reg _sv2v_0;
	parameter signed [31:0] ROW_BITS = 14;
	parameter signed [31:0] COL_BITS = 10;
	parameter signed [31:0] T_RCD = 3;
	parameter signed [31:0] T_RP = 3;
	parameter signed [31:0] T_RAS = 7;
	parameter signed [31:0] T_RC = 10;
	parameter signed [31:0] T_RTP = 2;
	parameter signed [31:0] T_WR = 3;
	parameter signed [31:0] T_CAS = 3;
	input wire clk;
	input wire rst_n;
	input wire cmd_valid;
	input wire [2:0] cmd_op;
	input wire [ROW_BITS - 1:0] cmd_row;
	input wire [COL_BITS - 1:0] cmd_col;
	output reg cmd_ready;
	output wire [2:0] bank_state;
	output wire [ROW_BITS - 1:0] open_row;
	output wire row_open;
	output wire row_hit;
	output reg phy_act;
	output reg phy_read;
	output reg phy_write;
	output reg phy_pre;
	output reg [ROW_BITS - 1:0] phy_row;
	output reg [COL_BITS - 1:0] phy_col;
	localparam [2:0] OP_ACT = 3'b001;
	localparam [2:0] OP_READ = 3'b010;
	localparam [2:0] OP_WRITE = 3'b011;
	localparam [2:0] OP_PRE = 3'b100;
	reg [2:0] state;
	reg [2:0] state_next;
	reg [3:0] cnt;
	reg [3:0] cnt_next;
	reg [3:0] ras_cnt;
	reg [3:0] ras_cnt_next;
	reg [ROW_BITS - 1:0] open_row_reg;
	reg [ROW_BITS - 1:0] open_row_next;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			cnt <= '0;
			ras_cnt <= '0;
			open_row_reg <= '0;
		end
		else begin
			state <= state_next;
			cnt <= cnt_next;
			ras_cnt <= ras_cnt_next;
			open_row_reg <= open_row_next;
		end
	assign bank_state = state;
	assign open_row = open_row_reg;
	assign row_open = ((state == 3'd2) || (state == 3'd3)) || (state == 3'd4);
	assign row_hit = row_open && (cmd_row == open_row_reg);
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		cnt_next = (cnt > 0 ? cnt - 1 : '0);
		ras_cnt_next = (ras_cnt > 0 ? ras_cnt - 1 : '0);
		open_row_next = open_row_reg;
		cmd_ready = 1'b0;
		phy_act = 1'b0;
		phy_read = 1'b0;
		phy_write = 1'b0;
		phy_pre = 1'b0;
		phy_row = '0;
		phy_col = '0;
		case (state)
			3'd0: begin
				cmd_ready = 1'b1;
				if (cmd_valid && (cmd_op == OP_ACT)) begin
					phy_act = 1'b1;
					phy_row = cmd_row;
					open_row_next = cmd_row;
					cnt_next = T_RCD[3:0] - 1;
					ras_cnt_next = T_RAS[3:0] - 1;
					state_next = 3'd1;
				end
			end
			3'd1:
				if (cnt == 0)
					state_next = 3'd2;
			3'd2: begin
				cmd_ready = 1'b1;
				if (cmd_valid) begin
					if (cmd_op == OP_READ) begin
						phy_read = 1'b1;
						phy_col = cmd_col;
						cnt_next = T_CAS[3:0] - 1;
						state_next = 3'd3;
					end
					else if (cmd_op == OP_WRITE) begin
						phy_write = 1'b1;
						phy_col = cmd_col;
						cnt_next = T_WR[3:0] - 1;
						state_next = 3'd4;
					end
					else if ((cmd_op == OP_PRE) && (ras_cnt == 0)) begin
						phy_pre = 1'b1;
						cnt_next = T_RP[3:0] - 1;
						state_next = 3'd5;
					end
				end
			end
			3'd3: begin
				if (cnt == 0)
					state_next = 3'd2;
				if (((cmd_valid && (cmd_op == OP_PRE)) && (cnt == 0)) && (ras_cnt == 0)) begin
					phy_pre = 1'b1;
					cnt_next = T_RP[3:0] - 1;
					state_next = 3'd5;
				end
			end
			3'd4: begin
				if (cnt == 0)
					state_next = 3'd2;
				if (((cmd_valid && (cmd_op == OP_PRE)) && (cnt == 0)) && (ras_cnt == 0)) begin
					phy_pre = 1'b1;
					cnt_next = T_RP[3:0] - 1;
					state_next = 3'd5;
				end
			end
			3'd5:
				if (cnt == 0) begin
					open_row_next = '0;
					state_next = 3'd0;
				end
			default: state_next = 3'd0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
