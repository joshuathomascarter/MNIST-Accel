module qos_arbiter (
	clk,
	rst_n,
	priority_i,
	bw_alloc_i,
	ar_valid_i,
	ar_ready_o,
	ar_addr_i,
	ar_id_i,
	ar_len_i,
	ar_size_i,
	ar_burst_i,
	aw_valid_i,
	aw_ready_o,
	aw_addr_i,
	aw_id_i,
	aw_len_i,
	aw_size_i,
	aw_burst_i,
	ar_valid_o,
	ar_ready_i,
	ar_addr_o,
	ar_id_o,
	ar_len_o,
	ar_size_o,
	ar_burst_o,
	ar_grant_id_o,
	aw_valid_o,
	aw_ready_i,
	aw_addr_o,
	aw_id_o,
	aw_len_o,
	aw_size_o,
	aw_burst_o,
	aw_grant_id_o,
	throttled_o
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_MASTERS = 4;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] ID_WIDTH = 4;
	parameter signed [31:0] TOKEN_DEPTH = 16;
	parameter signed [31:0] REFILL_PERIOD = 64;
	input wire clk;
	input wire rst_n;
	input wire [(NUM_MASTERS * 3) - 1:0] priority_i;
	input wire [(NUM_MASTERS * 8) - 1:0] bw_alloc_i;
	input wire [0:NUM_MASTERS - 1] ar_valid_i;
	output wire [0:NUM_MASTERS - 1] ar_ready_o;
	input wire [(NUM_MASTERS * ADDR_WIDTH) - 1:0] ar_addr_i;
	input wire [(NUM_MASTERS * ID_WIDTH) - 1:0] ar_id_i;
	input wire [(NUM_MASTERS * 8) - 1:0] ar_len_i;
	input wire [(NUM_MASTERS * 3) - 1:0] ar_size_i;
	input wire [(NUM_MASTERS * 2) - 1:0] ar_burst_i;
	input wire [0:NUM_MASTERS - 1] aw_valid_i;
	output wire [0:NUM_MASTERS - 1] aw_ready_o;
	input wire [(NUM_MASTERS * ADDR_WIDTH) - 1:0] aw_addr_i;
	input wire [(NUM_MASTERS * ID_WIDTH) - 1:0] aw_id_i;
	input wire [(NUM_MASTERS * 8) - 1:0] aw_len_i;
	input wire [(NUM_MASTERS * 3) - 1:0] aw_size_i;
	input wire [(NUM_MASTERS * 2) - 1:0] aw_burst_i;
	output wire ar_valid_o;
	input wire ar_ready_i;
	output wire [ADDR_WIDTH - 1:0] ar_addr_o;
	output wire [ID_WIDTH - 1:0] ar_id_o;
	output wire [7:0] ar_len_o;
	output wire [2:0] ar_size_o;
	output wire [1:0] ar_burst_o;
	output wire [$clog2(NUM_MASTERS) - 1:0] ar_grant_id_o;
	output wire aw_valid_o;
	input wire aw_ready_i;
	output wire [ADDR_WIDTH - 1:0] aw_addr_o;
	output wire [ID_WIDTH - 1:0] aw_id_o;
	output wire [7:0] aw_len_o;
	output wire [2:0] aw_size_o;
	output wire [1:0] aw_burst_o;
	output wire [$clog2(NUM_MASTERS) - 1:0] aw_grant_id_o;
	output wire [NUM_MASTERS - 1:0] throttled_o;
	localparam signed [31:0] M_BITS = $clog2(NUM_MASTERS);
	reg [$clog2(TOKEN_DEPTH):0] tokens [0:NUM_MASTERS - 1];
	reg [$clog2(REFILL_PERIOD) - 1:0] refill_cnt;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			refill_cnt <= '0;
		else if (refill_cnt == (REFILL_PERIOD - 1))
			refill_cnt <= '0;
		else
			refill_cnt <= refill_cnt + 1;
	wire refill_tick;
	assign refill_tick = refill_cnt == '0;
	genvar _gv_m_1;
	generate
		for (_gv_m_1 = 0; _gv_m_1 < NUM_MASTERS; _gv_m_1 = _gv_m_1 + 1) begin : g_tokens
			localparam m = _gv_m_1;
			always @(posedge clk or negedge rst_n)
				if (!rst_n)
					tokens[m] <= TOKEN_DEPTH;
				else begin
					if (refill_tick) begin
						if ((tokens[m] + bw_alloc_i[((NUM_MASTERS - 1) - m) * 8+:8]) > TOKEN_DEPTH)
							tokens[m] <= TOKEN_DEPTH;
						else
							tokens[m] <= tokens[m] + bw_alloc_i[((NUM_MASTERS - 1) - m) * 8+:8];
					end
					if ((ar_ready_o[m] && ar_valid_i[m]) || (aw_ready_o[m] && aw_valid_i[m])) begin
						if (tokens[m] > 0)
							tokens[m] <= tokens[m] - 1;
					end
				end
			assign throttled_o[m] = tokens[m] == 0;
		end
	endgenerate
	reg [7:0] age [0:NUM_MASTERS - 1];
	genvar _gv_m_2;
	generate
		for (_gv_m_2 = 0; _gv_m_2 < NUM_MASTERS; _gv_m_2 = _gv_m_2 + 1) begin : g_age
			localparam m = _gv_m_2;
			always @(posedge clk or negedge rst_n)
				if (!rst_n)
					age[m] <= '0;
				else begin
					if (ar_valid_i[m] || aw_valid_i[m]) begin
						if (age[m] < 8'hff)
							age[m] <= age[m] + 1;
					end
					else
						age[m] <= '0;
					if (ar_ready_o[m] || aw_ready_o[m])
						age[m] <= '0;
				end
		end
	endgenerate
	reg [3:0] eff_priority [0:NUM_MASTERS - 1];
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] m;
			for (m = 0; m < NUM_MASTERS; m = m + 1)
				begin
					eff_priority[m] = {1'b0, priority_i[((NUM_MASTERS - 1) - m) * 3+:3]};
					if (age[m] >= 8'd128)
						eff_priority[m] = 4'hf;
				end
		end
	end
	reg [M_BITS - 1:0] ar_winner;
	reg ar_winner_valid;
	reg [M_BITS - 1:0] ar_rr_ptr;
	reg [3:0] arb_best_pri;
	reg signed [31:0] arb_m;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			ar_rr_ptr <= '0;
		else if (ar_valid_o && ar_ready_i)
			ar_rr_ptr <= ar_winner + 1;
	always @(*) begin
		if (_sv2v_0)
			;
		ar_winner = '0;
		ar_winner_valid = 1'b0;
		arb_best_pri = '0;
		begin : sv2v_autoblock_2
			reg signed [31:0] pass;
			for (pass = 0; pass < NUM_MASTERS; pass = pass + 1)
				begin
					arb_m = (ar_rr_ptr + pass) % NUM_MASTERS;
					if (ar_valid_i[arb_m] && !throttled_o[arb_m]) begin
						if (!ar_winner_valid || (eff_priority[arb_m] > arb_best_pri)) begin
							arb_best_pri = eff_priority[arb_m];
							ar_winner = arb_m[M_BITS - 1:0];
							ar_winner_valid = 1'b1;
						end
					end
				end
		end
	end
	reg [M_BITS - 1:0] aw_winner;
	reg aw_winner_valid;
	reg [M_BITS - 1:0] aw_rr_ptr;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			aw_rr_ptr <= '0;
		else if (aw_valid_o && aw_ready_i)
			aw_rr_ptr <= aw_winner + 1;
	always @(*) begin
		if (_sv2v_0)
			;
		aw_winner = '0;
		aw_winner_valid = 1'b0;
		arb_best_pri = '0;
		begin : sv2v_autoblock_3
			reg signed [31:0] pass;
			for (pass = 0; pass < NUM_MASTERS; pass = pass + 1)
				begin
					arb_m = (aw_rr_ptr + pass) % NUM_MASTERS;
					if (aw_valid_i[arb_m] && !throttled_o[arb_m]) begin
						if (!aw_winner_valid || (eff_priority[arb_m] > arb_best_pri)) begin
							arb_best_pri = eff_priority[arb_m];
							aw_winner = arb_m[M_BITS - 1:0];
							aw_winner_valid = 1'b1;
						end
					end
				end
		end
	end
	assign ar_valid_o = ar_winner_valid;
	assign ar_addr_o = ar_addr_i[((NUM_MASTERS - 1) - ar_winner) * ADDR_WIDTH+:ADDR_WIDTH];
	assign ar_id_o = ar_id_i[((NUM_MASTERS - 1) - ar_winner) * ID_WIDTH+:ID_WIDTH];
	assign ar_len_o = ar_len_i[((NUM_MASTERS - 1) - ar_winner) * 8+:8];
	assign ar_size_o = ar_size_i[((NUM_MASTERS - 1) - ar_winner) * 3+:3];
	assign ar_burst_o = ar_burst_i[((NUM_MASTERS - 1) - ar_winner) * 2+:2];
	assign ar_grant_id_o = ar_winner;
	assign aw_valid_o = aw_winner_valid;
	assign aw_addr_o = aw_addr_i[((NUM_MASTERS - 1) - aw_winner) * ADDR_WIDTH+:ADDR_WIDTH];
	assign aw_id_o = aw_id_i[((NUM_MASTERS - 1) - aw_winner) * ID_WIDTH+:ID_WIDTH];
	assign aw_len_o = aw_len_i[((NUM_MASTERS - 1) - aw_winner) * 8+:8];
	assign aw_size_o = aw_size_i[((NUM_MASTERS - 1) - aw_winner) * 3+:3];
	assign aw_burst_o = aw_burst_i[((NUM_MASTERS - 1) - aw_winner) * 2+:2];
	assign aw_grant_id_o = aw_winner;
	genvar _gv_m_3;
	generate
		for (_gv_m_3 = 0; _gv_m_3 < NUM_MASTERS; _gv_m_3 = _gv_m_3 + 1) begin : g_ready
			localparam m = _gv_m_3;
			assign ar_ready_o[m] = (ar_winner_valid && (ar_winner == m[M_BITS - 1:0])) && ar_ready_i;
			assign aw_ready_o[m] = (aw_winner_valid && (aw_winner == m[M_BITS - 1:0])) && aw_ready_i;
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
