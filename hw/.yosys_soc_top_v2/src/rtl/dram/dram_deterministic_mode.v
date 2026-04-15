module dram_deterministic_mode (
	clk,
	rst_n,
	det_enable,
	ar_accepted,
	ar_id,
	dram_rvalid,
	dram_rdata,
	dram_rid,
	det_rvalid,
	det_rdata,
	det_rid,
	det_rlast,
	err_deadline_miss
);
	reg _sv2v_0;
	parameter signed [31:0] DATA_W = 32;
	parameter signed [31:0] ID_W = 4;
	parameter signed [31:0] FIXED_LATENCY = 16;
	parameter signed [31:0] MAX_OUTSTANDING = 4;
	input wire clk;
	input wire rst_n;
	input wire det_enable;
	input wire ar_accepted;
	input wire [ID_W - 1:0] ar_id;
	input wire dram_rvalid;
	input wire [DATA_W - 1:0] dram_rdata;
	input wire [ID_W - 1:0] dram_rid;
	output reg det_rvalid;
	output reg [DATA_W - 1:0] det_rdata;
	output reg [ID_W - 1:0] det_rid;
	output reg det_rlast;
	output reg err_deadline_miss;
	localparam signed [31:0] CTR_W = $clog2(FIXED_LATENCY + 1);
	localparam signed [31:0] SLOT_W = $clog2(MAX_OUTSTANDING);
	reg [MAX_OUTSTANDING - 1:0] sl_valid;
	reg [MAX_OUTSTANDING - 1:0] sl_data_ready;
	reg [CTR_W - 1:0] sl_countdown [0:MAX_OUTSTANDING - 1];
	reg [DATA_W - 1:0] sl_data [0:MAX_OUTSTANDING - 1];
	reg [ID_W - 1:0] sl_id [0:MAX_OUTSTANDING - 1];
	reg [SLOT_W - 1:0] free_idx;
	reg free_found;
	always @(*) begin
		if (_sv2v_0)
			;
		free_found = 1'b0;
		free_idx = '0;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < MAX_OUTSTANDING; i = i + 1)
				if (!sl_valid[i] && !free_found) begin
					free_found = 1'b1;
					free_idx = i;
				end
		end
	end
	reg [SLOT_W - 1:0] match_idx;
	reg match_found;
	always @(*) begin
		if (_sv2v_0)
			;
		match_found = 1'b0;
		match_idx = '0;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < MAX_OUTSTANDING; i = i + 1)
				if (((sl_valid[i] && !sl_data_ready[i]) && (sl_id[i] == dram_rid)) && !match_found) begin
					match_found = 1'b1;
					match_idx = i;
				end
		end
	end
	reg [SLOT_W - 1:0] fire_idx;
	reg fire_found;
	always @(*) begin
		if (_sv2v_0)
			;
		fire_found = 1'b0;
		fire_idx = '0;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < MAX_OUTSTANDING; i = i + 1)
				if ((sl_valid[i] && (sl_countdown[i] == '0)) && !fire_found) begin
					fire_found = 1'b1;
					fire_idx = i;
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			sl_valid <= '0;
			sl_data_ready <= '0;
			begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < MAX_OUTSTANDING; i = i + 1)
					begin
						sl_countdown[i] <= '0;
						sl_data[i] <= '0;
						sl_id[i] <= '0;
					end
			end
		end
		else begin
			if ((ar_accepted && free_found) && det_enable) begin
				sl_valid[free_idx] <= 1'b1;
				sl_data_ready[free_idx] <= 1'b0;
				sl_countdown[free_idx] <= FIXED_LATENCY - 1;
				sl_data[free_idx] <= '0;
				sl_id[free_idx] <= ar_id;
			end
			if ((dram_rvalid && match_found) && det_enable) begin
				sl_data_ready[match_idx] <= 1'b1;
				sl_data[match_idx] <= dram_rdata;
			end
			begin : sv2v_autoblock_5
				reg signed [31:0] i;
				for (i = 0; i < MAX_OUTSTANDING; i = i + 1)
					if (sl_valid[i] && (sl_countdown[i] != '0))
						sl_countdown[i] <= sl_countdown[i] - 1'b1;
			end
			if (fire_found && det_enable)
				sl_valid[fire_idx] <= 1'b0;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		if (!det_enable) begin
			det_rvalid = dram_rvalid;
			det_rdata = dram_rdata;
			det_rid = dram_rid;
			det_rlast = dram_rvalid;
			err_deadline_miss = 1'b0;
		end
		else begin
			det_rvalid = fire_found && sl_data_ready[fire_idx];
			det_rdata = (fire_found ? sl_data[fire_idx] : '0);
			det_rid = (fire_found ? sl_id[fire_idx] : '0);
			det_rlast = det_rvalid;
			err_deadline_miss = fire_found && !sl_data_ready[fire_idx];
		end
	end
	initial _sv2v_0 = 0;
endmodule
