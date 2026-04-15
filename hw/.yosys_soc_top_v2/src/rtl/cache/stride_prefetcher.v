module stride_prefetcher (
	clk,
	rst_n,
	miss_valid,
	miss_addr,
	pf_req_valid,
	pf_req_ready,
	pf_req_addr,
	pf_enable
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] TABLE_ENTRIES = 16;
	parameter signed [31:0] LINE_BYTES = 64;
	parameter signed [31:0] CONF_MAX = 3;
	input wire clk;
	input wire rst_n;
	input wire miss_valid;
	input wire [ADDR_WIDTH - 1:0] miss_addr;
	output wire pf_req_valid;
	input wire pf_req_ready;
	output wire [ADDR_WIDTH - 1:0] pf_req_addr;
	input wire pf_enable;
	localparam signed [31:0] LINE_ADDR_BITS = ADDR_WIDTH - $clog2(LINE_BYTES);
	localparam signed [31:0] IDX_BITS = $clog2(TABLE_ENTRIES);
	reg table_valid_r [0:TABLE_ENTRIES - 1];
	reg [LINE_ADDR_BITS - 1:0] table_last_line_r [0:TABLE_ENTRIES - 1];
	reg signed [LINE_ADDR_BITS - 1:0] table_stride_r [0:TABLE_ENTRIES - 1];
	reg [1:0] table_confidence_r [0:TABLE_ENTRIES - 1];
	wire [LINE_ADDR_BITS - 1:0] miss_line;
	assign miss_line = miss_addr[ADDR_WIDTH - 1:$clog2(LINE_BYTES)];
	wire [IDX_BITS - 1:0] tbl_idx;
	assign tbl_idx = miss_line[IDX_BITS - 1:0];
	wire cur_valid;
	wire [LINE_ADDR_BITS - 1:0] cur_last_line;
	wire signed [LINE_ADDR_BITS - 1:0] cur_stride;
	wire [1:0] cur_confidence;
	assign cur_valid = table_valid_r[tbl_idx];
	assign cur_last_line = table_last_line_r[tbl_idx];
	assign cur_stride = table_stride_r[tbl_idx];
	assign cur_confidence = table_confidence_r[tbl_idx];
	wire signed [LINE_ADDR_BITS - 1:0] observed_stride;
	assign observed_stride = miss_line - cur_last_line;
	wire stride_match;
	assign stride_match = (observed_stride == cur_stride) && cur_valid;
	wire [LINE_ADDR_BITS - 1:0] pf_line;
	assign pf_line = miss_line + cur_stride;
	reg [1:0] state;
	reg [1:0] state_next;
	wire should_prefetch;
	assign should_prefetch = (pf_enable && stride_match) && (cur_confidence == CONF_MAX[1:0]);
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 2'd0;
		else
			state <= state_next;
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		case (state)
			2'd0:
				if (miss_valid)
					state_next = 2'd1;
			2'd1:
				if (should_prefetch)
					state_next = 2'd2;
				else
					state_next = 2'd0;
			2'd2:
				if (pf_req_ready)
					state_next = 2'd0;
			default: state_next = 2'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < TABLE_ENTRIES; i = i + 1)
					table_valid_r[i] <= 1'b0;
			end
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < TABLE_ENTRIES; i = i + 1)
					table_last_line_r[i] <= '0;
			end
			begin : sv2v_autoblock_3
				reg signed [31:0] i;
				for (i = 0; i < TABLE_ENTRIES; i = i + 1)
					table_stride_r[i] <= '0;
			end
			begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < TABLE_ENTRIES; i = i + 1)
					table_confidence_r[i] <= '0;
			end
		end
		else if (state == 2'd1) begin
			table_last_line_r[tbl_idx] <= miss_line;
			if (!cur_valid) begin
				table_valid_r[tbl_idx] <= 1'b1;
				table_stride_r[tbl_idx] <= '0;
				table_confidence_r[tbl_idx] <= '0;
			end
			else if (stride_match) begin
				if (cur_confidence < CONF_MAX[1:0])
					table_confidence_r[tbl_idx] <= cur_confidence + 2'd1;
			end
			else begin
				table_stride_r[tbl_idx] <= observed_stride;
				table_confidence_r[tbl_idx] <= 2'd0;
			end
		end
	reg [ADDR_WIDTH - 1:0] pf_addr_reg;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			pf_addr_reg <= '0;
		else if ((state == 2'd1) && should_prefetch)
			pf_addr_reg <= {pf_line, {$clog2(LINE_BYTES) {1'b0}}};
	assign pf_req_valid = state == 2'd2;
	assign pf_req_addr = pf_addr_reg;
	initial _sv2v_0 = 0;
endmodule
