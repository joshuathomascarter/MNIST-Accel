module l1_cache_ctrl (
	clk,
	rst_n,
	cpu_req,
	cpu_gnt,
	cpu_addr,
	cpu_we,
	cpu_be,
	cpu_wdata,
	cpu_rvalid,
	cpu_rdata,
	mem_req,
	mem_gnt,
	mem_addr,
	mem_we,
	mem_wdata,
	mem_rvalid,
	mem_rdata,
	mem_last,
	cache_busy
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] NUM_SETS = 16;
	parameter signed [31:0] NUM_WAYS = 4;
	parameter signed [31:0] LINE_BYTES = 64;
	input wire clk;
	input wire rst_n;
	input wire cpu_req;
	output wire cpu_gnt;
	input wire [ADDR_WIDTH - 1:0] cpu_addr;
	input wire cpu_we;
	input wire [(DATA_WIDTH / 8) - 1:0] cpu_be;
	input wire [DATA_WIDTH - 1:0] cpu_wdata;
	output wire cpu_rvalid;
	output wire [DATA_WIDTH - 1:0] cpu_rdata;
	output wire mem_req;
	input wire mem_gnt;
	output wire [ADDR_WIDTH - 1:0] mem_addr;
	output wire mem_we;
	output wire [DATA_WIDTH - 1:0] mem_wdata;
	input wire mem_rvalid;
	input wire [DATA_WIDTH - 1:0] mem_rdata;
	output wire mem_last;
	output wire cache_busy;
	localparam signed [31:0] OFFSET_BITS = $clog2(LINE_BYTES);
	localparam signed [31:0] SET_BITS = $clog2(NUM_SETS);
	localparam signed [31:0] TAG_BITS = (ADDR_WIDTH - SET_BITS) - OFFSET_BITS;
	localparam signed [31:0] WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
	localparam signed [31:0] WORD_IDX_BITS = $clog2(WORDS_PER_LINE);
	function automatic signed [WORD_IDX_BITS - 1:0] sv2v_cast_60042_signed;
		input reg signed [WORD_IDX_BITS - 1:0] inp;
		sv2v_cast_60042_signed = inp;
	endfunction
	localparam [WORD_IDX_BITS - 1:0] LAST_BEAT = sv2v_cast_60042_signed(WORDS_PER_LINE - 1);
	reg [2:0] state;
	reg [2:0] state_next;
	reg [ADDR_WIDTH - 1:0] req_addr_r;
	reg req_we_r;
	reg [(DATA_WIDTH / 8) - 1:0] req_be_r;
	reg [DATA_WIDTH - 1:0] req_wdata_r;
	wire [TAG_BITS - 1:0] req_tag;
	wire [SET_BITS - 1:0] req_set;
	wire [OFFSET_BITS - 1:0] req_offset;
	assign req_tag = req_addr_r[ADDR_WIDTH - 1-:TAG_BITS];
	assign req_set = req_addr_r[OFFSET_BITS+:SET_BITS];
	assign req_offset = req_addr_r[OFFSET_BITS - 1:0];
	wire tag_lookup_hit;
	wire [$clog2(NUM_WAYS) - 1:0] tag_lookup_way;
	wire tag_lookup_dirty;
	wire tag_write_en;
	wire [$clog2(NUM_SETS) - 1:0] tag_write_set;
	wire [$clog2(NUM_WAYS) - 1:0] tag_write_way;
	wire [TAG_BITS - 1:0] tag_write_tag;
	wire tag_write_valid;
	wire tag_write_dirty;
	wire [TAG_BITS - 1:0] tag_rb_tag;
	wire tag_rb_dirty;
	wire tag_rb_valid;
	wire data_word_en;
	wire data_word_we;
	wire [DATA_WIDTH - 1:0] data_word_rdata;
	wire data_line_en;
	wire data_line_we;
	reg [(LINE_BYTES * 8) - 1:0] data_line_wdata;
	wire [(LINE_BYTES * 8) - 1:0] data_line_rdata;
	wire lru_access_valid;
	wire [$clog2(NUM_WAYS) - 1:0] lru_victim_way;
	reg [WORD_IDX_BITS - 1:0] beat_cnt;
	reg [(LINE_BYTES * 8) - 1:0] fill_buffer;
	reg [$clog2(NUM_WAYS) - 1:0] victim_way_r;
	l1_tag_array #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS),
		.LINE_BYTES(LINE_BYTES)
	) u_tags(
		.clk(clk),
		.rst_n(rst_n),
		.lookup_addr(req_addr_r),
		.lookup_hit(tag_lookup_hit),
		.lookup_way(tag_lookup_way),
		.lookup_dirty(tag_lookup_dirty),
		.write_en(tag_write_en),
		.write_set(tag_write_set),
		.write_way(tag_write_way),
		.write_tag(tag_write_tag),
		.write_valid(tag_write_valid),
		.write_dirty(tag_write_dirty),
		.inv_en(1'b0),
		.inv_set('0),
		.inv_way('0),
		.rb_set(req_set),
		.rb_way(victim_way_r),
		.rb_tag(tag_rb_tag),
		.rb_dirty(tag_rb_dirty),
		.rb_valid(tag_rb_valid)
	);
	l1_data_array #(
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS),
		.LINE_BYTES(LINE_BYTES),
		.WORD_WIDTH(DATA_WIDTH)
	) u_data(
		.clk(clk),
		.word_en(data_word_en),
		.word_we(data_word_we),
		.word_set(req_set),
		.word_way((tag_lookup_hit ? tag_lookup_way : victim_way_r)),
		.word_offset(req_offset),
		.word_be(req_be_r),
		.word_wdata(req_wdata_r),
		.word_rdata(data_word_rdata),
		.line_en(data_line_en),
		.line_we(data_line_we),
		.line_set(req_set),
		.line_way(victim_way_r),
		.line_wdata(data_line_wdata),
		.line_rdata(data_line_rdata)
	);
	l1_lru #(
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS)
	) u_lru(
		.clk(clk),
		.rst_n(rst_n),
		.access_valid(lru_access_valid),
		.access_set(req_set),
		.access_way((tag_lookup_hit ? tag_lookup_way : victim_way_r)),
		.query_set(req_set),
		.victim_way(lru_victim_way)
	);
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 3'd0;
		else
			state <= state_next;
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		case (state)
			3'd0:
				if (cpu_req)
					state_next = 3'd1;
			3'd1:
				if (tag_lookup_hit)
					state_next = 3'd0;
				else if (tag_rb_valid && tag_rb_dirty)
					state_next = 3'd2;
				else
					state_next = 3'd3;
			3'd2:
				if (mem_gnt && (beat_cnt == LAST_BEAT))
					state_next = 3'd3;
			3'd3:
				if (mem_rvalid && (beat_cnt == LAST_BEAT))
					state_next = 3'd4;
			3'd4: state_next = 3'd0;
			default: state_next = 3'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			req_addr_r <= '0;
			req_we_r <= 1'b0;
			req_be_r <= '0;
			req_wdata_r <= '0;
			beat_cnt <= '0;
			fill_buffer <= '0;
			victim_way_r <= '0;
		end
		else
			case (state)
				3'd0:
					if (cpu_req) begin
						req_addr_r <= cpu_addr;
						req_we_r <= cpu_we;
						req_be_r <= cpu_be;
						req_wdata_r <= cpu_wdata;
					end
				3'd1:
					if (!tag_lookup_hit) begin
						victim_way_r <= lru_victim_way;
						beat_cnt <= '0;
					end
				3'd2:
					if (mem_gnt)
						beat_cnt <= beat_cnt + 1;
				3'd3:
					if (mem_rvalid) begin
						fill_buffer[beat_cnt * DATA_WIDTH+:DATA_WIDTH] <= mem_rdata;
						beat_cnt <= beat_cnt + 1;
					end
				3'd4: beat_cnt <= '0;
				default:
					;
			endcase
	assign cpu_gnt = (state == 3'd0) && cpu_req;
	assign cpu_rvalid = ((state == 3'd1) && tag_lookup_hit) || (state == 3'd4);
	assign cpu_rdata = (state == 3'd4 ? fill_buffer[req_offset[$clog2(LINE_BYTES) - 1:2] * DATA_WIDTH+:DATA_WIDTH] : data_word_rdata);
	assign cache_busy = state != 3'd0;
	assign data_word_en = (state == 3'd1) && tag_lookup_hit;
	assign data_word_we = ((state == 3'd1) && tag_lookup_hit) && req_we_r;
	assign data_line_en = (state == 3'd2) || (state == 3'd4);
	assign data_line_we = state == 3'd4;
	always @(*) begin
		if (_sv2v_0)
			;
		data_line_wdata = fill_buffer;
		if ((state == 3'd4) && req_we_r) begin : sv2v_autoblock_1
			reg signed [31:0] b;
			for (b = 0; b < (DATA_WIDTH / 8); b = b + 1)
				if (req_be_r[b])
					data_line_wdata[{(req_offset[$clog2(LINE_BYTES) - 1:2] * DATA_WIDTH) + (b * 8)}+:8] = req_wdata_r[b * 8+:8];
		end
	end
	assign tag_write_en = (((state == 3'd1) && tag_lookup_hit) && req_we_r) || (state == 3'd4);
	assign tag_write_set = req_set;
	assign tag_write_way = (state == 3'd4 ? victim_way_r : tag_lookup_way);
	assign tag_write_tag = req_tag;
	assign tag_write_valid = 1'b1;
	assign tag_write_dirty = (state == 3'd4 ? req_we_r : ((state == 3'd1) && req_we_r ? 1'b1 : tag_lookup_dirty));
	assign lru_access_valid = ((state == 3'd1) && tag_lookup_hit) || (state == 3'd4);
	wire [ADDR_WIDTH - 1:0] wb_addr;
	assign wb_addr = {tag_rb_tag, req_set, {OFFSET_BITS {1'b0}}};
	assign mem_req = (state == 3'd2) || (state == 3'd3);
	assign mem_we = state == 3'd2;
	assign mem_addr = (state == 3'd2 ? wb_addr + {{(ADDR_WIDTH - WORD_IDX_BITS) - 2 {1'b0}}, beat_cnt, 2'b00} : {req_addr_r[ADDR_WIDTH - 1:OFFSET_BITS], {OFFSET_BITS {1'b0}}} + {{(ADDR_WIDTH - WORD_IDX_BITS) - 2 {1'b0}}, beat_cnt, 2'b00});
	assign mem_wdata = data_line_rdata[beat_cnt * DATA_WIDTH+:DATA_WIDTH];
	assign mem_last = beat_cnt == LAST_BEAT;
	initial _sv2v_0 = 0;
endmodule
