module l2_cache_top (
	clk,
	rst_n,
	s_axi_awvalid,
	s_axi_awready,
	s_axi_awaddr,
	s_axi_awid,
	s_axi_awlen,
	s_axi_awsize,
	s_axi_awburst,
	s_axi_wvalid,
	s_axi_wready,
	s_axi_wdata,
	s_axi_wstrb,
	s_axi_wlast,
	s_axi_bvalid,
	s_axi_bready,
	s_axi_bresp,
	s_axi_bid,
	s_axi_arvalid,
	s_axi_arready,
	s_axi_araddr,
	s_axi_arid,
	s_axi_arlen,
	s_axi_arsize,
	s_axi_arburst,
	s_axi_rvalid,
	s_axi_rready,
	s_axi_rdata,
	s_axi_rresp,
	s_axi_rid,
	s_axi_rlast,
	m_axi_awvalid,
	m_axi_awready,
	m_axi_awaddr,
	m_axi_awid,
	m_axi_awlen,
	m_axi_awsize,
	m_axi_awburst,
	m_axi_wvalid,
	m_axi_wready,
	m_axi_wdata,
	m_axi_wstrb,
	m_axi_wlast,
	m_axi_bvalid,
	m_axi_bready,
	m_axi_bresp,
	m_axi_bid,
	m_axi_arvalid,
	m_axi_arready,
	m_axi_araddr,
	m_axi_arid,
	m_axi_arlen,
	m_axi_arsize,
	m_axi_arburst,
	m_axi_rvalid,
	m_axi_rready,
	m_axi_rdata,
	m_axi_rresp,
	m_axi_rid,
	m_axi_rlast,
	pf_enable,
	cache_busy
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	parameter signed [31:0] ID_WIDTH = 4;
	parameter signed [31:0] NUM_SETS = 256;
	parameter signed [31:0] NUM_WAYS = 8;
	parameter signed [31:0] LINE_BYTES = 64;
	parameter signed [31:0] NUM_MSHR = 4;
	input wire clk;
	input wire rst_n;
	input wire s_axi_awvalid;
	output wire s_axi_awready;
	input wire [ADDR_WIDTH - 1:0] s_axi_awaddr;
	input wire [ID_WIDTH - 1:0] s_axi_awid;
	input wire [7:0] s_axi_awlen;
	input wire [2:0] s_axi_awsize;
	input wire [1:0] s_axi_awburst;
	input wire s_axi_wvalid;
	output wire s_axi_wready;
	input wire [DATA_WIDTH - 1:0] s_axi_wdata;
	input wire [(DATA_WIDTH / 8) - 1:0] s_axi_wstrb;
	input wire s_axi_wlast;
	output wire s_axi_bvalid;
	input wire s_axi_bready;
	output wire [1:0] s_axi_bresp;
	output wire [ID_WIDTH - 1:0] s_axi_bid;
	input wire s_axi_arvalid;
	output wire s_axi_arready;
	input wire [ADDR_WIDTH - 1:0] s_axi_araddr;
	input wire [ID_WIDTH - 1:0] s_axi_arid;
	input wire [7:0] s_axi_arlen;
	input wire [2:0] s_axi_arsize;
	input wire [1:0] s_axi_arburst;
	output wire s_axi_rvalid;
	input wire s_axi_rready;
	output wire [DATA_WIDTH - 1:0] s_axi_rdata;
	output wire [1:0] s_axi_rresp;
	output wire [ID_WIDTH - 1:0] s_axi_rid;
	output wire s_axi_rlast;
	output wire m_axi_awvalid;
	input wire m_axi_awready;
	output wire [ADDR_WIDTH - 1:0] m_axi_awaddr;
	output wire [ID_WIDTH - 1:0] m_axi_awid;
	output wire [7:0] m_axi_awlen;
	output wire [2:0] m_axi_awsize;
	output wire [1:0] m_axi_awburst;
	output wire m_axi_wvalid;
	input wire m_axi_wready;
	output wire [DATA_WIDTH - 1:0] m_axi_wdata;
	output wire [(DATA_WIDTH / 8) - 1:0] m_axi_wstrb;
	output wire m_axi_wlast;
	input wire m_axi_bvalid;
	output wire m_axi_bready;
	input wire [1:0] m_axi_bresp;
	input wire [ID_WIDTH - 1:0] m_axi_bid;
	output wire m_axi_arvalid;
	input wire m_axi_arready;
	output wire [ADDR_WIDTH - 1:0] m_axi_araddr;
	output wire [ID_WIDTH - 1:0] m_axi_arid;
	output wire [7:0] m_axi_arlen;
	output wire [2:0] m_axi_arsize;
	output wire [1:0] m_axi_arburst;
	input wire m_axi_rvalid;
	output wire m_axi_rready;
	input wire [DATA_WIDTH - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire [ID_WIDTH - 1:0] m_axi_rid;
	input wire m_axi_rlast;
	input wire pf_enable;
	output wire cache_busy;
	localparam signed [31:0] OFFSET_BITS = $clog2(LINE_BYTES);
	localparam signed [31:0] INDEX_BITS = $clog2(NUM_SETS);
	localparam signed [31:0] TAG_WIDTH = (ADDR_WIDTH - INDEX_BITS) - OFFSET_BITS;
	localparam signed [31:0] WAY_BITS = $clog2(NUM_WAYS);
	localparam signed [31:0] WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
	localparam signed [31:0] WORD_SEL_BITS = $clog2(WORDS_PER_LINE);
	function automatic [TAG_WIDTH - 1:0] addr_tag;
		input [ADDR_WIDTH - 1:0] addr;
		addr_tag = addr[ADDR_WIDTH - 1-:TAG_WIDTH];
	endfunction
	function automatic [INDEX_BITS - 1:0] addr_set;
		input [ADDR_WIDTH - 1:0] addr;
		addr_set = addr[OFFSET_BITS+:INDEX_BITS];
	endfunction
	function automatic [WORD_SEL_BITS - 1:0] addr_word;
		input [ADDR_WIDTH - 1:0] addr;
		addr_word = addr[$clog2(DATA_WIDTH / 8)+:WORD_SEL_BITS];
	endfunction
	reg [3:0] ctrl_state;
	reg [3:0] ctrl_state_next;
	reg [ADDR_WIDTH - 1:0] req_addr;
	reg [ID_WIDTH - 1:0] req_id;
	reg req_is_write;
	reg [DATA_WIDTH - 1:0] req_wdata;
	reg [(DATA_WIDTH / 8) - 1:0] req_wstrb;
	reg [7:0] req_len;
	reg wr_aw_pending;
	reg wr_w_pending;
	reg [ADDR_WIDTH - 1:0] wr_awaddr_r;
	reg [ID_WIDTH - 1:0] wr_awid_r;
	reg [7:0] wr_awlen_r;
	reg [DATA_WIDTH - 1:0] wr_wdata_r;
	reg [(DATA_WIDTH / 8) - 1:0] wr_wstrb_r;
	wire s_axi_aw_fire;
	wire s_axi_w_fire;
	wire write_req_ready;
	wire tag_lookup_valid;
	wire [INDEX_BITS - 1:0] tag_lookup_set;
	wire [TAG_WIDTH - 1:0] tag_lookup_tag;
	wire tag_lookup_hit;
	wire [$clog2(NUM_WAYS) - 1:0] tag_lookup_way;
	wire tag_lookup_dirty;
	wire tag_write_valid;
	wire [INDEX_BITS - 1:0] tag_write_set;
	wire [$clog2(NUM_WAYS) - 1:0] tag_write_way;
	wire [TAG_WIDTH - 1:0] tag_write_tag;
	wire tag_write_dirty;
	wire tag_inv_valid;
	wire [INDEX_BITS - 1:0] tag_inv_set;
	wire [$clog2(NUM_WAYS) - 1:0] tag_inv_way;
	wire tag_dc_valid;
	wire [INDEX_BITS - 1:0] tag_dc_set;
	wire [$clog2(NUM_WAYS) - 1:0] tag_dc_way;
	wire tag_dc_is_dirty;
	wire [TAG_WIDTH - 1:0] tag_dc_tag;
	wire data_rd_en;
	wire [INDEX_BITS - 1:0] data_rd_set;
	wire [$clog2(NUM_WAYS) - 1:0] data_rd_way;
	wire [WORD_SEL_BITS - 1:0] data_rd_word;
	wire [DATA_WIDTH - 1:0] data_rd_data;
	wire data_wr_en;
	wire [INDEX_BITS - 1:0] data_wr_set;
	wire [$clog2(NUM_WAYS) - 1:0] data_wr_way;
	wire [WORD_SEL_BITS - 1:0] data_wr_word;
	wire [DATA_WIDTH - 1:0] data_wr_data;
	wire [(DATA_WIDTH / 8) - 1:0] data_wr_be;
	wire data_line_rd_en;
	wire [INDEX_BITS - 1:0] data_line_rd_set;
	wire [$clog2(NUM_WAYS) - 1:0] data_line_rd_way;
	wire [WORD_SEL_BITS - 1:0] data_line_rd_word;
	wire [DATA_WIDTH - 1:0] data_line_rd_data;
	wire data_line_wr_en;
	wire [INDEX_BITS - 1:0] data_line_wr_set;
	wire [$clog2(NUM_WAYS) - 1:0] data_line_wr_way;
	wire [WORD_SEL_BITS - 1:0] data_line_wr_word;
	wire [DATA_WIDTH - 1:0] data_line_wr_data;
	wire mshr_alloc_valid;
	wire mshr_alloc_ready;
	wire [ADDR_WIDTH - 1:0] mshr_alloc_addr;
	wire [ID_WIDTH - 1:0] mshr_alloc_id;
	wire mshr_alloc_is_write;
	wire [$clog2(NUM_MSHR) - 1:0] mshr_alloc_idx;
	wire mshr_lookup_valid;
	wire [ADDR_WIDTH - 1:0] mshr_lookup_addr;
	wire mshr_lookup_hit;
	wire [$clog2(NUM_MSHR) - 1:0] mshr_lookup_idx;
	wire mshr_complete_valid;
	wire [$clog2(NUM_MSHR) - 1:0] mshr_complete_idx;
	wire [ADDR_WIDTH - 1:0] mshr_complete_addr;
	wire [ID_WIDTH - 1:0] mshr_complete_id;
	wire mshr_complete_is_write;
	wire mshr_full;
	wire mshr_empty;
	reg [NUM_WAYS - 2:0] plru_bits [0:NUM_SETS - 1];
	reg [$clog2(NUM_WAYS) - 1:0] victim_way;
	reg [NUM_WAYS - 2:0] plru_hit_bits_next;
	reg [NUM_WAYS - 2:0] plru_fill_bits_next;
	integer plru_victim_idx_c;
	integer plru_victim_level_c;
	integer plru_hit_idx_c;
	integer plru_hit_parent_c;
	integer plru_hit_level_c;
	integer plru_fill_idx_c;
	integer plru_fill_parent_c;
	integer plru_fill_level_c;
	integer plru_reset_set_i;
	always @(*) begin
		if (_sv2v_0)
			;
		plru_victim_idx_c = 0;
		for (plru_victim_level_c = 0; plru_victim_level_c < WAY_BITS; plru_victim_level_c = plru_victim_level_c + 1)
			if (plru_bits[addr_set(req_addr)][plru_victim_idx_c])
				plru_victim_idx_c = (2 * plru_victim_idx_c) + 2;
			else
				plru_victim_idx_c = (2 * plru_victim_idx_c) + 1;
		victim_way = plru_victim_idx_c - (NUM_WAYS - 1);
	end
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		plru_hit_bits_next = plru_bits[addr_set(req_addr)];
		plru_hit_idx_c = sv2v_cast_32_signed(tag_lookup_way) + (NUM_WAYS - 1);
		for (plru_hit_level_c = 0; plru_hit_level_c < WAY_BITS; plru_hit_level_c = plru_hit_level_c + 1)
			begin
				plru_hit_parent_c = (plru_hit_idx_c - 1) / 2;
				if (plru_hit_idx_c == ((2 * plru_hit_parent_c) + 1))
					plru_hit_bits_next[plru_hit_parent_c] = 1'b1;
				else
					plru_hit_bits_next[plru_hit_parent_c] = 1'b0;
				plru_hit_idx_c = plru_hit_parent_c;
			end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		plru_fill_bits_next = plru_bits[addr_set(req_addr)];
		plru_fill_idx_c = sv2v_cast_32_signed(tag_write_way) + (NUM_WAYS - 1);
		for (plru_fill_level_c = 0; plru_fill_level_c < WAY_BITS; plru_fill_level_c = plru_fill_level_c + 1)
			begin
				plru_fill_parent_c = (plru_fill_idx_c - 1) / 2;
				if (plru_fill_idx_c == ((2 * plru_fill_parent_c) + 1))
					plru_fill_bits_next[plru_fill_parent_c] = 1'b1;
				else
					plru_fill_bits_next[plru_fill_parent_c] = 1'b0;
				plru_fill_idx_c = plru_fill_parent_c;
			end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			for (plru_reset_set_i = 0; plru_reset_set_i < NUM_SETS; plru_reset_set_i = plru_reset_set_i + 1)
				plru_bits[plru_reset_set_i] <= '0;
		else begin
			if (((ctrl_state == 4'd2) || (ctrl_state == 4'd3)) && tag_lookup_hit)
				plru_bits[addr_set(req_addr)] <= plru_hit_bits_next;
			if (ctrl_state == 4'd9)
				plru_bits[addr_set(req_addr)] <= plru_fill_bits_next;
		end
	wire pf_miss_valid;
	wire [ADDR_WIDTH - 1:0] pf_miss_addr;
	wire pf_req_valid;
	wire pf_req_ready;
	wire [ADDR_WIDTH - 1:0] pf_req_addr;
	reg [WORD_SEL_BITS - 1:0] beat_cnt;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			beat_cnt <= '0;
		else
			case (ctrl_state)
				4'd7: beat_cnt <= '0;
				4'd6:
					if (m_axi_wvalid && m_axi_wready)
						beat_cnt <= beat_cnt + 1;
				4'd8:
					if (m_axi_rvalid && m_axi_rready)
						beat_cnt <= beat_cnt + 1;
				default:
					;
			endcase
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			ctrl_state <= 4'd0;
		else
			ctrl_state <= ctrl_state_next;
	assign s_axi_aw_fire = s_axi_awvalid && s_axi_awready;
	assign s_axi_w_fire = s_axi_wvalid && s_axi_wready;
	assign write_req_ready = (wr_aw_pending || s_axi_aw_fire) && (wr_w_pending || s_axi_w_fire);
	always @(*) begin
		if (_sv2v_0)
			;
		ctrl_state_next = ctrl_state;
		case (ctrl_state)
			4'd0:
				if (s_axi_arvalid && s_axi_arready)
					ctrl_state_next = 4'd1;
				else if (!s_axi_arvalid && write_req_ready)
					ctrl_state_next = 4'd1;
				else if ((pf_req_valid && !wr_aw_pending) && !wr_w_pending)
					ctrl_state_next = 4'd12;
			4'd1: ctrl_state_next = (tag_lookup_hit ? (req_is_write ? 4'd3 : 4'd2) : 4'd4);
			4'd2: ctrl_state_next = 4'd10;
			4'd3: ctrl_state_next = 4'd11;
			4'd4:
				if (mshr_lookup_hit)
					ctrl_state_next = 4'd0;
				else if (mshr_alloc_ready)
					ctrl_state_next = 4'd5;
				else
					ctrl_state_next = 4'd0;
			4'd5: ctrl_state_next = (tag_dc_is_dirty ? 4'd6 : 4'd7);
			4'd6:
				if ((m_axi_wvalid && m_axi_wready) && m_axi_wlast) begin
					if (m_axi_bvalid)
						ctrl_state_next = 4'd7;
				end
			4'd7:
				if (m_axi_arready)
					ctrl_state_next = 4'd8;
			4'd8:
				if (m_axi_rvalid && m_axi_rlast)
					ctrl_state_next = 4'd9;
			4'd9: ctrl_state_next = (req_is_write ? 4'd11 : 4'd10);
			4'd10:
				if (s_axi_rvalid && s_axi_rready)
					ctrl_state_next = 4'd0;
			4'd11:
				if (s_axi_bvalid && s_axi_bready)
					ctrl_state_next = 4'd0;
			4'd12:
				if (tag_lookup_hit || mshr_full)
					ctrl_state_next = 4'd0;
				else
					ctrl_state_next = 4'd5;
			default: ctrl_state_next = 4'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			req_addr <= '0;
			req_id <= '0;
			req_is_write <= 1'b0;
			req_wdata <= '0;
			req_wstrb <= '0;
			req_len <= '0;
			wr_aw_pending <= 1'b0;
			wr_w_pending <= 1'b0;
			wr_awaddr_r <= '0;
			wr_awid_r <= '0;
			wr_awlen_r <= '0;
			wr_wdata_r <= '0;
			wr_wstrb_r <= '0;
		end
		else if (ctrl_state == 4'd0) begin
			if (s_axi_arvalid && s_axi_arready) begin
				req_addr <= s_axi_araddr;
				req_id <= s_axi_arid;
				req_is_write <= 1'b0;
				req_wdata <= '0;
				req_wstrb <= '0;
				req_len <= s_axi_arlen;
			end
			else begin
				if (s_axi_aw_fire) begin
					wr_aw_pending <= 1'b1;
					wr_awaddr_r <= s_axi_awaddr;
					wr_awid_r <= s_axi_awid;
					wr_awlen_r <= s_axi_awlen;
				end
				if (s_axi_w_fire) begin
					wr_w_pending <= 1'b1;
					wr_wdata_r <= s_axi_wdata;
					wr_wstrb_r <= s_axi_wstrb;
				end
				if (!s_axi_arvalid && write_req_ready) begin
					req_addr <= (s_axi_aw_fire ? s_axi_awaddr : wr_awaddr_r);
					req_id <= (s_axi_aw_fire ? s_axi_awid : wr_awid_r);
					req_is_write <= 1'b1;
					req_wdata <= (s_axi_w_fire ? s_axi_wdata : wr_wdata_r);
					req_wstrb <= (s_axi_w_fire ? s_axi_wstrb : wr_wstrb_r);
					req_len <= (s_axi_aw_fire ? s_axi_awlen : wr_awlen_r);
					wr_aw_pending <= 1'b0;
					wr_w_pending <= 1'b0;
				end
				else if ((pf_req_valid && !wr_aw_pending) && !wr_w_pending) begin
					req_addr <= pf_req_addr;
					req_id <= '0;
					req_is_write <= 1'b0;
					req_wdata <= '0;
					req_wstrb <= '0;
					req_len <= '0;
				end
			end
		end
	assign s_axi_arready = ((ctrl_state == 4'd0) && !wr_aw_pending) && !wr_w_pending;
	assign s_axi_awready = ((ctrl_state == 4'd0) && !s_axi_arvalid) && !wr_aw_pending;
	assign s_axi_wready = ((ctrl_state == 4'd0) && !s_axi_arvalid) && !wr_w_pending;
	assign tag_lookup_valid = (ctrl_state == 4'd1) || (ctrl_state == 4'd12);
	assign tag_lookup_set = addr_set(req_addr);
	assign tag_lookup_tag = addr_tag(req_addr);
	assign tag_write_valid = ctrl_state == 4'd9;
	assign tag_write_set = addr_set(req_addr);
	assign tag_write_way = victim_way;
	assign tag_write_tag = addr_tag(req_addr);
	assign tag_write_dirty = req_is_write;
	assign tag_inv_valid = 1'b0;
	assign tag_inv_set = '0;
	assign tag_inv_way = '0;
	assign tag_dc_valid = ctrl_state == 4'd5;
	assign tag_dc_set = addr_set(req_addr);
	assign tag_dc_way = victim_way;
	assign data_rd_en = ctrl_state == 4'd2;
	assign data_rd_set = addr_set(req_addr);
	assign data_rd_way = tag_lookup_way;
	assign data_rd_word = addr_word(req_addr);
	assign data_wr_en = ctrl_state == 4'd3;
	assign data_wr_set = addr_set(req_addr);
	assign data_wr_way = tag_lookup_way;
	assign data_wr_word = addr_word(req_addr);
	assign data_wr_data = req_wdata;
	assign data_wr_be = req_wstrb;
	assign data_line_rd_en = ctrl_state == 4'd6;
	assign data_line_rd_set = addr_set(req_addr);
	assign data_line_rd_way = victim_way;
	assign data_line_rd_word = beat_cnt;
	assign data_line_wr_en = (ctrl_state == 4'd8) && m_axi_rvalid;
	assign data_line_wr_set = addr_set(req_addr);
	assign data_line_wr_way = victim_way;
	assign data_line_wr_word = beat_cnt;
	assign data_line_wr_data = m_axi_rdata;
	assign mshr_lookup_valid = ctrl_state == 4'd4;
	assign mshr_lookup_addr = req_addr;
	assign mshr_alloc_valid = (ctrl_state == 4'd4) && !mshr_lookup_hit;
	assign mshr_alloc_addr = req_addr;
	assign mshr_alloc_id = req_id;
	assign mshr_alloc_is_write = req_is_write;
	assign mshr_complete_valid = ctrl_state == 4'd9;
	assign mshr_complete_idx = mshr_alloc_idx;
	assign m_axi_arvalid = ctrl_state == 4'd7;
	assign m_axi_araddr = {req_addr[ADDR_WIDTH - 1:OFFSET_BITS], {OFFSET_BITS {1'b0}}};
	assign m_axi_arid = req_id;
	function automatic signed [7:0] sv2v_cast_8_signed;
		input reg signed [7:0] inp;
		sv2v_cast_8_signed = inp;
	endfunction
	assign m_axi_arlen = sv2v_cast_8_signed(WORDS_PER_LINE - 1);
	function automatic signed [2:0] sv2v_cast_3_signed;
		input reg signed [2:0] inp;
		sv2v_cast_3_signed = inp;
	endfunction
	assign m_axi_arsize = sv2v_cast_3_signed($clog2(DATA_WIDTH / 8));
	assign m_axi_arburst = 2'b01;
	assign m_axi_rready = ctrl_state == 4'd8;
	assign m_axi_awvalid = (ctrl_state == 4'd6) && (beat_cnt == '0);
	assign m_axi_awaddr = {tag_dc_tag, addr_set(req_addr), {OFFSET_BITS {1'b0}}};
	assign m_axi_awid = req_id;
	assign m_axi_awlen = sv2v_cast_8_signed(WORDS_PER_LINE - 1);
	assign m_axi_awsize = sv2v_cast_3_signed($clog2(DATA_WIDTH / 8));
	assign m_axi_awburst = 2'b01;
	assign m_axi_wvalid = ctrl_state == 4'd6;
	assign m_axi_wdata = data_line_rd_data;
	assign m_axi_wstrb = '1;
	assign m_axi_wlast = beat_cnt == (WORDS_PER_LINE - 1);
	assign m_axi_bready = 1'b1;
	assign s_axi_rvalid = ctrl_state == 4'd10;
	assign s_axi_rdata = data_rd_data;
	assign s_axi_rresp = 2'b00;
	assign s_axi_rid = req_id;
	assign s_axi_rlast = 1'b1;
	assign s_axi_bvalid = ctrl_state == 4'd11;
	assign s_axi_bresp = 2'b00;
	assign s_axi_bid = req_id;
	assign pf_miss_valid = (ctrl_state == 4'd4) && !mshr_lookup_hit;
	assign pf_miss_addr = req_addr;
	assign pf_req_ready = (((((ctrl_state == 4'd0) && !s_axi_arvalid) && !s_axi_awvalid) && !s_axi_wvalid) && !wr_aw_pending) && !wr_w_pending;
	assign cache_busy = ctrl_state != 4'd0;
	l2_tag_array #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS),
		.LINE_BYTES(LINE_BYTES)
	) u_tag(
		.clk(clk),
		.rst_n(rst_n),
		.lookup_valid(tag_lookup_valid),
		.lookup_set(tag_lookup_set),
		.lookup_tag(tag_lookup_tag),
		.lookup_hit(tag_lookup_hit),
		.lookup_way(tag_lookup_way),
		.lookup_dirty(tag_lookup_dirty),
		.write_valid(tag_write_valid),
		.write_set(tag_write_set),
		.write_way(tag_write_way),
		.write_tag(tag_write_tag),
		.write_dirty(tag_write_dirty),
		.inv_valid(tag_inv_valid),
		.inv_set(tag_inv_set),
		.inv_way(tag_inv_way),
		.dirty_check_valid(tag_dc_valid),
		.dirty_check_set(tag_dc_set),
		.dirty_check_way(tag_dc_way),
		.dirty_check_is_dirty(tag_dc_is_dirty),
		.dirty_check_tag(tag_dc_tag)
	);
	l2_data_array #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.DATA_WIDTH(DATA_WIDTH),
		.NUM_SETS(NUM_SETS),
		.NUM_WAYS(NUM_WAYS),
		.LINE_BYTES(LINE_BYTES)
	) u_data(
		.clk(clk),
		.rd_en(data_rd_en),
		.rd_set(data_rd_set),
		.rd_way(data_rd_way),
		.rd_word(data_rd_word),
		.rd_data(data_rd_data),
		.wr_en(data_wr_en),
		.wr_set(data_wr_set),
		.wr_way(data_wr_way),
		.wr_word(data_wr_word),
		.wr_data(data_wr_data),
		.wr_be(data_wr_be),
		.line_rd_en(data_line_rd_en),
		.line_rd_set(data_line_rd_set),
		.line_rd_way(data_line_rd_way),
		.line_rd_word(data_line_rd_word),
		.line_rd_data(data_line_rd_data),
		.line_wr_en(data_line_wr_en),
		.line_wr_set(data_line_wr_set),
		.line_wr_way(data_line_wr_way),
		.line_wr_word(data_line_wr_word),
		.line_wr_data(data_line_wr_data)
	);
	l2_mshr #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.NUM_ENTRIES(NUM_MSHR),
		.ID_WIDTH(ID_WIDTH),
		.LINE_BYTES(LINE_BYTES)
	) u_mshr(
		.clk(clk),
		.rst_n(rst_n),
		.alloc_valid(mshr_alloc_valid),
		.alloc_ready(mshr_alloc_ready),
		.alloc_addr(mshr_alloc_addr),
		.alloc_id(mshr_alloc_id),
		.alloc_is_write(mshr_alloc_is_write),
		.alloc_idx(mshr_alloc_idx),
		.lookup_valid(mshr_lookup_valid),
		.lookup_addr(mshr_lookup_addr),
		.lookup_hit(mshr_lookup_hit),
		.lookup_idx(mshr_lookup_idx),
		.complete_valid(mshr_complete_valid),
		.complete_idx(mshr_complete_idx),
		.complete_addr(mshr_complete_addr),
		.complete_id(mshr_complete_id),
		.complete_is_write(mshr_complete_is_write),
		.full(mshr_full),
		.empty(mshr_empty),
		.count()
	);
	stride_prefetcher #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.TABLE_ENTRIES(16),
		.LINE_BYTES(LINE_BYTES)
	) u_prefetcher(
		.clk(clk),
		.rst_n(rst_n),
		.miss_valid(pf_miss_valid),
		.miss_addr(pf_miss_addr),
		.pf_req_valid(pf_req_valid),
		.pf_req_ready(pf_req_ready),
		.pf_req_addr(pf_req_addr),
		.pf_enable(pf_enable)
	);
	initial _sv2v_0 = 0;
endmodule
