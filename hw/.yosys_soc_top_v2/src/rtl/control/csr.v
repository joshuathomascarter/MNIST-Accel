`default_nettype none
module csr (
	clk,
	rst_n,
	csr_wen,
	csr_ren,
	csr_addr,
	csr_wdata,
	csr_rdata,
	core_busy,
	core_done_tile_pulse,
	core_bank_sel_rd_A,
	core_bank_sel_rd_B,
	rx_illegal_cmd,
	perf_total_cycles,
	perf_active_cycles,
	perf_idle_cycles,
	perf_dma_bytes,
	perf_blocks_processed,
	perf_stall_cycles,
	result_data,
	dma_busy_in,
	dma_done_in,
	dma_bytes_xferred_in,
	start_pulse,
	abort_pulse,
	irq_en,
	M,
	N,
	K,
	Tm,
	Tn,
	Tk,
	m_idx,
	n_idx,
	k_idx,
	bank_sel_wr_A,
	bank_sel_wr_B,
	bank_sel_rd_A,
	bank_sel_rd_B,
	Sa_bits,
	Sw_bits,
	dma_src_addr,
	dma_dst_addr,
	dma_xfer_len,
	dma_start_pulse,
	act_dma_src_addr,
	act_dma_len,
	act_dma_start_pulse,
	bsr_config,
	bsr_num_blocks,
	bsr_block_rows,
	bsr_block_cols,
	bsr_ptr_addr,
	bsr_idx_addr,
	layer_total,
	layer_current,
	pool_en,
	output_h,
	output_w,
	layer_done_in,
	last_layer_done_in
);
	parameter ADDR_W = 8;
	parameter ENABLE_CLOCK_GATING = 1;
	input wire clk;
	input wire rst_n;
	input wire csr_wen;
	input wire csr_ren;
	input wire [ADDR_W - 1:0] csr_addr;
	input wire [31:0] csr_wdata;
	output reg [31:0] csr_rdata;
	input wire core_busy;
	input wire core_done_tile_pulse;
	input wire core_bank_sel_rd_A;
	input wire core_bank_sel_rd_B;
	input wire rx_illegal_cmd;
	input wire [31:0] perf_total_cycles;
	input wire [31:0] perf_active_cycles;
	input wire [31:0] perf_idle_cycles;
	input wire [31:0] perf_dma_bytes;
	input wire [31:0] perf_blocks_processed;
	input wire [31:0] perf_stall_cycles;
	input wire [127:0] result_data;
	input wire dma_busy_in;
	input wire dma_done_in;
	input wire [31:0] dma_bytes_xferred_in;
	output wire start_pulse;
	output wire abort_pulse;
	output wire irq_en;
	output wire [31:0] M;
	output wire [31:0] N;
	output wire [31:0] K;
	output wire [31:0] Tm;
	output wire [31:0] Tn;
	output wire [31:0] Tk;
	output wire [31:0] m_idx;
	output wire [31:0] n_idx;
	output wire [31:0] k_idx;
	output wire bank_sel_wr_A;
	output wire bank_sel_wr_B;
	output wire bank_sel_rd_A;
	output wire bank_sel_rd_B;
	output wire [31:0] Sa_bits;
	output wire [31:0] Sw_bits;
	output wire [31:0] dma_src_addr;
	output wire [31:0] dma_dst_addr;
	output wire [31:0] dma_xfer_len;
	output wire dma_start_pulse;
	output wire [31:0] act_dma_src_addr;
	output wire [31:0] act_dma_len;
	output wire act_dma_start_pulse;
	output wire [31:0] bsr_config;
	output wire [31:0] bsr_num_blocks;
	output wire [31:0] bsr_block_rows;
	output wire [31:0] bsr_block_cols;
	output wire [31:0] bsr_ptr_addr;
	output wire [31:0] bsr_idx_addr;
	output wire [7:0] layer_total;
	output wire [7:0] layer_current;
	output wire pool_en;
	output wire [15:0] output_h;
	output wire [15:0] output_w;
	input wire layer_done_in;
	input wire last_layer_done_in;
	wire clk_gated = (ENABLE_CLOCK_GATING ? clk : clk);
	localparam CTRL = 8'h00;
	localparam DIMS_M = 8'h04;
	localparam DIMS_N = 8'h08;
	localparam DIMS_K = 8'h0c;
	localparam TILES_Tm = 8'h10;
	localparam TILES_Tn = 8'h14;
	localparam TILES_Tk = 8'h18;
	localparam INDEX_m = 8'h1c;
	localparam INDEX_n = 8'h20;
	localparam INDEX_k = 8'h24;
	localparam BUFF = 8'h28;
	localparam SCALE_Sa = 8'h2c;
	localparam SCALE_Sw = 8'h30;
	localparam STATUS = 8'h3c;
	localparam PERF_TOTAL = 8'h40;
	localparam PERF_ACTIVE = 8'h44;
	localparam PERF_IDLE = 8'h48;
	localparam PERF_DMA_BYTES = 8'h4c;
	localparam PERF_BLOCKS_DONE = 8'h50;
	localparam PERF_STALL_CYCLES = 8'h54;
	localparam RESULT_0 = 8'h80;
	localparam RESULT_1 = 8'h84;
	localparam RESULT_2 = 8'h88;
	localparam RESULT_3 = 8'h8c;
	localparam [7:0] DMA_SRC_ADDR = 8'h90;
	localparam DMA_DST_ADDR = 8'h94;
	localparam DMA_XFER_LEN = 8'h98;
	localparam DMA_CTRL = 8'h9c;
	localparam DMA_BYTES_XFERRED = 8'hb8;
	localparam [7:0] ACT_DMA_SRC_ADDR = 8'ha0;
	localparam [7:0] ACT_DMA_LEN = 8'ha4;
	localparam [7:0] ACT_DMA_CTRL = 8'ha8;
	localparam BSR_CONFIG = 8'hc0;
	localparam BSR_NUM_BLOCKS = 8'hc4;
	localparam BSR_BLOCK_ROWS = 8'hc8;
	localparam BSR_BLOCK_COLS = 8'hcc;
	localparam BSR_STATUS = 8'hd0;
	localparam BSR_ERROR_CODE = 8'hd4;
	localparam BSR_PTR_ADDR = 8'hd8;
	localparam BSR_IDX_ADDR = 8'hdc;
	localparam LAYER_TOTAL = 8'he0;
	localparam LAYER_CURRENT = 8'he4;
	localparam LAYER_CONFIG = 8'he8;
	localparam LAYER_OUT_DIM = 8'hec;
	localparam LAYER_STATUS = 8'hf0;
	reg r_irq_en;
	reg [31:0] r_M;
	reg [31:0] r_N;
	reg [31:0] r_K;
	reg [31:0] r_Tm;
	reg [31:0] r_Tn;
	reg [31:0] r_Tk;
	reg [31:0] r_m_idx;
	reg [31:0] r_n_idx;
	reg [31:0] r_k_idx;
	reg r_bank_sel_wr_A;
	reg r_bank_sel_wr_B;
	reg [31:0] r_Sa_bits;
	reg [31:0] r_Sw_bits;
	reg st_done_tile;
	reg st_err_illegal;
	reg [31:0] r_dma_src_addr;
	reg [31:0] r_dma_dst_addr;
	reg [31:0] r_dma_xfer_len;
	reg st_dma_done;
	reg [31:0] r_bsr_config;
	reg [31:0] r_bsr_num_blocks;
	reg [31:0] r_bsr_block_rows;
	reg [31:0] r_bsr_block_cols;
	reg [31:0] r_bsr_ptr_addr;
	reg [31:0] r_bsr_idx_addr;
	reg st_bsr_done;
	reg st_bsr_error;
	reg [3:0] r_bsr_state;
	reg [31:0] r_act_dma_src_addr;
	reg [31:0] r_act_dma_len;
	reg st_act_dma_done;
	reg [7:0] r_layer_total;
	reg [7:0] r_layer_current;
	reg r_pool_en;
	reg [15:0] r_output_h;
	reg [15:0] r_output_w;
	reg st_layer_done;
	reg st_last_layer_done;
	always @(posedge clk_gated or negedge rst_n)
		if (!rst_n) begin
			r_irq_en <= 1'b0;
			r_M <= 0;
			r_N <= 0;
			r_K <= 0;
			r_Tm <= 0;
			r_Tn <= 0;
			r_Tk <= 0;
			r_m_idx <= 0;
			r_n_idx <= 0;
			r_k_idx <= 0;
			r_bank_sel_wr_A <= 1'b0;
			r_bank_sel_wr_B <= 1'b0;
			r_Sa_bits <= 32'h3f800000;
			r_Sw_bits <= 32'h3f800000;
			st_done_tile <= 1'b0;
			st_err_illegal <= 1'b0;
			r_dma_src_addr <= 32'h00000000;
			r_dma_dst_addr <= 32'h00000000;
			r_dma_xfer_len <= 32'h00000000;
			st_dma_done <= 1'b0;
			r_act_dma_src_addr <= 32'h00000000;
			r_act_dma_len <= 32'h00000000;
			st_act_dma_done <= 1'b0;
			r_layer_total <= 8'd1;
			r_layer_current <= 8'd0;
			r_pool_en <= 1'b0;
			r_output_h <= 16'd14;
			r_output_w <= 16'd14;
			st_layer_done <= 1'b0;
			st_last_layer_done <= 1'b0;
			r_bsr_config <= 32'h00000100;
			r_bsr_num_blocks <= 32'h00000000;
			r_bsr_block_rows <= 32'h00000000;
			r_bsr_block_cols <= 32'h00000000;
			r_bsr_ptr_addr <= 32'h00000000;
			r_bsr_idx_addr <= 32'h00000000;
			st_bsr_done <= 1'b0;
			st_bsr_error <= 1'b0;
			r_bsr_state <= 4'h0;
		end
		else begin
			if (core_done_tile_pulse)
				st_done_tile <= 1'b1;
			if (rx_illegal_cmd)
				st_err_illegal <= 1'b1;
			if (dma_done_in)
				st_dma_done <= 1'b1;
			if (layer_done_in)
				st_layer_done <= 1'b1;
			if (last_layer_done_in)
				st_last_layer_done <= 1'b1;
			if (csr_wen)
				case (csr_addr)
					CTRL: r_irq_en <= csr_wdata[2];
					DIMS_M: r_M <= csr_wdata;
					DIMS_N: r_N <= csr_wdata;
					DIMS_K: r_K <= csr_wdata;
					TILES_Tm: r_Tm <= csr_wdata;
					TILES_Tn: r_Tn <= csr_wdata;
					TILES_Tk: r_Tk <= csr_wdata;
					INDEX_m: r_m_idx <= csr_wdata;
					INDEX_n: r_n_idx <= csr_wdata;
					INDEX_k: r_k_idx <= csr_wdata;
					BUFF: begin
						r_bank_sel_wr_A <= csr_wdata[0];
						r_bank_sel_wr_B <= csr_wdata[1];
					end
					SCALE_Sa: r_Sa_bits <= csr_wdata;
					SCALE_Sw: r_Sw_bits <= csr_wdata;
					STATUS: begin
						if (csr_wdata[1])
							st_done_tile <= 1'b0;
						if (csr_wdata[9])
							st_err_illegal <= 1'b0;
					end
					DMA_SRC_ADDR: r_dma_src_addr <= csr_wdata;
					DMA_DST_ADDR: r_dma_dst_addr <= csr_wdata;
					DMA_XFER_LEN: r_dma_xfer_len <= csr_wdata;
					DMA_CTRL:
						if (csr_wdata[2])
							st_dma_done <= 1'b0;
					ACT_DMA_SRC_ADDR: r_act_dma_src_addr <= csr_wdata;
					ACT_DMA_LEN: r_act_dma_len <= csr_wdata;
					ACT_DMA_CTRL:
						if (csr_wdata[2])
							st_act_dma_done <= 1'b0;
					BSR_CONFIG: r_bsr_config <= csr_wdata;
					BSR_NUM_BLOCKS: r_bsr_num_blocks <= csr_wdata;
					BSR_BLOCK_ROWS: r_bsr_block_rows <= csr_wdata;
					BSR_BLOCK_COLS: r_bsr_block_cols <= csr_wdata;
					BSR_PTR_ADDR: r_bsr_ptr_addr <= csr_wdata;
					BSR_IDX_ADDR: r_bsr_idx_addr <= csr_wdata;
					LAYER_TOTAL: r_layer_total <= csr_wdata[7:0];
					LAYER_CURRENT: r_layer_current <= csr_wdata[7:0];
					LAYER_CONFIG: r_pool_en <= csr_wdata[0];
					LAYER_OUT_DIM: begin
						r_output_h <= csr_wdata[31:16];
						r_output_w <= csr_wdata[15:0];
					end
					LAYER_STATUS: begin
						if (csr_wdata[0])
							st_layer_done <= 1'b0;
						if (csr_wdata[1])
							st_last_layer_done <= 1'b0;
					end
					BSR_STATUS: begin
						if (csr_wdata[2])
							st_bsr_done <= 1'b0;
						if (csr_wdata[3])
							st_bsr_error <= 1'b0;
					end
					default:
						;
				endcase
		end
	wire w_start = (csr_wen && (csr_addr == CTRL)) && csr_wdata[0];
	wire w_abort = (csr_wen && (csr_addr == CTRL)) && csr_wdata[1];
	wire w_dma_start = (csr_wen && (csr_addr == DMA_CTRL)) && csr_wdata[0];
	wire w_act_dma_start = (csr_wen && (csr_addr == ACT_DMA_CTRL)) && csr_wdata[0];
	wire bsr_illegal = (r_bsr_block_rows == 0) || (r_bsr_block_cols == 0);
	wire dense_illegal = ((r_Tm == 0) || (r_Tn == 0)) || (r_Tk == 0);
	wire dims_illegal = bsr_illegal && dense_illegal;
	assign start_pulse = (w_start && !core_busy) && !dims_illegal;
	assign abort_pulse = w_abort;
	always @(posedge clk_gated or negedge rst_n)
		if (!rst_n)
			;
		else if (w_start && (core_busy || dims_illegal))
			st_err_illegal <= 1'b1;
	assign bank_sel_rd_A = core_bank_sel_rd_A;
	assign bank_sel_rd_B = core_bank_sel_rd_B;
	assign irq_en = r_irq_en;
	assign M = r_M;
	assign N = r_N;
	assign K = r_K;
	assign Tm = r_Tm;
	assign Tn = r_Tn;
	assign Tk = r_Tk;
	assign m_idx = r_m_idx;
	assign n_idx = r_n_idx;
	assign k_idx = r_k_idx;
	assign bank_sel_wr_A = r_bank_sel_wr_A;
	assign bank_sel_wr_B = r_bank_sel_wr_B;
	assign Sa_bits = r_Sa_bits;
	assign Sw_bits = r_Sw_bits;
	assign dma_src_addr = r_dma_src_addr;
	assign dma_dst_addr = r_dma_dst_addr;
	assign dma_xfer_len = r_dma_xfer_len;
	assign dma_start_pulse = w_dma_start;
	assign act_dma_src_addr = r_act_dma_src_addr;
	assign act_dma_len = r_act_dma_len;
	assign act_dma_start_pulse = w_act_dma_start;
	assign bsr_config = r_bsr_config;
	assign bsr_num_blocks = r_bsr_num_blocks;
	assign bsr_block_rows = r_bsr_block_rows;
	assign bsr_block_cols = r_bsr_block_cols;
	assign bsr_ptr_addr = r_bsr_ptr_addr;
	assign bsr_idx_addr = r_bsr_idx_addr;
	assign layer_total = r_layer_total;
	assign layer_current = r_layer_current;
	assign pool_en = r_pool_en;
	assign output_h = r_output_h;
	assign output_w = r_output_w;
	always @(*) begin
		csr_rdata = 32'h00000000;
		if (csr_ren)
			(* full_case, parallel_case *)
			case (csr_addr)
				CTRL: csr_rdata = {29'b00000000000000000000000000000, r_irq_en, 2'b00};
				DIMS_M: csr_rdata = r_M;
				DIMS_N: csr_rdata = r_N;
				DIMS_K: csr_rdata = r_K;
				TILES_Tm: csr_rdata = r_Tm;
				TILES_Tn: csr_rdata = r_Tn;
				TILES_Tk: csr_rdata = r_Tk;
				INDEX_m: csr_rdata = r_m_idx;
				INDEX_n: csr_rdata = r_n_idx;
				INDEX_k: csr_rdata = r_k_idx;
				BUFF: csr_rdata = {28'b0000000000000000000000000000, bank_sel_rd_B, bank_sel_rd_A, r_bank_sel_wr_B, r_bank_sel_wr_A};
				SCALE_Sa: csr_rdata = r_Sa_bits;
				SCALE_Sw: csr_rdata = r_Sw_bits;
				STATUS: csr_rdata = {28'b0000000000000000000000000000, st_err_illegal, 1'b0, st_done_tile, core_busy};
				PERF_TOTAL: csr_rdata = perf_total_cycles;
				PERF_ACTIVE: csr_rdata = perf_active_cycles;
				PERF_IDLE: csr_rdata = perf_idle_cycles;
				PERF_DMA_BYTES: csr_rdata = perf_dma_bytes;
				PERF_BLOCKS_DONE: csr_rdata = perf_blocks_processed;
				PERF_STALL_CYCLES: csr_rdata = perf_stall_cycles;
				RESULT_0: csr_rdata = result_data[31:0];
				RESULT_1: csr_rdata = result_data[63:32];
				RESULT_2: csr_rdata = result_data[95:64];
				RESULT_3: csr_rdata = result_data[127:96];
				DMA_SRC_ADDR: csr_rdata = r_dma_src_addr;
				DMA_DST_ADDR: csr_rdata = r_dma_dst_addr;
				DMA_XFER_LEN: csr_rdata = r_dma_xfer_len;
				DMA_CTRL: csr_rdata = {29'b00000000000000000000000000000, st_dma_done, dma_busy_in, 1'b0};
				DMA_BYTES_XFERRED: csr_rdata = dma_bytes_xferred_in;
				ACT_DMA_SRC_ADDR: csr_rdata = r_act_dma_src_addr;
				ACT_DMA_LEN: csr_rdata = r_act_dma_len;
				ACT_DMA_CTRL: csr_rdata = {29'b00000000000000000000000000000, st_act_dma_done, 2'b00};
				BSR_CONFIG: csr_rdata = r_bsr_config;
				BSR_NUM_BLOCKS: csr_rdata = r_bsr_num_blocks;
				BSR_BLOCK_ROWS: csr_rdata = r_bsr_block_rows;
				BSR_BLOCK_COLS: csr_rdata = r_bsr_block_cols;
				BSR_STATUS: csr_rdata = {24'h000000, r_bsr_state, st_bsr_error, st_bsr_done, 2'b01};
				BSR_ERROR_CODE: csr_rdata = 32'd0;
				BSR_PTR_ADDR: csr_rdata = r_bsr_ptr_addr;
				BSR_IDX_ADDR: csr_rdata = r_bsr_idx_addr;
				LAYER_TOTAL: csr_rdata = {24'b000000000000000000000000, r_layer_total};
				LAYER_CURRENT: csr_rdata = {24'b000000000000000000000000, r_layer_current};
				LAYER_CONFIG: csr_rdata = {31'b0000000000000000000000000000000, r_pool_en};
				LAYER_OUT_DIM: csr_rdata = {r_output_h, r_output_w};
				LAYER_STATUS: csr_rdata = {30'b000000000000000000000000000000, st_last_layer_done, st_layer_done};
				default: csr_rdata = 32'hdeadbeef;
			endcase
	end
endmodule
`default_nettype wire
