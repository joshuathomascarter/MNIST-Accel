`default_nettype none
module bsr_dma (
	clk,
	rst_n,
	start,
	src_addr,
	csr_num_rows,
	csr_total_blocks,
	done,
	busy,
	error,
	m_axi_arid,
	m_axi_araddr,
	m_axi_arlen,
	m_axi_arsize,
	m_axi_arburst,
	m_axi_arvalid,
	m_axi_arready,
	m_axi_rid,
	m_axi_rdata,
	m_axi_rresp,
	m_axi_rlast,
	m_axi_rvalid,
	m_axi_rready,
	row_ptr_we,
	row_ptr_addr,
	row_ptr_wdata,
	col_idx_we,
	col_idx_addr,
	col_idx_wdata,
	wgt_we,
	wgt_addr,
	wgt_wdata
);
	reg _sv2v_0;
	parameter AXI_ADDR_W = 32;
	parameter AXI_DATA_W = 64;
	parameter AXI_ID_W = 4;
	parameter STREAM_ID = 0;
	parameter BRAM_ADDR_W = 10;
	parameter BURST_LEN = 8'd15;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [AXI_ADDR_W - 1:0] src_addr;
	input wire [31:0] csr_num_rows;
	input wire [31:0] csr_total_blocks;
	output reg done;
	output reg busy;
	output reg error;
	output wire [AXI_ID_W - 1:0] m_axi_arid;
	output reg [AXI_ADDR_W - 1:0] m_axi_araddr;
	output reg [7:0] m_axi_arlen;
	output wire [2:0] m_axi_arsize;
	output wire [1:0] m_axi_arburst;
	output reg m_axi_arvalid;
	input wire m_axi_arready;
	input wire [AXI_ID_W - 1:0] m_axi_rid;
	input wire [AXI_DATA_W - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire m_axi_rlast;
	input wire m_axi_rvalid;
	output reg m_axi_rready;
	output reg row_ptr_we;
	output reg [BRAM_ADDR_W - 1:0] row_ptr_addr;
	output reg [31:0] row_ptr_wdata;
	output reg col_idx_we;
	output reg [BRAM_ADDR_W - 1:0] col_idx_addr;
	output reg [15:0] col_idx_wdata;
	output reg wgt_we;
	output reg [BRAM_ADDR_W + 6:0] wgt_addr;
	output reg [63:0] wgt_wdata;
	(* fsm_encoding = "one_hot" *) reg [3:0] state;
	reg [31:0] num_rows;
	reg [31:0] total_blocks;
	reg [AXI_ADDR_W - 1:0] current_axi_addr;
	reg [31:0] words_remaining;
	reg [63:0] rdata_reg;
	reg rlast_reg;
	localparam [2:0] AXI_SIZE_64 = 3'b011;
	localparam [1:0] AXI_BURST_INCR = 2'b01;
	assign m_axi_arid = STREAM_ID[AXI_ID_W - 1:0];
	assign m_axi_arsize = AXI_SIZE_64;
	assign m_axi_arburst = AXI_BURST_INCR;
	wire _unused_bsr_rid = &{1'b0, m_axi_rid};
	wire _unused_rdata_lo = &{1'b0, rdata_reg[15:0]};
	wire [10:0] page_max_beats = 11'd512 - {2'b00, current_axi_addr[11:3]};
	function automatic [7:0] safe_arlen;
		input [9:0] desired;
		if (({1'b0, desired} + 11'd1) > page_max_beats)
			safe_arlen = page_max_beats[7:0] - 8'd1;
		else
			safe_arlen = desired[7:0];
	endfunction
	function automatic [9:0] sv2v_cast_10;
		input reg [9:0] inp;
		sv2v_cast_10 = inp;
	endfunction
	wire [9:0] rowptr_arlen_short = sv2v_cast_10(((words_remaining + 32'd1) >> 1) - 32'd1);
	wire [9:0] colidx_arlen_short = sv2v_cast_10(((words_remaining + 32'd3) >> 2) - 32'd1);
	wire [9:0] wgt_arlen_short = words_remaining[9:0] - 10'd1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 4'd0;
			busy <= 1'b0;
			done <= 1'b0;
			error <= 1'b0;
			m_axi_arvalid <= 1'b0;
			m_axi_rready <= 1'b0;
			row_ptr_we <= 1'b0;
			col_idx_we <= 1'b0;
			wgt_we <= 1'b0;
			row_ptr_addr <= 0;
			col_idx_addr <= 0;
			wgt_addr <= 0;
			current_axi_addr <= 0;
			rdata_reg <= 0;
			rlast_reg <= 0;
		end
		else begin
			row_ptr_we <= 1'b0;
			col_idx_we <= 1'b0;
			wgt_we <= 1'b0;
			m_axi_arvalid <= 1'b0;
			case (state)
				4'd0: begin
					done <= 1'b0;
					if (start) begin
						busy <= 1'b1;
						error <= 1'b0;
						current_axi_addr <= src_addr;
						num_rows <= csr_num_rows;
						total_blocks <= csr_total_blocks;
						state <= 4'd1;
					end
				end
				4'd1: begin
					words_remaining <= num_rows + 1;
					row_ptr_addr <= {BRAM_ADDR_W {1'b1}};
					state <= 4'd2;
				end
				4'd2: begin
					if ((!m_axi_arvalid && !m_axi_rvalid) && (words_remaining > 0)) begin
						m_axi_araddr <= current_axi_addr;
						if (words_remaining > 32)
							m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
						else
							m_axi_arlen <= safe_arlen(rowptr_arlen_short);
						m_axi_arvalid <= 1'b1;
					end
					if (m_axi_arready && m_axi_arvalid)
						m_axi_arvalid <= 1'b0;
					m_axi_rready <= 1'b1;
					if (m_axi_rvalid) begin
						if (m_axi_rresp != 2'b00) begin
							error <= 1'b1;
							busy <= 1'b0;
							done <= 1'b1;
							state <= 4'd0;
						end
						else begin
							rdata_reg <= m_axi_rdata;
							rlast_reg <= m_axi_rlast;
							if (words_remaining > 0) begin
								row_ptr_we <= 1'b1;
								row_ptr_wdata <= m_axi_rdata[31:0];
								row_ptr_addr <= row_ptr_addr + 1;
								words_remaining <= words_remaining - 1;
							end
							if (words_remaining > 1) begin
								m_axi_rready <= 1'b0;
								state <= 4'd3;
							end
							else if (m_axi_rlast) begin
								current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
								if (words_remaining <= 1)
									state <= 4'd4;
							end
						end
					end
				end
				4'd3: begin
					m_axi_rready <= 1'b0;
					row_ptr_we <= 1'b1;
					row_ptr_wdata <= rdata_reg[63:32];
					row_ptr_addr <= row_ptr_addr + 1;
					words_remaining <= words_remaining - 1;
					if (rlast_reg) begin
						current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
						if (words_remaining <= 1)
							state <= 4'd4;
						else
							state <= 4'd2;
					end
					else
						state <= 4'd2;
				end
				4'd4: begin
					words_remaining <= total_blocks;
					col_idx_addr <= {BRAM_ADDR_W {1'b1}};
					state <= 4'd5;
				end
				4'd5: begin
					if ((!m_axi_arvalid && !m_axi_rvalid) && (words_remaining > 0)) begin
						m_axi_araddr <= current_axi_addr;
						if (words_remaining > 64)
							m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
						else
							m_axi_arlen <= safe_arlen(colidx_arlen_short);
						m_axi_arvalid <= 1'b1;
					end
					if (m_axi_arready && m_axi_arvalid)
						m_axi_arvalid <= 1'b0;
					m_axi_rready <= 1'b1;
					if (m_axi_rvalid) begin
						if (m_axi_rresp != 2'b00) begin
							error <= 1'b1;
							busy <= 1'b0;
							done <= 1'b1;
							state <= 4'd0;
						end
						else begin
							rdata_reg <= m_axi_rdata;
							rlast_reg <= m_axi_rlast;
							if (words_remaining > 0) begin
								col_idx_we <= 1'b1;
								col_idx_wdata <= m_axi_rdata[15:0];
								col_idx_addr <= col_idx_addr + 1;
								words_remaining <= words_remaining - 1;
							end
							if (words_remaining > 1) begin
								m_axi_rready <= 1'b0;
								state <= 4'd6;
							end
							else if (m_axi_rlast) begin
								current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
								if (words_remaining <= 1)
									state <= 4'd9;
							end
						end
					end
				end
				4'd6: begin
					m_axi_rready <= 1'b0;
					col_idx_we <= 1'b1;
					col_idx_wdata <= rdata_reg[31:16];
					col_idx_addr <= col_idx_addr + 1;
					words_remaining <= words_remaining - 1;
					if (words_remaining > 1)
						state <= 4'd7;
					else if (rlast_reg) begin
						current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
						state <= 4'd9;
					end
					else
						state <= 4'd5;
				end
				4'd7: begin
					m_axi_rready <= 1'b0;
					col_idx_we <= 1'b1;
					col_idx_wdata <= rdata_reg[47:32];
					col_idx_addr <= col_idx_addr + 1;
					words_remaining <= words_remaining - 1;
					if (words_remaining > 1)
						state <= 4'd8;
					else if (rlast_reg) begin
						current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
						state <= 4'd9;
					end
					else
						state <= 4'd5;
				end
				4'd8: begin
					m_axi_rready <= 1'b0;
					col_idx_we <= 1'b1;
					col_idx_wdata <= rdata_reg[63:48];
					col_idx_addr <= col_idx_addr + 1;
					words_remaining <= words_remaining - 1;
					if (rlast_reg) begin
						current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
						if (words_remaining <= 1)
							state <= 4'd9;
						else
							state <= 4'd5;
					end
					else
						state <= 4'd5;
				end
				4'd9: begin
					words_remaining <= total_blocks << 5;
					wgt_addr <= 0;
					state <= 4'd10;
				end
				4'd10: begin
					m_axi_rready <= 1'b1;
					if ((!m_axi_arvalid && !m_axi_rvalid) && (words_remaining > 0)) begin
						m_axi_araddr <= current_axi_addr;
						if (words_remaining > {24'd0, BURST_LEN})
							m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
						else
							m_axi_arlen <= safe_arlen(wgt_arlen_short);
						m_axi_arvalid <= 1'b1;
					end
					if (m_axi_arready && m_axi_arvalid)
						m_axi_arvalid <= 1'b0;
					if (m_axi_rvalid && m_axi_rready) begin
						if (m_axi_rresp != 2'b00) begin
							error <= 1'b1;
							busy <= 1'b0;
							done <= 1'b1;
							state <= 4'd0;
						end
						else begin
							wgt_we <= 1'b1;
							wgt_wdata <= m_axi_rdata;
							wgt_addr <= wgt_addr + 8;
							words_remaining <= words_remaining - 1;
							if (m_axi_rlast) begin
								current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
								if (words_remaining <= 1)
									state <= 4'd11;
							end
						end
					end
				end
				4'd11: begin
					busy <= 1'b0;
					done <= 1'b1;
					if (!start)
						state <= 4'd0;
				end
				default: state <= 4'd0;
			endcase
		end
	initial _sv2v_0 = 0;
endmodule
`default_nettype wire
