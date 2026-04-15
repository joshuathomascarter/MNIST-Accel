module tile_dma_gateway (
	clk,
	rst_n,
	noc_flit_in,
	noc_valid_in,
	noc_credit_out,
	noc_flit_out,
	noc_valid_out,
	noc_credit_in,
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
	m_axi_awid,
	m_axi_awaddr,
	m_axi_awlen,
	m_axi_awsize,
	m_axi_awburst,
	m_axi_awvalid,
	m_axi_awready,
	m_axi_wdata,
	m_axi_wstrb,
	m_axi_wlast,
	m_axi_wvalid,
	m_axi_wready,
	m_axi_bid,
	m_axi_bresp,
	m_axi_bvalid,
	m_axi_bready
);
	reg _sv2v_0;
	parameter signed [31:0] AXI_ADDR_W = 32;
	parameter signed [31:0] AXI_DATA_W = 32;
	parameter signed [31:0] AXI_ID_W = 4;
	parameter signed [31:0] MAX_BURST = 16;
	input wire clk;
	input wire rst_n;
	input wire [63:0] noc_flit_in;
	input wire noc_valid_in;
	output wire noc_credit_out;
	output reg [63:0] noc_flit_out;
	output reg noc_valid_out;
	input wire noc_credit_in;
	output wire [AXI_ID_W - 1:0] m_axi_arid;
	output wire [AXI_ADDR_W - 1:0] m_axi_araddr;
	output wire [7:0] m_axi_arlen;
	output wire [2:0] m_axi_arsize;
	output wire [1:0] m_axi_arburst;
	output wire m_axi_arvalid;
	input wire m_axi_arready;
	input wire [AXI_ID_W - 1:0] m_axi_rid;
	input wire [AXI_DATA_W - 1:0] m_axi_rdata;
	input wire [1:0] m_axi_rresp;
	input wire m_axi_rlast;
	input wire m_axi_rvalid;
	output wire m_axi_rready;
	output wire [AXI_ID_W - 1:0] m_axi_awid;
	output wire [AXI_ADDR_W - 1:0] m_axi_awaddr;
	output wire [7:0] m_axi_awlen;
	output wire [2:0] m_axi_awsize;
	output wire [1:0] m_axi_awburst;
	output wire m_axi_awvalid;
	input wire m_axi_awready;
	output wire [AXI_DATA_W - 1:0] m_axi_wdata;
	output wire [(AXI_DATA_W / 8) - 1:0] m_axi_wstrb;
	output wire m_axi_wlast;
	output wire m_axi_wvalid;
	input wire m_axi_wready;
	input wire [AXI_ID_W - 1:0] m_axi_bid;
	input wire [1:0] m_axi_bresp;
	input wire m_axi_bvalid;
	output wire m_axi_bready;
	reg [3:0] state;
	reg [3:0] state_next;
	localparam signed [31:0] BURST_LEN_W = (MAX_BURST <= 1 ? 1 : $clog2(MAX_BURST));
	localparam signed [31:0] BURST_COUNT_W = $clog2(MAX_BURST + 1);
	reg [AXI_ADDR_W - 1:0] req_addr;
	reg [BURST_LEN_W - 1:0] req_burst_len;
	reg [3:0] req_src_tile;
	reg req_is_write;
	reg [3:0] req_dst_node;
	reg [AXI_DATA_W - 1:0] rd_buf [0:MAX_BURST - 1];
	reg [BURST_COUNT_W - 1:0] rd_buf_count;
	reg [BURST_LEN_W - 1:0] rd_resp_idx;
	reg [AXI_DATA_W - 1:0] wr_buf [0:MAX_BURST - 1];
	reg [BURST_COUNT_W - 1:0] wr_buf_count;
	reg [BURST_COUNT_W - 1:0] wr_data_idx;
	localparam signed [31:0] BUF_DEPTH = 4;
	localparam signed [31:0] CREDIT_W = 3;
	reg [2:0] tx_credit_cnt;
	wire tx_flit_sent;
	assign tx_flit_sent = noc_valid_out && (tx_credit_cnt > 0);
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			tx_credit_cnt <= BUF_DEPTH;
		else
			case ({tx_flit_sent, noc_credit_in})
				2'b10: tx_credit_cnt <= tx_credit_cnt - 1;
				2'b01: tx_credit_cnt <= tx_credit_cnt + 1;
				default:
					;
			endcase
	assign noc_credit_out = noc_valid_in && (((state == 4'd0) || (state == 4'd5)) || ((state == 4'd1) && req_is_write));
	wire [1:0] flit_type;
	wire [3:0] flit_msg_type;
	wire [47:0] flit_payload;
	wire [3:0] flit_src;
	wire [3:0] flit_dst;
	assign flit_type = noc_flit_in[63:62];
	assign flit_src = noc_flit_in[61:58];
	assign flit_dst = noc_flit_in[57:54];
	assign flit_msg_type = noc_flit_in[51:48];
	assign flit_payload = noc_flit_in[47:0];
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 4'd0;
		else
			state <= state_next;
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		case (state)
			4'd0:
				if (noc_valid_in)
					state_next = 4'd1;
			4'd1:
				if (req_is_write)
					state_next = 4'd5;
				else
					state_next = 4'd2;
			4'd2:
				if (m_axi_arready)
					state_next = 4'd3;
			4'd3:
				if (m_axi_rvalid && m_axi_rready) begin
					if (rd_buf_count[BURST_LEN_W - 1:0] == req_burst_len)
						state_next = 4'd4;
					else
						state_next = 4'd2;
				end
			4'd4:
				if (tx_flit_sent && ({1'b0, rd_resp_idx} == (rd_buf_count - 1'b1)))
					state_next = 4'd0;
			4'd5:
				if (noc_valid_in && (wr_buf_count == {1'b0, req_burst_len}))
					state_next = 4'd6;
			4'd6:
				if (m_axi_awready)
					state_next = 4'd7;
			4'd7:
				if ((m_axi_wvalid && m_axi_wready) && m_axi_wlast)
					state_next = 4'd8;
			4'd8:
				if (m_axi_bvalid && m_axi_bready) begin
					if (wr_data_idx == ({1'b0, req_burst_len} + 1'b1))
						state_next = 4'd9;
					else
						state_next = 4'd6;
				end
			4'd9:
				if (tx_flit_sent)
					state_next = 4'd0;
			default: state_next = 4'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			req_addr <= '0;
			req_burst_len <= '0;
			req_src_tile <= '0;
			req_is_write <= 1'b0;
			req_dst_node <= '0;
			rd_buf_count <= '0;
			rd_resp_idx <= '0;
			wr_buf_count <= '0;
			wr_data_idx <= '0;
		end
		else
			case (state)
				4'd0:
					if (noc_valid_in && noc_credit_out) begin
						req_addr <= {flit_payload[47:16]};
						req_burst_len <= flit_payload[15:12];
						req_src_tile <= flit_src;
						req_dst_node <= flit_dst;
						req_is_write <= flit_msg_type == 4'h1;
						rd_buf_count <= '0;
						rd_resp_idx <= '0;
						wr_buf_count <= '0;
						wr_data_idx <= '0;
					end
				4'd3:
					if (m_axi_rvalid && m_axi_rready) begin
						rd_buf[rd_buf_count[BURST_LEN_W - 1:0]] <= m_axi_rdata;
						rd_buf_count <= rd_buf_count + 1;
					end
				4'd4:
					if (tx_flit_sent)
						rd_resp_idx <= rd_resp_idx + 1;
				4'd1:
					if ((req_is_write && noc_valid_in) && noc_credit_out) begin
						wr_buf[wr_buf_count[BURST_LEN_W - 1:0]] <= flit_payload[AXI_DATA_W - 1:0];
						wr_buf_count <= wr_buf_count + 1;
					end
				4'd5:
					if (noc_valid_in && noc_credit_out) begin
						wr_buf[wr_buf_count[BURST_LEN_W - 1:0]] <= flit_payload[AXI_DATA_W - 1:0];
						wr_buf_count <= wr_buf_count + 1;
					end
				4'd7:
					if (m_axi_wvalid && m_axi_wready)
						wr_data_idx <= wr_data_idx + 1;
				default:
					;
			endcase
	assign m_axi_arvalid = state == 4'd2;
	assign m_axi_araddr = req_addr + {26'b00000000000000000000000000, rd_buf_count[BURST_LEN_W - 1:0], 2'b00};
	assign m_axi_arid = {req_src_tile};
	assign m_axi_arlen = 8'd0;
	assign m_axi_arsize = 3'b010;
	assign m_axi_arburst = 2'b01;
	assign m_axi_rready = state == 4'd3;
	assign m_axi_awvalid = state == 4'd6;
	assign m_axi_awaddr = req_addr + {26'b00000000000000000000000000, wr_data_idx[BURST_LEN_W - 1:0], 2'b00};
	assign m_axi_awid = {req_src_tile};
	assign m_axi_awlen = 8'd0;
	assign m_axi_awsize = 3'b010;
	assign m_axi_awburst = 2'b01;
	assign m_axi_wvalid = state == 4'd7;
	assign m_axi_wdata = wr_buf[wr_data_idx[BURST_LEN_W - 1:0]];
	assign m_axi_wstrb = '1;
	assign m_axi_wlast = 1'b1;
	assign m_axi_bready = state == 4'd8;
	function automatic [7:0] sv2v_cast_8;
		input reg [7:0] inp;
		sv2v_cast_8 = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		noc_valid_out = 1'b0;
		noc_flit_out = '0;
		if ((state == 4'd4) && (tx_credit_cnt > 0)) begin
			noc_valid_out = 1'b1;
			noc_flit_out[61-:4] = req_dst_node;
			noc_flit_out[57-:4] = req_src_tile;
			noc_flit_out[53-:2] = '0;
			noc_flit_out[51-:4] = 4'h3;
			if (rd_buf_count == 1) begin
				noc_flit_out[63-:2] = 2'b11;
				noc_flit_out[47-:48] = {req_src_tile, sv2v_cast_8(rd_buf_count), 4'h0, rd_buf[0]};
			end
			else if (rd_resp_idx == '0) begin
				noc_flit_out[63-:2] = 2'b00;
				noc_flit_out[47-:48] = {req_src_tile, sv2v_cast_8(rd_buf_count), 4'h0, rd_buf[0]};
			end
			else if ({1'b0, rd_resp_idx} == (rd_buf_count - 1'b1)) begin
				noc_flit_out[63-:2] = 2'b10;
				noc_flit_out[47-:48] = {16'h0000, rd_buf[rd_resp_idx]};
			end
			else begin
				noc_flit_out[63-:2] = 2'b01;
				noc_flit_out[47-:48] = {16'h0000, rd_buf[rd_resp_idx]};
			end
		end
		else if ((state == 4'd9) && (tx_credit_cnt > 0)) begin
			noc_valid_out = 1'b1;
			noc_flit_out[63-:2] = 2'b11;
			noc_flit_out[61-:4] = req_dst_node;
			noc_flit_out[57-:4] = req_src_tile;
			noc_flit_out[53-:2] = '0;
			noc_flit_out[51-:4] = 4'h4;
			noc_flit_out[47-:48] = {req_src_tile, 44'h00000000000};
		end
	end
	initial _sv2v_0 = 0;
endmodule
