module noc_network_interface (
	clk,
	rst_n,
	clk_en,
	aw_valid,
	aw_ready,
	aw_addr,
	aw_len,
	aw_id,
	w_valid,
	w_ready,
	w_data,
	w_last,
	b_valid,
	b_ready,
	b_id,
	b_resp,
	ar_valid,
	ar_ready,
	ar_addr,
	ar_id,
	ar_len,
	r_valid,
	r_ready,
	r_data,
	r_id,
	r_resp,
	r_last,
	sparse_hint,
	reduce_inj_valid,
	reduce_inj_ready,
	reduce_inj_id,
	reduce_inj_expect,
	reduce_inj_dst,
	reduce_inj_val,
	noc_flit_out,
	noc_valid_out,
	noc_credit_in,
	noc_flit_in,
	noc_valid_in,
	noc_credit_out
);
	reg _sv2v_0;
	parameter signed [31:0] NODE_ID = 0;
	parameter signed [31:0] ADDR_WIDTH = 32;
	parameter signed [31:0] DATA_WIDTH = 32;
	localparam signed [31:0] noc_pkg_NUM_VCS = 4;
	parameter signed [31:0] NUM_VCS = noc_pkg_NUM_VCS;
	localparam signed [31:0] noc_pkg_MESH_COLS = 4;
	localparam signed [31:0] noc_pkg_MESH_ROWS = 4;
	localparam signed [31:0] noc_pkg_NUM_NODES = noc_pkg_MESH_ROWS * noc_pkg_MESH_COLS;
	parameter signed [31:0] GW_NODE_ID = noc_pkg_NUM_NODES - 1;
	input wire clk;
	input wire rst_n;
	input wire clk_en;
	input wire aw_valid;
	output reg aw_ready;
	input wire [ADDR_WIDTH - 1:0] aw_addr;
	input wire [7:0] aw_len;
	input wire [3:0] aw_id;
	input wire w_valid;
	output reg w_ready;
	input wire [DATA_WIDTH - 1:0] w_data;
	input wire w_last;
	output reg b_valid;
	input wire b_ready;
	output reg [3:0] b_id;
	output reg [1:0] b_resp;
	input wire ar_valid;
	output reg ar_ready;
	input wire [ADDR_WIDTH - 1:0] ar_addr;
	input wire [3:0] ar_id;
	input wire [7:0] ar_len;
	output reg r_valid;
	input wire r_ready;
	output reg [DATA_WIDTH - 1:0] r_data;
	output reg [3:0] r_id;
	output reg [1:0] r_resp;
	output reg r_last;
	input wire sparse_hint;
	input wire reduce_inj_valid;
	output wire reduce_inj_ready;
	input wire [7:0] reduce_inj_id;
	input wire [3:0] reduce_inj_expect;
	input wire [3:0] reduce_inj_dst;
	input wire [31:0] reduce_inj_val;
	output reg [63:0] noc_flit_out;
	output reg noc_valid_out;
	input wire [NUM_VCS - 1:0] noc_credit_in;
	input wire [63:0] noc_flit_in;
	input wire noc_valid_in;
	output reg [NUM_VCS - 1:0] noc_credit_out;
	localparam signed [31:0] noc_pkg_PAYLOAD_HI = 47;
	localparam signed [31:0] noc_pkg_PAYLOAD_LO = 0;
	localparam signed [31:0] PAYLOAD_BITS = (noc_pkg_PAYLOAD_HI - noc_pkg_PAYLOAD_LO) + 1;
	localparam signed [31:0] noc_pkg_NODE_BITS = $clog2(noc_pkg_NUM_NODES);
	function automatic [noc_pkg_NODE_BITS - 1:0] addr_to_node;
		input [ADDR_WIDTH - 1:0] addr;
		if (addr[31:28] >= 4'h4)
			addr_to_node = GW_NODE_ID;
		else
			addr_to_node = addr[31:28];
	endfunction
	reg [2:0] tx_state;
	localparam signed [31:0] noc_pkg_VC_BITS = 2;
	reg [1:0] tx_vc;
	reg [noc_pkg_NODE_BITS - 1:0] tx_dst;
	reg [3:0] tx_msg;
	reg [ADDR_WIDTH - 1:0] tx_addr_reg;
	reg [7:0] tx_len_reg;
	reg [3:0] tx_id_reg;
	localparam signed [31:0] noc_pkg_BUF_DEPTH = 4;
	reg [2:0] tx_credits [0:NUM_VCS - 1];
	wire tx_has_credit;
	function automatic signed [2:0] sv2v_cast_3_signed;
		input reg signed [2:0] inp;
		sv2v_cast_3_signed = inp;
	endfunction
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				tx_credits[v] <= sv2v_cast_3_signed(noc_pkg_BUF_DEPTH);
		end
		else if (clk_en) begin : sv2v_autoblock_2
			reg signed [31:0] v;
			for (v = 0; v < NUM_VCS; v = v + 1)
				begin : sv2v_autoblock_3
					reg inc;
					reg dec;
					inc = noc_credit_in[v];
					dec = noc_valid_out && (tx_vc == v);
					case ({inc, dec})
						2'b10: tx_credits[v] <= tx_credits[v] + 1;
						2'b01: tx_credits[v] <= tx_credits[v] - 1;
						default: tx_credits[v] <= tx_credits[v];
					endcase
				end
		end
	assign tx_has_credit = tx_credits[tx_vc] != '0;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			tx_state <= 3'd0;
			tx_vc <= '0;
			tx_dst <= '0;
			tx_msg <= 4'h0;
			tx_addr_reg <= '0;
			tx_len_reg <= '0;
			tx_id_reg <= '0;
		end
		else if (clk_en)
			case (tx_state)
				3'd0:
					if (reduce_inj_valid) begin
						tx_vc <= '0;
						tx_dst <= reduce_inj_dst;
						tx_state <= 3'd6;
					end
					else if (aw_valid) begin
						tx_dst <= addr_to_node(aw_addr);
						tx_vc <= (sparse_hint ? NUM_VCS - 1 : '0);
						tx_msg <= (sparse_hint ? 4'hb : 4'h1);
						tx_addr_reg <= aw_addr;
						tx_len_reg <= aw_len;
						tx_id_reg <= aw_id;
						tx_state <= 3'd1;
					end
					else if (ar_valid) begin
						tx_dst <= addr_to_node(ar_addr);
						tx_vc <= (sparse_hint ? NUM_VCS - 1 : '0);
						tx_msg <= (sparse_hint ? 4'hb : 4'h2);
						tx_addr_reg <= ar_addr;
						tx_len_reg <= ar_len;
						tx_id_reg <= ar_id;
						tx_state <= 3'd4;
					end
				3'd1:
					if (tx_has_credit)
						tx_state <= 3'd2;
				3'd2:
					if (tx_has_credit && w_valid) begin
						if (w_last)
							tx_state <= 3'd0;
					end
				3'd4:
					if (tx_has_credit)
						tx_state <= 3'd0;
				3'd6:
					if (tx_has_credit)
						tx_state <= 3'd0;
				default: tx_state <= 3'd0;
			endcase
	function automatic [63:0] noc_pkg_make_head_flit;
		input [3:0] src;
		input [3:0] dst;
		input [1:0] vc;
		input [3:0] mtype;
		input [47:0] payload;
		begin
			noc_pkg_make_head_flit[63-:2] = 2'b00;
			noc_pkg_make_head_flit[61-:4] = src;
			noc_pkg_make_head_flit[57-:4] = dst;
			noc_pkg_make_head_flit[53-:2] = vc;
			noc_pkg_make_head_flit[51-:4] = mtype;
			noc_pkg_make_head_flit[47-:48] = payload;
		end
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		noc_flit_out = '0;
		noc_valid_out = 1'b0;
		aw_ready = 1'b0;
		w_ready = 1'b0;
		ar_ready = 1'b0;
		case (tx_state)
			3'd0: begin
				aw_ready = 1'b1;
				ar_ready = !aw_valid && !reduce_inj_valid;
			end
			3'd1:
				if (tx_has_credit) begin
					noc_valid_out = 1'b1;
					noc_flit_out = noc_pkg_make_head_flit(NODE_ID, tx_dst, tx_vc, tx_msg, {tx_addr_reg, tx_len_reg[3:0], tx_id_reg, 8'h00});
				end
			3'd2:
				if (tx_has_credit && w_valid) begin
					noc_valid_out = 1'b1;
					noc_flit_out[61-:4] = NODE_ID;
					noc_flit_out[57-:4] = tx_dst;
					noc_flit_out[53-:2] = tx_vc;
					noc_flit_out[51-:4] = tx_msg;
					w_ready = 1'b1;
					if (w_last)
						noc_flit_out[63-:2] = 2'b10;
					else
						noc_flit_out[63-:2] = 2'b01;
					noc_flit_out[47-:48] = w_data;
				end
			3'd4:
				if (tx_has_credit) begin
					noc_valid_out = 1'b1;
					noc_flit_out = noc_pkg_make_head_flit(NODE_ID, tx_dst, tx_vc, tx_msg, {tx_addr_reg, tx_len_reg[3:0], tx_id_reg, 8'h00});
					noc_flit_out[63-:2] = 2'b11;
					ar_ready = 1'b1;
				end
			3'd6:
				if (tx_has_credit) begin
					noc_valid_out = 1'b1;
					noc_flit_out = noc_pkg_make_head_flit(NODE_ID, tx_dst, tx_vc, 4'h6, {reduce_inj_id, reduce_inj_expect, reduce_inj_val, 4'h0});
					noc_flit_out[63-:2] = 2'b11;
				end
			default:
				;
		endcase
	end
	reg [1:0] rx_state;
	reg [3:0] rx_id;
	reg [7:0] rx_beat_cnt;
	reg [7:0] rx_beat_total;
	reg rx_hold_valid;
	reg [DATA_WIDTH - 1:0] rx_hold_data;
	reg rx_hold_last;
	reg rx_accept_flit;
	always @(*) begin
		if (_sv2v_0)
			;
		rx_accept_flit = 1'b0;
		case (rx_state)
			2'd0:
				if ((noc_valid_in && ((noc_flit_in[63-:2] == 2'b00) || (noc_flit_in[63-:2] == 2'b11))) && ((noc_flit_in[51-:4] == 4'h3) || (noc_flit_in[51-:4] == 4'h4)))
					rx_accept_flit = 1'b1;
			2'd1:
				if (noc_valid_in && (!rx_hold_valid || r_ready))
					rx_accept_flit = 1'b1;
			default:
				;
		endcase
		noc_credit_out = '0;
		if (noc_valid_in && rx_accept_flit)
			noc_credit_out[noc_flit_in[53-:2]] = 1'b1;
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			rx_state <= 2'd0;
			rx_id <= '0;
			rx_beat_cnt <= '0;
			rx_beat_total <= '0;
			rx_hold_valid <= 1'b0;
			rx_hold_data <= '0;
			rx_hold_last <= 1'b0;
		end
		else if (clk_en)
			case (rx_state)
				2'd0:
					if (noc_valid_in && ((noc_flit_in[63-:2] == 2'b00) || (noc_flit_in[63-:2] == 2'b11)))
						case (noc_flit_in[51-:4])
							4'h3: begin
								rx_state <= 2'd1;
								rx_id <= noc_flit_in[47:44];
								rx_beat_cnt <= '0;
								rx_beat_total <= noc_flit_in[43:36];
								rx_hold_valid <= 1'b1;
								rx_hold_data <= noc_flit_in[31:0];
								rx_hold_last <= noc_flit_in[63-:2] == 2'b11;
							end
							4'h4: begin
								rx_state <= 2'd2;
								rx_id <= noc_flit_in[47:44];
							end
							default:
								;
						endcase
				2'd1:
					if (rx_hold_valid) begin
						if (r_ready) begin
							rx_beat_cnt <= rx_beat_cnt + 1'b1;
							if (rx_hold_last) begin
								rx_hold_valid <= 1'b0;
								rx_state <= 2'd0;
							end
							else if (noc_valid_in && rx_accept_flit) begin
								rx_hold_valid <= 1'b1;
								rx_hold_data <= noc_flit_in[31:0];
								rx_hold_last <= (noc_flit_in[63-:2] == 2'b10) || (noc_flit_in[63-:2] == 2'b11);
							end
							else
								rx_hold_valid <= 1'b0;
						end
					end
					else if (noc_valid_in && r_ready) begin
						rx_beat_cnt <= rx_beat_cnt + 1'b1;
						if ((noc_flit_in[63-:2] == 2'b10) || (noc_flit_in[63-:2] == 2'b11))
							rx_state <= 2'd0;
					end
				2'd2:
					if (b_ready)
						rx_state <= 2'd0;
				default: rx_state <= 2'd0;
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		r_valid = 1'b0;
		r_data = '0;
		r_id = rx_id;
		r_resp = 2'b00;
		r_last = 1'b0;
		b_valid = 1'b0;
		b_id = rx_id;
		b_resp = 2'b00;
		case (rx_state)
			2'd1:
				if (rx_hold_valid) begin
					r_valid = 1'b1;
					r_data = rx_hold_data;
					r_last = rx_hold_last;
				end
				else if (noc_valid_in) begin
					r_valid = 1'b1;
					r_data = noc_flit_in[31:0];
					r_last = (noc_flit_in[63-:2] == 2'b10) || (noc_flit_in[63-:2] == 2'b11);
				end
			2'd2: b_valid = 1'b1;
			default:
				;
		endcase
	end
	assign reduce_inj_ready = (tx_state == 3'd6) && tx_has_credit;
	initial _sv2v_0 = 0;
endmodule
