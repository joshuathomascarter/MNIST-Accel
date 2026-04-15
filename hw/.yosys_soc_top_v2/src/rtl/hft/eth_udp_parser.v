module eth_udp_parser (
	clk,
	rst_n,
	s_axis_tdata,
	s_axis_tvalid,
	s_axis_tlast,
	s_axis_tuser,
	s_axis_tready,
	m_axis_tdata,
	m_axis_tvalid,
	m_axis_tlast,
	m_axis_tready,
	src_ip,
	dst_ip,
	src_port,
	dst_port,
	udp_len,
	hdr_valid,
	frame_error
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [7:0] s_axis_tdata;
	input wire s_axis_tvalid;
	input wire s_axis_tlast;
	input wire s_axis_tuser;
	output reg s_axis_tready;
	output reg [7:0] m_axis_tdata;
	output reg m_axis_tvalid;
	output reg m_axis_tlast;
	input wire m_axis_tready;
	output reg [31:0] src_ip;
	output reg [31:0] dst_ip;
	output reg [15:0] src_port;
	output reg [15:0] dst_port;
	output reg [15:0] udp_len;
	output reg hdr_valid;
	output reg frame_error;
	localparam signed [31:0] ETH_LEN = 14;
	localparam signed [31:0] IP_LEN = 20;
	localparam signed [31:0] UDP_LEN = 8;
	localparam signed [31:0] HDR_TOTAL = (ETH_LEN + IP_LEN) + UDP_LEN;
	reg [2:0] state;
	reg [2:0] state_next;
	reg [15:0] byte_cnt;
	reg [15:0] byte_cnt_next;
	reg [15:0] ethertype_reg;
	reg [15:0] ethertype_next;
	reg [7:0] ip_proto_reg;
	reg [7:0] ip_proto_next;
	reg [31:0] src_ip_reg;
	reg [31:0] src_ip_next;
	reg [31:0] dst_ip_reg;
	reg [31:0] dst_ip_next;
	reg [15:0] src_port_reg;
	reg [15:0] src_port_next;
	reg [15:0] dst_port_reg;
	reg [15:0] dst_port_next;
	reg [15:0] udp_len_reg;
	reg [15:0] udp_len_next;
	reg [15:0] payload_left;
	reg [15:0] payload_left_next;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			byte_cnt <= '0;
			ethertype_reg <= '0;
			ip_proto_reg <= '0;
			src_ip_reg <= '0;
			dst_ip_reg <= '0;
			src_port_reg <= '0;
			dst_port_reg <= '0;
			udp_len_reg <= '0;
			payload_left <= '0;
		end
		else begin
			state <= state_next;
			byte_cnt <= byte_cnt_next;
			ethertype_reg <= ethertype_next;
			ip_proto_reg <= ip_proto_next;
			src_ip_reg <= src_ip_next;
			dst_ip_reg <= dst_ip_next;
			src_port_reg <= src_port_next;
			dst_port_reg <= dst_port_next;
			udp_len_reg <= udp_len_next;
			payload_left <= payload_left_next;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		byte_cnt_next = byte_cnt;
		ethertype_next = ethertype_reg;
		ip_proto_next = ip_proto_reg;
		src_ip_next = src_ip_reg;
		dst_ip_next = dst_ip_reg;
		src_port_next = src_port_reg;
		dst_port_next = dst_port_reg;
		udp_len_next = udp_len_reg;
		payload_left_next = payload_left;
		s_axis_tready = 1'b1;
		m_axis_tdata = '0;
		m_axis_tvalid = 1'b0;
		m_axis_tlast = 1'b0;
		hdr_valid = 1'b0;
		frame_error = 1'b0;
		src_ip = src_ip_reg;
		dst_ip = dst_ip_reg;
		src_port = src_port_reg;
		dst_port = dst_port_reg;
		udp_len = udp_len_reg;
		case (state)
			3'd0: begin
				byte_cnt_next = '0;
				if (s_axis_tvalid && !s_axis_tlast) begin
					state_next = 3'd1;
					byte_cnt_next = 16'd1;
				end
			end
			3'd1:
				if (s_axis_tvalid) begin
					byte_cnt_next = byte_cnt + 1;
					if (byte_cnt == 12)
						ethertype_next[15:8] = s_axis_tdata;
					if (byte_cnt == 13)
						ethertype_next[7:0] = s_axis_tdata;
					if (byte_cnt == 13) begin
						byte_cnt_next = '0;
						if (ethertype_next != 16'h0800)
							state_next = 3'd5;
						else
							state_next = 3'd2;
					end
					if (s_axis_tlast)
						state_next = 3'd0;
				end
			3'd2:
				if (s_axis_tvalid) begin
					byte_cnt_next = byte_cnt + 1;
					if (byte_cnt == 9)
						ip_proto_next = s_axis_tdata;
					case (byte_cnt)
						12: src_ip_next[31:24] = s_axis_tdata;
						13: src_ip_next[23:16] = s_axis_tdata;
						14: src_ip_next[15:8] = s_axis_tdata;
						15: src_ip_next[7:0] = s_axis_tdata;
						default:
							;
					endcase
					case (byte_cnt)
						16: dst_ip_next[31:24] = s_axis_tdata;
						17: dst_ip_next[23:16] = s_axis_tdata;
						18: dst_ip_next[15:8] = s_axis_tdata;
						19: dst_ip_next[7:0] = s_axis_tdata;
						default:
							;
					endcase
					if (byte_cnt == 19) begin
						byte_cnt_next = '0;
						if (ip_proto_next != 8'h11)
							state_next = 3'd5;
						else
							state_next = 3'd3;
					end
					if (s_axis_tlast)
						state_next = 3'd0;
				end
			3'd3:
				if (s_axis_tvalid) begin
					byte_cnt_next = byte_cnt + 1;
					case (byte_cnt)
						0: src_port_next[15:8] = s_axis_tdata;
						1: src_port_next[7:0] = s_axis_tdata;
						2: dst_port_next[15:8] = s_axis_tdata;
						3: dst_port_next[7:0] = s_axis_tdata;
						4: udp_len_next[15:8] = s_axis_tdata;
						5: udp_len_next[7:0] = s_axis_tdata;
						default:
							;
					endcase
					if (byte_cnt == 7) begin
						byte_cnt_next = '0;
						hdr_valid = 1'b1;
						payload_left_next = udp_len_next - 16'd8;
						state_next = 3'd4;
					end
					if (s_axis_tlast)
						state_next = 3'd0;
				end
			3'd4:
				if (s_axis_tvalid) begin
					m_axis_tdata = s_axis_tdata;
					m_axis_tvalid = 1'b1;
					payload_left_next = payload_left - 1;
					if ((payload_left == 1) || s_axis_tlast) begin
						m_axis_tlast = 1'b1;
						if (s_axis_tuser)
							frame_error = 1'b1;
						state_next = 3'd0;
					end
				end
			3'd5: begin
				frame_error = 1'b1;
				if (s_axis_tvalid && s_axis_tlast)
					state_next = 3'd0;
			end
			default: state_next = 3'd0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
