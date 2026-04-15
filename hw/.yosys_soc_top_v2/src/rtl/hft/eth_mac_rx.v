module eth_mac_rx (
	clk,
	rst_n,
	gmii_rxd,
	gmii_rx_dv,
	gmii_rx_er,
	m_axis_tdata,
	m_axis_tvalid,
	m_axis_tlast,
	m_axis_tuser,
	m_axis_tready
);
	reg _sv2v_0;
	parameter signed [31:0] DATA_W = 8;
	input wire clk;
	input wire rst_n;
	input wire [7:0] gmii_rxd;
	input wire gmii_rx_dv;
	input wire gmii_rx_er;
	output reg [DATA_W - 1:0] m_axis_tdata;
	output reg m_axis_tvalid;
	output reg m_axis_tlast;
	output reg m_axis_tuser;
	input wire m_axis_tready;
	function [31:0] crc32_byte;
		input [31:0] crc_in;
		input [7:0] data;
		reg [31:0] c;
		integer i;
		begin
			c = crc_in;
			for (i = 0; i < 8; i = i + 1)
				if (c[0] ^ data[i])
					c = {1'b0, c[31:1]} ^ 32'hedb88320;
				else
					c = {1'b0, c[31:1]};
			crc32_byte = c;
		end
	endfunction
	reg [2:0] state;
	reg [2:0] state_next;
	reg [31:0] crc_reg;
	reg [31:0] crc_next;
	reg [15:0] byte_cnt;
	reg [15:0] byte_cnt_next;
	reg [7:0] rx_data_d;
	reg rx_dv_d;
	localparam [31:0] CRC_RESIDUE = 32'hc704dd7b;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			rx_data_d <= '0;
			rx_dv_d <= 1'b0;
		end
		else begin
			rx_data_d <= gmii_rxd;
			rx_dv_d <= gmii_rx_dv;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			crc_reg <= 32'hffffffff;
			byte_cnt <= '0;
		end
		else begin
			state <= state_next;
			crc_reg <= crc_next;
			byte_cnt <= byte_cnt_next;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		crc_next = crc_reg;
		byte_cnt_next = byte_cnt;
		m_axis_tdata = '0;
		m_axis_tvalid = 1'b0;
		m_axis_tlast = 1'b0;
		m_axis_tuser = 1'b0;
		case (state)
			3'd0: begin
				crc_next = 32'hffffffff;
				byte_cnt_next = '0;
				if (gmii_rx_dv && (gmii_rxd == 8'h55))
					state_next = 3'd1;
			end
			3'd1:
				if (!gmii_rx_dv)
					state_next = 3'd0;
				else if (gmii_rxd == 8'hd5)
					state_next = 3'd2;
				else if (gmii_rxd != 8'h55)
					state_next = 3'd4;
			3'd2:
				if (!rx_dv_d)
					state_next = 3'd3;
				else begin
					m_axis_tdata = rx_data_d;
					m_axis_tvalid = 1'b1;
					crc_next = crc32_byte(crc_reg, rx_data_d);
					byte_cnt_next = byte_cnt + 1;
				end
			3'd3: begin
				m_axis_tvalid = 1'b1;
				m_axis_tlast = 1'b1;
				m_axis_tdata = '0;
				m_axis_tuser = (crc_reg != CRC_RESIDUE ? 1'b1 : 1'b0);
				state_next = 3'd0;
			end
			3'd4:
				if (!gmii_rx_dv)
					state_next = 3'd0;
			default: state_next = 3'd0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
