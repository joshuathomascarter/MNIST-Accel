module axi_addr_decoder (
	addr,
	slave_sel,
	decode_error
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] NUM_SLAVES = 8;
	parameter [31:0] DECODE_BITS = 4;
	parameter [31:0] DECODE_MSB = 31;
	input wire [ADDR_WIDTH - 1:0] addr;
	output reg [NUM_SLAVES - 1:0] slave_sel;
	output reg decode_error;
	localparam [31:0] DECODE_LSB = (DECODE_MSB - DECODE_BITS) + 1;
	wire [DECODE_BITS - 1:0] region;
	assign region = addr[DECODE_MSB:DECODE_LSB];
	wire _unused_addr_lsbs = &{1'b0, addr[DECODE_LSB - 1:0]};
	always @(*) begin
		if (_sv2v_0)
			;
		slave_sel = '0;
		decode_error = 1'b0;
		case (region)
			4'h0: slave_sel[0] = 1'b1;
			4'h1: slave_sel[1] = 1'b1;
			4'h2: slave_sel[2] = 1'b1;
			4'h3: slave_sel[3] = 1'b1;
			4'h4: slave_sel[4] = 1'b1;
			4'h6: slave_sel[4] = 1'b1;
			4'h7: slave_sel[1] = 1'b1;
			default: begin
				slave_sel[NUM_SLAVES - 1] = 1'b1;
				decode_error = 1'b1;
			end
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
