module obi_to_axi (
	clk,
	rst_n,
	obi_gnt,
	obi_req,
	obi_addr,
	obi_we,
	obi_be,
	obi_wdata,
	obi_burst,
	obi_rvalid,
	obi_rdata,
	obi_err,
	axi_awvalid,
	axi_awready,
	axi_awaddr,
	axi_awid,
	axi_wvalid,
	axi_wready,
	axi_wdata,
	axi_wstrb,
	axi_wlast,
	axi_bvalid,
	axi_bready,
	axi_bresp,
	axi_bid,
	axi_arvalid,
	axi_arready,
	axi_araddr,
	axi_arid,
	axi_rvalid,
	axi_rready,
	axi_rdata,
	axi_rresp,
	axi_rid
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] ID_WIDTH = 4;
	input wire clk;
	input wire rst_n;
	output wire obi_gnt;
	input wire obi_req;
	input wire [ADDR_WIDTH - 1:0] obi_addr;
	input wire obi_we;
	input wire [(DATA_WIDTH / 8) - 1:0] obi_be;
	input wire [DATA_WIDTH - 1:0] obi_wdata;
	input wire [1:0] obi_burst;
	output wire obi_rvalid;
	output wire [DATA_WIDTH - 1:0] obi_rdata;
	output wire obi_err;
	output wire axi_awvalid;
	input wire axi_awready;
	output wire [ADDR_WIDTH - 1:0] axi_awaddr;
	output wire [ID_WIDTH - 1:0] axi_awid;
	output wire axi_wvalid;
	input wire axi_wready;
	output wire [DATA_WIDTH - 1:0] axi_wdata;
	output wire [(DATA_WIDTH / 8) - 1:0] axi_wstrb;
	output wire axi_wlast;
	input wire axi_bvalid;
	output wire axi_bready;
	input wire [1:0] axi_bresp;
	input wire [ID_WIDTH - 1:0] axi_bid;
	output wire axi_arvalid;
	input wire axi_arready;
	output wire [ADDR_WIDTH - 1:0] axi_araddr;
	output wire [ID_WIDTH - 1:0] axi_arid;
	input wire axi_rvalid;
	output wire axi_rready;
	input wire [DATA_WIDTH - 1:0] axi_rdata;
	input wire [1:0] axi_rresp;
	input wire [ID_WIDTH - 1:0] axi_rid;
	reg [2:0] state;
	reg [2:0] next_state;
	reg [ADDR_WIDTH - 1:0] addr_reg;
	reg [DATA_WIDTH - 1:0] wdata_reg;
	reg [(DATA_WIDTH / 8) - 1:0] strb_reg;
	reg we_reg;
	assign obi_gnt = obi_req;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			addr_reg <= '0;
			wdata_reg <= '0;
			strb_reg <= '0;
			we_reg <= 1'b0;
		end
		else begin
			state <= next_state;
			if (obi_req && obi_gnt) begin
				addr_reg <= obi_addr;
				wdata_reg <= obi_wdata;
				strb_reg <= obi_be;
				we_reg <= obi_we;
			end
		end
	always @(*) begin
		if (_sv2v_0)
			;
		next_state = state;
		case (state)
			3'd0:
				if (obi_req) begin
					if (obi_we)
						next_state = 3'd1;
					else
						next_state = 3'd4;
				end
			3'd1:
				if (axi_awready)
					next_state = 3'd2;
			3'd2:
				if (axi_wready)
					next_state = 3'd3;
			3'd3:
				if (axi_bvalid)
					next_state = 3'd0;
			3'd4:
				if (axi_arready)
					next_state = 3'd5;
			3'd5:
				if (axi_rvalid)
					next_state = 3'd0;
			default: next_state = 3'd0;
		endcase
	end
	assign axi_awvalid = state == 3'd1;
	assign axi_awaddr = addr_reg;
	assign axi_awid = '0;
	assign axi_wvalid = state == 3'd2;
	assign axi_wdata = wdata_reg;
	assign axi_wstrb = strb_reg;
	assign axi_wlast = 1'b1;
	assign axi_bready = state == 3'd3;
	assign axi_arvalid = state == 3'd4;
	assign axi_araddr = addr_reg;
	assign axi_arid = '0;
	assign axi_rready = state == 3'd5;
	assign obi_rvalid = axi_rvalid;
	assign obi_rdata = axi_rdata;
	assign obi_err = ((state == 3'd3) && (axi_bresp != 2'b00)) || ((state == 3'd5) && (axi_rresp != 2'b00));
	initial _sv2v_0 = 0;
endmodule
