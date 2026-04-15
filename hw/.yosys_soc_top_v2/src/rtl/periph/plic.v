module plic (
	clk,
	rst_n,
	irq_i,
	irq_o,
	awvalid,
	awready,
	awaddr,
	awsize,
	awburst,
	awid,
	wvalid,
	wready,
	wdata,
	wstrb,
	wlast,
	bvalid,
	bready,
	bresp,
	bid,
	arvalid,
	arready,
	araddr,
	arsize,
	arburst,
	arid,
	rvalid,
	rready,
	rdata,
	rresp,
	rid,
	rlast
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] NUM_SOURCES = 32;
	parameter [31:0] NUM_TARGETS = 1;
	input wire clk;
	input wire rst_n;
	input wire [NUM_SOURCES - 1:0] irq_i;
	output wire [NUM_TARGETS - 1:0] irq_o;
	input wire awvalid;
	output wire awready;
	input wire [ADDR_WIDTH - 1:0] awaddr;
	input wire [2:0] awsize;
	input wire [1:0] awburst;
	input wire [3:0] awid;
	input wire wvalid;
	output wire wready;
	input wire [DATA_WIDTH - 1:0] wdata;
	input wire [(DATA_WIDTH / 8) - 1:0] wstrb;
	input wire wlast;
	output wire bvalid;
	input wire bready;
	output wire [1:0] bresp;
	output wire [3:0] bid;
	input wire arvalid;
	output wire arready;
	input wire [ADDR_WIDTH - 1:0] araddr;
	input wire [2:0] arsize;
	input wire [1:0] arburst;
	input wire [3:0] arid;
	output wire rvalid;
	input wire rready;
	output reg [DATA_WIDTH - 1:0] rdata;
	output wire [1:0] rresp;
	output wire [3:0] rid;
	output wire rlast;
	reg [2:0] priorities [0:NUM_SOURCES - 1];
	reg [NUM_SOURCES - 1:0] pending;
	reg [NUM_SOURCES - 1:0] enable;
	reg [2:0] threshold [0:NUM_TARGETS - 1];
	reg [NUM_SOURCES - 1:0] claimed;
	reg [NUM_SOURCES - 1:0] claim_set;
	reg [NUM_SOURCES - 1:0] claim_clr;
	reg [3:0] aw_id;
	reg [3:0] ar_id;
	reg [ADDR_WIDTH - 1:0] ar_addr_r;
	reg b_pending;
	reg ar_valid;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			pending <= '0;
			claimed <= '0;
		end
		else begin
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < NUM_SOURCES; i = i + 1)
					if (irq_i[i] && enable[i])
						pending[i] <= 1'b1;
					else if (claimed[i])
						pending[i] <= 1'b0;
			end
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < NUM_SOURCES; i = i + 1)
					if (claim_clr[i])
						claimed[i] <= 1'b0;
					else if (claim_set[i])
						claimed[i] <= 1'b1;
			end
		end
	wire [NUM_SOURCES - 1:0] eligible;
	reg [NUM_SOURCES - 1:0] highest_priority_interrupt;
	reg [2:0] highest_priority;
	reg has_interrupt;
	always @(*) begin
		if (_sv2v_0)
			;
		highest_priority = 3'b000;
		highest_priority_interrupt = '0;
		has_interrupt = 1'b0;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = NUM_SOURCES - 1; i >= 0; i = i - 1)
				if (pending[i] && (priorities[i] > threshold[0])) begin
					if (priorities[i] > highest_priority) begin
						highest_priority = priorities[i];
						highest_priority_interrupt = 1 << i;
						has_interrupt = 1'b1;
					end
				end
		end
	end
	assign irq_o[0] = has_interrupt;
	assign awready = !b_pending;
	assign wready = awvalid && !b_pending;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			b_pending <= 1'b0;
			aw_id <= '0;
		end
		else if (b_pending && bready)
			b_pending <= 1'b0;
		else if ((awvalid && wvalid) && !b_pending) begin
			b_pending <= 1'b1;
			aw_id <= awid;
		end
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	always @(posedge clk)
		if ((awvalid && wvalid) && !b_pending) begin
			if ((awaddr[31:7] == '0) && (sv2v_cast_32_signed(awaddr[6:2]) < NUM_SOURCES)) begin
				if (wstrb[0])
					priorities[awaddr[6:2]][2:0] <= wdata[2:0];
			end
			if (awaddr[31:6] == 26'h0000080) begin
				if (awaddr[5:2] == 0) begin
					if (wstrb[0])
						enable[7:0] <= wdata[7:0];
					if (wstrb[1])
						enable[15:8] <= wdata[15:8];
					if (wstrb[2])
						enable[23:16] <= wdata[23:16];
					if (wstrb[3])
						enable[31:24] <= wdata[31:24];
				end
			end
			if (awaddr[31:2] == 30'h00080000) begin
				if (wstrb[0])
					threshold[0][2:0] <= wdata[2:0];
			end
			if (awaddr[31:2] == 30'h00080001) begin
				if (wstrb[0])
					;
			end
		end
	always @(*) begin
		if (_sv2v_0)
			;
		claim_clr = '0;
		if ((((awvalid && wvalid) && !b_pending) && (awaddr[31:2] == 30'h00080001)) && wstrb[0]) begin : sv2v_autoblock_4
			reg signed [31:0] i;
			for (i = 0; i < NUM_SOURCES; i = i + 1)
				if (i == sv2v_cast_32_signed(wdata[4:0]))
					claim_clr[i] = 1'b1;
		end
	end
	assign bvalid = b_pending;
	assign bresp = 2'b00;
	assign bid = aw_id;
	assign arready = 1'b1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			ar_valid <= 1'b0;
			ar_id <= '0;
			ar_addr_r <= '0;
		end
		else if (arvalid && arready) begin
			ar_valid <= 1'b1;
			ar_id <= arid;
			ar_addr_r <= araddr;
		end
		else if (rvalid && rready)
			ar_valid <= 1'b0;
	assign rvalid = ar_valid;
	assign rid = ar_id;
	assign rresp = 2'b00;
	assign rlast = 1'b1;
	always @(*) begin
		if (_sv2v_0)
			;
		claim_set = '0;
		case (ar_addr_r)
			32'h00000000: rdata = {29'b00000000000000000000000000000, priorities[0]};
			32'h00000004: rdata = {29'b00000000000000000000000000000, priorities[1]};
			32'h00000008: rdata = {29'b00000000000000000000000000000, priorities[2]};
			32'h0000000c: rdata = {29'b00000000000000000000000000000, priorities[3]};
			32'h00001000: rdata = pending[31:0];
			32'h00002000: rdata = enable[31:0];
			32'h00200000: rdata = {29'b00000000000000000000000000000, threshold[0]};
			32'h00200004: begin
				rdata = '0;
				begin : sv2v_autoblock_5
					reg signed [31:0] i;
					for (i = NUM_SOURCES - 1; i >= 0; i = i - 1)
						if (highest_priority_interrupt[i]) begin
							rdata = i;
							claim_set[i] = 1'b1;
						end
				end
			end
			default: rdata = '0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
