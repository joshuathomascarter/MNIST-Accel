module uart_ctrl (
	clk,
	rst_n,
	rx,
	tx,
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
	rlast,
	irq_o
);
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] CLK_FREQ = 50000000;
	parameter [31:0] DEFAULT_BAUD = 115200;
	input wire clk;
	input wire rst_n;
	input wire rx;
	output reg tx;
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
	output wire irq_o;
	localparam [7:0] TX_DATA = 8'h00;
	localparam [7:0] RX_DATA = 8'h04;
	localparam [7:0] STATUS = 8'h08;
	localparam [7:0] CTRL = 8'h0c;
	localparam [31:0] TXFIFO_FULL = 0;
	localparam [31:0] RXFIFO_EMPTY = 1;
	localparam [31:0] TX_BUSY = 2;
	reg [15:0] baud_divisor;
	function automatic [15:0] sv2v_cast_16;
		input reg [15:0] inp;
		sv2v_cast_16 = inp;
	endfunction
	localparam [15:0] DEFAULT_DIVISOR = sv2v_cast_16(CLK_FREQ / DEFAULT_BAUD);
	reg [7:0] tx_fifo [0:15];
	reg [3:0] tx_wr_ptr;
	reg [3:0] tx_rd_ptr;
	reg [4:0] tx_count;
	wire tx_fifo_full;
	wire tx_fifo_empty;
	reg [7:0] rx_fifo [0:15];
	reg [3:0] rx_wr_ptr;
	reg [3:0] rx_rd_ptr;
	reg [4:0] rx_count;
	wire rx_fifo_full;
	wire rx_fifo_empty;
	reg [9:0] tx_shift;
	reg [3:0] tx_bit_cnt;
	reg tx_busy;
	reg [15:0] tx_baud_cnt;
	reg tx_start;
	reg [9:0] rx_shift;
	reg [3:0] rx_bit_cnt;
	reg [15:0] rx_baud_cnt;
	reg [7:0] rx_data_captured;
	reg rx_valid;
	reg rx_in_sync;
	reg rx_in_sync_r;
	reg [3:0] aw_id;
	reg [3:0] ar_id;
	reg [7:0] ar_addr_r;
	reg b_pending;
	reg ar_valid;
	assign tx_fifo_full = tx_count == 5'd16;
	assign tx_fifo_empty = tx_count == 5'd0;
	assign rx_fifo_full = rx_count == 5'd16;
	assign rx_fifo_empty = rx_count == 5'd0;
	wire tx_consume;
	assign tx_consume = !tx_busy && !tx_fifo_empty;
	wire tx_produce;
	assign tx_produce = (((awvalid && wvalid) && !b_pending) && (awaddr[7:0] == TX_DATA)) && !tx_fifo_full;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			tx_busy <= 1'b0;
			tx_bit_cnt <= '0;
			tx_baud_cnt <= '0;
			tx_shift <= '0;
			tx <= 1'b1;
			tx_start <= 1'b0;
			tx_rd_ptr <= '0;
		end
		else begin
			if (tx_start || (tx_busy && (tx_baud_cnt == 0)))
				tx_baud_cnt <= baud_divisor - 1;
			else if (tx_baud_cnt != 0)
				tx_baud_cnt <= tx_baud_cnt - 1;
			if (tx_consume) begin
				tx_busy <= 1'b1;
				tx_start <= 1'b1;
				tx_shift <= {1'b1, tx_fifo[tx_rd_ptr], 1'b0};
				tx_bit_cnt <= 4'd10;
				tx_rd_ptr <= tx_rd_ptr + 1;
			end
			else
				tx_start <= 1'b0;
			if (tx_busy && (tx_baud_cnt == 0)) begin
				tx <= tx_shift[0];
				tx_shift <= {1'b1, tx_shift[9:1]};
				tx_bit_cnt <= tx_bit_cnt - 1;
				if (tx_bit_cnt == 1)
					tx_busy <= 1'b0;
			end
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			rx_in_sync <= 1'b1;
			rx_in_sync_r <= 1'b1;
		end
		else begin
			rx_in_sync_r <= rx;
			rx_in_sync <= rx_in_sync_r;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			rx_bit_cnt <= '0;
			rx_baud_cnt <= '0;
			rx_shift <= '0;
			rx_data_captured <= '0;
			rx_valid <= 1'b0;
		end
		else begin
			rx_valid <= 1'b0;
			if (((rx_bit_cnt == 0) && (rx_in_sync_r == 1'b0)) && (rx_in_sync == 1'b1)) begin
				rx_bit_cnt <= 4'd8;
				rx_baud_cnt <= (baud_divisor + (baud_divisor >> 1)) - 1;
				rx_shift <= '0;
			end
			else if (rx_bit_cnt != 0) begin
				if (rx_baud_cnt == 0) begin
					rx_baud_cnt <= baud_divisor - 1;
					if (rx_bit_cnt == 1) begin
						rx_data_captured <= {rx_in_sync, rx_shift[9:3]};
						rx_valid <= 1'b1;
						rx_bit_cnt <= '0;
					end
					else begin
						rx_shift <= {rx_in_sync, rx_shift[9:1]};
						rx_bit_cnt <= rx_bit_cnt - 1;
					end
				end
				else
					rx_baud_cnt <= rx_baud_cnt - 1;
			end
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			rx_wr_ptr <= '0;
		else if (rx_valid && !rx_fifo_full) begin
			rx_fifo[rx_wr_ptr] <= rx_data_captured;
			rx_wr_ptr <= rx_wr_ptr + 1;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			rx_count <= '0;
		else
			(* full_case, parallel_case *)
			case ({rx_valid && !rx_fifo_full, ((rvalid && rready) && (ar_addr_r == RX_DATA)) && !rx_fifo_empty})
				2'b10: rx_count <= rx_count + 1;
				2'b01: rx_count <= rx_count - 1;
				2'b11: rx_count <= rx_count;
				default: rx_count <= rx_count;
			endcase
	assign awready = !b_pending;
	assign wready = awvalid && !b_pending;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			baud_divisor <= DEFAULT_DIVISOR;
			b_pending <= 1'b0;
			aw_id <= '0;
			tx_wr_ptr <= '0;
		end
		else if (b_pending && bready)
			b_pending <= 1'b0;
		else if ((awvalid && wvalid) && !b_pending) begin
			b_pending <= 1'b1;
			aw_id <= awid;
			case (awaddr[7:0])
				TX_DATA:
					if (!tx_fifo_full) begin
						tx_fifo[tx_wr_ptr] <= wdata[7:0];
						tx_wr_ptr <= tx_wr_ptr + 1;
					end
				CTRL:
					if (wdata[31:16] != 16'd0)
						baud_divisor <= wdata[31:16];
				default:
					;
			endcase
		end
	assign bvalid = b_pending;
	assign bresp = 2'b00;
	assign bid = aw_id;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			tx_count <= '0;
		else
			(* full_case, parallel_case *)
			case ({tx_produce, tx_consume})
				2'b10: tx_count <= tx_count + 1;
				2'b01: tx_count <= tx_count - 1;
				default: tx_count <= tx_count;
			endcase
	assign arready = 1'b1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			ar_valid <= 1'b0;
			ar_id <= '0;
			ar_addr_r <= '0;
			rdata <= '0;
		end
		else if (arvalid && arready) begin
			ar_valid <= 1'b1;
			ar_id <= arid;
			ar_addr_r <= araddr[7:0];
			case (araddr[7:0])
				TX_DATA: rdata <= {24'b000000000000000000000000, tx_fifo[tx_rd_ptr]};
				RX_DATA: rdata <= {24'b000000000000000000000000, rx_fifo[rx_rd_ptr]};
				STATUS: rdata <= {29'b00000000000000000000000000000, tx_busy, rx_fifo_empty, tx_fifo_full};
				CTRL: rdata <= {16'b0000000000000000, baud_divisor};
				default: rdata <= '0;
			endcase
		end
		else if (rvalid && rready)
			ar_valid <= 1'b0;
	assign rvalid = ar_valid;
	assign rid = ar_id;
	assign rresp = 2'b00;
	assign rlast = 1'b1;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			rx_rd_ptr <= '0;
		else if (((rvalid && rready) && (ar_addr_r == RX_DATA)) && !rx_fifo_empty)
			rx_rd_ptr <= rx_rd_ptr + 1;
	assign irq_o = !rx_fifo_empty;
endmodule
