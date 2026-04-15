module soc_top_v2_asic_sim_wrapper (
	clk,
	rst_n,
	uart_rx,
	uart_tx,
	gpio_o,
	gpio_i,
	gpio_oe,
	irq_external,
	irq_timer,
	accel_busy,
	accel_done,
	dram_ctrl_busy
);
	parameter BOOT_ROM_FILE = "firmware.hex";
	parameter DRAM_INIT_FILE = "";
	parameter signed [31:0] MEM_WORDS = 524288;
	input wire clk;
	input wire rst_n;
	input wire uart_rx;
	output wire uart_tx;
	output wire [7:0] gpio_o;
	input wire [7:0] gpio_i;
	output wire [7:0] gpio_oe;
	output wire irq_external;
	output wire irq_timer;
	output wire accel_busy;
	output wire accel_done;
	output wire dram_ctrl_busy;
	wire [7:0] dram_phy_act;
	wire [7:0] dram_phy_read;
	wire [7:0] dram_phy_write;
	wire [7:0] dram_phy_pre;
	wire [13:0] dram_phy_row;
	wire [9:0] dram_phy_col;
	wire dram_phy_ref;
	wire [31:0] dram_phy_wdata;
	wire [3:0] dram_phy_wstrb;
	wire [31:0] dram_phy_rdata;
	wire dram_phy_rdata_valid;
	soc_top_v2 #(
		.BOOT_ROM_FILE(BOOT_ROM_FILE),
		.CLK_FREQ(50000000),
		.UART_BAUD(115200),
		.MESH_ROWS(4),
		.MESH_COLS(4),
		.SPARSE_VC_ALLOC(1'b0),
		.INNET_REDUCE(1'b0)
	) u_soc(
		.clk(clk),
		.rst_n(rst_n),
		.uart_rx(uart_rx),
		.uart_tx(uart_tx),
		.gpio_o(gpio_o),
		.gpio_i(gpio_i),
		.gpio_oe(gpio_oe),
		.irq_external(irq_external),
		.irq_timer(irq_timer),
		.accel_busy(accel_busy),
		.accel_done(accel_done),
		.dram_phy_act(dram_phy_act),
		.dram_phy_read(dram_phy_read),
		.dram_phy_write(dram_phy_write),
		.dram_phy_pre(dram_phy_pre),
		.dram_phy_row(dram_phy_row),
		.dram_phy_col(dram_phy_col),
		.dram_phy_ref(dram_phy_ref),
		.dram_phy_wdata(dram_phy_wdata),
		.dram_phy_wstrb(dram_phy_wstrb),
		.dram_phy_rdata(dram_phy_rdata),
		.dram_phy_rdata_valid(dram_phy_rdata_valid),
		.dram_ctrl_busy(dram_ctrl_busy)
	);
	dram_phy_simple_mem #(
		.NUM_BANKS(8),
		.ROW_BITS(14),
		.COL_BITS(10),
		.DATA_W(32),
		.MEM_WORDS(MEM_WORDS),
		.INIT_FILE(DRAM_INIT_FILE)
	) u_dram_mem(
		.clk(clk),
		.rst_n(rst_n),
		.dram_phy_act(dram_phy_act),
		.dram_phy_read(dram_phy_read),
		.dram_phy_write(dram_phy_write),
		.dram_phy_pre(dram_phy_pre),
		.dram_phy_row(dram_phy_row),
		.dram_phy_col(dram_phy_col),
		.dram_phy_ref(dram_phy_ref),
		.dram_phy_wdata(dram_phy_wdata),
		.dram_phy_wstrb(dram_phy_wstrb),
		.dram_phy_rdata(dram_phy_rdata),
		.dram_phy_rdata_valid(dram_phy_rdata_valid)
	);
endmodule
