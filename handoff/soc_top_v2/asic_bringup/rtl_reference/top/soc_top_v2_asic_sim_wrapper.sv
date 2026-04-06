// =============================================================================
// soc_top_v2_asic_sim_wrapper.sv — ASIC bringup wrapper with simple DRAM model
// =============================================================================

`timescale 1ns/1ps

module soc_top_v2_asic_sim_wrapper #(
  parameter string BOOT_ROM_FILE = "firmware.hex",
  parameter string DRAM_INIT_FILE = "",
  parameter int MEM_WORDS = 16384
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        uart_rx,
  output logic        uart_tx,
  output logic [7:0]  gpio_o,
  input  logic [7:0]  gpio_i,
  output logic [7:0]  gpio_oe,
  output logic        irq_external,
  output logic        irq_timer,
  output logic        accel_busy,
  output logic        accel_done,
  output logic        dram_ctrl_busy
);

  logic [7:0]  dram_phy_act;
  logic [7:0]  dram_phy_read;
  logic [7:0]  dram_phy_write;
  logic [7:0]  dram_phy_pre;
  logic [13:0] dram_phy_row;
  logic [9:0]  dram_phy_col;
  logic        dram_phy_ref;
  logic [31:0] dram_phy_wdata;
  logic [3:0]  dram_phy_wstrb;
  logic [31:0] dram_phy_rdata;
  logic        dram_phy_rdata_valid;

  soc_top_v2 #(
    .BOOT_ROM_FILE   (BOOT_ROM_FILE),
    .CLK_FREQ        (50_000_000),
    .UART_BAUD       (115_200),
    .MESH_ROWS       (4),
    .MESH_COLS       (4),
    .SPARSE_VC_ALLOC (1'b0),
    .INNET_REDUCE    (1'b0)
  ) u_soc (
    .clk                  (clk),
    .rst_n                (rst_n),
    .uart_rx              (uart_rx),
    .uart_tx              (uart_tx),
    .gpio_o               (gpio_o),
    .gpio_i               (gpio_i),
    .gpio_oe              (gpio_oe),
    .irq_external         (irq_external),
    .irq_timer            (irq_timer),
    .accel_busy           (accel_busy),
    .accel_done           (accel_done),
    .dram_phy_act         (dram_phy_act),
    .dram_phy_read        (dram_phy_read),
    .dram_phy_write       (dram_phy_write),
    .dram_phy_pre         (dram_phy_pre),
    .dram_phy_row         (dram_phy_row),
    .dram_phy_col         (dram_phy_col),
    .dram_phy_ref         (dram_phy_ref),
    .dram_phy_wdata       (dram_phy_wdata),
    .dram_phy_wstrb       (dram_phy_wstrb),
    .dram_phy_rdata       (dram_phy_rdata),
    .dram_phy_rdata_valid (dram_phy_rdata_valid),
    .dram_ctrl_busy       (dram_ctrl_busy)
  );

  dram_phy_simple_mem #(
    .NUM_BANKS (8),
    .ROW_BITS  (14),
    .COL_BITS  (10),
    .DATA_W    (32),
    .MEM_WORDS (MEM_WORDS),
    .INIT_FILE (DRAM_INIT_FILE)
  ) u_dram_mem (
    .clk                  (clk),
    .rst_n                (rst_n),
    .dram_phy_act         (dram_phy_act),
    .dram_phy_read        (dram_phy_read),
    .dram_phy_write       (dram_phy_write),
    .dram_phy_pre         (dram_phy_pre),
    .dram_phy_row         (dram_phy_row),
    .dram_phy_col         (dram_phy_col),
    .dram_phy_ref         (dram_phy_ref),
    .dram_phy_wdata       (dram_phy_wdata),
    .dram_phy_wstrb       (dram_phy_wstrb),
    .dram_phy_rdata       (dram_phy_rdata),
    .dram_phy_rdata_valid (dram_phy_rdata_valid)
  );

endmodule
