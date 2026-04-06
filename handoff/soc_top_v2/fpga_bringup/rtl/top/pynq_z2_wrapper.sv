// =============================================================================
// pynq_z2_wrapper.sv — PYNQ-Z2 board wrapper for soc_top_v2
// =============================================================================
// Generates a 50 MHz core clock from the 125 MHz PL reference oscillator using
// a Xilinx MMCME2_BASE primitive, maps SoC I/O to board peripherals, and
// stubs out the DRAM PHY interface with a loopback / constant response until
// a real memory controller (MIG or PS DDR bridge) is integrated.
//
// Board mapping (active pin assignments in pynq_z2.xdc):
//   125 MHz reference      → H16 (PL clock)
//   Active-low reset       → BTN0 (active press → reset)
//   UART TX/RX             → PMODA[1:2]  (Y18 / Y19)
//   GPIO_O[3:0]            → LD0-LD3     (R14, P14, N16, M14)
//   GPIO_I[1:0]            → SW0-SW1     (M20, M19)
//   GPIO_I[3:2]            → BTN2-BTN3   (directly)
//   accel_busy / accel_done→ RGB LED 4   (red / green)
//
// Defines:
//   XILINX_FPGA  — set by Vivado project / filelist verilog_define
//
// Author: auto-generated board stub — fill in real pin I/O as needed
// =============================================================================

`timescale 1ns/1ps
/* verilator lint_off PINCONNECTEMPTY */

module pynq_z2_wrapper #(
  parameter string BOOT_ROM_FILE = "firmware.hex",
  parameter string DRAM_INIT_FILE = "",
  parameter int MEM_WORDS = 16384
) (
  // ---- Board clock & reset ------------------------------------------------
  input  logic       clk_125m,      // 125 MHz PL reference (H16)
  input  logic       btn0_n,        // Active-low reset (directly active-press)

  // ---- UART via PMOD-A ----------------------------------------------------
  input  logic       pmoda_rx,      // PMODA pin 2 → SoC UART RX
  output logic       pmoda_tx,      // PMODA pin 1 → SoC UART TX

  // ---- LEDs ---------------------------------------------------------------
  output logic [3:0] led,           // LD0–LD3

  // ---- RGB LED 4 (active-low accent) -------------------------------------
  output logic       led4_r,        // soc accel_busy
  output logic       led4_g,        // soc accel_done

  // ---- Switches -----------------------------------------------------------
  input  logic [1:0] sw,            // SW0–SW1

  // ---- Buttons (active-high on PYNQ-Z2) ----------------------------------
  input  logic       btn2,
  input  logic       btn3
);

  // =========================================================================
  // Clock generation — MMCME2_BASE  (125 MHz → 50 MHz)
  // =========================================================================
  //  VCO = 125 MHz × (MULT / DIV_IN) = 125 × 8 / 1 = 1000 MHz
  //  CLKOUT0 = VCO / DIVOUT0 = 1000 / 20 = 50 MHz
  // =========================================================================
  logic clk_50m;
  logic clk_fb;
  logic mmcm_locked;

`ifdef VERILATOR
  assign clk_50m = clk_125m;
  assign clk_fb = 1'b0;
  assign mmcm_locked = 1'b1;
`else
  MMCME2_BASE #(
    .CLKIN1_PERIOD   (8.000),   // 125 MHz input
    .CLKFBOUT_MULT_F (8.0),    // VCO = 1000 MHz
    .CLKOUT0_DIVIDE_F(20.0),   // 50 MHz output
    .DIVCLK_DIVIDE   (1)
  ) u_mmcm (
    .CLKIN1   (clk_125m),
    .CLKFBIN  (clk_fb),
    .CLKFBOUT (clk_fb),
    .CLKOUT0  (clk_50m),
    .CLKOUT1  (),
    .CLKOUT2  (),
    .CLKOUT3  (),
    .CLKOUT4  (),
    .CLKOUT5  (),
    .CLKOUT6  (),
    .CLKOUT0B (),
    .CLKOUT1B (),
    .CLKOUT2B (),
    .CLKOUT3B (),
    .LOCKED   (mmcm_locked),
    .PWRDWN   (1'b0),
    .RST      (~btn0_n)        // MMCM reset = button press
  );
`endif

  // =========================================================================
  // Reset synchroniser — hold SoC in reset until MMCM is locked
  // =========================================================================
  logic [3:0] rst_shift;
  logic       rst_n;

  always_ff @(posedge clk_50m or negedge btn0_n) begin
    if (!btn0_n)
      rst_shift <= 4'b0;
    else if (!mmcm_locked)
      rst_shift <= 4'b0;
    else
      rst_shift <= {rst_shift[2:0], 1'b1};
  end

  assign rst_n = rst_shift[3];

  // =========================================================================
  // GPIO wiring
  // =========================================================================
  logic [7:0] gpio_o;
  logic [7:0] gpio_oe;  // unused on board — directly driving LEDs
  logic [7:0] gpio_i_pad;

  assign led = gpio_o[3:0];

  assign gpio_i_pad = {2'b0, btn3, btn2, 2'b0, sw};

  // =========================================================================
  // DRAM backing store — persistent word-addressable memory behind dram_ctrl
  // =========================================================================
  logic [7:0]  dram_phy_act;
  logic [7:0]  dram_phy_read;
  logic [7:0]  dram_phy_write;
  logic [7:0]  dram_phy_pre;
  logic [13:0] dram_phy_row;
  logic [9:0]  dram_phy_col;
  logic        dram_phy_ref;
  logic [31:0] dram_phy_wdata;
  logic [3:0]  dram_phy_wstrb;
  logic [31:0] dram_rdata;
  logic        dram_rdata_valid;

  dram_phy_simple_mem #(
    .NUM_BANKS (8),
    .ROW_BITS  (14),
    .COL_BITS  (10),
    .DATA_W    (32),
    .MEM_WORDS (MEM_WORDS),
    .INIT_FILE (DRAM_INIT_FILE)
  ) u_dram_mem (
    .clk                  (clk_50m),
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
    .dram_phy_rdata       (dram_rdata),
    .dram_phy_rdata_valid (dram_rdata_valid)
  );

  // =========================================================================
  // Accelerator status LEDs  (active-LOW on PYNQ-Z2 RGB LEDs)
  // =========================================================================
  logic accel_busy, accel_done;
  assign led4_r = ~accel_busy;   // red = busy
  assign led4_g = ~accel_done;   // green = done

  // =========================================================================
  // SoC instance
  // =========================================================================
  soc_top_v2 #(
    .BOOT_ROM_FILE (BOOT_ROM_FILE),
    .CLK_FREQ      (50_000_000),
    .UART_BAUD     (115_200),
    .MESH_ROWS     (4),
    .MESH_COLS     (4),
    .SPARSE_VC_ALLOC(1'b0),
    .INNET_REDUCE  (1'b0)
  ) u_soc (
    .clk                (clk_50m),
    .rst_n              (rst_n),

    // UART
    .uart_rx            (pmoda_rx),
    .uart_tx            (pmoda_tx),

    // GPIO
    .gpio_o             (gpio_o),
    .gpio_i             (gpio_i_pad),
    .gpio_oe            (gpio_oe),

    // Interrupts (directly active on SoC side; no board routing needed)
    .irq_external       (),
    .irq_timer          (),

    // Accelerator status
    .accel_busy         (accel_busy),
    .accel_done         (accel_done),

    // DRAM PHY — backed by local bringup memory
    .dram_phy_act       (dram_phy_act),
    .dram_phy_read      (dram_phy_read),
    .dram_phy_write     (dram_phy_write),
    .dram_phy_pre       (dram_phy_pre),
    .dram_phy_row       (dram_phy_row),
    .dram_phy_col       (dram_phy_col),
    .dram_phy_ref       (dram_phy_ref),
    .dram_phy_wdata     (dram_phy_wdata),
    .dram_phy_wstrb     (dram_phy_wstrb),
    .dram_phy_rdata     (dram_rdata),
    .dram_phy_rdata_valid(dram_rdata_valid),
    .dram_ctrl_busy     ()
  );

endmodule
