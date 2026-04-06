// =============================================================================
// zcu104_wrapper.sv — ZCU104 board wrapper for soc_top_v2
// =============================================================================
// Target: Xilinx Zynq UltraScale+ xczu7ev-ffvc1156-2-e (ZCU104 Eval Board)
//
// Generates a 50 MHz core clock from the 125 MHz PL differential reference
// oscillator using an IBUFDS + MMCME4_ADV primitive, maps SoC I/O to board
// peripherals, and backs the DRAM PHY interface with a local SRAM for initial
// bringup.  Replace dram_phy_simple_mem with a PS DDR4 AXI bridge for full
// 2 GB memory access during real demo.
//
// Board mapping (active pin assignments in zcu104.xdc):
//   125 MHz PL diff clock   → E12(P)/D12(N)  (DIFF_SSTL12, fixed oscillator)
//   CPU reset button         → M11             (active-high press)
//   UART TX/RX               → PMOD J160 pins  (directly — attach USB-TTL)
//   GPIO_O[3:0]              → LED0–LED3       (D5, D6, A5, B5)
//   GPIO_I[1:0]              → DIP_SW0–1       (A17, A16)
//   GPIO_I[3:2]              → PB_SW2–3        (B3, C3)
//   accel_busy               → LED0 accent     (directly active-high)
//   accel_done               → LED1 accent     (directly active-high)
//
// NOTE: Verify all pin assignments against the ZCU104 schematic (UG1267) and
//       the AMD master XDC before synthesis.  Bank voltages matter — the LEDs
//       and buttons are in 3.3 V HD banks; DIP switches may be in 1.8 V banks.
//
// Author: auto-generated board wrapper for ZCU104
// =============================================================================

`timescale 1ns/1ps
/* verilator lint_off PINCONNECTEMPTY */

module zcu104_wrapper #(
  parameter string BOOT_ROM_FILE = "firmware.hex",
  parameter string DRAM_INIT_FILE = "",
  parameter int MEM_WORDS = 524288       // 2 MB bringup SRAM (512K × 32-bit)
) (
  // ---- Board differential clock (125 MHz fixed oscillator) ----------------
  input  logic       sysclk_125_p,   // E12  (DIFF_SSTL12)
  input  logic       sysclk_125_n,   // D12  (DIFF_SSTL12)

  // ---- CPU reset button (active-high on ZCU104) ---------------------------
  input  logic       cpu_reset,      // M11  (LVCMOS33, active-high press)

  // ---- UART via PMOD J160 ------------------------------------------------
  //  Pin 1 (top row, leftmost) = TX out
  //  Pin 2 (top row, second)   = RX in
  input  logic       pmod_rx,        // PMOD J160 pin 2 — SoC UART RX
  output logic       pmod_tx,        // PMOD J160 pin 1 — SoC UART TX

  // ---- LEDs (active-high on ZCU104) ---------------------------------------
  output logic [3:0] led,            // LED0–LED3  (D5, D6, A5, B5)

  // ---- DIP switches -------------------------------------------------------
  input  logic [3:0] dip_sw,         // DIP_SW0–3  (A17, A16, B16, B15)

  // ---- Push buttons (active-high on ZCU104) -------------------------------
  input  logic       btn_n,          // North  (C4)
  input  logic       btn_s,          // South  (B3)
  input  logic       btn_e,          // East   (C3)
  input  logic       btn_w           // West   (A4)
);

  // =========================================================================
  //  Differential-to-single-ended clock buffer
  // =========================================================================
  logic clk_125m;

`ifdef VERILATOR
  assign clk_125m = sysclk_125_p;
`else
  IBUFDS #(
    .DIFF_TERM ("FALSE")
  ) u_ibufds_clk (
    .I  (sysclk_125_p),
    .IB (sysclk_125_n),
    .O  (clk_125m)
  );
`endif

  // =========================================================================
  //  Clock generation — MMCME4_ADV  (125 MHz → 50 MHz)
  // =========================================================================
  //  VCO = 125 MHz × (MULT / DIV_IN) = 125 × 8 / 1 = 1000 MHz
  //  CLKOUT0 = VCO / DIVOUT0 = 1000 / 20 = 50 MHz
  // =========================================================================
  logic clk_50m;
  logic clk_fb;
  logic mmcm_locked;

`ifdef VERILATOR
  assign clk_50m     = clk_125m;
  assign clk_fb      = 1'b0;
  assign mmcm_locked = 1'b1;
`else
  MMCME4_ADV #(
    .CLKIN1_PERIOD    (8.000),   // 125 MHz input
    .CLKFBOUT_MULT_F  (8.0),    // VCO = 1000 MHz
    .CLKOUT0_DIVIDE_F (20.0),   // 50 MHz output
    .DIVCLK_DIVIDE    (1),
    .STARTUP_WAIT     ("FALSE")
  ) u_mmcm (
    .CLKIN1      (clk_125m),
    .CLKIN2      (1'b0),
    .CLKINSEL    (1'b1),          // Select CLKIN1
    .CLKFBIN     (clk_fb),
    .CLKFBOUT    (clk_fb),
    .CLKOUT0     (clk_50m),
    .CLKOUT1     (),
    .CLKOUT2     (),
    .CLKOUT3     (),
    .CLKOUT4     (),
    .CLKOUT5     (),
    .CLKOUT6     (),
    .CLKOUT0B    (),
    .CLKOUT1B    (),
    .CLKOUT2B    (),
    .CLKOUT3B    (),
    .CLKFBOUTB   (),
    .LOCKED      (mmcm_locked),
    .PWRDWN      (1'b0),
    .RST         (cpu_reset),     // MMCM reset = CPU reset button press
    // Dynamic reconfiguration — unused, tie off
    .DADDR       (7'h0),
    .DCLK        (1'b0),
    .DEN         (1'b0),
    .DI          (16'h0),
    .DO          (),
    .DRDY        (),
    .DWE         (1'b0),
    // Phase shift — unused
    .PSCLK       (1'b0),
    .PSEN        (1'b0),
    .PSINCDEC    (1'b0),
    .PSDONE      (),
    // Clock detect
    .CLKINSTOPPED(),
    .CLKFBSTOPPED(),
    .CDDCREQ     (1'b0),
    .CDDCDONE    ()
  );
`endif

  // =========================================================================
  //  Reset synchroniser — hold SoC in reset until MMCM is locked
  // =========================================================================
  logic [3:0] rst_shift;
  logic       rst_n;

  always_ff @(posedge clk_50m or posedge cpu_reset) begin
    if (cpu_reset)
      rst_shift <= 4'b0;
    else if (!mmcm_locked)
      rst_shift <= 4'b0;
    else
      rst_shift <= {rst_shift[2:0], 1'b1};
  end

  assign rst_n = rst_shift[3];

  // =========================================================================
  //  GPIO wiring
  // =========================================================================
  logic [7:0] gpio_o;
  logic [7:0] gpio_oe;  // unused on board — directly driving LEDs
  logic [7:0] gpio_i_pad;

  assign led = gpio_o[3:0];

  //  gpio_i[1:0] = DIP switches 0–1
  //  gpio_i[3:2] = push buttons South / East
  //  gpio_i[7:4] = DIP switches 2–3, tied low
  assign gpio_i_pad = {2'b0, dip_sw[3:2], btn_e, btn_s, dip_sw[1:0]};

  // =========================================================================
  //  DRAM backing store
  // =========================================================================
  //  For initial bringup: word-addressable SRAM behind dram_ctrl.
  //  MEM_WORDS = 524288 → 2 MB — enough for the full MNIST model (~1.3 MB).
  //
  //  For full demo: replace this block with a Zynq PS DDR4 AXI bridge
  //  (via AXI HP/HPC port in the PS block design).  The SoC's dram_phy_*
  //  signals are converted to AXI transactions by dram_ctrl; wrap those in
  //  an AXI interconnect and route to the PS DDR4 controller for 2 GB access.
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
  //  Accelerator status — directly active-high LEDs on ZCU104
  // =========================================================================
  logic accel_busy, accel_done;
  // Accent: active-high on ZCU104 (unlike PYNQ-Z2 which is active-low)
  // accel_busy/done are also visible in gpio_o if firmware sets GPIO bits.

  // =========================================================================
  //  SoC instance — full 4×4 mesh (16 tiles)
  // =========================================================================
  soc_top_v2 #(
    .BOOT_ROM_FILE  (BOOT_ROM_FILE),
    .CLK_FREQ       (50_000_000),
    .UART_BAUD      (115_200),
    .MESH_ROWS      (4),
    .MESH_COLS      (4),
    .SPARSE_VC_ALLOC(1'b0),
    .INNET_REDUCE   (1'b0)
  ) u_soc (
    .clk                  (clk_50m),
    .rst_n                (rst_n),

    // UART — via PMOD
    .uart_rx              (pmod_rx),
    .uart_tx              (pmod_tx),

    // GPIO
    .gpio_o               (gpio_o),
    .gpio_i               (gpio_i_pad),
    .gpio_oe              (gpio_oe),

    // Interrupts
    .irq_external         (),
    .irq_timer            (),

    // Accelerator status
    .accel_busy           (accel_busy),
    .accel_done           (accel_done),

    // DRAM PHY — backed by local bringup SRAM
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
    .dram_phy_rdata_valid (dram_rdata_valid),
    .dram_ctrl_busy       ()
  );

endmodule
