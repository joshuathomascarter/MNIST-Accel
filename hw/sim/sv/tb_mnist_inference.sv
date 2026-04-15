// =============================================================================
// tb_mnist_inference.sv — End-to-End MNIST Inference Testbench
// =============================================================================
// Boots soc_top_v2 with real inference firmware and real DRAM weight/activation
// data, then checks that the SoC correctly classifies a handwritten "7".
//
// Data path exercised (full stack):
//   DRAM (dram_init.hex) → DRAM controller → NoC → tile DMA → scratchpad
//   → systolic 16×16 INT8 MAC → scratchpad → tile DMA → NoC → DRAM
//   → CPU readback → argmax → UART "Predicted: 7" → PASS
//
// Files required (relative to the sim working directory):
//   firmware_inference.hex   — compiled fw/main_inference.c  (FC2-only)
//   dram_init.hex            — data/dram_init.hex (FC2 weights + activations)
//
// EXPECTED RESULT:
//   true label  = 7
//   hw predict  = 7  (golden_reference.json confirms)
//   GPIO[3:0]   = 4'h7, GPIO[7:4] = 4'hF (done flag from firmware)
//   UART line   = "PASS: matches golden"
//
// HOW TO RUN: see run_e2e_inference.sh in the repo root.
// Or manually (from repo root):
//   $ ./run_e2e_inference.sh
// (filelist.f includes dram_phy_simple_mem.sv — no extra -f needed)
// =============================================================================
`timescale 1ns/1ps
/* verilator lint_off PROCASSINIT */
/* verilator lint_off UNUSEDSIGNAL  */
/* verilator lint_off SYNCASYNCNET  */
/* verilator lint_off BLKSEQ        */

module tb_mnist_inference;

  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------
  // Inference takes ~40k cycles per K-tile × 9 K-tiles + readback overhead.
  // 2M cycles gives 4× headroom at 50 MHz (40 ms sim time).
  localparam int    MAX_CYCLES      = 2_000_000;
  localparam string FW_HEX_DEFAULT  = "fw/firmware_inference.hex";
  localparam string DRAM_HEX_DEFAULT = "data/dram_init.hex";
  localparam int    CLK_HALF_NS     = 10;       // 50 MHz

  localparam int    UART_BIT_CYCLES = 50_000_000 / 115_200;  // ~434 cycles

  // Expected outputs from golden_reference.json
  localparam int          EXPECTED_DIGIT    = 7;
  localparam logic [7:0]  EXPECTED_GPIO     = 8'hF7;  // 0xF0 | digit

  // ---------------------------------------------------------------------------
  // Runtime file overrides via +firmware / +dram_init plusargs
  // ---------------------------------------------------------------------------
  string fw_hex;
  string dram_hex;

  initial begin
    if (!$value$plusargs("firmware=%s",  fw_hex))   fw_hex   = FW_HEX_DEFAULT;
    if (!$value$plusargs("dram_init=%s", dram_hex)) dram_hex = DRAM_HEX_DEFAULT;
    $display("[TB] firmware : %s", fw_hex);
    $display("[TB] dram_init: %s", dram_hex);
  end

  // ---------------------------------------------------------------------------
  // Clock & reset
  // ---------------------------------------------------------------------------
  logic clk   = 1'b0;
  logic rst_n = 1'b0;

  always #(CLK_HALF_NS) clk = ~clk;

  // ---------------------------------------------------------------------------
  // DUT I/O
  // ---------------------------------------------------------------------------
  logic        uart_rx   = 1'b1;
  logic        uart_tx;
  logic [7:0]  gpio_o;
  logic [7:0]  gpio_i    = 8'h0;
  logic [7:0]  gpio_oe;
  logic        irq_external = 1'b0;
  logic        irq_timer    = 1'b0;
  logic        accel_busy;
  logic        accel_done;

  // DRAM PHY bus
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
  logic        dram_ctrl_busy;

  // ---------------------------------------------------------------------------
  // DUT: soc_top_v2
  // ---------------------------------------------------------------------------
  soc_top_v2 #(
    .BOOT_ROM_FILE (FW_HEX_DEFAULT),
    .CLK_FREQ      (50_000_000),
    .UART_BAUD     (115_200),
    .MESH_ROWS     (4),
    .MESH_COLS     (4)
  ) dut (
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

  // ---------------------------------------------------------------------------
  // DRAM model: dram_phy_simple_mem loaded with real weights + activations
  // ---------------------------------------------------------------------------
  dram_phy_simple_mem #(
    .NUM_BANKS  (8),
    .ROW_BITS   (14),
    .COL_BITS   (10),
    .DATA_W     (32),
    .MEM_WORDS  (524288),
    .INIT_FILE  (DRAM_HEX_DEFAULT)
  ) u_dram (
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

  // ---------------------------------------------------------------------------
  // Cycle counter & timeout watchdog
  // ---------------------------------------------------------------------------
  int cycle_count = 0;

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 1;
    if (cycle_count >= MAX_CYCLES) begin
      $display("[TB] TIMEOUT: %0d cycles without completion.", cycle_count);
      $display("[TB] RESULT: FAIL (timeout)");
      $finish;
    end
  end

  // ---------------------------------------------------------------------------
  // UART monitor — capture bytes → lines, mirror to $display
  // ---------------------------------------------------------------------------
  logic        uart_capturing  = 1'b0;
  int          uart_bit_cnt    = 0;
  int          uart_sample_pt;
  logic [7:0]  uart_shift      = 8'h0;
  int          uart_rx_cnt     = 0;
  logic        uart_prev_tx    = 1'b1;
  int          uart_chars_rx   = 0;

  string uart_line        = "";
  string uart_predicted   = "";   // captured value after "Predicted: "
  bit    uart_pass_seen   = 0;
  bit    uart_fail_seen   = 0;
  bit    uart_done_seen   = 0;
  string last_uart_line   = "";

  // Cycle count at key events
  int t_inference_start   = 0;
  int t_inference_done    = 0;

  byte unsigned uart_tx_bytes[$];

  initial uart_sample_pt = UART_BIT_CYCLES / 2;

  always_ff @(posedge clk) begin
    uart_prev_tx <= uart_tx;

    // Detect start bit (falling edge)
    if (!uart_capturing && uart_prev_tx && !uart_tx) begin
      uart_capturing <= 1'b1;
      uart_bit_cnt   <= UART_BIT_CYCLES + uart_sample_pt;
      uart_rx_cnt    <= 0;
      uart_shift     <= 8'h0;
    end

    if (uart_capturing) begin
      if (uart_bit_cnt == 0) begin
        uart_bit_cnt <= UART_BIT_CYCLES - 1;
        if (uart_rx_cnt < 8) begin
          uart_shift  <= {uart_tx, uart_shift[7:1]};
          uart_rx_cnt <= uart_rx_cnt + 1;
        end else begin
          // Stop bit — character complete
          uart_capturing <= 1'b0;
          uart_chars_rx  <= uart_chars_rx + 1;
          uart_tx_bytes.push_back(byte'(uart_shift));

          if (uart_shift == 8'h0A) begin
            // Newline — flush line to display and check keywords
            $display("[UART @%0d] %s", cycle_count, uart_line);

            // Check for key result strings
            if (uart_line == "PASS: matches golden") begin
              uart_pass_seen = 1;
              t_inference_done = cycle_count;
            end
            if (uart_line.substr(0, 4) == "FAIL:") begin
              uart_fail_seen = 1;
              t_inference_done = cycle_count;
            end
            if (uart_line == "Done.") begin
              uart_done_seen = 1;
            end
            // Extract predicted digit (line starts with "Predicted: ")
            if (uart_line.substr(0, 10) == "Predicted: ") begin
              uart_predicted = uart_line.substr(11, uart_line.len() - 1);
            end

            last_uart_line = uart_line;
            uart_line = "";
          end else if (uart_shift == 8'h0D || uart_shift == 8'h00) begin
            // ignore CR / NUL
          end else begin
            uart_line = {uart_line,
                         string'(((uart_shift >= 8'h20) && (uart_shift <= 8'h7e))
                                  ? uart_shift : 8'h2e)};
          end
        end
      end else begin
        uart_bit_cnt <= uart_bit_cnt - 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Detect when firmware starts the inference (GPIO[0] = 1 after boot)
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (gpio_o[0] && t_inference_start == 0 && rst_n)
      t_inference_start <= cycle_count;
  end

  // ---------------------------------------------------------------------------
  // Test result tracking
  // ---------------------------------------------------------------------------
  int tests_passed = 0;
  int tests_failed = 0;

  task automatic check(
    input string test_name,
    input logic  condition,
    input string fail_msg = ""
  );
    if (condition) begin
      $display("[PASS] %-50s @cycle %0d", test_name, cycle_count);
      tests_passed++;
    end else begin
      $display("[FAIL] %-50s @cycle %0d  %s", test_name, cycle_count, fail_msg);
      tests_failed++;
    end
  endtask

  // ---------------------------------------------------------------------------
  // Main test sequence
  // ---------------------------------------------------------------------------
  initial begin
    $display("=================================================================");
    $display("  tb_mnist_inference — End-to-End MNIST Inference Testbench");
    $display("  soc_top_v2 + dram_phy_simple_mem (dram_init.hex)");
    $display("  Expected: digit 7 classified correctly");
    $display("=================================================================");

    // ---- Reset ----
    rst_n = 1'b0;
    repeat (40) @(posedge clk);
    rst_n = 1'b1;
    $display("[TB] Reset released at cycle %0d", cycle_count);

    // ---- Wait for firmware boot message ----
    begin
      int boot_wait = 0;
      while (uart_chars_rx == 0 && boot_wait < 300_000) begin
        @(posedge clk);
        boot_wait++;
      end
      check("UART boot message received",
            uart_chars_rx > 0,
            "No UART output after 300k cycles — check boot ROM path");
    end

    // ---- Wait for inference to complete (Done. line or timeout) ----
    // FC2 on 9 K-tiles: each K-tile = ~2 DMA loads + compute + store.
    // At 50 MHz with ~100-cycle DMA latency: ~9 × 400 cycles = 3600 compute
    // cycles + UART overhead. 1.5M cycles is very conservative.
    begin
      int infer_wait = 0;
      while (!uart_done_seen && infer_wait < 1_500_000) begin
        @(posedge clk);
        infer_wait++;
      end
      check("Inference completed (Done. seen on UART)",
            uart_done_seen,
            "Firmware never printed 'Done.' — inference may have stalled");
    end

    // ---- Wait a few more cycles for GPIO to settle ----
    repeat (500) @(posedge clk);

    // ---- Result checks ----
    $display("\n=== INFERENCE RESULTS ===");

    check("UART PASS: matches golden",
          uart_pass_seen,
          uart_fail_seen ? "Firmware printed FAIL — wrong classification" :
                           "Neither PASS nor FAIL seen on UART");

    check("GPIO done flag set (GPIO[7:4]=F)",
          gpio_o[7:4] == 4'hF,
          $sformatf("GPIO=0x%02x — expected upper nibble 0xF", gpio_o));

    check("GPIO digit correct (GPIO[3:0]=7)",
          gpio_o[3:0] == 4'(EXPECTED_DIGIT),
          $sformatf("GPIO[3:0]=0x%x — expected 0x%x", gpio_o[3:0], EXPECTED_DIGIT));

    // ---- Performance report ----
    if (t_inference_start > 0 && t_inference_done > 0) begin
      automatic int infer_cycles = t_inference_done - t_inference_start;
      $display("\n=== PERFORMANCE ===");
      $display("  Inference cycles   : %0d", infer_cycles);
      $display("  @ 50 MHz           : %.2f us", real'(infer_cycles) / 50.0);
      $display("  Throughput         : %.1f inferences/sec",
               50_000_000.0 / real'(infer_cycles));
    end

    // ---- Summary ----
    $display("\n=================================================================");
    $display("  TESTS PASSED: %0d / %0d", tests_passed, tests_passed + tests_failed);
    if (tests_failed == 0)
      $display("  RESULT: PASS — digit 7 correctly classified end-to-end");
    else
      $display("  RESULT: FAIL — %0d check(s) failed", tests_failed);
    $display("=================================================================");

    $finish;
  end

endmodule
