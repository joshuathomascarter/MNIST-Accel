// =============================================================================
// tb_e2e_inference.sv — End-to-End MNIST Inference Testbench
// =============================================================================
// Proves the complete datapath with real weights and real input data:
//
//   DRAM (dram_init.hex, real INT8 weights + activations)
//     → DRAM controller → AXI crossbar → NoC → tile DMA → scratchpad
//     → systolic 16×16 INT8 → scratchpad → tile DMA → NoC
//     → DRAM → CPU readback → UART output
//
// Verification:
//   1. UART must print "Predicted: N"
//   2. UART must print "PASS: matches golden"
//   3. GPIO must output 0xF0 | predicted_digit
//   4. Inference must complete within timeout
//
// Uses dram_phy_simple_mem (not a random stub) so DRAM reads return
// actual weight and activation data from gen_dram_init.py.
//
// HOW TO RUN:
//   $ verilator --sv --binary --timing -f filelist.f \
//       tb_e2e_inference.sv --top-module tb_e2e_inference --Mdir obj_dir_e2e
//   $ ./obj_dir_e2e/Vtb_e2e_inference
// =============================================================================
`timescale 1ns/1ps
/* verilator lint_off PROCASSINIT */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off SYNCASYNCNET */
/* verilator lint_off BLKSEQ */

module tb_e2e_inference #(
  parameter bit INNET_REDUCE = 1'b0
);

  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------
  localparam int    MAX_CYCLES    = 50_000_000;  // 1 s @ 50 MHz (multi-tile needs more)
  localparam string BOOT_ROM_HEX  = "firmware_inference.hex";
  localparam string DRAM_INIT_HEX = "dram_init.hex";
  localparam int    CLK_HALF_NS   = 10;          // 50 MHz clock
  localparam int    FC2_M_ORIG_DBG = 10;
  localparam int    OUTPUT_OFFSET_DBG = 'h00000480;
  localparam int    OUTPUT_TILE_WORDS_DBG = 16 * 16;
  localparam int    OUTPUT_NUM_K_TILES_DBG = 9;

  // ---------------------------------------------------------------------------
  // Clock & Reset
  // ---------------------------------------------------------------------------
  logic clk   = 0;
  logic rst_n = 0;

  always #(CLK_HALF_NS) clk = ~clk;

  // ---------------------------------------------------------------------------
  // DUT I/O
  // ---------------------------------------------------------------------------
  logic        uart_rx   = 1;   // Idle high
  logic        uart_tx;

  logic [7:0]  gpio_o;
  logic [7:0]  gpio_i    = 8'h0;
  logic [7:0]  gpio_oe;

  logic        irq_external;
  logic        irq_timer;

  logic        accel_busy;
  logic        accel_done;

  // DRAM PHY signals
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
    .BOOT_ROM_FILE (BOOT_ROM_HEX),
    .CLK_FREQ      (50_000_000),
    .UART_BAUD     (115_200),
    .MESH_ROWS     (4),
    .MESH_COLS     (4),
    .INNET_REDUCE  (INNET_REDUCE)
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
  // DRAM backing store: real weights + activations from dram_init.hex
  // ---------------------------------------------------------------------------
  dram_phy_simple_mem #(
    .NUM_BANKS (8),
    .ROW_BITS  (14),
    .COL_BITS  (10),
    .DATA_W    (32),
    .MEM_WORDS (524288),
    .INIT_FILE (DRAM_INIT_HEX)
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
  // UART monitor: capture and print characters from firmware
  // ---------------------------------------------------------------------------
  localparam int UART_BIT_CYCLES = 50_000_000 / 115_200;  // ~434 cycles

  logic uart_capturing = 0;
  int   uart_bit_cnt   = 0;
  int   uart_sample_pt = UART_BIT_CYCLES / 2;
  logic [7:0] uart_shift = 8'h0;
  int   uart_rx_cnt = 0;
  logic uart_prev_tx = 1;
  int   uart_chars_rx = 0;
  logic accel_seen_busy = 0;

  string uart_line = "";
  string last_uart_line = "";

  // Track specific UART outputs for verification
  logic saw_predicted     = 0;
  logic saw_pass          = 0;
  logic saw_done          = 0;
  int   predicted_digit   = -1;

  always_ff @(posedge clk) begin
    uart_prev_tx <= uart_tx;

    if (!uart_capturing && uart_prev_tx && !uart_tx) begin
      // Start bit detected (falling edge)
      uart_capturing <= 1;
      // Sample the first data bit 1.5 bit-times after the start edge.
      uart_bit_cnt   <= UART_BIT_CYCLES + uart_sample_pt;
      uart_rx_cnt    <= 0;
      uart_shift     <= 8'h0;
    end

    if (uart_capturing) begin
      if (uart_bit_cnt == 0) begin
        uart_bit_cnt <= UART_BIT_CYCLES - 1;
        if (uart_rx_cnt < 8) begin
          uart_shift   <= {uart_tx, uart_shift[7:1]};  // LSB first
          uart_rx_cnt  <= uart_rx_cnt + 1;
        end else begin
          // Stop bit — character complete
          uart_capturing <= 0;
          uart_chars_rx  <= uart_chars_rx + 1;
          // Print character
          if (uart_shift == 8'h0A || uart_shift == 8'h0D) begin
            if (uart_line.len() != 0) begin
              $display("[UART] %s", uart_line);

              // ---- Parse key output lines ----
              // "Predicted: N"
              if (uart_line.len() >= 12 &&
                  uart_line.substr(0, 9) == "Predicted:") begin
                saw_predicted <= 1;
                // Extract digit: skip "Predicted: " (11 chars)
                if (uart_line.len() > 11) begin
                  string digit_str;
                  digit_str = uart_line.substr(11, uart_line.len()-1);
                  // Simple single-digit parse
                  if (digit_str.len() >= 1) begin
                    predicted_digit <= int'(digit_str.getc(0)) - int'("0");
                  end
                end
              end

              // "PASS: matches golden"
              if (uart_line.len() >= 4 &&
                  uart_line.substr(0, 3) == "PASS") begin
                saw_pass <= 1;
              end

              // "Done."
              if (uart_line.len() >= 4 &&
                  uart_line.substr(0, 3) == "Done") begin
                saw_done <= 1;
              end

              last_uart_line = uart_line;
              uart_line = "";
            end
          end else if (uart_shift == 8'h00) begin
            // Ignore NUL
          end else begin
            uart_line = {uart_line,
                         string'(((uart_shift >= 8'h20) && (uart_shift <= 8'h7e)) ? uart_shift : 8'h2e)};
          end
        end
      end else begin
        uart_bit_cnt <= uart_bit_cnt - 1;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      accel_seen_busy <= 1'b0;
    end else begin
      if (accel_busy)
        accel_seen_busy <= 1'b1;
      else if (accel_done)
        accel_seen_busy <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------
  // Cycle counter & timeout
  // ---------------------------------------------------------------------------
  int cycle_count = 0;

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 1;
    if ($test$plusargs("TB_PROGRESS") && cycle_count != 0 && (cycle_count % 500000) == 0) begin
      $display("[TB-PROGRESS] cycle=%0d accel_busy=%0b accel_done=%0b dram_busy=%0b gpio=%02x uart_chars=%0d last_uart=\"%s\"",
               cycle_count, accel_busy, accel_done, dram_ctrl_busy, gpio_o, uart_chars_rx, last_uart_line);
    end
    if (cycle_count >= MAX_CYCLES) begin
      $display("[TB] TIMEOUT: %0d cycles exceeded without test completion.", MAX_CYCLES);
      $display("[TB] Last UART line: \"%s\"", last_uart_line);
      $finish;
    end
  end

  // ---------------------------------------------------------------------------
  // Performance tracking
  // ---------------------------------------------------------------------------
  int accel_busy_start  = 0;
  int accel_busy_end    = 0;
  int accel_busy_cycles = 0;
  int dram_read_count   = 0;
  int dram_write_count  = 0;

  always_ff @(posedge clk) begin
    if ($rose(accel_busy))
      accel_busy_start <= cycle_count;
    if ($fell(accel_busy)) begin
      accel_busy_end    <= cycle_count;
      accel_busy_cycles <= cycle_count - accel_busy_start;
    end
    if (|dram_phy_read)
      dram_read_count <= dram_read_count + 1;
    if (|dram_phy_write)
      dram_write_count <= dram_write_count + 1;
  end

  // ---------------------------------------------------------------------------
  // Test tracking
  // ---------------------------------------------------------------------------
  int tests_passed = 0;
  int tests_failed = 0;

  task automatic check(
    input string  test_name,
    input logic   condition,
    input string  fail_msg = ""
  );
    if (condition) begin
      $display("[PASS] %-50s  @cycle %0d", test_name, cycle_count);
      tests_passed++;
    end else begin
      $display("[FAIL] %-50s  @cycle %0d  %s", test_name, cycle_count, fail_msg);
      tests_failed++;
    end
  endtask

  task automatic dump_output_debug;
    int dump_tile_idx;
    int logits [0:15];
    int fd;
    string dump_path;
    begin
      for (int cls = 0; cls < 16; cls++)
        logits[cls] = 0;

      // Write dump to file to avoid stdout buffering issues
      if (!$value$plusargs("TB_DUMP_FILE=%s", dump_path))
        dump_path = "/tmp/dram_dump.txt";
      fd = $fopen(dump_path, "w");
      if (fd == 0) begin
        $display("[TB-DUMP] ERROR: Cannot open %s", dump_path);
        return;
      end

      $fwrite(fd, "[TB-DUMP] Reduced logits from DRAM output tiles:\n");
      for (int kt = 0; kt < OUTPUT_NUM_K_TILES_DBG; kt++) begin
        $fwrite(fd, "[TB-DUMP] tile%0d col_sums:", kt);
        for (int cls = 0; cls < 16; cls++) begin
          int tile_sum;
          tile_sum = 0;
          for (int row = 0; row < 16; row++) begin
            tile_sum += $signed(u_dram.mem[OUTPUT_OFFSET_DBG +
                                           kt * OUTPUT_TILE_WORDS_DBG +
                                           row * 16 + cls]);
          end
          logits[cls] += tile_sum;
          $fwrite(fd, " %08x", tile_sum);
        end
        $fwrite(fd, "\n");
      end

      $fwrite(fd, "[TB-DUMP] logits:");
      for (int cls = 0; cls < FC2_M_ORIG_DBG; cls++)
        $fwrite(fd, " %08x", logits[cls]);
      $fwrite(fd, "\n");

      dump_tile_idx = -1;
      if ($value$plusargs("TB_DUMP_TILE=%d", dump_tile_idx) &&
          dump_tile_idx >= 0 && dump_tile_idx < OUTPUT_NUM_K_TILES_DBG) begin
        for (int row = 0; row < 16; row++) begin
          $fwrite(fd, "[TB-DUMP] tile%0d row%0d:", dump_tile_idx, row);
          for (int col = 0; col < 16; col++) begin
            $fwrite(fd, " %08x", u_dram.mem[OUTPUT_OFFSET_DBG +
                                         dump_tile_idx * OUTPUT_TILE_WORDS_DBG +
                                         row * 16 + col]);
          end
          $fwrite(fd, "\n");
        end
      end

      $fclose(fd);
      $display("[TB-DUMP] Dump written to %s", dump_path);
    end
  endtask

  task automatic dump_tile0_scratchpad;
    begin
      $display("[TB-DUMP] Tile0 scratchpad words 0..383:");
      for (int addr = 0; addr < 384; addr++) begin
        if ((addr % 8) == 0)
          $write("[TB-DUMP] sp[%0d]:", addr);
        $write(" %08x", dut.u_tile_array.gen_tile[0].u_tile.u_sp.mem[addr]);
        if ((addr % 8) == 7)
          $write("\n");
      end
    end
  endtask

  // ---------------------------------------------------------------------------
  // Main test sequence
  // ---------------------------------------------------------------------------
  initial begin
    $display("=" * 72);
    $display("  tb_e2e_inference — End-to-End MNIST Classification");
    $display("  soc_top_v2 + dram_phy_simple_mem (real weights)");
    $display("  Expected: firmware matches generated golden reference");
    $display("=" * 72);

    // ------------------------------------------------------------------
    // PHASE 1: Reset
    // ------------------------------------------------------------------
    $display("\n[PHASE 1] Reset ...");
    rst_n = 0;
    repeat (20) @(posedge clk);

    check("accel_busy=0 in reset",   !accel_busy);
    check("accel_done=0 in reset",   !accel_done);

    // ------------------------------------------------------------------
    // PHASE 2: Boot
    // ------------------------------------------------------------------
    $display("\n[PHASE 2] CPU boot ...");
    rst_n = 1;
    repeat (100) @(posedge clk);
    check("CLK running", clk === 1'b1 || clk === 1'b0);

    // ------------------------------------------------------------------
    // PHASE 3: Wait for UART activity (firmware hello message)
    // ------------------------------------------------------------------
    $display("\n[PHASE 3] Waiting for firmware UART output ...");
    begin
      int uart_wait = 0;
      while (uart_chars_rx == 0 && uart_wait < 500_000) begin
        @(posedge clk);
        uart_wait++;
      end
      check("UART TX active within 500k cycles", uart_chars_rx > 0,
            $sformatf("No UART after %0d cycles", uart_wait));
    end

    // ------------------------------------------------------------------
    // PHASE 4: Wait for inference completion
    // ------------------------------------------------------------------
    $display("\n[PHASE 4] Waiting for inference ...");
    begin
      // Wait for firmware to signal done via GPIO (0xFx pattern)
      int infer_wait = 0;
      while (!saw_done && infer_wait < 48_000_000) begin
        @(posedge clk);
        infer_wait++;
      end

      check("Firmware printed 'Done.'", saw_done,
            $sformatf("Did not see 'Done.' after %0d cycles", infer_wait));
    end

    // ------------------------------------------------------------------
    // PHASE 5: Verify classification result
    // ------------------------------------------------------------------
    $display("\n[PHASE 5] Verifying classification ...");
    repeat (100) @(posedge clk);

    check("UART printed 'Predicted: N'", saw_predicted);
        check($sformatf("Predicted digit valid (0-9): %0d", predicted_digit),
          predicted_digit >= 0 && predicted_digit < 10,
          $sformatf("Got %0d", predicted_digit));
    check("Firmware self-check PASS", saw_pass,
          "Did not see 'PASS: matches golden'");

    // ------------------------------------------------------------------
    // PHASE 6: GPIO output check
    // ------------------------------------------------------------------
    $display("\n[PHASE 6] GPIO output ...");
        begin
      int expected_gpio;
      expected_gpio = (predicted_digit & 'h0F) | 8'hF0;
      check($sformatf("GPIO done flag asserted (gpio=0x%02x)", gpio_o),
        gpio_o[7:4] == 4'hF,
        $sformatf("gpio_o=0x%02x", gpio_o));
      check($sformatf("GPIO = 0x%02x (matches predicted digit %0d)", gpio_o, predicted_digit),
        saw_predicted && gpio_o == expected_gpio[7:0],
        $sformatf("gpio_o=0x%02x expected=0x%02x", gpio_o, expected_gpio[7:0]));
        end

    // ------------------------------------------------------------------
    // PHASE 7: Accelerator handshake check
    // ------------------------------------------------------------------
    $display("\n[PHASE 7] Accelerator handshake ...");
    check("accel_busy de-asserted after completion", !accel_busy);

    // ------------------------------------------------------------------
    // RESULTS & PERFORMANCE
    // ------------------------------------------------------------------
    $display("\n" + "=" * 72);
    $display("  RESULTS: %0d passed, %0d failed", tests_passed, tests_failed);
    $display("=" * 72);

    if (tests_failed == 0) begin
      $display("  *** ALL TESTS PASSED — E2E MNIST classification correct ***");
    end else begin
      $display("  *** FAILURES: %0d test(s) failed ***", tests_failed);
    end

    $display("\n  PERFORMANCE:");
    $display("    Total simulation cycles : %0d", cycle_count);
    $display("    Accel busy cycles       : %0d", accel_busy_cycles);
    $display("    DRAM read transactions  : %0d", dram_read_count);
    $display("    DRAM write transactions : %0d", dram_write_count);
    $display("    Simulated wall time     : ~%0.2f ms @ 50 MHz",
             real'(cycle_count) / 50_000.0);
    if (accel_busy_cycles > 0)
      $display("    Inference latency       : ~%0.2f us @ 50 MHz",
               real'(accel_busy_cycles) / 50.0);
    $display("[E2E-METRIC] INNET_REDUCE=%0d total_cycles=%0d accel_cycles=%0d dram_reads=%0d dram_writes=%0d predicted=%0d",
             int'(INNET_REDUCE), cycle_count, accel_busy_cycles,
             dram_read_count, dram_write_count, predicted_digit);
    if ($test$plusargs("TB_DUMP_OUTPUT"))
      dump_output_debug();
    if ($test$plusargs("TB_DUMP_SP"))
      dump_tile0_scratchpad();
    $fflush();
    $display("=" * 72);

    $finish;
  end

  // ---------------------------------------------------------------------------
  // Waveform dump
  // ---------------------------------------------------------------------------
  initial begin
    if ($test$plusargs("waves")) begin
      $dumpfile("tb_e2e_inference.vcd");
      $dumpvars(0, tb_e2e_inference);
    end
  end

  // ---------------------------------------------------------------------------
  // Assertions
  // ---------------------------------------------------------------------------
  property p_dram_no_rw_collision;
    @(posedge clk) disable iff (!rst_n)
    !(|dram_phy_read && |dram_phy_write);
  endproperty

  assert property (p_dram_no_rw_collision)
    else $error("[ASSERT] DRAM read/write collision at cycle %0d", cycle_count);

endmodule
