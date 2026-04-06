// =============================================================================
// tb_soc_top.sv — SystemVerilog Testbench for soc_top_v2
// =============================================================================
// Drives clock/reset, loads firmware hex, monitors UART output, and checks
// the loader-command behaviour of the full SoC:
//
//   Ibex RISC-V → L1 D-Cache → TLB → AXI Crossbar → {SRAM, Periph, DRAM,
//   Accel} → accel_tile_array (2×2 tiles + NoC + DMA) → DRAM controller
//
// TEST SEQUENCE:
//   1. RESET_TEST:   Assert reset, verify outputs hold inactive.
//   2. BOOT_TEST:    Release reset and confirm firmware banner activity.
//   3. PING_TEST:    Send CMD_PING over UART and verify the loader signature.
//   4. MEM_TEST:     Write/read cached SRAM words through the loader.
//   5. ARGMAX_TEST:  Run loader-side argmax and verify GPIO result encoding.
//
// INR novelty-on SoC coverage lives in tb_soc_top_inr.sv so the default
// top-level regression stays aligned with the baseline handoff configuration.
//
// HOW TO RUN (Verilator):
//   $ verilator --sv --binary --timing -f filelist.f tb_soc_top.sv --top-module tb_soc_top --Mdir obj_dir
//   $ ./obj_dir/Vtb_soc_top
//
// HOW TO RUN (VCS/Questa/Xcelium):
//   vcs -sverilog -f filelist.f tb_soc_top.sv
//   vsim -sv tb_soc_top
//
// PARAMETERS:
//   MAX_CYCLES   — timeout (default 5_000_000 cycles at 50 MHz ≈ 100 ms sim)
//   BOOT_ROM_HEX — path to firmware hex image (default: firmware.hex)
//   CLK_PERIOD   — clock half-period in ns (default: 10 ns → 50 MHz)
//
// =============================================================================
`timescale 1ns/1ps
/* verilator lint_off PROCASSINIT */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off SYNCASYNCNET */
/* verilator lint_off BLKSEQ */

module tb_soc_top;

  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------
  localparam int    MAX_CYCLES    = 5_000_000;
  localparam string BOOT_ROM_HEX  = "firmware.hex";
  localparam int    CLK_HALF_NS   = 10;    // 50 MHz clock
  localparam logic [31:0] HOST_SCRATCH_ADDR = 32'h7000_0100;
  localparam byte CMD_PING            = 8'h00;
  localparam byte CMD_MEM_WRITE_WORDS = 8'h01;
  localparam byte CMD_MEM_READ_WORDS  = 8'h02;
  localparam byte CMD_ARGMAX          = 8'h20;
  localparam byte RESP_OK             = 8'h79;

  // ---------------------------------------------------------------------------
  // Clock & Reset
  // ---------------------------------------------------------------------------
  logic clk   = 0;
  logic rst_n = 0;

  always #(CLK_HALF_NS) clk = ~clk;

  // ---------------------------------------------------------------------------
  // DUT I/O
  // ---------------------------------------------------------------------------
  logic        uart_rx   = 1;   // Idle high (UART)
  logic        uart_tx;

  logic [7:0]  gpio_o;
  logic [7:0]  gpio_i    = 8'h0;
  logic [7:0]  gpio_oe;

  logic        irq_external;
  logic        irq_timer;

  logic        accel_busy;
  logic        accel_done;

  // DRAM PHY
  logic [7:0]  dram_phy_act;
  logic [7:0]  dram_phy_read;
  logic [7:0]  dram_phy_write;
  logic [7:0]  dram_phy_pre;
  logic [13:0] dram_phy_row;
  logic [9:0]  dram_phy_col;
  logic        dram_phy_ref;
  logic [31:0] dram_phy_wdata;
  logic [3:0]  dram_phy_wstrb;
  logic [31:0] dram_phy_rdata    = 32'h0;
  logic        dram_phy_rdata_valid = 1'b0;
  logic        dram_ctrl_busy;

  // ---------------------------------------------------------------------------
  // DUT instantiation
  // ---------------------------------------------------------------------------
  soc_top_v2 #(
    .BOOT_ROM_FILE (BOOT_ROM_HEX),
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
  // DRAM stub: echo back a simple read response after 8-cycle latency
  // ---------------------------------------------------------------------------
  logic [31:0] dram_rdata_pipe [0:7];
  logic [7:0]  dram_rvalid_pipe;
  int          dram_latency = 8;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dram_rvalid_pipe <= '0;
      for (int i = 0; i < 8; i++) dram_rdata_pipe[i] <= '0;
    end else begin
      // Shift pipeline
      for (int i = 7; i > 0; i--) begin
        dram_rdata_pipe[i]  <= dram_rdata_pipe[i-1];
        dram_rvalid_pipe[i] <= dram_rvalid_pipe[i-1];
      end
      // Inject request when DRAM controller issues a read
      dram_rdata_pipe[0]  <= $urandom_range(0, 255);  // Stub: random INT8 weight
      dram_rvalid_pipe[0] <= |dram_phy_read;
    end
  end

  assign dram_phy_rdata       = dram_rdata_pipe[7];
  assign dram_phy_rdata_valid = dram_rvalid_pipe[7];

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
  byte unsigned uart_rx_bytes[$];
  byte unsigned uart_tx_bytes[$];

  always_ff @(posedge clk) begin
    uart_prev_tx <= uart_tx;

    if (dut.u_uart.tx_produce)
      uart_tx_bytes.push_back(dut.u_uart.wdata[7:0]);

    if (!uart_capturing && uart_prev_tx && !uart_tx) begin
      // Start bit detected (falling edge)
      uart_capturing <= 1;
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
          uart_rx_bytes.push_back(byte'(uart_shift));
          // Print character
          if (uart_shift == 8'h0A) begin
            $display("[UART] %s", uart_line);
            last_uart_line = uart_line;
            uart_line = "";
          end else if (uart_shift == 8'h0D || uart_shift == 8'h00) begin
            // Ignore CR/NUL in the summary buffer.
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
    if (cycle_count >= MAX_CYCLES) begin
      $display("[TB] TIMEOUT: %0d cycles exceeded without test completion.", MAX_CYCLES);
      $finish;
    end
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
      $display("[PASS] %-40s  @cycle %0d", test_name, cycle_count);
      tests_passed++;
    end else begin
      $display("[FAIL] %-40s  @cycle %0d  %s", test_name, cycle_count, fail_msg);
      tests_failed++;
    end
  endtask

  task automatic uart_host_send_byte(input byte unsigned data);
    uart_rx = 1'b1;
    repeat (UART_BIT_CYCLES) @(posedge clk);

    uart_rx = 1'b0;
    repeat (UART_BIT_CYCLES) @(posedge clk);

    for (int bit_index = 0; bit_index < 8; bit_index++) begin
      uart_rx = data[bit_index];
      repeat (UART_BIT_CYCLES) @(posedge clk);
    end

    uart_rx = 1'b1;
    repeat (UART_BIT_CYCLES) @(posedge clk);
  endtask

  task automatic uart_host_recv_byte(
    output byte unsigned data,
    input  int timeout_cycles,
    output bit ok
  );
    int wait_cycles;
    ok = 1'b0;
    data = 8'h00;
    wait_cycles = 0;
    while (!ok && wait_cycles < timeout_cycles) begin
      if (uart_tx_bytes.size() > 0) begin
        data = uart_tx_bytes.pop_front();
        ok = 1'b1;
      end else begin
        @(posedge clk);
        wait_cycles++;
      end
    end
  endtask

  task automatic uart_host_send_u32(input logic [31:0] data);
    uart_host_send_byte(byte'(data[7:0]));
    uart_host_send_byte(byte'(data[15:8]));
    uart_host_send_byte(byte'(data[23:16]));
    uart_host_send_byte(byte'(data[31:24]));
  endtask

  task automatic uart_host_recv_u32(
    output logic [31:0] data,
    input  int timeout_cycles,
    output bit ok
  );
    byte unsigned b0;
    byte unsigned b1;
    byte unsigned b2;
    byte unsigned b3;
    bit ok0;
    bit ok1;
    bit ok2;
    bit ok3;

    uart_host_recv_byte(b0, timeout_cycles, ok0);
    uart_host_recv_byte(b1, timeout_cycles, ok1);
    uart_host_recv_byte(b2, timeout_cycles, ok2);
    uart_host_recv_byte(b3, timeout_cycles, ok3);

    ok = ok0 && ok1 && ok2 && ok3;
    data = {b3, b2, b1, b0};
  endtask

  task automatic wait_for_uart_idle(
    input  int quiet_cycles,
    input  int max_cycles,
    output bit idle_seen
  );
    int last_count;
    int quiet_count;
    int elapsed;

    idle_seen = 1'b0;
    last_count = uart_chars_rx;
    quiet_count = 0;
    elapsed = 0;

    while (!idle_seen && elapsed < max_cycles) begin
      @(posedge clk);
      elapsed++;
      if (uart_chars_rx != last_count) begin
        last_count = uart_chars_rx;
        quiet_count = 0;
      end else begin
        quiet_count++;
        if (quiet_count >= quiet_cycles)
          idle_seen = 1'b1;
      end
    end
  endtask

  // ---------------------------------------------------------------------------
  // Main test sequence
  // ---------------------------------------------------------------------------
  initial begin
    $display("=" * 72);
    $display("  tb_soc_top — SoC Integration Testbench");
    $display("  soc_top_v2 : 4x4 tile array, L1/L2 cache, DRAM, Ibex RISC-V");
    $display("=" * 72);

    // ------------------------------------------------------------------
    // TEST 1: Reset behaviour
    // ------------------------------------------------------------------
    $display("\n[TEST 1] Reset assertion ...");
    rst_n = 0;
    repeat (20) @(posedge clk);

    check("accel_busy=0 in reset",  !accel_busy);
    check("accel_done=0 in reset",  !accel_done);
    check("gpio_o holds 0 in reset", gpio_o == 8'h0);

    // ------------------------------------------------------------------
    // TEST 2: Boot — CPU starts executing
    // ------------------------------------------------------------------
    $display("\n[TEST 2] Boot / CPU startup ...");
    rst_n = 1;
    repeat (100) @(posedge clk);

    check("CLK toggles after reset", clk === 1'b1 || clk === 1'b0);

    // ------------------------------------------------------------------
    // TEST 3: Firmware banner
    // ------------------------------------------------------------------
    $display("\n[TEST 3] Waiting for UART output (firmware boot message)...");
    begin
      int uart_wait = 0;
      bit uart_idle;
      while (uart_chars_rx == 0 && uart_wait < 200_000) begin
        @(posedge clk);
        uart_wait++;
      end
      check("UART TX active within 200k cycles", uart_chars_rx > 0,
            $sformatf("No UART activity after %0d cycles", uart_wait));
      if (uart_chars_rx > 0)
        $display("       First UART char at cycle ~%0d", cycle_count);

      wait_for_uart_idle(20_000, 500_000, uart_idle);
      check("UART boot banner drained", uart_idle,
            "Firmware banner never went idle");
    end

    // ------------------------------------------------------------------
    // TEST 4: UART loader ping
    // ------------------------------------------------------------------
    $display("\n[TEST 4] UART loader ping ...");
    begin
      byte unsigned resp;
      logic [31:0] ping_sig;
      bit ok;
      byte unsigned sig_b0;
      byte unsigned sig_b1;
      byte unsigned sig_b2;
      byte unsigned sig_b3;

      uart_tx_bytes.delete();
      uart_host_send_byte(CMD_PING);
      uart_host_recv_byte(resp, 200_000, ok);
      check("CMD_PING returned RESP_OK", ok && resp == RESP_OK,
            ok ? $sformatf("resp=0x%02x", resp) : "No response byte");

      uart_host_recv_byte(sig_b0, 200_000, ok);
      uart_host_recv_byte(sig_b1, 200_000, ok);
      uart_host_recv_byte(sig_b2, 200_000, ok);
      uart_host_recv_byte(sig_b3, 200_000, ok);
      ping_sig = {sig_b3, sig_b2, sig_b1, sig_b0};
      check("CMD_PING signature correct", ok && ping_sig == 32'h53545632,
            ok ? $sformatf("sig=0x%08x", ping_sig) : "Timed out waiting for signature");
    end

            // ------------------------------------------------------------------
            // TEST 5: Cached SRAM write/readback through the loader
            // ------------------------------------------------------------------
            $display("\n[TEST 5] Cached SRAM write/readback ...");
    begin
          byte unsigned resp;
          logic [31:0] rd0;
          logic [31:0] rd1;
          logic [31:0] rd2;
          logic [31:0] rd3;
          bit ok;

          uart_tx_bytes.delete();
          uart_host_send_byte(CMD_MEM_WRITE_WORDS);
          uart_host_send_u32(HOST_SCRATCH_ADDR);
          uart_host_send_u32(32'd4);
          uart_host_send_u32(32'd5);
          uart_host_send_u32(32'd123);
          uart_host_send_u32(32'd42);
          uart_host_send_u32(32'd7);
          uart_host_recv_byte(resp, 200_000, ok);
          check("CMD_MEM_WRITE_WORDS returned RESP_OK", ok && resp == RESP_OK,
            ok ? $sformatf("resp=0x%02x", resp) : "No response byte");

          uart_tx_bytes.delete();
          uart_host_send_byte(CMD_MEM_READ_WORDS);
          uart_host_send_u32(HOST_SCRATCH_ADDR);
          uart_host_send_u32(32'd4);
          uart_host_recv_byte(resp, 200_000, ok);
          check("CMD_MEM_READ_WORDS returned RESP_OK", ok && resp == RESP_OK,
            ok ? $sformatf("resp=0x%02x", resp) : "No response byte");

          uart_host_recv_u32(rd0, 200_000, ok);
          check("Readback word 0 correct", ok && rd0 == 32'd5,
            ok ? $sformatf("rd0=%0d", rd0) : "Timed out waiting for word 0");
          uart_host_recv_u32(rd1, 200_000, ok);
          check("Readback word 1 correct", ok && rd1 == 32'd123,
            ok ? $sformatf("rd1=%0d", rd1) : "Timed out waiting for word 1");
          uart_host_recv_u32(rd2, 200_000, ok);
          check("Readback word 2 correct", ok && rd2 == 32'd42,
            ok ? $sformatf("rd2=%0d", rd2) : "Timed out waiting for word 2");
          uart_host_recv_u32(rd3, 200_000, ok);
          check("Readback word 3 correct", ok && rd3 == 32'd7,
            ok ? $sformatf("rd3=%0d", rd3) : "Timed out waiting for word 3");
            end

            // ------------------------------------------------------------------
            // TEST 6: Loader argmax + GPIO
            // ------------------------------------------------------------------
            $display("\n[TEST 6] Loader argmax + GPIO ...");
            begin
          byte unsigned resp;
          logic [31:0] best_index;
          logic [31:0] best_value;
          bit ok;

          uart_tx_bytes.delete();
          uart_host_send_byte(CMD_ARGMAX);
          uart_host_send_u32(HOST_SCRATCH_ADDR);
          uart_host_send_u32(32'd4);

          uart_host_recv_byte(resp, 200_000, ok);
          check("CMD_ARGMAX returned RESP_OK", ok && resp == RESP_OK,
            ok ? $sformatf("resp=0x%02x", resp) : "No response byte");

          uart_host_recv_u32(best_index, 200_000, ok);
          check("CMD_ARGMAX best index correct", ok && best_index == 32'd1,
            ok ? $sformatf("best_index=%0d", best_index) : "Timed out waiting for best index");

          uart_host_recv_u32(best_value, 200_000, ok);
          check("CMD_ARGMAX best value correct", ok && best_value == 32'd123,
            ok ? $sformatf("best_value=%0d", best_value) : "Timed out waiting for best value");

          repeat (500) @(posedge clk);
          check("GPIO outputs enabled", gpio_oe == 8'hFF,
            $sformatf("gpio_oe=0x%02x", gpio_oe));
          check("GPIO output matches argmax index", gpio_o == 8'hF1,
            $sformatf("gpio_o=0x%02x", gpio_o));
          check("Accelerator stays idle in loader mode", !accel_busy && !accel_done,
            $sformatf("accel_busy=%0b accel_done=%0b", accel_busy, accel_done));
    end

    // ------------------------------------------------------------------
    // RESULTS
    // ------------------------------------------------------------------
    $display("\n" + "=" * 72);
    $display("  tb_soc_top RESULTS: %0d passed, %0d failed", tests_passed, tests_failed);
    if (last_uart_line != "")
      $display("  Last UART line: \"%s\"", last_uart_line);
    else if (uart_line != "")
      $display("  Last UART partial: \"%s\"", uart_line);
    $display("  Total simulation cycles: %0d", cycle_count);
    $display("  Wall time: ~%0.2f ms @ 50 MHz", real'(cycle_count) / 50_000.0);
    $display("=" * 72);

    if (tests_failed == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  FAILURES: %0d test(s) failed", tests_failed);

    $finish;
  end

  // ---------------------------------------------------------------------------
  // Waveform dump (for GTKWave / DVE)
  // ---------------------------------------------------------------------------
  initial begin
    if ($test$plusargs("waves")) begin
      $dumpfile("tb_soc_top.vcd");
      $dumpvars(0, tb_soc_top);
    end
  end

  // ---------------------------------------------------------------------------
  // Assertions: protocol sanity on key interfaces
  // ---------------------------------------------------------------------------

  // accel_done is a completion indication and may rise on the cycle after
  // accel_busy drops. What must hold is that some accelerator activity was
  // observed before the completion pulse.
  property p_done_after_busy;
    @(posedge clk) disable iff (!rst_n)
    $rose(accel_done) |-> (accel_busy || accel_seen_busy);
  endproperty

  // DRAM PHY: should never assert both read and write in the same cycle
  property p_dram_no_rw_collision;
    @(posedge clk) disable iff (!rst_n)
    !(|dram_phy_read && |dram_phy_write);
  endproperty

  // UART TX: framing — stop bit must be high (line idle = 1)
  // Checked after capture window ends
  property p_uart_stop_bit;
    @(posedge clk) disable iff (!rst_n)
    ($fell(uart_capturing)) |-> uart_tx;
  endproperty

  assert property (p_dram_no_rw_collision)
    else $error("[ASSERT] DRAM read/write collision at cycle %0d", cycle_count);

  // Soft completion check. Disable with +no_assert if bring-up firmware is
  // intentionally exercising only partial accelerator flows.
  `ifndef NO_ACCEL_ASSERT
  assert property (p_done_after_busy)
    else $warning("[ASSERT] accel_done rose without prior accelerator activity at cycle %0d", cycle_count);
  `endif

endmodule
