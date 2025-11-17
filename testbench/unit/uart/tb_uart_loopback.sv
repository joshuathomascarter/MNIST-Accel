// -----------------------------------------------------------------------------
// tb_uart_loopback.sv — Self-checking UART TX<->RX loopback
//  - Drives random & directed bytes into TX, wires TX->RX, checks RX output
//  - Parameterizable clock/baud; supports integer divider or NCO paths
//  - Includes basic assertions and error counters
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module tb_uart_loopback;

  // --- Params you can tweak quickly
  localparam int unsigned CLK_HZ       = 50_000_000;
  localparam real         CLK_PERIODNS = 1e9 / CLK_HZ;
  localparam int unsigned BAUD         = 115_200;
  localparam int unsigned OVERSAMPLE   = 16;
  localparam int unsigned PARITY       = 0;  // 0:none
  localparam int unsigned STOP_BITS    = 1;
  localparam bit          USE_NCO      = 0;
  localparam int unsigned DATA_BITS    = 8;

  // --- Clock / reset
  logic clk = 0;
  always #(CLK_PERIODNS/2.0) clk = ~clk;

  logic rst_n = 0;

  // --- DUTs
  logic        i_valid;
  logic        i_ready;
  logic [DATA_BITS-1:0]  i_data;

  logic        tx_line;     // wire between TX and RX
  logic [DATA_BITS-1:0]  rx_data;
  logic        rx_valid;
  logic        rx_frm_err, rx_par_err;

  uart_tx #(
    .DATA_BITS(DATA_BITS),
    .CLK_HZ(CLK_HZ), .BAUD(BAUD), .OVERSAMPLE(OVERSAMPLE),
    .PARITY(PARITY), .STOP_BITS(STOP_BITS), .USE_NCO(USE_NCO)
  ) dut_tx (
    .i_clk(clk), .i_rst_n(rst_n),
    .i_data(i_data), .i_valid(i_valid), .i_ready(i_ready),
    .o_tx(tx_line)
  );

  uart_rx #(
    .DATA_BITS(DATA_BITS),
    .CLK_HZ(CLK_HZ), .BAUD(BAUD), .OVERSAMPLE(OVERSAMPLE),
    .PARITY(PARITY), .STOP_BITS(STOP_BITS), .USE_NCO(USE_NCO)
  ) dut_rx (
    .i_clk(clk), .i_rst_n(rst_n),
    .i_rx(tx_line),
    .o_data(rx_data), .o_valid(rx_valid),
    .o_frm_err(rx_frm_err), .o_par_err(rx_par_err)
  );

  // --- Scoreboard
  int sent_cnt = 0;
  int recv_cnt = 0;
  int err_cnt  = 0;

  byte queue[$]; // expected bytes (FIFO)

  // --- Simple task to send one byte
  task automatic send_byte(input byte b);
    // wait until TX ready
    @(posedge clk);
    wait (i_ready);
    i_data  <= b;
    i_valid <= 1'b1;
    @(posedge clk);
    i_valid <= 1'b0;
    queue.push_back(b);
    sent_cnt++;
  endtask

  // --- Drive & monitor
  initial begin
    // init
    i_valid = 0; i_data = 8'h00;
    rst_n   = 0;
    repeat (10) @(posedge clk);
    rst_n   = 1;

    // Directed sequence
    send_byte(8'h00);
    send_byte(8'h55);
    send_byte(8'hAA);
    send_byte(8'hFF);

    // Random burst
    for (int i = 0; i < 64; i++) begin
      send_byte($urandom_range(0,255));
    end

    // Let last frames drain
    repeat (20000) @(posedge clk);

    // Final report
    $display("[TB] Sent=%0d, Recv=%0d, Errors=%0d", sent_cnt, recv_cnt, err_cnt);
    if (err_cnt == 0 && recv_cnt == sent_cnt) begin
      $display("[TB] PASS");
    end else begin
      $error("[TB] FAIL");
    end
    $finish;
  end

  // --- RX monitor + checks
  always_ff @(posedge clk) begin
    if (rst_n && rx_valid) begin
      recv_cnt++;
      if (queue.size() == 0) begin
        $error("[TB] RX got byte with empty expected queue!");
        err_cnt++;
      end else begin
        byte exp = queue.pop_front();
        if (rx_data !== exp) begin
          $error("[TB] Mismatch: got %02x exp %02x", rx_data, exp);
          err_cnt++;
        end
        if (rx_frm_err) begin
          $error("[TB] Framing error flagged on byte %02x", rx_data);
          err_cnt++;
        end
        if (rx_par_err) begin
          $error("[TB] Parity error flagged on byte %02x", rx_data);
          err_cnt++;
        end
      end
    end
  end

  // --- Assertions (basic)
`ifndef SYNTHESIS
  // TX must hold idle high when not sending
  property p_tx_idle;
    @(posedge clk) disable iff (!rst_n)
      (dut_tx.state == dut_tx.IDLE) |-> (tx_line == 1'b1);
  endproperty
  assert property (p_tx_idle);

  // RX valid implies RX outputs are stable that cycle
  property p_rx_valid_stable;
    @(posedge clk) disable iff (!rst_n)
      rx_valid |-> $stable(rx_data);
  endproperty
  assert property (p_rx_valid_stable);
`endif

endmodule
`default_nettype wire
