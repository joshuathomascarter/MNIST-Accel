`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_tb;

  localparam int N_ROWS    = 16;
  localparam int N_COLS    = 16;
  localparam int DATA_W    = 8;
  localparam int ACC_W     = 32;
  localparam int SP_DEPTH  = 1024;
  localparam int SP_DATA_W = 32;
  localparam int NUM_VCS   = noc_pkg::NUM_VCS;

  localparam logic [7:0] OP_COMPUTE = 8'h03;
  localparam int ACT_BASE = 0;
  localparam int WGT_BASE = 128;
  localparam int OUT_BASE = 256;

  logic                  clk;
  logic                  rst_n;
  flit_t                 noc_flit_out;
  logic                  noc_valid_out;
  logic [NUM_VCS-1:0]    noc_credit_in;
  flit_t                 noc_flit_in;
  logic                  noc_valid_in;
  logic [NUM_VCS-1:0]    noc_credit_out;
  logic [31:0]           csr_wdata;
  logic [7:0]            csr_addr;
  logic                  csr_wen;
  logic [31:0]           csr_rdata;
  logic                  barrier_req;
  logic                  barrier_done;
  logic                  tile_busy;
  logic                  tile_done;

  accel_tile #(
    .TILE_ID   (0),
    .N_ROWS    (N_ROWS),
    .N_COLS    (N_COLS),
    .DATA_W    (DATA_W),
    .ACC_W     (ACC_W),
    .SP_DEPTH  (SP_DEPTH),
    .SP_DATA_W (SP_DATA_W),
    .NUM_VCS   (NUM_VCS)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .noc_flit_out   (noc_flit_out),
    .noc_valid_out  (noc_valid_out),
    .noc_credit_in  (noc_credit_in),
    .noc_flit_in    (noc_flit_in),
    .noc_valid_in   (noc_valid_in),
    .noc_credit_out (noc_credit_out),
    .csr_wdata      (csr_wdata),
    .csr_addr       (csr_addr),
    .csr_wen        (csr_wen),
    .csr_rdata      (csr_rdata),
    .barrier_req    (barrier_req),
    .barrier_done   (barrier_done),
    .tile_busy      (tile_busy),
    .tile_done      (tile_done)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic csr_write(input logic [7:0] addr, input logic [31:0] data);
    begin
      @(posedge clk);
      csr_addr  = addr;
      csr_wdata = data;
      csr_wen   = 1'b1;
      @(posedge clk);
      csr_wen   = 1'b0;
    end
  endtask

  task automatic check_result_word(input int word_idx, input logic [31:0] expected);
    begin
      csr_addr = 8'(32'h30 + (word_idx * 4));
      #1;
      if (csr_rdata !== expected)
        $fatal(1, "CSR result word %0d mismatch: got %0d expected %0d", word_idx, csr_rdata, expected);
    end
  endtask

  initial begin
    noc_credit_in = '1;
    noc_flit_in   = '0;
    noc_valid_in  = 1'b0;
    csr_wdata     = '0;
    csr_addr      = '0;
    csr_wen       = 1'b0;
    barrier_done  = 1'b0;
    rst_n         = 1'b0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    for (int vec = 0; vec < N_ROWS; vec++) begin
      for (int word = 0; word < 4; word++) begin
        dut.u_sp.mem[ACT_BASE + (vec * 4) + word] = 32'h01010101;
        dut.u_sp.mem[WGT_BASE + (vec * 4) + word] = 32'h01010101;
      end
    end

    csr_write(8'h04, ACT_BASE);
    csr_write(8'h08, WGT_BASE);
    csr_write(8'h0C, OUT_BASE);
    csr_write(8'h00, 32'(OP_COMPUTE));

    repeat (2000) begin
      @(posedge clk);
      if (tile_done)
        break;
    end

    if (!tile_done)
      $fatal(1, "Timed out waiting for accel_tile compute to finish");

    check_result_word(0, 32'd16);
    check_result_word(1, 32'd16);
    check_result_word(2, 32'd16);
    check_result_word(3, 32'd16);

    for (int out_idx = 0; out_idx < N_ROWS * N_COLS; out_idx++) begin
      if (dut.u_sp.mem[OUT_BASE + out_idx] !== 32'd16)
        $fatal(1, "Scratchpad output word %0d mismatch: got %0d expected %0d",
               out_idx, dut.u_sp.mem[OUT_BASE + out_idx], 32'd16);
    end

    $display("accel_tile_tb PASS: real tile compute completed and spilled 256 outputs");
    $finish;
  end

  initial begin
    #50000;
    $fatal(1, "accel_tile_tb timeout");
  end

endmodule