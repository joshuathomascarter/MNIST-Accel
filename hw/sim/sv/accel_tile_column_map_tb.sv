`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_column_map_tb;

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

  function automatic logic [31:0] pack4(input int b0, input int b1, input int b2, input int b3);
    begin
      pack4 = {8'(b3), 8'(b2), 8'(b1), 8'(b0)};
    end
  endfunction

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

    // Activation block: first streamed row is all ones; remaining rows are zero.
    dut.u_sp.mem[ACT_BASE + 0] = 32'h01010101;
    dut.u_sp.mem[ACT_BASE + 1] = 32'h01010101;
    dut.u_sp.mem[ACT_BASE + 2] = 32'h01010101;
    dut.u_sp.mem[ACT_BASE + 3] = 32'h01010101;
    for (int vec = 1; vec < N_ROWS; vec++) begin
      for (int word = 0; word < 4; word++)
        dut.u_sp.mem[ACT_BASE + (vec * 4) + word] = 32'h00000000;
    end

    // Weight block: every row has distinct ascending column values 1..16.
    for (int row = 0; row < N_ROWS; row++) begin
      dut.u_sp.mem[WGT_BASE + row * 4 + 0] = pack4(1, 2, 3, 4);
      dut.u_sp.mem[WGT_BASE + row * 4 + 1] = pack4(5, 6, 7, 8);
      dut.u_sp.mem[WGT_BASE + row * 4 + 2] = pack4(9, 10, 11, 12);
      dut.u_sp.mem[WGT_BASE + row * 4 + 3] = pack4(13, 14, 15, 16);
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

    for (int row = 0; row < N_ROWS; row++) begin
      for (int col = 0; col < N_COLS; col++) begin
        logic [31:0] got;
        got = dut.u_sp.mem[OUT_BASE + row * N_COLS + col];
        if (got !== 32'(col + 1)) begin
          $display("row %0d col %0d got %0d expected %0d", row, col, got, col + 1);
          $fatal(1, "Column mapping mismatch at row %0d col %0d", row, col);
        end
      end
    end

    $display("accel_tile_column_map_tb PASS: direct tile compute preserves column order");
    $finish;
  end

  initial begin
    #50000;
    $fatal(1, "accel_tile_column_map_tb timeout");
  end

endmodule