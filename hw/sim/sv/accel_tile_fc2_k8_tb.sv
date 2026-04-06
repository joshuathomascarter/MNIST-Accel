`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_fc2_k8_tb;

  localparam int N_ROWS    = 16;
  localparam int N_COLS    = 16;
  localparam int DATA_W    = 8;
  localparam int ACC_W     = 32;
  localparam int SP_DEPTH  = 1024;
  localparam int SP_DATA_W = 32;
  localparam int NUM_VCS   = noc_pkg::NUM_VCS;

  localparam logic [7:0] OP_COMPUTE = 8'h03;
  localparam int ACT_BASE = 64;
  localparam int WGT_BASE = 0;
  localparam int OUT_BASE = 128;

  localparam int BLOCK_WORDS       = 64;
  localparam int FC2_WEIGHT_OFFSET = 16'h0000;
  localparam int ACT_OFFSET        = 16'h0240;
  localparam int K_TILE            = 8;
  localparam int MEM_WORDS         = 524288;

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
  logic [31:0]           dram_words [0:MEM_WORDS-1];

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

  initial begin
    noc_credit_in = '1;
    noc_flit_in   = '0;
    noc_valid_in  = 1'b0;
    csr_wdata     = '0;
    csr_addr      = '0;
    csr_wen       = 1'b0;
    barrier_done  = 1'b0;
    rst_n         = 1'b0;

    $readmemh("/Users/joshcarter/MNIST-Accel/data/dram_init.hex", dram_words);

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    for (int i = 0; i < BLOCK_WORDS; i++) begin
      dut.u_sp.mem[WGT_BASE + i] = dram_words[FC2_WEIGHT_OFFSET + (K_TILE * BLOCK_WORDS) + i];
      dut.u_sp.mem[ACT_BASE + i] = dram_words[ACT_OFFSET + (K_TILE * BLOCK_WORDS) + i];
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

    begin
      logic [31:0] col_sums [0:N_COLS-1];
      logic [31:0] expected [0:N_COLS-1];

      expected[0]  = 32'hfffff2de;
      expected[1]  = 32'h0000019d;
      expected[2]  = 32'hfffff26a;
      expected[3]  = 32'hfffff804;
      expected[4]  = 32'h00000727;
      expected[5]  = 32'hfffffd27;
      expected[6]  = 32'h0000051d;
      expected[7]  = 32'h000014de;
      expected[8]  = 32'hfffffffe;
      expected[9]  = 32'h00000a32;
      for (int col = 10; col < N_COLS; col++)
        expected[col] = 32'h00000000;

      for (int col = 0; col < N_COLS; col++) begin
        col_sums[col] = '0;
        for (int row = 0; row < N_ROWS; row++)
          col_sums[col] = col_sums[col] + dut.u_sp.mem[OUT_BASE + row * N_COLS + col];
      end

      for (int col = 0; col < N_COLS; col++) begin
        if (col_sums[col] !== expected[col]) begin
          $display("col %0d got %08h expected %08h", col, col_sums[col], expected[col]);
          $fatal(1, "FC2 k8 partial mismatch at col %0d", col);
        end
      end
    end

    $display("accel_tile_fc2_k8_tb PASS: direct tile matches expected FC2 k8 partial logits");
    $finish;
  end

  initial begin
    #50000;
    $fatal(1, "accel_tile_fc2_k8_tb timeout");
  end

endmodule