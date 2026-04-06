`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_dma_load_tb;

  localparam int NUM_VCS   = noc_pkg::NUM_VCS;
  localparam int SP_DEPTH  = 1024;
  localparam int SP_BASE   = 32;
  localparam logic [7:0] OP_LOAD = 8'h01;

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
    .SP_DEPTH  (SP_DEPTH),
    .SP_DATA_W (32),
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

  task automatic send_read_resp_flit(
    input flit_type_e flit_type,
    input logic [47:0] payload
  );
    begin
      @(posedge clk);
      noc_flit_in.flit_type = flit_type;
      noc_flit_in.src_id    = 4'd15;
      noc_flit_in.dst_id    = 4'd0;
      noc_flit_in.vc_id     = '0;
      noc_flit_in.msg_type  = MSG_READ_RESP;
      noc_flit_in.payload   = payload;
      noc_valid_in          = 1'b1;
      @(posedge clk);
      noc_valid_in          = 1'b0;
      noc_flit_in           = '0;
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

    csr_write(8'h04, 32'h4000_0000);
    csr_write(8'h08, 32'd4);
    csr_write(8'h0C, SP_BASE);
    csr_write(8'h00, 32'(OP_LOAD));

    wait (noc_valid_out);

    send_read_resp_flit(FLIT_HEAD, {4'h0, 8'd4, 4'h0, 32'h04030201});
    send_read_resp_flit(FLIT_BODY, {16'h0, 32'h08070605});
    send_read_resp_flit(FLIT_BODY, {16'h0, 32'h0c0b0a09});
    send_read_resp_flit(FLIT_TAIL, {16'h0, 32'h100f0e0d});

    repeat (20) @(posedge clk);

    if (tile_busy)
      $fatal(1, "Tile still busy after injected read response stream");

    if (dut.u_sp.mem[SP_BASE + 0] !== 32'h04030201)
      $fatal(1, "word0 mismatch: %08h", dut.u_sp.mem[SP_BASE + 0]);
    if (dut.u_sp.mem[SP_BASE + 1] !== 32'h08070605)
      $fatal(1, "word1 mismatch: %08h", dut.u_sp.mem[SP_BASE + 1]);
    if (dut.u_sp.mem[SP_BASE + 2] !== 32'h0c0b0a09)
      $fatal(1, "word2 mismatch: %08h", dut.u_sp.mem[SP_BASE + 2]);
    if (dut.u_sp.mem[SP_BASE + 3] !== 32'h100f0e0d)
      $fatal(1, "word3 mismatch: %08h", dut.u_sp.mem[SP_BASE + 3]);

    $display("accel_tile_dma_load_tb PASS: NI/controller preserve read-response word order");
    $finish;
  end

  initial begin
    #50000;
    $fatal(1, "accel_tile_dma_load_tb timeout");
  end

endmodule