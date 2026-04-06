`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_array_mesh_tb;

  localparam int MESH_ROWS  = 4;
  localparam int MESH_COLS  = 4;
  localparam int NUM_TILES  = MESH_ROWS * MESH_COLS;
  localparam int N_ROWS     = 16;
  localparam int N_COLS     = 16;
  localparam int DATA_W     = 8;
  localparam int ACC_W      = 32;
  localparam int SP_DEPTH   = 1024;
  localparam int AXI_ADDR_W = 32;
  localparam int AXI_DATA_W = 32;
  localparam int AXI_ID_W   = 4;
  localparam int MEM_WORDS  = 4096;

  localparam logic [7:0] OP_LOAD    = 8'h01;
  localparam logic [7:0] OP_STORE   = 8'h02;
  localparam logic [7:0] OP_COMPUTE = 8'h03;

  localparam logic [31:0] EXT_MEM_BASE = 32'h4000_0000;
  localparam logic [31:0] ACT_EXT_BASE = EXT_MEM_BASE + 32'h0000;
  localparam logic [31:0] WGT_EXT_BASE = EXT_MEM_BASE + 32'h0400;
  localparam logic [31:0] OUT_EXT_BASE = EXT_MEM_BASE + 32'h0800;

  localparam int ACT_SP_BASE = 0;
  localparam int WGT_SP_BASE = 128;
  localparam int OUT_SP_BASE = 256;

  localparam int ACT_WORDS = (N_ROWS * N_ROWS * DATA_W) / AXI_DATA_W;
  localparam int WGT_WORDS = (N_ROWS * N_COLS * DATA_W) / AXI_DATA_W;
  localparam int OUT_WORDS = N_ROWS * N_COLS;

  logic                   clk;
  logic                   rst_n;

  logic                   inr_meta_cfg_valid [NUM_TILES];
  logic [7:0]             inr_meta_cfg_reduce_id [NUM_TILES];
  logic [3:0]             inr_meta_cfg_target [NUM_TILES];
  logic                   inr_meta_cfg_enable [NUM_TILES];

  logic [AXI_ADDR_W-1:0]  s_axi_awaddr;
  logic                   s_axi_awvalid;
  logic                   s_axi_awready;
  logic [AXI_DATA_W-1:0]  s_axi_wdata;
  logic [3:0]             s_axi_wstrb;
  logic                   s_axi_wvalid;
  logic                   s_axi_wready;
  logic [1:0]             s_axi_bresp;
  logic                   s_axi_bvalid;
  logic                   s_axi_bready;
  logic [AXI_ADDR_W-1:0]  s_axi_araddr;
  logic                   s_axi_arvalid;
  logic                   s_axi_arready;
  logic [AXI_DATA_W-1:0]  s_axi_rdata;
  logic [1:0]             s_axi_rresp;
  logic                   s_axi_rvalid;
  logic                   s_axi_rready;

  logic [AXI_ID_W-1:0]    m_axi_arid;
  logic [AXI_ADDR_W-1:0]  m_axi_araddr;
  logic [7:0]             m_axi_arlen;
  logic [2:0]             m_axi_arsize;
  logic [1:0]             m_axi_arburst;
  logic                   m_axi_arvalid;
  logic                   m_axi_arready;
  logic [AXI_ID_W-1:0]    m_axi_rid;
  logic [AXI_DATA_W-1:0]  m_axi_rdata;
  logic [1:0]             m_axi_rresp;
  logic                   m_axi_rlast;
  logic                   m_axi_rvalid;
  logic                   m_axi_rready;
  logic [AXI_ID_W-1:0]    m_axi_awid;
  logic [AXI_ADDR_W-1:0]  m_axi_awaddr;
  logic [7:0]             m_axi_awlen;
  logic [2:0]             m_axi_awsize;
  logic [1:0]             m_axi_awburst;
  logic                   m_axi_awvalid;
  logic                   m_axi_awready;
  logic [AXI_DATA_W-1:0]  m_axi_wdata;
  logic [AXI_DATA_W/8-1:0] m_axi_wstrb;
  logic                   m_axi_wlast;
  logic                   m_axi_wvalid;
  logic                   m_axi_wready;
  logic [AXI_ID_W-1:0]    m_axi_bid;
  logic [1:0]             m_axi_bresp;
  logic                   m_axi_bvalid;
  logic                   m_axi_bready;

  logic [NUM_TILES-1:0]   tile_busy_o;
  logic [NUM_TILES-1:0]   tile_done_o;

  logic [31:0]            ext_mem [0:MEM_WORDS-1];

  logic                   rd_active;
  logic [31:0]            rd_addr_reg;
  logic [7:0]             rd_len_reg;
  logic [7:0]             rd_idx;
  logic [3:0]             rd_id_reg;

  logic                   wr_active;
  logic [31:0]            wr_addr_reg;
  logic [7:0]             wr_len_reg;
  logic [7:0]             wr_idx;
  logic [3:0]             wr_id_reg;
  logic                   b_pending;

  function automatic int ext_mem_index(input logic [31:0] addr);
    begin
      ext_mem_index = int'((addr - EXT_MEM_BASE) >> 2);
    end
  endfunction

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic axi_write(input logic [15:0] addr, input logic [31:0] data);
    begin
      @(posedge clk);
      s_axi_awaddr  <= AXI_ADDR_W'(addr);
      s_axi_awvalid <= 1'b1;
      s_axi_wdata   <= data;
      s_axi_wstrb   <= 4'hF;
      s_axi_wvalid  <= 1'b1;
      s_axi_bready  <= 1'b1;

      wait (s_axi_awready && s_axi_wready);

      @(negedge clk);
      s_axi_awvalid <= 1'b0;
      s_axi_wvalid  <= 1'b0;

      while (!s_axi_bvalid)
        @(posedge clk);

      @(negedge clk);
      s_axi_bready <= 1'b0;

      if ($test$plusargs("CSR_TRACE"))
        $display("[CSR] write addr=%04h data=%08h", addr, data);
    end
  endtask

  task automatic dump_tile0_cmd_state(input string label);
    begin
      $display("[CMD] %s pending=%0b opcode=%02h arg0=%08h arg1=%08h arg2=%08h ctrl_state=%0d busy=%0b done=%0b",
               label,
               dut.gen_tile[0].u_tile.cmd_pending,
               dut.gen_tile[0].u_tile.cmd_opcode,
               dut.gen_tile[0].u_tile.arg0_reg,
               dut.gen_tile[0].u_tile.arg1_reg,
               dut.gen_tile[0].u_tile.arg2_reg,
               dut.gen_tile[0].u_tile.u_ctrl.state,
               tile_busy_o[0],
               tile_done_o[0]);
    end
  endtask

  task automatic issue_cmd(
    input logic [7:0] opcode,
    input logic [31:0] arg0,
    input logic [31:0] arg1,
    input logic [31:0] arg2
  );
    begin
      axi_write(16'h0004, arg0);
      axi_write(16'h0008, arg1);
      axi_write(16'h000C, arg2);
      axi_write(16'h0000, 32'(opcode));
      if ($test$plusargs("CSR_TRACE"))
        dump_tile0_cmd_state("after issue_cmd");
    end
  endtask

  task automatic wait_for_done(input string label);
    begin
      repeat (50000) begin
        @(posedge clk);
        if (tile_done_o[0])
          return;
      end
      $fatal(1, "%s timed out waiting for tile_done_o[0]", label);
    end
  endtask

  accel_tile_array #(
    .MESH_ROWS  (MESH_ROWS),
    .MESH_COLS  (MESH_COLS),
    .N_ROWS     (N_ROWS),
    .N_COLS     (N_COLS),
    .DATA_W     (DATA_W),
    .ACC_W      (ACC_W),
    .SP_DEPTH   (SP_DEPTH),
    .AXI_ADDR_W (AXI_ADDR_W),
    .AXI_DATA_W (AXI_DATA_W),
    .AXI_ID_W   (AXI_ID_W)
  ) dut (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .inr_meta_cfg_valid    (inr_meta_cfg_valid),
    .inr_meta_cfg_reduce_id(inr_meta_cfg_reduce_id),
    .inr_meta_cfg_target   (inr_meta_cfg_target),
    .inr_meta_cfg_enable   (inr_meta_cfg_enable),
    .s_axi_awaddr          (s_axi_awaddr),
    .s_axi_awvalid         (s_axi_awvalid),
    .s_axi_awready         (s_axi_awready),
    .s_axi_wdata           (s_axi_wdata),
    .s_axi_wstrb           (s_axi_wstrb),
    .s_axi_wvalid          (s_axi_wvalid),
    .s_axi_wready          (s_axi_wready),
    .s_axi_bresp           (s_axi_bresp),
    .s_axi_bvalid          (s_axi_bvalid),
    .s_axi_bready          (s_axi_bready),
    .s_axi_araddr          (s_axi_araddr),
    .s_axi_arvalid         (s_axi_arvalid),
    .s_axi_arready         (s_axi_arready),
    .s_axi_rdata           (s_axi_rdata),
    .s_axi_rresp           (s_axi_rresp),
    .s_axi_rvalid          (s_axi_rvalid),
    .s_axi_rready          (s_axi_rready),
    .m_axi_arid            (m_axi_arid),
    .m_axi_araddr          (m_axi_araddr),
    .m_axi_arlen           (m_axi_arlen),
    .m_axi_arsize          (m_axi_arsize),
    .m_axi_arburst         (m_axi_arburst),
    .m_axi_arvalid         (m_axi_arvalid),
    .m_axi_arready         (m_axi_arready),
    .m_axi_rid             (m_axi_rid),
    .m_axi_rdata           (m_axi_rdata),
    .m_axi_rresp           (m_axi_rresp),
    .m_axi_rlast           (m_axi_rlast),
    .m_axi_rvalid          (m_axi_rvalid),
    .m_axi_rready          (m_axi_rready),
    .m_axi_awid            (m_axi_awid),
    .m_axi_awaddr          (m_axi_awaddr),
    .m_axi_awlen           (m_axi_awlen),
    .m_axi_awsize          (m_axi_awsize),
    .m_axi_awburst         (m_axi_awburst),
    .m_axi_awvalid         (m_axi_awvalid),
    .m_axi_awready         (m_axi_awready),
    .m_axi_wdata           (m_axi_wdata),
    .m_axi_wstrb           (m_axi_wstrb),
    .m_axi_wlast           (m_axi_wlast),
    .m_axi_wvalid          (m_axi_wvalid),
    .m_axi_wready          (m_axi_wready),
    .m_axi_bid             (m_axi_bid),
    .m_axi_bresp           (m_axi_bresp),
    .m_axi_bvalid          (m_axi_bvalid),
    .m_axi_bready          (m_axi_bready),
    .tile_busy_o           (tile_busy_o),
    .tile_done_o           (tile_done_o)
  );

  assign m_axi_arready = !rd_active && !wr_active && !b_pending;
  assign m_axi_rid     = rd_id_reg;
  assign m_axi_rresp   = 2'b00;
  assign m_axi_rvalid  = rd_active;
  assign m_axi_rdata   = ext_mem[ext_mem_index(rd_addr_reg) + int'(rd_idx)];
  assign m_axi_rlast   = (rd_idx == rd_len_reg);

  assign m_axi_awready = !wr_active && !rd_active && !b_pending;
  assign m_axi_wready  = wr_active;
  assign m_axi_bid     = wr_id_reg;
  assign m_axi_bresp   = 2'b00;
  assign m_axi_bvalid  = b_pending;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_active   <= 1'b0;
      rd_addr_reg <= '0;
      rd_len_reg  <= '0;
      rd_idx      <= '0;
      rd_id_reg   <= '0;
      wr_active   <= 1'b0;
      wr_addr_reg <= '0;
      wr_len_reg  <= '0;
      wr_idx      <= '0;
      wr_id_reg   <= '0;
      b_pending   <= 1'b0;
    end else begin
      if (m_axi_arvalid && m_axi_arready) begin
        rd_active   <= 1'b1;
        rd_addr_reg <= m_axi_araddr;
        rd_len_reg  <= m_axi_arlen;
        rd_idx      <= '0;
        rd_id_reg   <= m_axi_arid;
      end else if (rd_active && m_axi_rvalid && m_axi_rready) begin
        if (m_axi_rlast)
          rd_active <= 1'b0;
        else
          rd_idx <= rd_idx + 1'b1;
      end

      if (m_axi_awvalid && m_axi_awready) begin
        wr_active   <= 1'b1;
        wr_addr_reg <= m_axi_awaddr;
        wr_len_reg  <= m_axi_awlen;
        wr_idx      <= '0;
        wr_id_reg   <= m_axi_awid;
      end

      if (wr_active && m_axi_wvalid && m_axi_wready) begin
        ext_mem[ext_mem_index(wr_addr_reg) + int'(wr_idx)] <= m_axi_wdata;
        if (m_axi_wlast) begin
          wr_active <= 1'b0;
          b_pending <= 1'b1;
        end else begin
          wr_idx <= wr_idx + 1'b1;
        end
      end

      if (b_pending && m_axi_bvalid && m_axi_bready)
        b_pending <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if ($test$plusargs("CSR_TRACE") && dut.tile_csr_wen[0]) begin
      $display("[CSR-WEN] axi_state=%0d addr=%02h data=%08h",
               dut.axi_s_state,
               dut.csr_addr_r,
               dut.csr_wdata_r);
    end
    if ($test$plusargs("CSR_TRACE") && dut.gen_tile[0].u_tile.cmd_issue) begin
      $display("[CMD-ISSUE] opcode=%02h arg0=%08h arg1=%08h arg2=%08h ctrl_state=%0d",
               dut.gen_tile[0].u_tile.cmd_opcode,
               dut.gen_tile[0].u_tile.arg0_reg,
               dut.gen_tile[0].u_tile.arg1_reg,
               dut.gen_tile[0].u_tile.arg2_reg,
               dut.gen_tile[0].u_tile.u_ctrl.state);
    end
  end

  initial begin
    rst_n        = 1'b0;
    s_axi_awaddr = '0;
    s_axi_awvalid = 1'b0;
    s_axi_wdata  = '0;
    s_axi_wstrb  = '0;
    s_axi_wvalid = 1'b0;
    s_axi_bready = 1'b0;
    s_axi_araddr = '0;
    s_axi_arvalid = 1'b0;
    s_axi_rready = 1'b0;

    for (int t = 0; t < NUM_TILES; t++) begin
      inr_meta_cfg_valid[t]     = 1'b0;
      inr_meta_cfg_reduce_id[t] = '0;
      inr_meta_cfg_target[t]    = '0;
      inr_meta_cfg_enable[t]    = 1'b0;
    end

    for (int idx = 0; idx < MEM_WORDS; idx++)
      ext_mem[idx] = '0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    ext_mem[ext_mem_index(ACT_EXT_BASE) + 0] = 32'h01010101;
    ext_mem[ext_mem_index(ACT_EXT_BASE) + 1] = 32'h01010101;
    ext_mem[ext_mem_index(ACT_EXT_BASE) + 2] = 32'h01010101;
    ext_mem[ext_mem_index(ACT_EXT_BASE) + 3] = 32'h01010101;
    for (int vec = 1; vec < N_ROWS; vec++) begin
      for (int word = 0; word < 4; word++)
        ext_mem[ext_mem_index(ACT_EXT_BASE) + vec * 4 + word] = 32'h00000000;
    end

    for (int row = 0; row < N_ROWS; row++) begin
      ext_mem[ext_mem_index(WGT_EXT_BASE) + row * 4 + 0] = 32'h04030201;
      ext_mem[ext_mem_index(WGT_EXT_BASE) + row * 4 + 1] = 32'h08070605;
      ext_mem[ext_mem_index(WGT_EXT_BASE) + row * 4 + 2] = 32'h0c0b0a09;
      ext_mem[ext_mem_index(WGT_EXT_BASE) + row * 4 + 3] = 32'h100f0e0d;
    end

    for (int out_idx = 0; out_idx < OUT_WORDS; out_idx++)
      ext_mem[ext_mem_index(OUT_EXT_BASE) + out_idx] = 32'hDEADBEEF;

    issue_cmd(OP_LOAD, ACT_EXT_BASE, ACT_WORDS, ACT_SP_BASE);
    dump_tile0_cmd_state("after activation issue");
    wait_for_done("activation load");
    dump_tile0_cmd_state("after activation done");

    for (int word = 0; word < 4; word++) begin
      if (dut.gen_tile[0].u_tile.u_sp.mem[ACT_SP_BASE + word] !== 32'h01010101)
        $fatal(1,
               "Activation scratchpad word %0d mismatch after mesh load: got %08h",
               word,
               dut.gen_tile[0].u_tile.u_sp.mem[ACT_SP_BASE + word]);
    end

    issue_cmd(OP_LOAD, WGT_EXT_BASE, WGT_WORDS, WGT_SP_BASE);
    dump_tile0_cmd_state("after weight issue");
    wait_for_done("weight load");
    dump_tile0_cmd_state("after weight done");

    if (dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 0] !== 32'h04030201)
      $fatal(1,
             "Weight scratchpad word 0 mismatch after mesh load: got %08h",
             dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 0]);
    if (dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 1] !== 32'h08070605)
      $fatal(1,
             "Weight scratchpad word 1 mismatch after mesh load: got %08h",
             dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 1]);
    if (dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 2] !== 32'h0c0b0a09)
      $fatal(1,
             "Weight scratchpad word 2 mismatch after mesh load: got %08h",
             dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 2]);
    if (dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 3] !== 32'h100f0e0d)
      $fatal(1,
             "Weight scratchpad word 3 mismatch after mesh load: got %08h",
             dut.gen_tile[0].u_tile.u_sp.mem[WGT_SP_BASE + 3]);

    issue_cmd(OP_COMPUTE, ACT_SP_BASE, WGT_SP_BASE, OUT_SP_BASE);
    wait_for_done("tile compute");

    issue_cmd(OP_STORE, OUT_EXT_BASE, OUT_WORDS, OUT_SP_BASE);
    wait_for_done("result store");

    for (int row = 0; row < N_ROWS; row++) begin
      for (int col = 0; col < N_COLS; col++) begin
        logic [31:0] got;
        got = ext_mem[ext_mem_index(OUT_EXT_BASE) + row * N_COLS + col];
        if (got !== 32'(col + 1)) begin
          $display("row %0d col %0d got %0d expected %0d", row, col, got, col + 1);
          $fatal(1, "Mesh path mismatch at row %0d col %0d", row, col);
        end
      end
    end

    $display("accel_tile_array_mesh_tb PASS: mesh + shared gateway preserve load/compute/store ordering");
    $finish;
  end

  initial begin
    #2000000;
    $fatal(1, "accel_tile_array_mesh_tb timeout");
  end

endmodule