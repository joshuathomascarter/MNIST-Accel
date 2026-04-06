`timescale 1ns/1ps

import noc_pkg::*;

module gateway_dram_read_order_tb;

  logic        clk;
  logic        rst_n;

  flit_t       noc_flit_in;
  logic        noc_valid_in;
  logic        noc_credit_out;
  flit_t       noc_flit_out;
  logic        noc_valid_out;
  logic        noc_credit_in;

  logic [3:0]  m_axi_arid;
  logic [31:0] m_axi_araddr;
  logic [7:0]  m_axi_arlen;
  logic [2:0]  m_axi_arsize;
  logic [1:0]  m_axi_arburst;
  logic        m_axi_arvalid;
  logic        m_axi_arready;
  logic [3:0]  m_axi_rid;
  logic [31:0] m_axi_rdata;
  logic [1:0]  m_axi_rresp;
  logic        m_axi_rlast;
  logic        m_axi_rvalid;
  logic        m_axi_rready;
  logic [3:0]  m_axi_awid;
  logic [31:0] m_axi_awaddr;
  logic [7:0]  m_axi_awlen;
  logic [2:0]  m_axi_awsize;
  logic [1:0]  m_axi_awburst;
  logic        m_axi_awvalid;
  logic        m_axi_awready;
  logic [31:0] m_axi_wdata;
  logic [3:0]  m_axi_wstrb;
  logic        m_axi_wlast;
  logic        m_axi_wvalid;
  logic        m_axi_wready;
  logic [3:0]  m_axi_bid;
  logic [1:0]  m_axi_bresp;
  logic        m_axi_bvalid;
  logic        m_axi_bready;

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
  logic        ctrl_busy;

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic expect_flit(
    input flit_type_e expected_type,
    input logic [31:0] expected_data,
    input string label
  );
    begin
      while (1) begin
        @(posedge clk);
        #1;
        if (noc_valid_out) begin
          if (noc_flit_out.flit_type !== expected_type)
            $fatal(1, "%s flit_type mismatch: got %0d expected %0d", label, noc_flit_out.flit_type, expected_type);
          if (noc_flit_out.payload[31:0] !== expected_data)
            $fatal(1, "%s data mismatch: %08h", label, noc_flit_out.payload[31:0]);
          return;
        end
      end
    end
  endtask

  tile_dma_gateway #(
    .AXI_ADDR_W (32),
    .AXI_DATA_W (32),
    .AXI_ID_W   (4),
    .MAX_BURST  (16)
  ) dut_gw (
    .clk          (clk),
    .rst_n        (rst_n),
    .noc_flit_in  (noc_flit_in),
    .noc_valid_in (noc_valid_in),
    .noc_credit_out(noc_credit_out),
    .noc_flit_out (noc_flit_out),
    .noc_valid_out(noc_valid_out),
    .noc_credit_in(noc_credit_in),
    .m_axi_arid   (m_axi_arid),
    .m_axi_araddr (m_axi_araddr),
    .m_axi_arlen  (m_axi_arlen),
    .m_axi_arsize (m_axi_arsize),
    .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_rid    (m_axi_rid),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rlast  (m_axi_rlast),
    .m_axi_rvalid (m_axi_rvalid),
    .m_axi_rready (m_axi_rready),
    .m_axi_awid   (m_axi_awid),
    .m_axi_awaddr (m_axi_awaddr),
    .m_axi_awlen  (m_axi_awlen),
    .m_axi_awsize (m_axi_awsize),
    .m_axi_awburst(m_axi_awburst),
    .m_axi_awvalid(m_axi_awvalid),
    .m_axi_awready(m_axi_awready),
    .m_axi_wdata  (m_axi_wdata),
    .m_axi_wstrb  (m_axi_wstrb),
    .m_axi_wlast  (m_axi_wlast),
    .m_axi_wvalid (m_axi_wvalid),
    .m_axi_wready (m_axi_wready),
    .m_axi_bid    (m_axi_bid),
    .m_axi_bresp  (m_axi_bresp),
    .m_axi_bvalid (m_axi_bvalid),
    .m_axi_bready (m_axi_bready)
  );

  dram_ctrl_top #(
    .AXI_ADDR_W  (32),
    .AXI_DATA_W  (32),
    .AXI_ID_W    (4),
    .NUM_BANKS   (8),
    .ROW_BITS    (14),
    .COL_BITS    (10),
    .BANK_BITS   (3),
    .QUEUE_DEPTH (16),
    .ADDR_MODE   (0)
  ) dut_dram_ctrl (
    .clk                  (clk),
    .rst_n                (rst_n),
    .s_axi_awvalid        (m_axi_awvalid),
    .s_axi_awready        (m_axi_awready),
    .s_axi_awaddr         (m_axi_awaddr),
    .s_axi_awid           (m_axi_awid),
    .s_axi_awlen          (m_axi_awlen),
    .s_axi_awsize         (m_axi_awsize),
    .s_axi_wvalid         (m_axi_wvalid),
    .s_axi_wready         (m_axi_wready),
    .s_axi_wdata          (m_axi_wdata),
    .s_axi_wstrb          (m_axi_wstrb),
    .s_axi_wlast          (m_axi_wlast),
    .s_axi_bvalid         (m_axi_bvalid),
    .s_axi_bready         (m_axi_bready),
    .s_axi_bresp          (m_axi_bresp),
    .s_axi_bid            (m_axi_bid),
    .s_axi_arvalid        (m_axi_arvalid),
    .s_axi_arready        (m_axi_arready),
    .s_axi_araddr         (m_axi_araddr),
    .s_axi_arid           (m_axi_arid),
    .s_axi_arlen          (m_axi_arlen),
    .s_axi_arsize         (m_axi_arsize),
    .s_axi_rvalid         (m_axi_rvalid),
    .s_axi_rready         (m_axi_rready),
    .s_axi_rdata          (m_axi_rdata),
    .s_axi_rresp          (m_axi_rresp),
    .s_axi_rid            (m_axi_rid),
    .s_axi_rlast          (m_axi_rlast),
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
    .ctrl_busy            (ctrl_busy)
  );

  dram_phy_simple_mem #(
    .NUM_BANKS (8),
    .ROW_BITS  (14),
    .COL_BITS  (10),
    .DATA_W    (32),
    .MEM_WORDS (1024),
    .INIT_FILE ("")
  ) dut_mem (
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

  initial begin
    noc_flit_in   = '0;
    noc_valid_in  = 1'b0;
    noc_credit_in = 1'b0;
    rst_n         = 1'b0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    dut_mem.mem[0] = 32'h04030201;
    dut_mem.mem[1] = 32'h08070605;
    dut_mem.mem[2] = 32'h0c0b0a09;
    dut_mem.mem[3] = 32'h100f0e0d;

    @(posedge clk);
    noc_flit_in.flit_type = FLIT_HEADTAIL;
    noc_flit_in.src_id    = 4'd0;
    noc_flit_in.dst_id    = 4'd15;
    noc_flit_in.vc_id     = '0;
    noc_flit_in.msg_type  = MSG_READ_REQ;
    noc_flit_in.payload   = {32'h00000000, 4'd3, 4'd0, 8'h00};
    noc_valid_in          = 1'b1;
    @(posedge clk);
    noc_valid_in          = 1'b0;
    noc_flit_in           = '0;

    expect_flit(FLIT_HEAD, 32'h04030201, "head");
    expect_flit(FLIT_BODY, 32'h08070605, "body1");
    expect_flit(FLIT_BODY, 32'h0c0b0a09, "body2");
    expect_flit(FLIT_TAIL, 32'h100f0e0d, "tail");

    $display("gateway_dram_read_order_tb PASS: gateway+DRAM preserve word byte order");
    $finish;
  end

  initial begin
    #200000;
    $fatal(1, "gateway_dram_read_order_tb timeout");
  end

endmodule