`timescale 1ns/1ps

import noc_pkg::*;

module accel_tile_gateway_tb;

  localparam int N_ROWS    = 16;
  localparam int N_COLS    = 16;
  localparam int DATA_W    = 8;
  localparam int ACC_W     = 32;
  localparam int SP_DEPTH  = 1024;
  localparam int SP_DATA_W = 32;
  localparam int NUM_VCS   = noc_pkg::NUM_VCS;
  localparam int MEM_WORDS = 2048;
  localparam int REQ_FIFO_DEPTH = 4;
  localparam int REQ_FIFO_PTR_W = $clog2(REQ_FIFO_DEPTH);

  localparam logic [7:0] OP_LOAD    = 8'h01;
  localparam logic [7:0] OP_STORE   = 8'h02;
  localparam logic [7:0] OP_COMPUTE = 8'h03;

  localparam int ACT_SP_BASE = 0;
  localparam int WGT_SP_BASE = 128;
  localparam int OUT_SP_BASE = 256;

  localparam logic [31:0] EXT_MEM_BASE = 32'h4000_0000;
  localparam logic [31:0] ACT_EXT_BASE = EXT_MEM_BASE + 32'h0000;
  localparam logic [31:0] WGT_EXT_BASE = EXT_MEM_BASE + 32'h0400;
  localparam logic [31:0] OUT_EXT_BASE = EXT_MEM_BASE + 32'h0800;

  localparam int ACT_WORDS = (N_ROWS * N_ROWS * DATA_W) / SP_DATA_W;
  localparam int WGT_WORDS = (N_ROWS * N_COLS * DATA_W) / SP_DATA_W;
  localparam int OUT_WORDS = N_ROWS * N_COLS;

  logic               clk;
  logic               rst_n;

  flit_t              tile_noc_flit_out;
  logic               tile_noc_valid_out;
  logic [NUM_VCS-1:0] tile_noc_credit_in;
  flit_t              tile_noc_flit_in;
  logic               tile_noc_valid_in;
  logic [NUM_VCS-1:0] tile_noc_credit_out;

  logic [31:0]        csr_wdata;
  logic [7:0]         csr_addr;
  logic               csr_wen;
  logic [31:0]        csr_rdata;
  logic               barrier_req;
  logic               barrier_done;
  logic               tile_busy;
  logic               tile_done;

  logic               gw_noc_credit_out;
  flit_t              gw_noc_flit_out;
  logic               gw_noc_valid_out;
  logic               gw_noc_credit_in;

  logic [3:0]         m_axi_arid;
  logic [31:0]        m_axi_araddr;
  logic [7:0]         m_axi_arlen;
  logic [2:0]         m_axi_arsize;
  logic [1:0]         m_axi_arburst;
  logic               m_axi_arvalid;
  logic               m_axi_arready;

  logic [3:0]         m_axi_rid;
  logic [31:0]        m_axi_rdata;
  logic [1:0]         m_axi_rresp;
  logic               m_axi_rlast;
  logic               m_axi_rvalid;
  logic               m_axi_rready;

  logic [3:0]         m_axi_awid;
  logic [31:0]        m_axi_awaddr;
  logic [7:0]         m_axi_awlen;
  logic [2:0]         m_axi_awsize;
  logic [1:0]         m_axi_awburst;
  logic               m_axi_awvalid;
  logic               m_axi_awready;

  logic [31:0]        m_axi_wdata;
  logic [3:0]         m_axi_wstrb;
  logic               m_axi_wlast;
  logic               m_axi_wvalid;
  logic               m_axi_wready;

  logic [3:0]         m_axi_bid;
  logic [1:0]         m_axi_bresp;
  logic               m_axi_bvalid;
  logic               m_axi_bready;

  flit_t              req_fifo [0:REQ_FIFO_DEPTH-1];
  logic [REQ_FIFO_PTR_W-1:0] req_fifo_rd_ptr;
  logic [REQ_FIFO_PTR_W-1:0] req_fifo_wr_ptr;
  logic [$clog2(REQ_FIFO_DEPTH+1)-1:0] req_fifo_count;
  logic               req_fifo_push;
  logic               req_fifo_pop;

  logic [31:0]        ext_mem [0:MEM_WORDS-1];

  logic               rd_active;
  logic [31:0]        rd_addr_reg;
  logic [7:0]         rd_len_reg;
  logic [7:0]         rd_idx;
  logic [3:0]         rd_id_reg;

  logic               wr_active;
  logic [31:0]        wr_addr_reg;
  logic [7:0]         wr_len_reg;
  logic [7:0]         wr_idx;
  logic [3:0]         wr_id_reg;
  logic               b_pending;

  function automatic int ext_mem_index(input logic [31:0] addr);
    begin
      ext_mem_index = int'((addr - EXT_MEM_BASE) >> 2);
    end
  endfunction

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

  task automatic issue_cmd(
    input logic [7:0] opcode,
    input logic [31:0] arg0,
    input logic [31:0] arg1,
    input logic [31:0] arg2
  );
    begin
      csr_write(8'h04, arg0);
      csr_write(8'h08, arg1);
      csr_write(8'h0C, arg2);
      csr_write(8'h00, 32'(opcode));
    end
  endtask

  task automatic wait_for_done(input string label);
    begin
      repeat (20000) begin
        @(posedge clk);
        if (tile_done)
          return;
      end
      $fatal(1, "%s timed out waiting for tile_done", label);
    end
  endtask

  initial clk = 1'b0;
  always #5 clk = ~clk;

  assign tile_noc_flit_in  = gw_noc_flit_out;
  assign tile_noc_valid_in = gw_noc_valid_out;
  assign gw_noc_credit_in  = |tile_noc_credit_out;
  assign req_fifo_push     = tile_noc_valid_out;
  assign req_fifo_pop      = (req_fifo_count != '0) && gw_noc_credit_out;

  always_comb begin
    tile_noc_credit_in = '0;
    if (req_fifo_pop)
      tile_noc_credit_in[req_fifo[req_fifo_rd_ptr].vc_id] = 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      req_fifo_rd_ptr <= '0;
      req_fifo_wr_ptr <= '0;
      req_fifo_count  <= '0;
    end else begin
      if (req_fifo_push && !req_fifo_pop && (req_fifo_count == $clog2(REQ_FIFO_DEPTH+1)'(REQ_FIFO_DEPTH)))
        $fatal(1, "Request FIFO overflow between tile and gateway");

      case ({req_fifo_push, req_fifo_pop})
        2'b10: begin
          req_fifo[req_fifo_wr_ptr] <= tile_noc_flit_out;
          req_fifo_wr_ptr           <= req_fifo_wr_ptr + 1'b1;
          req_fifo_count            <= req_fifo_count + 1'b1;
        end

        2'b01: begin
          req_fifo_rd_ptr <= req_fifo_rd_ptr + 1'b1;
          req_fifo_count  <= req_fifo_count - 1'b1;
        end

        2'b11: begin
          req_fifo[req_fifo_wr_ptr] <= tile_noc_flit_out;
          req_fifo_wr_ptr           <= req_fifo_wr_ptr + 1'b1;
          req_fifo_rd_ptr           <= req_fifo_rd_ptr + 1'b1;
        end

        default: ;
      endcase
    end
  end

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
    .noc_flit_out   (tile_noc_flit_out),
    .noc_valid_out  (tile_noc_valid_out),
    .noc_credit_in  (tile_noc_credit_in),
    .noc_flit_in    (tile_noc_flit_in),
    .noc_valid_in   (tile_noc_valid_in),
    .noc_credit_out (tile_noc_credit_out),
    .csr_wdata      (csr_wdata),
    .csr_addr       (csr_addr),
    .csr_wen        (csr_wen),
    .csr_rdata      (csr_rdata),
    .barrier_req    (barrier_req),
    .barrier_done   (barrier_done),
    .tile_busy      (tile_busy),
    .tile_done      (tile_done)
  );

  tile_dma_gateway #(
    .AXI_ADDR_W (32),
    .AXI_DATA_W (32),
    .AXI_ID_W   (4),
    .MAX_BURST  (16)
  ) u_gateway (
    .clk          (clk),
    .rst_n        (rst_n),
    .noc_flit_in  (req_fifo[req_fifo_rd_ptr]),
    .noc_valid_in (req_fifo_count != '0),
    .noc_credit_out(gw_noc_credit_out),
    .noc_flit_out (gw_noc_flit_out),
    .noc_valid_out(gw_noc_valid_out),
    .noc_credit_in(gw_noc_credit_in),
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
      rd_active  <= 1'b0;
      rd_addr_reg <= '0;
      rd_len_reg <= '0;
      rd_idx     <= '0;
      rd_id_reg  <= '0;
      wr_active  <= 1'b0;
      wr_addr_reg <= '0;
      wr_len_reg <= '0;
      wr_idx     <= '0;
      wr_id_reg  <= '0;
      b_pending  <= 1'b0;
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

  initial begin
    csr_wdata    = '0;
    csr_addr     = '0;
    csr_wen      = 1'b0;
    barrier_done = 1'b0;
    rst_n        = 1'b0;

    for (int idx = 0; idx < MEM_WORDS; idx++)
      ext_mem[idx] = '0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    for (int vec = 0; vec < N_ROWS; vec++) begin
      for (int word = 0; word < 4; word++) begin
        ext_mem[ext_mem_index(ACT_EXT_BASE) + (vec * 4) + word] = 32'h01010101;
        ext_mem[ext_mem_index(WGT_EXT_BASE) + (vec * 4) + word] = 32'h01010101;
      end
    end

    for (int out_idx = 0; out_idx < OUT_WORDS; out_idx++)
      ext_mem[ext_mem_index(OUT_EXT_BASE) + out_idx] = 32'hDEADBEEF;

    issue_cmd(OP_LOAD, ACT_EXT_BASE, ACT_WORDS, ACT_SP_BASE);
    wait_for_done("activation load");

    issue_cmd(OP_LOAD, WGT_EXT_BASE, WGT_WORDS, WGT_SP_BASE);
    wait_for_done("weight load");

    issue_cmd(OP_COMPUTE, ACT_SP_BASE, WGT_SP_BASE, OUT_SP_BASE);
    wait_for_done("tile compute");

    issue_cmd(OP_STORE, OUT_EXT_BASE, OUT_WORDS, OUT_SP_BASE);
    wait_for_done("result store");

    for (int out_idx = 0; out_idx < OUT_WORDS; out_idx++) begin
      if (ext_mem[ext_mem_index(OUT_EXT_BASE) + out_idx] !== 32'd16) begin
        $fatal(1,
               "External memory output word %0d mismatch: got %0d expected %0d",
               out_idx,
               ext_mem[ext_mem_index(OUT_EXT_BASE) + out_idx],
               32'd16);
      end
    end

    $display("accel_tile_gateway_tb PASS: tile load/compute/store completed through DMA gateway");
    $finish;
  end

  initial begin
    #300000;
    $fatal(1, "accel_tile_gateway_tb timeout");
  end

endmodule
