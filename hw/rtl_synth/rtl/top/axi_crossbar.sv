// ===========================================================================
// axi_crossbar.sv — AXI Crossbar: 2 Masters × 8 Slaves
// ===========================================================================
// Instantiates axi_addr_decoder (per-master) and axi_arbiter (per-slave)
// as standalone sub-modules, then wires the datapath MUX.
//
// Refactored from monolithic inline logic for testability and reuse.
// ===========================================================================

module axi_crossbar #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned ID_WIDTH = 4,
  parameter int unsigned NUM_MASTERS = 2,
  parameter int unsigned NUM_SLAVES = 8
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // Master ports (input from masters)
  input  logic [NUM_MASTERS-1:0]   m_awvalid,
  output logic [NUM_MASTERS-1:0]   m_awready,
  input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] m_awaddr,
  input  logic [NUM_MASTERS-1:0][ID_WIDTH-1:0]   m_awid,

  input  logic [NUM_MASTERS-1:0]   m_wvalid,
  output logic [NUM_MASTERS-1:0]   m_wready,
  input  logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] m_wdata,
  input  logic [NUM_MASTERS-1:0][DATA_WIDTH/8-1:0] m_wstrb,
  input  logic [NUM_MASTERS-1:0]   m_wlast,

  output logic [NUM_MASTERS-1:0]   m_bvalid,
  input  logic [NUM_MASTERS-1:0]   m_bready,
  output logic [NUM_MASTERS-1:0][1:0] m_bresp,
  output logic [NUM_MASTERS-1:0][ID_WIDTH-1:0] m_bid,

  input  logic [NUM_MASTERS-1:0]   m_arvalid,
  output logic [NUM_MASTERS-1:0]   m_arready,
  input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] m_araddr,
  input  logic [NUM_MASTERS-1:0][ID_WIDTH-1:0]   m_arid,

  output logic [NUM_MASTERS-1:0]   m_rvalid,
  input  logic [NUM_MASTERS-1:0]   m_rready,
  output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] m_rdata,
  output logic [NUM_MASTERS-1:0][1:0] m_rresp,
  output logic [NUM_MASTERS-1:0][ID_WIDTH-1:0] m_rid,
  output logic [NUM_MASTERS-1:0]   m_rlast,

  // Slave ports (output to slaves)
  output logic [NUM_SLAVES-1:0]    s_awvalid,
  input  logic [NUM_SLAVES-1:0]    s_awready,
  output logic [NUM_SLAVES-1:0][ADDR_WIDTH-1:0] s_awaddr,
  output logic [NUM_SLAVES-1:0][ID_WIDTH-1:0]   s_awid,

  output logic [NUM_SLAVES-1:0]    s_wvalid,
  input  logic [NUM_SLAVES-1:0]    s_wready,
  output logic [NUM_SLAVES-1:0][DATA_WIDTH-1:0] s_wdata,
  output logic [NUM_SLAVES-1:0][DATA_WIDTH/8-1:0] s_wstrb,
  output logic [NUM_SLAVES-1:0]    s_wlast,

  input  logic [NUM_SLAVES-1:0]    s_bvalid,
  output logic [NUM_SLAVES-1:0]    s_bready,
  input  logic [NUM_SLAVES-1:0][1:0] s_bresp,
  input  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_bid,

  output logic [NUM_SLAVES-1:0]    s_arvalid,
  input  logic [NUM_SLAVES-1:0]    s_arready,
  output logic [NUM_SLAVES-1:0][ADDR_WIDTH-1:0] s_araddr,
  output logic [NUM_SLAVES-1:0][ID_WIDTH-1:0]   s_arid,

  input  logic [NUM_SLAVES-1:0]    s_rvalid,
  output logic [NUM_SLAVES-1:0]    s_rready,
  input  logic [NUM_SLAVES-1:0][DATA_WIDTH-1:0] s_rdata,
  input  logic [NUM_SLAVES-1:0][1:0] s_rresp,
  input  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_rid,
  input  logic [NUM_SLAVES-1:0]    s_rlast
);

  // -----------------------------------------------------------------------
  // 1. Address Decode — one decoder per master, per channel (AW and AR)
  // -----------------------------------------------------------------------
  logic [NUM_SLAVES-1:0] m_aw_target [NUM_MASTERS];
  logic [NUM_SLAVES-1:0] m_ar_target [NUM_MASTERS];

  genvar mi;
  generate
    for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_decoders
      axi_addr_decoder #(
        .ADDR_WIDTH(ADDR_WIDTH), .NUM_SLAVES(NUM_SLAVES)
      ) u_aw_dec (
        .addr        (m_awaddr[mi]),
        .slave_sel   (m_aw_target[mi]),
        .decode_error(/* unused — error sink is slave 7 */)
      );

      axi_addr_decoder #(
        .ADDR_WIDTH(ADDR_WIDTH), .NUM_SLAVES(NUM_SLAVES)
      ) u_ar_dec (
        .addr        (m_araddr[mi]),
        .slave_sel   (m_ar_target[mi]),
        .decode_error()
      );
    end
  endgenerate

  // -----------------------------------------------------------------------
  // 2. Per-slave Arbitration — one arbiter per slave, per channel
  // -----------------------------------------------------------------------
  logic [NUM_MASTERS-1:0] aw_grant [NUM_SLAVES];
  logic [NUM_MASTERS-1:0] ar_grant [NUM_SLAVES];
  localparam int unsigned MIDX_W = $clog2(NUM_MASTERS);
  logic [MIDX_W-1:0]     aw_grant_idx [NUM_SLAVES];
  logic [MIDX_W-1:0]     ar_grant_idx [NUM_SLAVES];

  genvar si;
  generate
    for (si = 0; si < NUM_SLAVES; si++) begin : gen_arbiters
      // Build request vectors: which masters target this slave?
      logic [NUM_MASTERS-1:0] aw_req, ar_req;
      for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_req
        assign aw_req[mi] = m_aw_target[mi][si] && m_awvalid[mi];
        assign ar_req[mi] = m_ar_target[mi][si] && m_arvalid[mi];
      end

      axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_aw_arb (
        .clk            (clk),
        .rst_n          (rst_n),
        .req            (aw_req),
        .handshake_done (s_awvalid[si] && s_awready[si]),
        .grant          (aw_grant[si]),
        .grant_idx      (aw_grant_idx[si])
      );

      axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_ar_arb (
        .clk            (clk),
        .rst_n          (rst_n),
        .req            (ar_req),
        .handshake_done (s_arvalid[si] && s_arready[si]),
        .grant          (ar_grant[si]),
        .grant_idx      (ar_grant_idx[si])
      );
    end
  endgenerate

  // -----------------------------------------------------------------------
  // 3. Datapath MUX — forward master→slave based on arbiter grant
  // -----------------------------------------------------------------------
  generate
    for (si = 0; si < NUM_SLAVES; si++) begin : gen_slave_mux
      // Write address channel
      always_comb begin
        s_awvalid[si] = |aw_grant[si];
        s_awaddr[si]  = m_awaddr[aw_grant_idx[si]];
        s_awid[si]    = m_awid[aw_grant_idx[si]];
      end

      // Write data channel — follows the AW grant
      always_comb begin
        s_wvalid[si] = 1'b0;
        s_wdata[si]  = '0;
        s_wstrb[si]  = '0;
        s_wlast[si]  = 1'b0;
        if (|aw_grant[si]) begin
          s_wvalid[si] = m_wvalid[aw_grant_idx[si]];
          s_wdata[si]  = m_wdata[aw_grant_idx[si]];
          s_wstrb[si]  = m_wstrb[aw_grant_idx[si]];
          s_wlast[si]  = m_wlast[aw_grant_idx[si]];
        end
      end

      // Read address channel
      always_comb begin
        s_arvalid[si] = |ar_grant[si];
        s_araddr[si]  = m_araddr[ar_grant_idx[si]];
        s_arid[si]    = m_arid[ar_grant_idx[si]];
      end

      // Slave-side ready (forward from the granted master)
      assign s_bready[si] = |aw_grant[si] ? m_bready[aw_grant_idx[si]] : m_bready[0];
      assign s_rready[si] = |ar_grant[si] ? m_rready[ar_grant_idx[si]] : m_rready[0];
    end
  endgenerate

  // -----------------------------------------------------------------------
  // 4. Ready back-propagation — master gets ready only from its target slave
  // -----------------------------------------------------------------------
  always_comb begin
    m_awready = '0;
    m_wready  = '0;
    m_arready = '0;
    for (int m = 0; m < NUM_MASTERS; m++) begin
      for (int s = 0; s < NUM_SLAVES; s++) begin
        if (m_aw_target[m][s] && aw_grant[s][m])
          m_awready[m] = s_awready[s];
        if (m_aw_target[m][s] && aw_grant[s][m])
          m_wready[m] = s_wready[s];
        if (m_ar_target[m][s] && ar_grant[s][m])
          m_arready[m] = s_arready[s];
      end
    end
  end

  // -----------------------------------------------------------------------
  // 5. Response routing — slave responses back to masters
  // -----------------------------------------------------------------------
  generate
    for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_responses
      always_comb begin
        m_bvalid[mi] = 1'b0;
        m_bresp[mi]  = 2'b00;
        m_bid[mi]    = '0;
        m_rvalid[mi] = 1'b0;
        m_rdata[mi]  = '0;
        m_rresp[mi]  = 2'b00;
        m_rid[mi]    = '0;
        m_rlast[mi]  = 1'b0;

        for (int s = 0; s < NUM_SLAVES; s++) begin
          if (s_bvalid[s] && aw_grant[s][mi]) begin
            m_bvalid[mi] = 1'b1;
            m_bresp[mi]  = s_bresp[s];
            m_bid[mi]    = s_bid[s];
          end
          if (s_rvalid[s] && ar_grant[s][mi]) begin
            m_rvalid[mi] = 1'b1;
            m_rdata[mi]  = s_rdata[s];
            m_rresp[mi]  = s_rresp[s];
            m_rid[mi]    = s_rid[s];
            m_rlast[mi]  = s_rlast[s];
          end
        end
      end
    end
  endgenerate

endmodule : axi_crossbar
