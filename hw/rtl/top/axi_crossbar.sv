// ===========================================================================
// axi_crossbar.sv — AXI Crossbar: 2 Masters × 8 Slaves
// ===========================================================================
// Instantiates axi_addr_decoder (per-master) and axi_arbiter (per-slave)
// as standalone sub-modules, then wires the datapath MUX.
//
// Refactored from monolithic inline logic for testability and reuse.
// ===========================================================================

/* verilator lint_off PINCONNECTEMPTY */
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
  input  logic [NUM_MASTERS-1:0][7:0] m_awlen,

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
  input  logic [NUM_MASTERS-1:0][7:0] m_arlen,

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
  output logic [NUM_SLAVES-1:0][7:0] s_awlen,

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
  output logic [NUM_SLAVES-1:0][7:0] s_arlen,

  input  logic [NUM_SLAVES-1:0]    s_rvalid,
  output logic [NUM_SLAVES-1:0]    s_rready,
  input  logic [NUM_SLAVES-1:0][DATA_WIDTH-1:0] s_rdata,
  input  logic [NUM_SLAVES-1:0][1:0] s_rresp,
  input  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_rid,
  input  logic [NUM_SLAVES-1:0]    s_rlast
);

  // -----------------------------------------------------------------------
  // Constants
  // -----------------------------------------------------------------------
  localparam int unsigned MIDX_W = $clog2(NUM_MASTERS);

  // -----------------------------------------------------------------------
  // Helper: one-hot to binary index (local copy for use in datapath MUX)
  // -----------------------------------------------------------------------
  function automatic [MIDX_W-1:0] onehot_to_idx(input [NUM_MASTERS-1:0] oh);
    onehot_to_idx = '0;
    for (int i = 0; i < NUM_MASTERS; i++)
      if (oh[i]) onehot_to_idx = MIDX_W'(i);
  endfunction

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
        s_awlen[si]   = m_awlen[aw_grant_idx[si]];
      end

      // Write data channel — follows AW grant OR latched resp owner
      //
      // The AW arbiter releases the grant after the AW handshake completes,
      // but the W data phase may still be pending.  Use the combination of
      // live aw_grant (before AW handshake) and aw_resp_owner (after AW
      // handshake, until bvalid) so the W channel stays routed for the
      // entire write transaction.
      logic [NUM_MASTERS-1:0] w_owner;
      logic                   w_owner_valid;
      logic [MIDX_W-1:0]      w_owner_idx;

      always_comb begin
        // Gate live grant with s_awready: W only routes when AW is being
        // accepted THIS cycle (simultaneous AW+W) OR was already accepted
        // (aw_resp_active).  Without the s_awready gate, m_wready fires for
        // master M even though the slave hasn't yet accepted AW, causing W
        // data to be consumed before the address — an AXI protocol violation.
        w_owner       = (aw_grant[si] & {NUM_MASTERS{s_awready[si]}}) |
                        (aw_resp_owner[si] & {NUM_MASTERS{aw_resp_active[si]}});
        w_owner_valid = |w_owner;
        w_owner_idx   = onehot_to_idx(w_owner);
      end

      always_comb begin
        s_wvalid[si] = 1'b0;
        s_wdata[si]  = '0;
        s_wstrb[si]  = '0;
        s_wlast[si]  = 1'b0;
        if (w_owner_valid) begin
          s_wvalid[si] = m_wvalid[w_owner_idx];
          s_wdata[si]  = m_wdata[w_owner_idx];
          s_wstrb[si]  = m_wstrb[w_owner_idx];
          s_wlast[si]  = m_wlast[w_owner_idx];
        end
      end

      // Read address channel
      always_comb begin
        s_arvalid[si] = |ar_grant[si];
        s_araddr[si]  = m_araddr[ar_grant_idx[si]];
        s_arid[si]    = m_arid[ar_grant_idx[si]];
        s_arlen[si]   = m_arlen[ar_grant_idx[si]];
      end

      // Slave-side ready (forward from the owning master for response channels)
      always_comb begin
        s_bready[si] = 1'b0;
        s_rready[si] = 1'b0;
        for (int m = 0; m < NUM_MASTERS; m++) begin
          s_bready[si] |= aw_resp_owner[si][m] & aw_resp_active[si] & m_bready[m];
          s_rready[si] |= ar_resp_owner[si][m] & ar_resp_active[si] & m_rready[m];
        end
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // 4. Ready back-propagation — master gets ready only from its target slave
  //
  // OR-accumulate idiom: since at most one slave can be granted per master
  // the |= produces the same result as priority-if but synthesises as a
  // balanced OR tree (O(log N) vs O(N) priority chain).
  // -----------------------------------------------------------------------
  always_comb begin
    m_awready = '0;
    m_wready  = '0;
    m_arready = '0;
    for (int m = 0; m < NUM_MASTERS; m++) begin
      for (int s = 0; s < NUM_SLAVES; s++) begin
        m_awready[m] |= (m_aw_target[m][s] & aw_grant[s][m]) ? s_awready[s] : 1'b0;
        m_wready[m]  |= ((aw_grant[s][m] & s_awready[s]) | (aw_resp_owner[s][m] & aw_resp_active[s])) ? s_wready[s]  : 1'b0;
        m_arready[m] |= (m_ar_target[m][s] & ar_grant[s][m]) ? s_arready[s] : 1'b0;
      end
    end
  end

  // -----------------------------------------------------------------------
  // 4b. Outstanding-transaction tracking
  //
  // The arbiter releases the grant after the AR/AW handshake, but the
  // slave's R/B response arrives later.  We latch the winning master
  // index at handshake time and hold it until the response channel
  // completes (rlast for reads, bvalid for writes).
  // -----------------------------------------------------------------------
  logic [NUM_MASTERS-1:0] aw_resp_owner [NUM_SLAVES]; // one-hot
  logic [NUM_MASTERS-1:0] ar_resp_owner [NUM_SLAVES]; // one-hot
  logic                   aw_resp_active [NUM_SLAVES];
  logic                   ar_resp_active [NUM_SLAVES];

  generate
    for (si = 0; si < NUM_SLAVES; si++) begin : gen_resp_track
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          aw_resp_owner[si]  <= '0;
          ar_resp_owner[si]  <= '0;
          aw_resp_active[si] <= 1'b0;
          ar_resp_active[si] <= 1'b0;
        end else begin
          // Latch write owner on AW handshake
          if (s_awvalid[si] && s_awready[si]) begin
            aw_resp_owner[si]  <= aw_grant[si];
            aw_resp_active[si] <= 1'b1;
          end else if (aw_resp_active[si] && s_bvalid[si] && s_bready[si]) begin
            aw_resp_active[si] <= 1'b0;
          end

          // Latch read owner on AR handshake
          if (s_arvalid[si] && s_arready[si]) begin
            ar_resp_owner[si]  <= ar_grant[si];
            ar_resp_active[si] <= 1'b1;
          end else if (ar_resp_active[si] && s_rvalid[si] && s_rready[si] && s_rlast[si]) begin
            ar_resp_active[si] <= 1'b0;
          end
        end
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // 5. Response routing — slave responses back to masters
  //
  // Uses resp_owner (latched at handshake time) instead of live grant,
  // so R/B responses reach the originating master even after the arbiter
  // has released the grant.
  // -----------------------------------------------------------------------
  generate
    for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_responses
      always_comb begin
        m_bvalid[mi] = 1'b0;
        m_bresp[mi]  = '0;
        m_bid[mi]    = '0;
        m_rvalid[mi] = 1'b0;
        m_rdata[mi]  = '0;
        m_rresp[mi]  = '0;
        m_rid[mi]    = '0;
        m_rlast[mi]  = 1'b0;

        for (int s = 0; s < NUM_SLAVES; s++) begin
          // Write response — use latched owner
          m_bvalid[mi] |= s_bvalid[s] & aw_resp_owner[s][mi] & aw_resp_active[s];
          m_bresp[mi]  |= (aw_resp_owner[s][mi] & aw_resp_active[s]) ? s_bresp[s] : 2'b0;
          m_bid[mi]    |= (aw_resp_owner[s][mi] & aw_resp_active[s]) ? s_bid[s]   : '0;
          // Read response — use latched owner
          m_rvalid[mi] |= s_rvalid[s] & ar_resp_owner[s][mi] & ar_resp_active[s];
          m_rdata[mi]  |= (ar_resp_owner[s][mi] & ar_resp_active[s]) ? s_rdata[s] : '0;
          m_rresp[mi]  |= (ar_resp_owner[s][mi] & ar_resp_active[s]) ? s_rresp[s] : 2'b0;
          m_rid[mi]    |= (ar_resp_owner[s][mi] & ar_resp_active[s]) ? s_rid[s]   : '0;
          m_rlast[mi]  |= s_rlast[s] & ar_resp_owner[s][mi] & ar_resp_active[s];
        end
      end
    end
  endgenerate

endmodule : axi_crossbar
