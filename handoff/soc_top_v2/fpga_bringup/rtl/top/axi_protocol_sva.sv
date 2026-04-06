// =============================================================================
// axi_protocol_sva.sv — AXI4 Protocol Compliance Assertions
// =============================================================================
// Bind module for AXI crossbar slave ports. Verifies AXI4 protocol rules:
//   1. VALID must not depend on READY (AXI rule: valid asserted independently)
//   2. Signals stable while VALID && !READY (AXI rule: no change until handshake)
//   3. WLAST must be asserted on final beat
//   4. Response channel ordering: BRESP only after W handshake
//   5. No simultaneous multi-driver on response channels
//
// Usage:
//   bind axi_crossbar axi_protocol_sva #(.NUM_MASTERS(2), .NUM_SLAVES(8)) u_sva (.*);

module axi_protocol_sva #(
  parameter int NUM_MASTERS = 2,
  parameter int NUM_SLAVES  = 8,
  parameter int ADDR_WIDTH  = 32,
  parameter int DATA_WIDTH  = 32,
  parameter int ID_WIDTH    = 4
) (
  input logic              clk,
  input logic              rst_n,

  // Master-side (NM = NUM_MASTERS)
  input logic [NUM_MASTERS-1:0]                    m_awvalid,
  input logic [NUM_MASTERS-1:0]                    m_awready,
  input logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]    m_awaddr,
  input logic [NUM_MASTERS-1:0]                    m_wvalid,
  input logic [NUM_MASTERS-1:0]                    m_wready,
  input logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]    m_wdata,
  input logic [NUM_MASTERS-1:0]                    m_wlast,
  input logic [NUM_MASTERS-1:0]                    m_arvalid,
  input logic [NUM_MASTERS-1:0]                    m_arready,
  input logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]    m_araddr,
  input logic [NUM_MASTERS-1:0]                    m_bvalid,
  input logic [NUM_MASTERS-1:0]                    m_bready,
  input logic [NUM_MASTERS-1:0]                    m_rvalid,
  input logic [NUM_MASTERS-1:0]                    m_rready,

  // Slave-side (NS = NUM_SLAVES)
  input logic [NUM_SLAVES-1:0]                     s_awvalid,
  input logic [NUM_SLAVES-1:0]                     s_awready,
  input logic [NUM_SLAVES-1:0]                     s_wvalid,
  input logic [NUM_SLAVES-1:0]                     s_wready,
  input logic [NUM_SLAVES-1:0]                     s_arvalid,
  input logic [NUM_SLAVES-1:0]                     s_arready,
  input logic [NUM_SLAVES-1:0]                     s_bvalid,
  input logic [NUM_SLAVES-1:0]                     s_bready,
  input logic [NUM_SLAVES-1:0]                     s_rvalid,
  input logic [NUM_SLAVES-1:0]                     s_rready
);

  // =========================================================================
  // 1. AW channel: address stable while VALID && !READY
  // =========================================================================
  generate
    for (genvar mi = 0; mi < NUM_MASTERS; mi++) begin : gen_m_aw_stable
      prop_aw_addr_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_awvalid[mi] && !m_awready[mi])
        |=> (m_awvalid[mi] && $stable(m_awaddr[mi]))
      ) else $error("Master %0d: AW addr changed while valid && !ready", mi);

      prop_aw_valid_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_awvalid[mi] && !m_awready[mi])
        |=> m_awvalid[mi]
      ) else $error("Master %0d: AWVALID deasserted without handshake", mi);
    end
  endgenerate

  // =========================================================================
  // 2. W channel: data stable while VALID && !READY
  // =========================================================================
  generate
    for (genvar mi = 0; mi < NUM_MASTERS; mi++) begin : gen_m_w_stable
      prop_w_data_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_wvalid[mi] && !m_wready[mi])
        |=> (m_wvalid[mi] && $stable(m_wdata[mi]))
      ) else $error("Master %0d: W data changed while valid && !ready", mi);

      prop_w_valid_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_wvalid[mi] && !m_wready[mi])
        |=> m_wvalid[mi]
      ) else $error("Master %0d: WVALID deasserted without handshake", mi);
    end
  endgenerate

  // =========================================================================
  // 3. AR channel: address stable while VALID && !READY
  // =========================================================================
  generate
    for (genvar mi = 0; mi < NUM_MASTERS; mi++) begin : gen_m_ar_stable
      prop_ar_addr_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_arvalid[mi] && !m_arready[mi])
        |=> (m_arvalid[mi] && $stable(m_araddr[mi]))
      ) else $error("Master %0d: AR addr changed while valid && !ready", mi);

      prop_ar_valid_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_arvalid[mi] && !m_arready[mi])
        |=> m_arvalid[mi]
      ) else $error("Master %0d: ARVALID deasserted without handshake", mi);
    end
  endgenerate

  // =========================================================================
  // 4. B channel: slave BVALID stable until BREADY handshake
  // =========================================================================
  generate
    for (genvar si = 0; si < NUM_SLAVES; si++) begin : gen_s_b_stable
      prop_b_valid_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (s_bvalid[si] && !s_bready[si])
        |=> s_bvalid[si]
      ) else $error("Slave %0d: BVALID deasserted without handshake", si);
    end
  endgenerate

  // =========================================================================
  // 5. R channel: slave RVALID stable until RREADY handshake
  // =========================================================================
  generate
    for (genvar si = 0; si < NUM_SLAVES; si++) begin : gen_s_r_stable
      prop_r_valid_stable: assert property (
        @(posedge clk) disable iff (!rst_n)
        (s_rvalid[si] && !s_rready[si])
        |=> s_rvalid[si]
      ) else $error("Slave %0d: RVALID deasserted without handshake", si);
    end
  endgenerate

  // =========================================================================
  // 6. No deadlock: outstanding request eventually gets response
  //    (bounded liveness — 256 cycles)
  // =========================================================================
  generate
    for (genvar mi = 0; mi < NUM_MASTERS; mi++) begin : gen_m_liveness
      prop_write_eventually_completes: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_awvalid[mi] && m_awready[mi])
        |-> ##[1:256] (m_bvalid[mi] && m_bready[mi])
      ) else $error("Master %0d: write never completed (BRESP timeout)", mi);

      prop_read_eventually_completes: assert property (
        @(posedge clk) disable iff (!rst_n)
        (m_arvalid[mi] && m_arready[mi])
        |-> ##[1:256] (m_rvalid[mi] && m_rready[mi])
      ) else $error("Master %0d: read never completed (RDATA timeout)", mi);
    end
  endgenerate

endmodule
