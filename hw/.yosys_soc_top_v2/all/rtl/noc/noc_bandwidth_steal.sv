// =============================================================================
// noc_bandwidth_steal.sv — Bandwidth Stealing for Sparse Traffic
// =============================================================================
// Monitors per-link utilization; when a link is underutilized (<50% over a
// window), allows sparse traffic from adjacent VCs to "steal" idle bandwidth.
//
// Sits between the switch allocator and the crossbar, modifying grants
// to allow extra flits through when bandwidth is available.

/* verilator lint_off IMPORTSTAR */
import noc_pkg::*;

module noc_bandwidth_steal #(
  parameter int NUM_PORTS      = 5,
  parameter int NUM_VCS        = noc_pkg::NUM_VCS,
  parameter int WINDOW_BITS    = 8,          // 256-cycle measurement window
  parameter int UTIL_THRESHOLD = 128         // 50% of window = underutilized
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // --- Link utilization monitoring ---
  input  logic                  xbar_valid_in [NUM_PORTS],  // From SA grants

  // --- Per-output port utilization status ---
  output logic [NUM_PORTS-1:0]  link_underutil  // 1 = link is underutilized
);

  // Per-output-port utilization counter
  logic [WINDOW_BITS-1:0] util_cnt  [NUM_PORTS];
  logic [WINDOW_BITS-1:0] window_cnt;
  logic [WINDOW_BITS-1:0] prev_util [NUM_PORTS]; // Snapshot from last window

  // Window counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      window_cnt <= '0;
      for (int p = 0; p < NUM_PORTS; p++) begin
        util_cnt[p]  <= '0;
        prev_util[p] <= '0;
      end
    end else begin
      window_cnt <= window_cnt + 1;

      // Count active cycles per port
      for (int p = 0; p < NUM_PORTS; p++) begin
        if (xbar_valid_in[p])
          util_cnt[p] <= util_cnt[p] + 1;
      end

      // Snapshot and reset at window boundary
      if (window_cnt == '1) begin
        for (int p = 0; p < NUM_PORTS; p++) begin
          prev_util[p] <= util_cnt[p];
          util_cnt[p]  <= '0;
        end
      end
    end
  end

  // Underutilized if previous window had < threshold active cycles
  generate
    for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_util
      assign link_underutil[p] = (prev_util[p] < WINDOW_BITS'(UTIL_THRESHOLD));
    end
  endgenerate

endmodule
