// =============================================================================
// noc_credit_counter.sv — Credit-Based Flow Control Counter
// =============================================================================
// Each output VC maintains a credit counter. When we send a flit on that VC
// we decrement. When the downstream router frees a buffer slot it sends a
// credit back, and we increment. Counter == 0 means no space → cannot send.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_credit_counter #(
  parameter int BUF_DEPTH = noc_pkg::BUF_DEPTH,
  parameter int NUM_VCS   = noc_pkg::NUM_VCS
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // Credit return from downstream (one-hot per VC)
  input  logic [NUM_VCS-1:0]     credit_in,

  // Flit sent on this port (grant from switch allocator)
  input  logic                   flit_sent,
  input  logic [VC_BITS-1:0]     flit_vc,

  // Per-VC credit availability
  output logic [NUM_VCS-1:0]     has_credit
);

  localparam int CNT_BITS = $clog2(BUF_DEPTH + 1);

  logic [CNT_BITS-1:0] count [NUM_VCS];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int v = 0; v < NUM_VCS; v++)
        count[v] <= CNT_BITS'(BUF_DEPTH);  // Start full (downstream buffers empty)
    end else begin
      for (int v = 0; v < NUM_VCS; v++) begin
        logic inc, dec;
        inc = credit_in[v];
        dec = (flit_sent && (flit_vc == VC_BITS'(v)));

        case ({inc, dec})
          2'b10:   count[v] <= count[v] + CNT_BITS'(1);
          2'b01:   count[v] <= count[v] - CNT_BITS'(1);
          default: count[v] <= count[v];  // 00 or 11 cancel
        endcase
      end
    end
  end

  // Has credit if counter > 0
  generate
    for (genvar v = 0; v < NUM_VCS; v++) begin : gen_has_credit
      assign has_credit[v] = (count[v] != '0);
    end
  endgenerate

endmodule
