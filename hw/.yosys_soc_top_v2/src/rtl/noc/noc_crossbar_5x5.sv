// =============================================================================
// noc_crossbar_5x5.sv — 5×5 Crossbar Switch
// =============================================================================
// Pure combinational mux: each output port selects one input port's flit
// based on xbar_sel from the switch allocator. No buffering.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_crossbar_5x5 #(
  parameter int NUM_PORTS = 5
) (
  // Input flits (one per input port)
  input  flit_t                  in_flit   [NUM_PORTS],
  input  logic [VC_BITS-1:0]    in_vc     [NUM_PORTS],  // Allocated output VC

  // Crossbar control (from switch allocator)
  input  logic [PORT_BITS-1:0]  xbar_sel  [NUM_PORTS],  // per output: which input
  input  logic                  xbar_valid[NUM_PORTS],   // per output: valid

  // Output flits (one per output port)
  output flit_t                 out_flit  [NUM_PORTS],
  output logic                  out_valid [NUM_PORTS]
);

  always_comb begin
    for (int op = 0; op < NUM_PORTS; op++) begin
      if (xbar_valid[op]) begin
        out_flit[op]       = in_flit[xbar_sel[op]];
        // Rewrite VC field to the allocated output VC
        out_flit[op].vc_id = in_vc[xbar_sel[op]];
        out_valid[op]      = 1'b1;
      end else begin
        out_flit[op]  = '0;
        out_valid[op] = 1'b0;
      end
    end
  end

endmodule
