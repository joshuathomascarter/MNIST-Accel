// =============================================================================
// noc_vc_allocator_static_prio.sv — Static Priority VC Allocator
// =============================================================================
// Comparison baseline: sparse requests always beat dense requests.
// No round-robin within the same priority class — strict priority.
// This is the simplest QoS scheme: sparse = high priority, dense = low.
//
// Problem: dense traffic starves under sustained sparse load.
// This allocator exists to demonstrate that our sparsity-aware approach
// (with reservation + round-robin) provides better fairness.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_vc_allocator_static_prio #(
  parameter int NUM_PORTS = 5,
  parameter int NUM_VCS   = noc_pkg::NUM_VCS
) (
  input  logic                     clk,
  input  logic                     rst_n,

  input  logic [NUM_VCS-1:0]       req       [NUM_PORTS],
  input  logic [PORT_BITS-1:0]     req_port  [NUM_PORTS][NUM_VCS],
  input  msg_type_e                req_msg   [NUM_PORTS][NUM_VCS],
  input  logic [NUM_VCS-1:0]       vc_busy   [NUM_PORTS],

  output logic [NUM_VCS-1:0]       grant     [NUM_PORTS],
  output logic [VC_BITS-1:0]       grant_vc  [NUM_PORTS][NUM_VCS],

  input  logic [NUM_VCS-1:0]       release_vc [NUM_PORTS],
  input  logic [VC_BITS-1:0]       release_id [NUM_PORTS][NUM_VCS]
);

  logic [NUM_VCS-1:0] vc_free [NUM_PORTS];

  always_comb begin
    for (int op = 0; op < NUM_PORTS; op++)
      vc_free[op] = ~vc_busy[op];
  end

  // Helper: is this a sparse-related message?
  function automatic is_sparse_msg;
    input msg_type_e mt;
    begin
      is_sparse_msg = (mt == MSG_SPARSE_HINT) || (mt == MSG_SCATTER) ||
                      (mt == MSG_REDUCE)      || (mt == MSG_BARRIER);
    end
  endfunction

  logic [VC_BITS-1:0] free_idx_c;
  logic               found_free_c;
  logic               granted_c;

  // ---------------------------------------------------------------------------
  // Strict priority allocation: sparse first, then dense.
  // No round-robin — always picks lowest-indexed free VC and lowest-indexed
  // requesting port. Simple but unfair.
  // ---------------------------------------------------------------------------
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        grant[ip][iv]    = 1'b0;
        grant_vc[ip][iv] = '0;
      end

    for (int op = 0; op < NUM_PORTS; op++) begin
      // Find first free VC on this port
      found_free_c = 1'b0;
      free_idx_c   = '0;

      for (int v = 0; v < NUM_VCS; v++) begin
        if (!found_free_c && vc_free[op][v]) begin
          found_free_c = 1'b1;
          free_idx_c   = VC_BITS'(v);
        end
      end

      if (found_free_c) begin
        // Pass 1: Try sparse requests (HIGH priority)
        granted_c = 1'b0;

        for (int ip = 0; ip < NUM_PORTS; ip++) begin
          for (int iv = 0; iv < NUM_VCS; iv++) begin
            if (!granted_c && req[ip][iv] &&
                (req_port[ip][iv] == PORT_BITS'(op)) &&
                is_sparse_msg(req_msg[ip][iv])) begin
              grant[ip][iv]    = 1'b1;
              grant_vc[ip][iv] = free_idx_c;
              granted_c        = 1'b1;
            end
          end
        end

        // Pass 2: If no sparse granted, try dense (LOW priority)
        if (!granted_c) begin
          for (int ip = 0; ip < NUM_PORTS; ip++) begin
            for (int iv = 0; iv < NUM_VCS; iv++) begin
              if (!granted_c && req[ip][iv] &&
                  (req_port[ip][iv] == PORT_BITS'(op)) &&
                  !is_sparse_msg(req_msg[ip][iv])) begin
                grant[ip][iv]    = 1'b1;
                grant_vc[ip][iv] = free_idx_c;
                granted_c        = 1'b1;
              end
            end
          end
        end
      end
    end
  end

endmodule
