// =============================================================================
// noc_switch_allocator.sv — iSLIP-style Switch Allocator
// =============================================================================
// Two-phase separable allocator (simplified iSLIP):
//   Phase 1: Each output port picks one requesting input (round-robin)
//   Phase 2: Each input port picks one granted output (round-robin)
// This gives O(1) allocation with fairness guarantees.
//
// Operates per-VC: an input port can have multiple active VCs,
// each needing switch bandwidth independently.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off BLKSEQ */
import noc_pkg::*;

module noc_switch_allocator #(
  parameter int NUM_PORTS = 5,
  parameter int NUM_VCS   = noc_pkg::NUM_VCS
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // --- Request: input port ip, VC iv wants output port target_port ---
  input  logic [NUM_VCS-1:0]       sa_req        [NUM_PORTS],
  input  logic [PORT_BITS-1:0]     sa_target     [NUM_PORTS][NUM_VCS],

  // --- Credit availability on output VCs ---
  input  logic [NUM_VCS-1:0]       out_has_credit [NUM_PORTS],

  // --- Allocated output VC for each (ip, iv) from VC allocator ---
  input  logic [NUM_PORTS-1:0][NUM_VCS-1:0][VC_BITS-1:0] alloc_vc,

  // --- Grant: one flit per input port, one flit per output port ---
  output logic [NUM_VCS-1:0]       sa_grant      [NUM_PORTS],

  // --- Crossbar control ---
  output logic [PORT_BITS-1:0]     xbar_sel      [NUM_PORTS], // per output: which input
  output logic                     xbar_valid    [NUM_PORTS]  // per output: data valid
);

  // ---------------------------------------------------------------------------
  // Phase 1: Output arbitration — each output port picks one input
  // ---------------------------------------------------------------------------
  // Flatten requests: for output port op, who is requesting it?
  logic [NUM_PORTS*NUM_VCS-1:0] op_req_flat [NUM_PORTS];

  // Round-robin pointers per output port
  logic [NUM_PORTS-1:0][$clog2(NUM_PORTS*NUM_VCS)-1:0] op_rr;
  localparam int TOTAL = NUM_PORTS * NUM_VCS;
  logic [$clog2(TOTAL)-1:0] flat_idx_c;
  integer rotated_c;
  logic   done_c;
  logic [$clog2(NUM_PORTS+1)-1:0] selected_op_c; // +1 so max value < NUM_PORTS avoids OOB
  integer winner_c;

  // Phase 1 intermediate grants
  logic p1_grant_valid [NUM_PORTS]; // per output port
  logic [PORT_BITS-1:0] p1_grant_ip [NUM_PORTS]; // which input port won
  logic [VC_BITS-1:0]   p1_grant_iv [NUM_PORTS]; // which VC won

  always_comb begin
    // Build flat request vectors per output port
    for (int op = 0; op < NUM_PORTS; op++) begin
      op_req_flat[op] = '0;
      for (int ip = 0; ip < NUM_PORTS; ip++) begin
        for (int iv = 0; iv < NUM_VCS; iv++) begin
          flat_idx_c = $clog2(TOTAL)'(ip * NUM_VCS + iv);
          // Request valid if: has flit, wants this output, has credit on allocated VC
          op_req_flat[op][flat_idx_c] = sa_req[ip][iv] &&
                                      (sa_target[ip][iv] == PORT_BITS'(op)) &&
                                      out_has_credit[op][alloc_vc[ip][iv]];
        end
      end
    end

    // Round-robin select per output port
    for (int op = 0; op < NUM_PORTS; op++) begin
      p1_grant_valid[op] = 1'b0;
      p1_grant_ip[op]    = '0;
      p1_grant_iv[op]    = '0;

      for (int t = 0; t < TOTAL; t++) begin
        rotated_c = (int'(op_rr[op]) + t) % TOTAL;
        if (!p1_grant_valid[op] && op_req_flat[op][rotated_c]) begin
          p1_grant_valid[op] = 1'b1;
          p1_grant_ip[op]    = PORT_BITS'(rotated_c / NUM_VCS);
          p1_grant_iv[op]    = VC_BITS'(rotated_c % NUM_VCS);
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Phase 2: Input arbitration — each input port accepts at most one grant
  // ---------------------------------------------------------------------------
  logic [NUM_PORTS-1:0][$clog2(NUM_PORTS)-1:0] ip_rr; // per-input RR pointer

  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++) begin
      sa_grant[ip] = '0;
    end
    for (int op = 0; op < NUM_PORTS; op++) begin
      xbar_sel[op]   = '0;
      xbar_valid[op] = 1'b0;
    end

    // For each input port, find the first (by RR priority) output that granted it
    for (int ip = 0; ip < NUM_PORTS; ip++) begin
      done_c = 1'b0;
      for (int r = 0; r < NUM_PORTS; r++) begin
        selected_op_c = $clog2(NUM_PORTS+1)'((int'(ip_rr[ip]) + r) % NUM_PORTS);
        // Use explicit per-port comparison to avoid dynamic part-select OOB warnings
        for (int opo = 0; opo < NUM_PORTS; opo++) begin
          if (!done_c && (opo == int'(selected_op_c)) &&
              p1_grant_valid[opo] && (p1_grant_ip[opo] == PORT_BITS'(ip))) begin
            sa_grant[ip][p1_grant_iv[opo]] = 1'b1;
            xbar_sel[opo]                  = PORT_BITS'(ip);
            xbar_valid[opo]                = 1'b1;
            done_c                         = 1'b1;
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Pointer updates
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      op_rr <= '0;
      ip_rr <= '0;
    end else begin
      // Output port pointers: advance past the winner
      for (int op = 0; op < NUM_PORTS; op++) begin
        if (xbar_valid[op]) begin
          winner_c = int'(p1_grant_ip[op]) * NUM_VCS + int'(p1_grant_iv[op]);
          op_rr[op] <= $clog2(TOTAL)'((winner_c + 1) % TOTAL);
        end
      end

      // Input port pointers: advance past the accepted output
      for (int ip = 0; ip < NUM_PORTS; ip++) begin
        if (sa_grant[ip] != '0) begin
          for (int op = 0; op < NUM_PORTS; op++) begin
            if (xbar_valid[op] && (xbar_sel[op] == PORT_BITS'(ip)))
              ip_rr[ip] <= $clog2(NUM_PORTS)'((op + 1) % NUM_PORTS);
          end
        end
      end
    end
  end

endmodule
