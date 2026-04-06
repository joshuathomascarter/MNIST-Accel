// =============================================================================
// noc_vc_allocator.sv — Baseline Round-Robin VC Allocator
// =============================================================================
// When a HEAD flit arrives and needs an output VC on the target output port,
// this module allocates a free VC at that output. Uses round-robin fairness.
//
// Same interface as noc_vc_allocator_sparse.sv for drop-in comparison.
//
// Flow:
//   1. Input port presents request: "I have a HEAD flit wanting output port P"
//   2. This module checks which VCs on output port P are free
//   3. Grants one VC using round-robin priority per output port
//   4. The allocated VC is locked until TAIL flit traverses

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module noc_vc_allocator #(
  parameter int NUM_PORTS = 5,
  parameter int NUM_VCS   = noc_pkg::NUM_VCS
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // --- Request interface (from input ports) ---
  // req[ip][iv] = 1 → input port ip, VC iv wants an output VC
  // req_port[ip][iv] = target output port
  input  logic [NUM_VCS-1:0]       req       [NUM_PORTS],
  input  logic [PORT_BITS-1:0]     req_port  [NUM_PORTS][NUM_VCS],

  // --- Output VC status ---
  // vc_busy[op][ov] = 1 → output port op, VC ov is occupied by a packet
  input  logic [NUM_VCS-1:0]       vc_busy   [NUM_PORTS],

  // --- Grant interface ---
  // grant[ip][iv] = 1 → request granted
  // grant_vc[ip][iv] = allocated output VC index
  output logic [NUM_VCS-1:0]       grant     [NUM_PORTS],
  output logic [VC_BITS-1:0]       grant_vc  [NUM_PORTS][NUM_VCS],

  // --- Release interface (TAIL flit consumed → free the output VC) ---
  input  logic [NUM_VCS-1:0]       release_vc [NUM_PORTS],         // one-hot per output port
  input  logic [VC_BITS-1:0]       release_id [NUM_PORTS][NUM_VCS] // which VC to free
);

  // ---------------------------------------------------------------------------
  // Per-output-port round-robin priority pointer
  // ---------------------------------------------------------------------------
  logic [VC_BITS-1:0] rr_ptr [NUM_PORTS];

  // Collect free VCs per output port
  logic [NUM_VCS-1:0] vc_free [NUM_PORTS];
  logic [VC_BITS-1:0] free_vc_idx_c;
  logic               found_free_c;
  logic               granted_one_c;
  logic               any_grant_c;
  integer             rotated_c;

  always_comb begin
    for (int op = 0; op < NUM_PORTS; op++)
      vc_free[op] = ~vc_busy[op];
  end

  // ---------------------------------------------------------------------------
  // Allocation logic — one grant per output port per cycle
  // ---------------------------------------------------------------------------
  // For each output port, collect all requestors, pick one by round-robin,
  // then assign the lowest-indexed free VC.

  always_comb begin
    // Default grants
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        grant[ip][iv]    = 1'b0;
        grant_vc[ip][iv] = '0;
      end

    for (int op = 0; op < NUM_PORTS; op++) begin
      // Find first free VC on this output port (round-robin start)
      found_free_c = 1'b0;
      free_vc_idx_c = '0;
      granted_one_c = 1'b0;

      for (int v = 0; v < NUM_VCS; v++) begin
        rotated_c = (int'(rr_ptr[op]) + v) % NUM_VCS;
        if (!found_free_c && vc_free[op][rotated_c]) begin
          found_free_c = 1'b1;
          free_vc_idx_c = VC_BITS'(rotated_c);
        end
      end

      if (found_free_c) begin
        // Find first requesting (ip, iv) pair targeting this output port
        for (int ip = 0; ip < NUM_PORTS; ip++) begin
          for (int iv = 0; iv < NUM_VCS; iv++) begin
            if (!granted_one_c && req[ip][iv] && (req_port[ip][iv] == PORT_BITS'(op))) begin
              grant[ip][iv]    = 1'b1;
              grant_vc[ip][iv] = free_vc_idx_c;
              granted_one_c    = 1'b1;
            end
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Round-robin pointer update
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int op = 0; op < NUM_PORTS; op++)
        rr_ptr[op] <= '0;
    end else begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        // Advance pointer when any grant is made on this output
        any_grant_c = 1'b0;
        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (grant[ip][iv] && (req_port[ip][iv] == PORT_BITS'(op)))
              any_grant_c = 1'b1;

        if (any_grant_c)
          rr_ptr[op] <= (rr_ptr[op] == VC_BITS'(NUM_VCS - 1)) ? '0 : rr_ptr[op] + 1;
      end
    end
  end

endmodule
