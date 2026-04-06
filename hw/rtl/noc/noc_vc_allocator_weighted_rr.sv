// =============================================================================
// noc_vc_allocator_weighted_rr.sv — Weighted Round-Robin VC Allocator
// =============================================================================
// Comparison scheme: sparse requests get W_SPARSE weight (default 3),
// dense requests get W_DENSE weight (default 1).
//
// Each weight means "how many consecutive grants before yielding."
// After serving W_SPARSE sparse requests, must serve W_DENSE dense
// even if more sparse are waiting. This prevents starvation but
// doesn't provide guaranteed low-latency since sparse can still be
// blocked for W_DENSE cycles.
//
// This is a software-configurable approach — weights must be tuned
// per workload. Our sparsity-aware approach adapts automatically.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_vc_allocator_weighted_rr #(
  parameter int NUM_PORTS = 5,
  parameter int NUM_VCS   = noc_pkg::NUM_VCS,
  parameter int W_SPARSE  = 3,   // Sparse gets 3x more grants
  parameter int W_DENSE   = 1    // Dense gets 1x grants
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

  function automatic is_sparse_msg;
    input msg_type_e mt;
    begin
      is_sparse_msg = (mt == MSG_SPARSE_HINT) || (mt == MSG_SCATTER) ||
                      (mt == MSG_REDUCE)      || (mt == MSG_BARRIER);
    end
  endfunction

  // ---------------------------------------------------------------------------
  // Per-output-port state: round-robin pointer + weight counter
  // ---------------------------------------------------------------------------
  logic [VC_BITS-1:0] rr_ptr [NUM_PORTS];

  // Weight counter: counts down from W_SPARSE or W_DENSE
  // serve_sparse = 1 → currently in "sparse turn" phase
  logic        serve_sparse [NUM_PORTS];
  logic [3:0]  weight_cnt   [NUM_PORTS];
  logic [VC_BITS-1:0] free_idx_c;
  logic               found_free_c;
  logic               granted_c;
  logic               any_granted_c;
  integer             rotated_c;

  // ---------------------------------------------------------------------------
  // Allocation logic
  // ---------------------------------------------------------------------------
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        grant[ip][iv]    = 1'b0;
        grant_vc[ip][iv] = '0;
      end

    for (int op = 0; op < NUM_PORTS; op++) begin
      found_free_c = 1'b0;
      free_idx_c   = '0;

      for (int v = 0; v < NUM_VCS; v++) begin
        rotated_c = (int'(rr_ptr[op]) + v) % NUM_VCS;
        if (!found_free_c && vc_free[op][rotated_c]) begin
          found_free_c = 1'b1;
          free_idx_c   = VC_BITS'(rotated_c);
        end
      end

      if (found_free_c) begin
        granted_c = 1'b0;

        if (serve_sparse[op]) begin
          // Sparse turn — try sparse first
          for (int ip = 0; ip < NUM_PORTS; ip++)
            for (int iv = 0; iv < NUM_VCS; iv++)
              if (!granted_c && req[ip][iv] &&
                  (req_port[ip][iv] == PORT_BITS'(op)) &&
                  is_sparse_msg(req_msg[ip][iv])) begin
                grant[ip][iv]    = 1'b1;
                grant_vc[ip][iv] = free_idx_c;
                granted_c        = 1'b1;
              end

          // If no sparse available, serve dense anyway (work-conserving)
          if (!granted_c) begin
            for (int ip = 0; ip < NUM_PORTS; ip++)
              for (int iv = 0; iv < NUM_VCS; iv++)
                if (!granted_c && req[ip][iv] &&
                    (req_port[ip][iv] == PORT_BITS'(op))) begin
                  grant[ip][iv]    = 1'b1;
                  grant_vc[ip][iv] = free_idx_c;
                  granted_c        = 1'b1;
                end
          end
        end else begin
          // Dense turn — try dense first
          for (int ip = 0; ip < NUM_PORTS; ip++)
            for (int iv = 0; iv < NUM_VCS; iv++)
              if (!granted_c && req[ip][iv] &&
                  (req_port[ip][iv] == PORT_BITS'(op)) &&
                  !is_sparse_msg(req_msg[ip][iv])) begin
                grant[ip][iv]    = 1'b1;
                grant_vc[ip][iv] = free_idx_c;
                granted_c        = 1'b1;
              end

          if (!granted_c) begin
            for (int ip = 0; ip < NUM_PORTS; ip++)
              for (int iv = 0; iv < NUM_VCS; iv++)
                if (!granted_c && req[ip][iv] &&
                    (req_port[ip][iv] == PORT_BITS'(op))) begin
                  grant[ip][iv]    = 1'b1;
                  grant_vc[ip][iv] = free_idx_c;
                  granted_c        = 1'b1;
                end
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Sequential: update rr_ptr and weight counters
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        rr_ptr[op]       <= '0;
        serve_sparse[op] <= 1'b1;  // Start in sparse phase
        weight_cnt[op]   <= 4'(W_SPARSE);
      end
    end else begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        // Check if any grant was made on this output port
        any_granted_c = 1'b0;
        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (grant[ip][iv] && (req_port[ip][iv] == PORT_BITS'(op)))
              any_granted_c = 1'b1;

        if (any_granted_c) begin
          rr_ptr[op] <= (rr_ptr[op] == VC_BITS'(NUM_VCS - 1)) ? '0 : rr_ptr[op] + 1;

          if (weight_cnt[op] <= 4'd1) begin
            // Switch phase
            serve_sparse[op] <= ~serve_sparse[op];
            weight_cnt[op]   <= serve_sparse[op] ? 4'(W_DENSE) : 4'(W_SPARSE);
          end else begin
            weight_cnt[op] <= weight_cnt[op] - 4'd1;
          end
        end
      end
    end
  end

endmodule
