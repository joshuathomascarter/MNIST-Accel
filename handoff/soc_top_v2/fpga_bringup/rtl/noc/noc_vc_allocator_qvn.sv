// =============================================================================
// noc_vc_allocator_qvn.sv — ARM QVN-Style VC Allocator
// =============================================================================
// Comparison scheme modeled after ARM's Quality Virtual Networks (QVN).
//
// Concept: VCs are statically partitioned into virtual networks (VNs).
// Each traffic class is assigned a VN at DESIGN TIME via CSR configuration.
//
//   VN 0 = VCs {0, 1} → dense data traffic
//   VN 1 = VCs {2, 3} → sparse/control traffic
//
// Key differences from our sparsity-aware approach:
// 1. Static partition: even split (2+2) regardless of workload
// 2. No adaptation: partition never changes at runtime
// 3. Software-configured: requires CSR writes to classify traffic
// 4. Wasted capacity: if one class is idle, its VCs sit empty
//
// Our approach beats QVN because:
// - Asymmetric partition (3 dense + 1 reserved) better matches typical ratio
// - Sparse overflow to any VC prevents waste
// - Adaptive threshold enables/disables based on actual traffic

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_vc_allocator_qvn #(
  parameter int NUM_PORTS    = 5,
  parameter int NUM_VCS      = noc_pkg::NUM_VCS,
  parameter int VN_SPLIT     = NUM_VCS / 2  // VCs 0..VN_SPLIT-1 = dense, VN_SPLIT..NUM_VCS-1 = sparse
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
  // Per-VN round-robin pointers
  // ---------------------------------------------------------------------------
  logic [VC_BITS-1:0] rr_dense  [NUM_PORTS];
  logic [VC_BITS-1:0] rr_sparse [NUM_PORTS];

  // Hoisted from always_comb bodies — Verilator does not allow logic
  // declarations after statements inside for-loop begin...end blocks.
  logic               found_dense_vc, found_sparse_vc;
  logic [VC_BITS-1:0] dense_vc_idx, sparse_vc_idx;
  logic               granted_dense, granted_sparse;
  // Also hoisted for always_ff rr-update block
  logic               dense_granted_ff, sparse_granted_ff;

  // ---------------------------------------------------------------------------
  // Allocation: strictly partition VCs between VNs.
  // Dense can ONLY use VCs 0..VN_SPLIT-1
  // Sparse can ONLY use VCs VN_SPLIT..NUM_VCS-1
  // No overflow — this is the QVN limitation.
  // ---------------------------------------------------------------------------
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        grant[ip][iv]    = 1'b0;
        grant_vc[ip][iv] = '0;
      end

    for (int op = 0; op < NUM_PORTS; op++) begin
      // ---------- Dense VN (VCs 0 to VN_SPLIT-1) ----------
      found_dense_vc = 1'b0;
      dense_vc_idx   = '0;

      for (int v = 0; v < VN_SPLIT; v++) begin
        int rotated;
        rotated = (int'(rr_dense[op]) + v) % VN_SPLIT;
        if (!found_dense_vc && vc_free[op][rotated]) begin
          found_dense_vc = 1'b1;
          dense_vc_idx   = VC_BITS'(rotated);
        end
      end

      if (found_dense_vc) begin
        granted_dense = 1'b0;

        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (!granted_dense && req[ip][iv] &&
                (req_port[ip][iv] == PORT_BITS'(op)) &&
                !is_sparse_msg(req_msg[ip][iv])) begin
              grant[ip][iv]    = 1'b1;
              grant_vc[ip][iv] = dense_vc_idx;
              granted_dense    = 1'b1;
            end
      end

      // ---------- Sparse VN (VCs VN_SPLIT to NUM_VCS-1) ----------
      found_sparse_vc = 1'b0;
      sparse_vc_idx   = '0;

      for (int v = 0; v < (NUM_VCS - VN_SPLIT); v++) begin
        int rotated;
        rotated = VN_SPLIT + ((int'(rr_sparse[op]) + v) % (NUM_VCS - VN_SPLIT));
        if (!found_sparse_vc && vc_free[op][rotated]) begin
          found_sparse_vc = 1'b1;
          sparse_vc_idx   = VC_BITS'(rotated);
        end
      end

      if (found_sparse_vc) begin
        granted_sparse = 1'b0;

        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (!granted_sparse && req[ip][iv] &&
                (req_port[ip][iv] == PORT_BITS'(op)) &&
                is_sparse_msg(req_msg[ip][iv])) begin
              grant[ip][iv]    = 1'b1;
              grant_vc[ip][iv] = sparse_vc_idx;
              granted_sparse   = 1'b1;
            end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // RR pointer update
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        rr_dense[op]  <= '0;
        rr_sparse[op] <= '0;
      end
    end else begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        dense_granted_ff  = 1'b0;
        sparse_granted_ff = 1'b0;

        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (grant[ip][iv] && (req_port[ip][iv] == PORT_BITS'(op))) begin
              if (!is_sparse_msg(req_msg[ip][iv]))
                dense_granted_ff = 1'b1;
              else
                sparse_granted_ff = 1'b1;
            end

        if (dense_granted_ff)
          rr_dense[op] <= (rr_dense[op] == VC_BITS'(VN_SPLIT - 1)) ? '0 : rr_dense[op] + 1;
        if (sparse_granted_ff)
          rr_sparse[op] <= (rr_sparse[op] == VC_BITS'(NUM_VCS - VN_SPLIT - 1)) ?
                           '0 : rr_sparse[op] + 1;
      end
    end
  end

endmodule
