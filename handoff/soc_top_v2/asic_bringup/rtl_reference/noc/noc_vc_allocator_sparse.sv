// =============================================================================
// noc_vc_allocator_sparse.sv — Sparsity-Aware VC Allocator (Drop-in)
// =============================================================================
// Same interface as noc_vc_allocator.sv (baseline round-robin) so the router
// can swap between them via SPARSE_VC_ALLOC parameter.
//
// Novel contribution: uses sparsity hints embedded in HEAD flits to make
// smarter VC allocation decisions:
//
//   1. PRIORITY BOOST: Flits carrying sparse data (MSG_SPARSE_HINT or
//      scatter/reduce messages) get higher priority for VC allocation.
//      Rationale: sparse traffic has bursty, latency-sensitive patterns.
//
//   2. VC RESERVATION: One VC per output port is "reserved" for sparse
//      traffic. Dense traffic can use it only when all other VCs are busy.
//      This prevents head-of-line blocking of sparse flits behind dense data.
//
//   3. ADAPTIVE THRESHOLD: A running counter tracks sparse vs. dense traffic
//      ratio. When sparse fraction > 50%, reservation is enabled; otherwise
//      all VCs are treated equally (degrades to baseline behavior).

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_vc_allocator_sparse #(
  parameter int NUM_PORTS     = 5,
  parameter int NUM_VCS       = noc_pkg::NUM_VCS,
  parameter int RESERVED_VC   = NUM_VCS - 1  // Highest VC reserved for sparse
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // --- Request interface (same as baseline) ---
  input  logic [NUM_VCS-1:0]       req       [NUM_PORTS],
  input  logic [PORT_BITS-1:0]     req_port  [NUM_PORTS][NUM_VCS],

  // --- Sparsity hints (additional input over baseline) ---
  // msg_type of the HEAD flit requesting allocation
  input  msg_type_e                req_msg   [NUM_PORTS][NUM_VCS],

  // --- Output VC status ---
  input  logic [NUM_VCS-1:0]       vc_busy   [NUM_PORTS],

  // --- Grant interface ---
  output logic [NUM_VCS-1:0]       grant     [NUM_PORTS],
  output logic [VC_BITS-1:0]       grant_vc  [NUM_PORTS][NUM_VCS],

  // --- Release interface ---
  input  logic [NUM_VCS-1:0]       release_vc [NUM_PORTS],
  input  logic [VC_BITS-1:0]       release_id [NUM_PORTS][NUM_VCS]
);

  // ---------------------------------------------------------------------------
  // Sparsity traffic tracker (adaptive threshold)
  // ---------------------------------------------------------------------------
  logic [15:0] sparse_count;
  logic [15:0] total_count;
  logic        reservation_active;
  logic [15:0] sparse_req_count;
  logic [15:0] total_req_count;
  logic [15:0] sparse_count_base;
  logic [15:0] total_count_base;
  logic [16:0] sparse_count_next;
  logic [16:0] total_count_next;

  // Reservation active when sparse traffic > 50%
  assign reservation_active = (total_count > 16'd64) &&
                              (sparse_count > (total_count >> 1));

  always_comb begin
    sparse_req_count  = '0;
    total_req_count   = '0;
    sparse_count_base = sparse_count;
    total_count_base  = total_count;

    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++)
        if (req[ip][iv]) begin
          total_req_count = total_req_count + 16'd1;
          if (is_sparse_msg(req_msg[ip][iv]))
            sparse_req_count = sparse_req_count + 16'd1;
        end

    if (total_count == 16'hFFFF) begin
      sparse_count_base = sparse_count >> 1;
      total_count_base  = total_count >> 1;
    end

    sparse_count_next = {1'b0, sparse_count_base} + {1'b0, sparse_req_count};
    total_count_next  = {1'b0, total_count_base} + {1'b0, total_req_count};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sparse_count <= '0;
      total_count  <= '0;
    end else begin
      sparse_count <= sparse_count_next[16] ? 16'hFFFF : sparse_count_next[15:0];
      total_count  <= total_count_next[16] ? 16'hFFFF : total_count_next[15:0];
    end
  end

  // Helper: is this a sparse-related message?
  function automatic is_sparse_msg;
    input msg_type_e mt;
    begin
      is_sparse_msg = (mt == MSG_SPARSE_HINT) || (mt == MSG_SCATTER) ||
                      (mt == MSG_REDUCE)      || (mt == MSG_BARRIER);
    end
  endfunction

  // ---------------------------------------------------------------------------
  // Per-output-port round-robin priority pointers (two: sparse + dense)
  // ---------------------------------------------------------------------------
  logic [VC_BITS-1:0] rr_sparse [NUM_PORTS];
  logic [VC_BITS-1:0] rr_dense  [NUM_PORTS];

  // Free VCs per output port
  logic [NUM_VCS-1:0] vc_free [NUM_PORTS];

  // Hoisted from always_comb bodies — Verilator does not allow logic
  // declarations after statements inside for-loop begin...end blocks.
  logic               found_sparse, found_dense;
  logic [VC_BITS-1:0] picked_vc;
  logic               picked_ok;

  always_comb begin
    for (int op = 0; op < NUM_PORTS; op++)
      vc_free[op] = ~vc_busy[op];
  end

  // ---------------------------------------------------------------------------
  // Two-pass allocation
  // Pass 1: Sparse requests get priority, can use reserved VC
  // Pass 2: Dense requests, cannot use reserved VC if reservation active
  // ---------------------------------------------------------------------------
  always_comb begin
    found_sparse = 1'b0;
    found_dense  = 1'b0;
    picked_vc    = '0;
    picked_ok    = 1'b0;

    // Defaults
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        grant[ip][iv]    = 1'b0;
        grant_vc[ip][iv] = '0;
      end

    for (int op = 0; op < NUM_PORTS; op++) begin
      logic [NUM_VCS-1:0] avail_sparse, avail_dense;
      avail_sparse = vc_free[op];  // Sparse can use ALL VCs
      avail_dense  = reservation_active ?
                     (vc_free[op] & ~(NUM_VCS'(1) << RESERVED_VC)) :
                     vc_free[op];  // Dense cannot use reserved when active

      // --- Pass 1: Sparse requests ---
      found_sparse = 1'b0;

      for (int ip = 0; ip < NUM_PORTS; ip++) begin
        for (int iv = 0; iv < NUM_VCS; iv++) begin
          if (!found_sparse &&
              req[ip][iv] &&
              (req_port[ip][iv] == PORT_BITS'(op)) &&
              is_sparse_msg(req_msg[ip][iv]) &&
              (avail_sparse != '0)) begin

            // Find free VC (round-robin from rr_sparse)
            picked_ok = 1'b0;
            picked_vc = '0;

            for (int v = 0; v < NUM_VCS; v++) begin
              if (!picked_ok && avail_sparse[(int'(rr_sparse[op]) + v) % NUM_VCS]) begin
                picked_ok = 1'b1;
                picked_vc = VC_BITS'((int'(rr_sparse[op]) + v) % NUM_VCS);
              end
            end

            if (picked_ok) begin
              grant[ip][iv]             = 1'b1;
              grant_vc[ip][iv]          = picked_vc;
              avail_sparse[picked_vc]   = 1'b0;
              avail_dense[picked_vc]    = 1'b0;
              found_sparse              = 1'b1;
            end
          end
        end
      end

      // --- Pass 2: Dense requests (remaining capacity) ---
      found_dense = 1'b0;

      for (int ip = 0; ip < NUM_PORTS; ip++) begin
        for (int iv = 0; iv < NUM_VCS; iv++) begin
          if (!found_dense &&
              req[ip][iv] &&
              (req_port[ip][iv] == PORT_BITS'(op)) &&
              !is_sparse_msg(req_msg[ip][iv]) &&
              !grant[ip][iv] &&  // Not already granted in pass 1
              (avail_dense != '0)) begin

            picked_ok = 1'b0;
            picked_vc = '0;

            for (int v = 0; v < NUM_VCS; v++) begin
              if (!picked_ok && avail_dense[(int'(rr_dense[op]) + v) % NUM_VCS]) begin
                picked_ok = 1'b1;
                picked_vc = VC_BITS'((int'(rr_dense[op]) + v) % NUM_VCS);
              end
            end

            if (picked_ok) begin
              grant[ip][iv]    = 1'b1;
              grant_vc[ip][iv] = picked_vc;
              found_dense      = 1'b1;
            end
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
      for (int op = 0; op < NUM_PORTS; op++) begin
        rr_sparse[op] <= '0;
        rr_dense[op]  <= '0;
      end
    end else begin
      for (int op = 0; op < NUM_PORTS; op++) begin
        for (int ip = 0; ip < NUM_PORTS; ip++)
          for (int iv = 0; iv < NUM_VCS; iv++)
            if (grant[ip][iv] && (req_port[ip][iv] == PORT_BITS'(op))) begin
              if (is_sparse_msg(req_msg[ip][iv]))
                rr_sparse[op] <= (grant_vc[ip][iv] == VC_BITS'(NUM_VCS - 1)) ?
                                 '0 : grant_vc[ip][iv] + 1;
              else
                rr_dense[op] <= (grant_vc[ip][iv] == VC_BITS'(NUM_VCS - 1)) ?
                                '0 : grant_vc[ip][iv] + 1;
            end
      end
    end
  end

endmodule
