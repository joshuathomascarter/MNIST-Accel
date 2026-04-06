// =============================================================================
// noc_innet_reduce.sv — In-Network Reduction Engine (per-router)
// =============================================================================
// Intercepts MSG_REDUCE flits passing through this router and accumulates
// partial sums in a small scratchpad, forwarding only the final reduced
// result when all expected contributors have arrived.
//
// This is the hardware analogue of SwitchML / SHARP in-network computing,
// applied specifically to sparse INT32 partial-sum reduction across the
// 4×4 systolic tile mesh.
//
// HOW IT WORKS:
//   1. Each tile generates one MSG_REDUCE FLIT_HEADTAIL per row partial sum.
//      The HEAD flit payload carries:
//        [47:40] reduce_id    — identifies the reduction wave / row group
//        [39:36] reduce_expect — total expected contributors (from BSR row_ptr)
//        [35:4]  reduce_val   — INT32 partial sum
//   2. When ANY input port receives a MSG_REDUCE flit destined for the
//      reduce_root AND passing through this router, the engine captures it.
//   3. The engine accumulates into an 8-entry scratchpad (indexed by
//      reduce_id[2:0]).  Each entry stores: {expected, count, acc_value}.
//   4. When count reaches expected, the engine emits ONE new FLIT_HEADTAIL
//      carrying the accumulated value — instead of the N original flits.
//      Traffic reduction: up to (N-1)/N, e.g. 15/16 for 16-tile reduction.
//   5. Flits intercepted are signalled via `intercept_mask` so the router
//      switch allocator does NOT forward them to the crossbar.
//
// SCRATCHPAD:
//   8 entries × 64 bits state per router.  Supports 8 simultaneous
//   in-flight reduction groups (e.g., 8 different output channels).
//   Eviction is automatic when count == expected (entry cleared).
//
// DEADLOCK SAFETY:
//   - Only absorbs FLIT_HEADTAIL (single-flit) MSG_REDUCE packets.
//   - Multi-flit MSG_REDUCE are passed through unmodified (safe fallback).
//   - Scratchpad timeout (256 cycles) clears stale entries to avoid lockup.
//
// INTEGRATION:
//   Instantiated inside noc_router.sv when INNET_REDUCE == 1.
//   Sits between input-port outputs and crossbar inputs:
//     router → [noc_innet_reduce] → crossbar → output links
//
// =============================================================================

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_innet_reduce #(
  parameter int NODE_ID   = 0,
  parameter int NUM_PORTS = 5,
  parameter int MESH_ROWS = noc_pkg::MESH_ROWS,
  parameter int MESH_COLS = noc_pkg::MESH_COLS,
  parameter int SP_DEPTH  = noc_pkg::INNET_SP_DEPTH,
  parameter int REDUCE_VC = noc_pkg::NUM_VCS - 1
) (
  input  logic  clk,
  input  logic  rst_n,

  // Optional per-group subtree metadata override.
  input  logic  cfg_valid,
  input  logic [7:0] cfg_reduce_id,
  input  logic [3:0] cfg_target,
  input  logic  cfg_enable,

  // Enable flag (could be tied to CSR)
  input  logic  enable,

  // -------------------------------------------------------------------------
  // Flit stream from the router crossbar outputs (5 ports)
  // One flit per output port per cycle — the flit about to leave the router
  // -------------------------------------------------------------------------
  input  flit_t flit_in    [NUM_PORTS],
  input  logic  valid_in   [NUM_PORTS],
  input  logic [PORT_BITS-1:0] src_port_in [NUM_PORTS],

  // -------------------------------------------------------------------------
  // Intercept signals — when asserted for port p, that flit is absorbed here
  // and must NOT be forwarded through the crossbar
  // -------------------------------------------------------------------------
  output logic  intercept  [NUM_PORTS],

  // -------------------------------------------------------------------------
  // Inject output — the engine injects one accumulated flit per cycle
  // onto the router output link selected by normal XY routing
  // -------------------------------------------------------------------------
  output flit_t inject_flit,
  output logic  inject_valid,
  input  logic  inject_ready   // selected output link + credit available
);

  // =========================================================================
  // Scratchpad entry type
  // =========================================================================
  typedef struct packed {
    logic        valid;
    logic [7:0]  reduce_id;
    logic [3:0]  expected;
    logic [3:0]  target;         // contributors expected at this router/subtree
    logic [3:0]  count;          // contributors represented so far
    logic [31:0] acc_val;        // accumulated INT32 partial sum
    logic [3:0]  dst_id;         // original destination (reduce root)
    logic [1:0]  vc_id;          // VC to use on forwarded flit
    logic [7:0]  age;            // timeout counter
  } sp_entry_t;

  sp_entry_t scratchpad [SP_DEPTH];
  logic       meta_valid [256];
  logic [3:0] meta_target[256];

  // =========================================================================
  // Combinational: for each input port, decide if flit is interceptable
  // =========================================================================
  // A flit is "interceptable" iff:
  //   - engine is enabled
  //   - it is a single-flit (FLIT_HEADTAIL) MSG_REDUCE packet
  //   - this router is NOT the destination (if dst == this router it goes
  //     to the reduce_engine at the NI, not intercepted here)
  // =========================================================================
  logic is_reduce_single [NUM_PORTS];
  logic can_intercept    [NUM_PORTS];
  logic has_tree_child   [NUM_PORTS];
  logic engine_ready;
  logic [7:0]  p_rid   [NUM_PORTS];   // reduce_id field
  logic [3:0]  p_exp   [NUM_PORTS];   // reduce_expect field
  logic [3:0]  p_tgt   [NUM_PORTS];   // subtree target at this router
  logic [3:0]  p_cnt   [NUM_PORTS];   // contributors represented by this flit
  logic [31:0] p_val   [NUM_PORTS];   // reduce_val field
  logic        p_meta_hit [NUM_PORTS];
  logic [ROW_BITS-1:0] p_dst_row [NUM_PORTS];
  logic [COL_BITS-1:0] p_dst_col [NUM_PORTS];

  localparam logic [ROW_BITS-1:0] CUR_ROW = node_row(NODE_BITS'(NODE_ID));
  localparam logic [COL_BITS-1:0] CUR_COL = node_col(NODE_BITS'(NODE_ID));

  function automatic logic [3:0] subtree_target;
    input logic [ROW_BITS-1:0] dst_row;
    input logic [COL_BITS-1:0] dst_col;
    input logic [3:0] expected;
    int target;
    begin
      target = 1;

      // XY routing: horizontal toward dst_col, then vertical on dst_col.
      // Count the contributors whose routed path still has to merge here.
      // Clamp by the group expected count so smaller reductions do not wait
      // for the full geometric subtree when fewer contributors are active.
      if (CUR_COL != dst_col) begin
        if (CUR_COL > dst_col)
          target = MESH_COLS - int'(CUR_COL);
        else
          target = int'(CUR_COL) + 1;
      end else if (CUR_ROW != dst_row) begin
        if (CUR_ROW > dst_row)
          target = (MESH_ROWS - int'(CUR_ROW)) * MESH_COLS;
        else
          target = (int'(CUR_ROW) + 1) * MESH_COLS;
      end

      if (target > int'(expected))
        target = int'(expected);

      subtree_target = target[3:0];
    end
  endfunction

  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++) begin
      is_reduce_single[p] = enable
                          && valid_in[p]
                          && (flit_in[p].flit_type == FLIT_HEADTAIL)
                          && (flit_in[p].msg_type  == MSG_REDUCE)
                          && (flit_in[p].dst_id    != NODE_BITS'(NODE_ID));
      p_rid[p] = flit_in[p].payload[REDUCE_ID_HI      : REDUCE_ID_LO];
      p_exp[p] = flit_in[p].payload[REDUCE_EXPECT_HI  : REDUCE_EXPECT_LO];
      p_cnt[p] = (flit_in[p].payload[REDUCE_COUNT_HI  : REDUCE_COUNT_LO] == 4'h0)
               ? 4'h1
               : flit_in[p].payload[REDUCE_COUNT_HI  : REDUCE_COUNT_LO];
      p_val[p] = flit_in[p].payload[REDUCE_VAL_HI     : REDUCE_VAL_LO];
      p_dst_row[p] = node_row(flit_in[p].dst_id);
      p_dst_col[p] = node_col(flit_in[p].dst_id);
      p_meta_hit[p] = meta_valid[p_rid[p]];
      p_tgt[p] = p_meta_hit[p]
               ? meta_target[p_rid[p]]
               : subtree_target(p_dst_row[p], p_dst_col[p], p_exp[p]);

      has_tree_child[p] = (p_tgt[p] > p_cnt[p]);
    end
  end

  // =========================================================================
  // Scratchpad lookup: for each port find the entry that matches its rid
  // Returns SP_DEPTH if no match (miss)
  // =========================================================================
  logic [$clog2(SP_DEPTH):0]  sp_hit_idx [NUM_PORTS];  // SP_DEPTH = miss
  logic                       sp_hit     [NUM_PORTS];

  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++) begin
      sp_hit_idx[p] = SP_DEPTH;  // default: miss
      sp_hit[p]     = 1'b0;
      for (int e = 0; e < SP_DEPTH; e++) begin
        if (scratchpad[e].valid && (scratchpad[e].reduce_id == p_rid[p])) begin
          sp_hit_idx[p] = $clog2(SP_DEPTH)'(e);
          sp_hit[p]     = 1'b1;
        end
      end
    end
  end

  // empty slot finder (lowest-indexed free entry)
  logic [$clog2(SP_DEPTH)-1:0]  free_slot;
  logic                          has_free;

  always_comb begin
    free_slot = '0;
    has_free  = 1'b0;
    for (int e = SP_DEPTH-1; e >= 0; e--) begin
      if (!scratchpad[e].valid) begin
        free_slot = $clog2(SP_DEPTH)'(e);
        has_free  = 1'b1;
      end
    end
  end

  // =========================================================================
  // Priority arbiter: one port processed per cycle (round-robin)
  // =========================================================================
  logic [$clog2(NUM_PORTS)-1:0] rr_ptr;

  logic [$clog2(NUM_PORTS)-1:0] chosen_port;
  logic                          have_work;

  assign engine_ready = (inj_state == INJ_IDLE) && !emit_pending;

  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++) begin
      can_intercept[p] = is_reduce_single[p]
                       && engine_ready
                       && (sp_hit[p] ||
                           (has_tree_child[p] && has_free && (p_cnt[p] < p_exp[p])));
    end
  end

  always_comb begin
    chosen_port = '0;
    have_work   = 1'b0;
    // Round-robin starting from rr_ptr
    for (int k = 0; k < NUM_PORTS; k++) begin
      automatic int p = (int'(rr_ptr) + k) % NUM_PORTS;
      if (can_intercept[p] && !have_work) begin
        chosen_port = $clog2(NUM_PORTS)'(p);
        have_work   = 1'b1;
      end
    end
  end

  // Intercept: only the chosen port is intercepted this cycle
  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++)
      intercept[p] = have_work && (p == int'(chosen_port));
  end

  // =========================================================================
  // Inject FSM:  IDLE → accumulate / emit
  // =========================================================================
  // We operate in 1-cycle burst: process chosen_port, update scratchpad,
  // and if ready to emit then drive inject_flit.
  // Emission may take >1 cycle if inject_ready is low.
  // =========================================================================
  typedef enum logic [1:0] {
    INJ_IDLE,
    INJ_WAIT_CREDIT,
    INJ_EMIT
  } inj_state_e;

  inj_state_e inj_state;

  flit_t    emit_flit_r;
  logic     emit_pending;

  // ---------- scratchpad update (sequential) ----------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rr_ptr       <= '0;
      inj_state    <= INJ_IDLE;
      emit_flit_r  <= '0;
      emit_pending <= 1'b0;
      for (int e = 0; e < SP_DEPTH; e++) begin
        scratchpad[e].valid <= 1'b0;
        scratchpad[e].age   <= '0;
      end
      for (int rid = 0; rid < 256; rid++) begin
        meta_valid[rid]  <= 1'b0;
        meta_target[rid] <= '0;
      end
    end else begin

      if (cfg_valid) begin
        meta_valid[cfg_reduce_id]  <= cfg_enable;
        meta_target[cfg_reduce_id] <= cfg_target;
      end

      // ---- Age tracking / timeout ----
      for (int e = 0; e < SP_DEPTH; e++) begin
        if (scratchpad[e].valid) begin
          if (scratchpad[e].age == 8'hFF) begin
            // Timeout: clear stale entry to prevent deadlock
            scratchpad[e].valid <= 1'b0;
            scratchpad[e].age   <= '0;
          end else begin
            scratchpad[e].age <= scratchpad[e].age + 8'h1;
          end
        end
      end

      // ---- Process chosen port ----
      if (have_work) begin
        rr_ptr <= (rr_ptr == $clog2(NUM_PORTS)'(NUM_PORTS-1)) ? '0 : rr_ptr + 1'b1;

        if (sp_hit[chosen_port]) begin
          // --- HIT: accumulate ---
          automatic int idx = int'(sp_hit_idx[chosen_port]);
          scratchpad[idx].acc_val <= scratchpad[idx].acc_val + p_val[chosen_port];
          scratchpad[idx].count   <= scratchpad[idx].count + p_cnt[chosen_port];
          scratchpad[idx].age     <= '0;  // reset timeout

          if (scratchpad[idx].count + p_cnt[chosen_port] >= scratchpad[idx].target) begin
            // Local subtree complete — emit a partial toward the root.
            emit_flit_r <= make_reduce_flit(
              NODE_BITS'(NODE_ID),
              scratchpad[idx].dst_id,
              VC_BITS'(REDUCE_VC),
              scratchpad[idx].reduce_id,
              scratchpad[idx].expected,
              scratchpad[idx].acc_val + p_val[chosen_port],
              scratchpad[idx].count + p_cnt[chosen_port]
            );
            scratchpad[idx].valid <= 1'b0;
            emit_pending          <= 1'b1;
            inj_state             <= inject_ready ? INJ_IDLE : INJ_WAIT_CREDIT;
          end
        end else begin
          // --- MISS: store first contribution ---
          if (has_free) begin
            scratchpad[free_slot].valid     <= 1'b1;
            scratchpad[free_slot].reduce_id <= p_rid[chosen_port];
            scratchpad[free_slot].expected  <= p_exp[chosen_port];
            scratchpad[free_slot].target    <= p_tgt[chosen_port];
            scratchpad[free_slot].count     <= p_cnt[chosen_port];
            scratchpad[free_slot].acc_val   <= p_val[chosen_port];
            scratchpad[free_slot].dst_id    <= flit_in[chosen_port].dst_id;
            scratchpad[free_slot].vc_id     <= VC_BITS'(REDUCE_VC);
            scratchpad[free_slot].age       <= '0;
          end
          // If no free slot: flit passes through unmodified (safe fallback)
        end
      end

      // ---- Inject state machine ----
      case (inj_state)
        INJ_IDLE: begin
          // Emission handled inline above
          if (emit_pending && inject_ready) begin
            emit_pending <= 1'b0;
          end
        end

        INJ_WAIT_CREDIT: begin
          if (inject_ready) begin
            emit_pending <= 1'b0;
            inj_state    <= INJ_IDLE;
          end
        end

        default: inj_state <= INJ_IDLE;
      endcase
    end
  end

  // ---------- Output assignments ----------
  assign inject_valid = emit_pending;
  assign inject_flit  = emit_flit_r;

endmodule
