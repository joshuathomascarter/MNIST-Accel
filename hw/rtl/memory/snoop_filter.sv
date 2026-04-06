// =============================================================================
// snoop_filter.sv — Snoop Filter (Inclusive Directory Cache)
// =============================================================================
// Tracks which L1 caches hold copies of cache lines. Reduces snoop traffic
// by filtering unnecessary snoops — only sends snoops to caches that
// actually hold the line.
//
// Standalone demo module for the coherence subsystem.

/* verilator lint_off IMPORTSTAR */
import coherence_pkg::*;

module snoop_filter #(
  parameter int NUM_ENTRIES = 128,     // Number of tracked lines
  parameter int NUM_NODES   = 4,       // Number of L1 caches
  parameter int ADDR_WIDTH  = COH_ADDR_W,
  parameter int LINE_OFFSET = 5        // log2(32 bytes/line)
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Lookup request ---
  input  logic                   lookup_valid,
  output logic                   lookup_ready,
  input  logic [ADDR_WIDTH-1:0]  lookup_addr,
  input  logic [COH_NODE_W-1:0]  lookup_src,      // Who is requesting?

  // --- Lookup result ---
  output logic                   lookup_hit,
  output logic [NUM_NODES-1:0]   sharer_mask,     // Which caches to snoop
  output logic [1:0]             line_state,

  // --- Update interface (after coherence transaction completes) ---
  input  logic                   update_valid,
  input  logic [ADDR_WIDTH-1:0]  update_addr,
  input  logic [1:0]             update_state,
  input  logic [COH_NODE_W-1:0]  update_node,
  input  logic                   update_add,      // 1=add to sharers, 0=remove
  input  logic                   update_clear_all // Clear all sharers
);

  localparam int IDX_BITS = $clog2(NUM_ENTRIES);
  localparam int TAG_BITS = ADDR_WIDTH - LINE_OFFSET - IDX_BITS;

  // =========================================================================
  // Filter storage (set-associative, 2-way for simplicity)
  // =========================================================================
  localparam int NUM_WAYS = 2;
  localparam int WAY_BITS = $clog2(NUM_WAYS);

  logic                    sf_valid [NUM_ENTRIES][NUM_WAYS];
  logic [TAG_BITS-1:0]     sf_tag [NUM_ENTRIES][NUM_WAYS];
  logic [1:0]              sf_state [NUM_ENTRIES][NUM_WAYS];
  logic [NUM_NODES-1:0]    sf_sharers [NUM_ENTRIES][NUM_WAYS];

  // LRU bit per set (0 = way 0 is LRU)
  logic lru [NUM_ENTRIES];

  // =========================================================================
  // Index/tag extraction
  // =========================================================================
  logic [IDX_BITS-1:0] lu_idx;
  logic [TAG_BITS-1:0] lu_tag;
  logic [IDX_BITS-1:0] up_idx;
  logic [TAG_BITS-1:0] up_tag;

  assign lu_idx = lookup_addr[LINE_OFFSET +: IDX_BITS];
  assign lu_tag = lookup_addr[LINE_OFFSET + IDX_BITS +: TAG_BITS];
  assign up_idx = update_addr[LINE_OFFSET +: IDX_BITS];
  assign up_tag = update_addr[LINE_OFFSET + IDX_BITS +: TAG_BITS];

  // =========================================================================
  // Lookup (combinational)
  // =========================================================================
  logic [NUM_WAYS-1:0] way_hit;
  logic any_hit;
  logic [WAY_BITS-1:0] hit_way;
  logic                update_hit_found;
  logic [WAY_BITS-1:0] update_hit_way;
  logic [WAY_BITS-1:0] update_victim_way;
  logic [NUM_NODES-1:0] update_node_mask;

  assign update_node_mask = NUM_NODES'(1) << update_node;

  always_comb begin
    any_hit     = 1'b0;
    hit_way     = '0;
    sharer_mask = '0;
    line_state  = MESI_I;

    for (int w = 0; w < NUM_WAYS; w++) begin
      way_hit[w] = sf_valid[lu_idx][w] && (sf_tag[lu_idx][w] == lu_tag);
      if (way_hit[w]) begin
        any_hit     = 1'b1;
        hit_way     = WAY_BITS'(w);
        sharer_mask = sf_sharers[lu_idx][w];
        line_state  = sf_state[lu_idx][w];
      end
    end

    lookup_hit = any_hit;

    // Exclude requester from snoop mask
    if (any_hit)
      sharer_mask[lookup_src] = 1'b0;
  end

  always_comb begin
    update_hit_found = 1'b0;
    update_hit_way = '0;
    update_victim_way = lru[up_idx] ? '0 : WAY_BITS'(1);

    for (int w = 0; w < NUM_WAYS; w++) begin
      if (sf_valid[up_idx][w] && (sf_tag[up_idx][w] == up_tag)) begin
        update_hit_found = 1'b1;
        update_hit_way = WAY_BITS'(w);
      end
      if (!sf_valid[up_idx][w])
        update_victim_way = WAY_BITS'(w);
    end
  end

  assign lookup_ready = 1'b1; // Always ready (combinational lookup)

  // =========================================================================
  // Update (sequential)
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_ENTRIES; i++) begin
        for (int w = 0; w < NUM_WAYS; w++) begin
          sf_valid[i][w] <= 1'b0;
          sf_tag[i][w] <= '0;
          sf_state[i][w] <= MESI_I;
          sf_sharers[i][w] <= '0;
        end
        lru[i] <= 1'b0;
      end
    end else begin
      // LRU update on lookup hit
      if (lookup_valid && any_hit)
        lru[lu_idx] <= ~hit_way[0];

      // Explicit update
      if (update_valid) begin
        if (update_hit_found) begin
          sf_state[up_idx][update_hit_way] <= update_state;

          if (update_clear_all)
            sf_sharers[up_idx][update_hit_way] <= '0;
          else if (update_add)
            sf_sharers[up_idx][update_hit_way] <=
              sf_sharers[up_idx][update_hit_way] | update_node_mask;
          else
            sf_sharers[up_idx][update_hit_way] <=
              sf_sharers[up_idx][update_hit_way] & ~update_node_mask;

          // If state goes Invalid and no sharers, invalidate entry
          if (update_state == MESI_I)
            sf_valid[up_idx][update_hit_way] <= 1'b0;
        end

        // Allocate new entry if not found and adding
        if (!update_hit_found && update_add) begin
          sf_valid[up_idx][update_victim_way] <= 1'b1;
          sf_tag[up_idx][update_victim_way] <= up_tag;
          sf_state[up_idx][update_victim_way] <= update_state;
          sf_sharers[up_idx][update_victim_way] <= update_node_mask;
          lru[up_idx] <= ~update_victim_way[0];
        end
      end
    end
  end

endmodule
