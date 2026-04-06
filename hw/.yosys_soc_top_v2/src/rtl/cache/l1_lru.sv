// =============================================================================
// l1_lru.sv — True LRU Replacement Policy Tracker
// =============================================================================
// Tracks access ordering for a 4-way set-associative cache.
// Uses a 6-bit LRU state per set (encodes all 4! = 24 orderings).
// Provides the least-recently-used way on eviction.

module l1_lru #(
  parameter int NUM_SETS = 16,
  parameter int NUM_WAYS = 4
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // Access port — update LRU on cache hit or fill
  input  logic                          access_valid,
  input  logic [$clog2(NUM_SETS)-1:0]   access_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   access_way,

  // Query port — which way to evict?
  input  logic [$clog2(NUM_SETS)-1:0]   query_set,
  output logic [$clog2(NUM_WAYS)-1:0]   victim_way
);

  // -----------------------------------------------------------------------
  // LRU state: 3 comparison bits per set for 4 ways (tree-PLRU).
  //
  //        bit[2]
  //       /      \
  //    bit[1]   bit[0]
  //    / \       / \
  //  w0  w1    w2  w3
  //
  //  bit = 0 → go left (that subtree was used more recently)
  //  bit = 1 → go right
  //  To find victim: follow the tree in the direction bits point.
  //  On access: set bits to point AWAY from the accessed way.
  // -----------------------------------------------------------------------

  logic [2:0] lru_bits [NUM_SETS];

  // -----------------------------------------------------------------------
  // Victim selection (combinational) — follow tree from root
  // -----------------------------------------------------------------------
  always_comb begin
    case (lru_bits[query_set])
      3'b000:  victim_way = 2'd0;
      3'b001:  victim_way = 2'd2;
      3'b010:  victim_way = 2'd1;
      3'b011:  victim_way = 2'd3;
      3'b100:  victim_way = 2'd0;
      3'b101:  victim_way = 2'd2;
      3'b110:  victim_way = 2'd1;
      3'b111:  victim_way = 2'd3;
      default: victim_way = 2'd0;
    endcase

    // Proper tree-PLRU decode:
    // Follow bit[2] → left(0)/right(1)
    //   if left:  follow bit[1] → way 0 or 1
    //   if right: follow bit[0] → way 2 or 3
    if (!lru_bits[query_set][2]) begin
      // Left subtree
      victim_way = lru_bits[query_set][1] ? 2'd1 : 2'd0;
    end else begin
      // Right subtree
      victim_way = lru_bits[query_set][0] ? 2'd3 : 2'd2;
    end
  end

  // -----------------------------------------------------------------------
  // Update LRU — point bits AWAY from accessed way
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_SETS; i++) begin
        lru_bits[i] <= 3'b000;
      end
    end else if (access_valid) begin
      case (access_way)
        2'd0: begin
          lru_bits[access_set][2] <= 1'b1;  // point right (away from 0)
          lru_bits[access_set][1] <= 1'b1;  // point to way 1 (away from 0)
        end
        2'd1: begin
          lru_bits[access_set][2] <= 1'b1;  // point right
          lru_bits[access_set][1] <= 1'b0;  // point to way 0 (away from 1)
        end
        2'd2: begin
          lru_bits[access_set][2] <= 1'b0;  // point left (away from 2)
          lru_bits[access_set][0] <= 1'b1;  // point to way 3 (away from 2)
        end
        2'd3: begin
          lru_bits[access_set][2] <= 1'b0;  // point left
          lru_bits[access_set][0] <= 1'b0;  // point to way 2 (away from 3)
        end
        default: ;
      endcase
    end
  end

endmodule
