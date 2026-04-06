// =============================================================================
// l2_tag_array.sv — L2 Cache Tag Storage
// =============================================================================
// Dual-ported tag SRAM: one read port and one write port.
// Each entry: valid, dirty, tag bits.
// Supports multi-way associative lookup with parallel comparators.

module l2_tag_array #(
  parameter int ADDR_WIDTH  = 32,
  parameter int NUM_SETS    = 256,
  parameter int NUM_WAYS    = 8,
  parameter int LINE_BYTES  = 64,
  localparam int OFFSET_BITS = $clog2(LINE_BYTES),
  localparam int INDEX_BITS  = $clog2(NUM_SETS),
  localparam int TAG_WIDTH   = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS,
  localparam int WAY_BITS    = $clog2(NUM_WAYS)
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- Lookup port ---
  input  logic              lookup_valid,
  input  logic [INDEX_BITS-1:0]        lookup_set,
  input  logic [TAG_WIDTH-1:0]        lookup_tag,
  output logic              lookup_hit,
  output logic [WAY_BITS-1:0]          lookup_way,
  output logic              lookup_dirty,

  // --- Write / update port ---
  input  logic              write_valid,
  input  logic [INDEX_BITS-1:0]        write_set,
  input  logic [WAY_BITS-1:0]          write_way,
  input  logic [TAG_WIDTH-1:0]        write_tag,
  input  logic              write_dirty,

  // --- Invalidate port (for coherence or flush) ---
  input  logic              inv_valid,
  input  logic [INDEX_BITS-1:0]        inv_set,
  input  logic [WAY_BITS-1:0]          inv_way,

  // --- Dirty-set for writeback check ---
  input  logic              dirty_check_valid,
  input  logic [INDEX_BITS-1:0]        dirty_check_set,
  input  logic [WAY_BITS-1:0]          dirty_check_way,
  output logic              dirty_check_is_dirty,
  output logic [TAG_WIDTH-1:0]        dirty_check_tag
);

  // Tag storage — split into parallel arrays for broader synthesis support.
  logic                  valid_bits [NUM_SETS][NUM_WAYS];
  logic                  dirty_bits [NUM_SETS][NUM_WAYS];
  logic [TAG_WIDTH-1:0]  tag_bits   [NUM_SETS][NUM_WAYS];

  // =========================================================================
  // Lookup — parallel tag compare
  // =========================================================================
  always_comb begin
    lookup_hit   = 1'b0;
    lookup_way   = '0;
    lookup_dirty = 1'b0;

    if (lookup_valid) begin
      for (int w = 0; w < NUM_WAYS; w++) begin
        if (valid_bits[lookup_set][w] &&
            tag_bits[lookup_set][w] == lookup_tag) begin
          lookup_hit   = 1'b1;
          lookup_way   = w[$clog2(NUM_WAYS)-1:0];
          lookup_dirty = dirty_bits[lookup_set][w];
        end
      end
    end
  end

  // =========================================================================
  // Dirty check (for eviction decision)
  // =========================================================================
  always_comb begin
    dirty_check_is_dirty = 1'b0;
    dirty_check_tag      = '0;
    if (dirty_check_valid) begin
      dirty_check_is_dirty = dirty_bits[dirty_check_set][dirty_check_way];
      dirty_check_tag      = tag_bits[dirty_check_set][dirty_check_way];
    end
  end

  // =========================================================================
  // Write / Invalidate
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int s = 0; s < NUM_SETS; s++)
        for (int w = 0; w < NUM_WAYS; w++) begin
          valid_bits[s][w] <= 1'b0;
          dirty_bits[s][w] <= 1'b0;
          tag_bits[s][w]   <= '0;
        end
    end else begin
      // Invalidation takes priority
      if (inv_valid)
        valid_bits[inv_set][inv_way] <= 1'b0;

      if (write_valid) begin
        valid_bits[write_set][write_way] <= 1'b1;
        dirty_bits[write_set][write_way] <= write_dirty;
        tag_bits[write_set][write_way]   <= write_tag;
      end
    end
  end

endmodule
