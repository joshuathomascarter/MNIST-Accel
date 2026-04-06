// =============================================================================
// l1_tag_array.sv — L1 Data Cache Tag Storage
// =============================================================================
// 16 sets × 4 ways. Each entry: valid, dirty, tag.
// Tag = addr[31 : (SET_BITS + OFFSET_BITS)]
//
// For 4KB cache: 16 sets × 4 ways × 64B lines
//   OFFSET_BITS = 6 (64 bytes per line)
//   SET_BITS    = 4 (16 sets)
//   TAG_BITS    = 32 - 6 - 4 = 22

/* verilator lint_off UNUSEDSIGNAL */

module l1_tag_array #(
  parameter int ADDR_WIDTH  = 32,
  parameter int NUM_SETS    = 16,
  parameter int NUM_WAYS    = 4,
  parameter int LINE_BYTES  = 64
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // Lookup port (combinational read)
  input  logic [ADDR_WIDTH-1:0]         lookup_addr,
  output logic                          lookup_hit,
  output logic [$clog2(NUM_WAYS)-1:0]   lookup_way,
  output logic                          lookup_dirty,

  // Write port — set valid+tag on fill, set dirty on store
  input  logic                          write_en,
  input  logic [$clog2(NUM_SETS)-1:0]   write_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   write_way,
  input  logic [TAG_BITS-1:0]           write_tag,
  input  logic                          write_valid,
  input  logic                          write_dirty,

  // Invalidate port
  input  logic                          inv_en,
  input  logic [$clog2(NUM_SETS)-1:0]   inv_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   inv_way,

  // Read back tag+dirty for writeback
  input  logic [$clog2(NUM_SETS)-1:0]   rb_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   rb_way,
  output logic [TAG_BITS-1:0]           rb_tag,
  output logic                          rb_dirty,
  output logic                          rb_valid
);

  // -----------------------------------------------------------------------
  // Derived parameters
  // -----------------------------------------------------------------------
  localparam int OFFSET_BITS = $clog2(LINE_BYTES);   // 6
  localparam int SET_BITS    = $clog2(NUM_SETS);      // 4
  localparam int TAG_BITS    = ADDR_WIDTH - SET_BITS - OFFSET_BITS;  // 22

  // -----------------------------------------------------------------------
  // Address decomposition
  // -----------------------------------------------------------------------
  logic [TAG_BITS-1:0]  addr_tag;
  logic [SET_BITS-1:0]  addr_set;

  assign addr_tag = lookup_addr[ADDR_WIDTH-1 -: TAG_BITS];
  assign addr_set = lookup_addr[OFFSET_BITS +: SET_BITS];

  // -----------------------------------------------------------------------
  // Storage arrays
  // -----------------------------------------------------------------------
  logic                valid [NUM_SETS][NUM_WAYS];
  logic                dirty [NUM_SETS][NUM_WAYS];
  logic [TAG_BITS-1:0] tags  [NUM_SETS][NUM_WAYS];

  // -----------------------------------------------------------------------
  // Tag comparison (combinational)
  // -----------------------------------------------------------------------
  always_comb begin
    lookup_hit = 1'b0;
    lookup_way = '0;
    lookup_dirty = 1'b0;

    for (int w = 0; w < NUM_WAYS; w++) begin
      if (valid[addr_set][w] && (tags[addr_set][w] == addr_tag)) begin
        lookup_hit   = 1'b1;
        lookup_way   = w[$clog2(NUM_WAYS)-1:0];
        lookup_dirty = dirty[addr_set][w];
      end
    end
  end

  // -----------------------------------------------------------------------
  // Read-back port (for writeback address reconstruction)
  // -----------------------------------------------------------------------
  assign rb_tag   = tags[rb_set][rb_way];
  assign rb_dirty = dirty[rb_set][rb_way];
  assign rb_valid = valid[rb_set][rb_way];

  // -----------------------------------------------------------------------
  // Write / Invalidate
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int s = 0; s < NUM_SETS; s++) begin
        for (int w = 0; w < NUM_WAYS; w++) begin
          valid[s][w] <= 1'b0;
          dirty[s][w] <= 1'b0;
          tags[s][w]  <= '0;
        end
      end
    end else begin
      if (write_en) begin
        valid[write_set][write_way] <= write_valid;
        dirty[write_set][write_way] <= write_dirty;
        tags[write_set][write_way]  <= write_tag;
      end
      if (inv_en) begin
        valid[inv_set][inv_way] <= 1'b0;
      end
    end
  end

endmodule
