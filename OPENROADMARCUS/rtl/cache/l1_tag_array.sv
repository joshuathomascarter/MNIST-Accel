// =============================================================================
// l1_tag_array.sv — L1 Data Cache Tag Storage (SRAM-backed)
// =============================================================================
// 16 sets × 4 ways.  Each entry: valid(1), dirty(1), tag[21:0] = 24 bits.
//
// Storage: 4 × sram_1rw_wrapper (one per way).
//   SRAM address:  {8'b0, set[3:0]}   → 12-bit address, depth 16 used
//   SRAM word:     {8'b0, valid, dirty, tag[21:0]} packed into 32 bits
//
// Read latency: 1 cycle (registered SRAM output).
// The cache controller issues the read in S_TAG_READ and evaluates the
// result in S_COMPARE_TAG one cycle later.
//
// Write / Invalidate are single-cycle (synchronous write to SRAM).
//
// Tag comparison (hit detection) is done combinationally from the
// REGISTERED sram output — no behavioral array, no decoder fanout.

/* verilator lint_off UNUSEDSIGNAL */

module l1_tag_array #(
  parameter int ADDR_WIDTH  = 32,
  parameter int NUM_SETS    = 16,
  parameter int NUM_WAYS    = 4,
  parameter int LINE_BYTES  = 64
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ── Read port (1-cycle registered) ────────────────────────────────────────
  // Issue: assert rd_en with rd_set.  Results appear 1 cycle later on
  // rd_hit / rd_way / rd_dirty / rd_tag / rd_valid.
  input  logic                          rd_en,
  input  logic [$clog2(NUM_SETS)-1:0]   rd_set,
  input  logic [TAG_BITS-1:0]           rd_tag_cmp,   // tag to compare against

  output logic                          rd_hit,
  output logic [$clog2(NUM_WAYS)-1:0]   rd_way,
  output logic                          rd_dirty,

  // ── Read-back port (for writeback address reconstruction, 1-cycle) ────────
  input  logic                          rb_en,
  input  logic [$clog2(NUM_SETS)-1:0]   rb_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   rb_way_sel,
  output logic [TAG_BITS-1:0]           rb_tag,
  output logic                          rb_dirty,
  output logic                          rb_valid,

  // ── Write port ─────────────────────────────────────────────────────────────
  input  logic                          write_en,
  input  logic [$clog2(NUM_SETS)-1:0]   write_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   write_way,
  input  logic [TAG_BITS-1:0]           write_tag,
  input  logic                          write_valid,
  input  logic                          write_dirty,

  // ── Invalidate port ────────────────────────────────────────────────────────
  input  logic                          inv_en,
  input  logic [$clog2(NUM_SETS)-1:0]   inv_set,
  input  logic [$clog2(NUM_WAYS)-1:0]   inv_way
);

  // -------------------------------------------------------------------------
  // Derived parameters
  // -------------------------------------------------------------------------
  localparam int OFFSET_BITS = $clog2(LINE_BYTES);   // 6
  localparam int SET_BITS    = $clog2(NUM_SETS);      // 4
  localparam int TAG_BITS    = ADDR_WIDTH - SET_BITS - OFFSET_BITS;  // 22

  // SRAM word packing: { (32-2-TAG_BITS)'b0, valid, dirty, tag[TAG_BITS-1:0] }
  localparam int WORD_VALID_BIT = TAG_BITS + 1;
  localparam int WORD_DIRTY_BIT = TAG_BITS;
  localparam int WORD_TAG_HI    = TAG_BITS - 1;
  localparam int WORD_TAG_LO    = 0;

  // -------------------------------------------------------------------------
  // Per-way SRAM signals
  // -------------------------------------------------------------------------
  logic [11:0] sram_addr  [0:NUM_WAYS-1];
  logic [31:0] sram_wdata [0:NUM_WAYS-1];
  logic        sram_en    [0:NUM_WAYS-1];
  logic        sram_we    [0:NUM_WAYS-1];
  logic [31:0] sram_rdata [0:NUM_WAYS-1];

  // -------------------------------------------------------------------------
  // SRAM port arbitration
  // For each way:
  //   • write or invalidate takes priority (en=1, we=1)
  //   • read-back uses set from rb_set (en=1, we=0)
  //   • lookup read uses set from rd_set (en=rd_en, we=0)
  //   Note: The controller guarantees write/inv and read are mutually
  //   exclusive (write happens in S_REFILL_DONE, read in S_TAG_READ).
  // -------------------------------------------------------------------------
  genvar w;
  generate
    for (w = 0; w < NUM_WAYS; w++) begin : gen_ways
      always_comb begin
        if (write_en && (write_way == w[$clog2(NUM_WAYS)-1:0])) begin
          sram_en   [w] = 1'b1;
          sram_we   [w] = 1'b1;
          sram_addr [w] = {{(12-SET_BITS){1'b0}}, write_set};
          sram_wdata[w] = {8'b0,
                           write_valid,
                           write_dirty,
                           write_tag[TAG_BITS-1:0]};
        end else if (inv_en && (inv_way == w[$clog2(NUM_WAYS)-1:0])) begin
          sram_en   [w] = 1'b1;
          sram_we   [w] = 1'b1;
          sram_addr [w] = {{(12-SET_BITS){1'b0}}, inv_set};
          sram_wdata[w] = 32'b0;   // valid=0, dirty=0, tag=0
        end else if (rb_en) begin
          sram_en   [w] = 1'b1;
          sram_we   [w] = 1'b0;
          sram_addr [w] = {{(12-SET_BITS){1'b0}}, rb_set};
          sram_wdata[w] = 32'b0;
        end else begin
          sram_en   [w] = rd_en;
          sram_we   [w] = 1'b0;
          sram_addr [w] = {{(12-SET_BITS){1'b0}}, rd_set};
          sram_wdata[w] = 32'b0;
        end
      end

      sram_1rw_wrapper u_sram (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (sram_en   [w]),
        .we    (sram_we   [w]),
        .addr  (sram_addr [w]),
        .wdata (sram_wdata[w]),
        .rdata (sram_rdata[w])
      );
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Tag comparison — combinational from REGISTERED SRAM outputs
  // (SRAM outputs are valid 1 cycle after the read was issued)
  // -------------------------------------------------------------------------
  // Capture which tag we're comparing against (registered with the read)
  logic [TAG_BITS-1:0]           rd_tag_cmp_r;
  logic [$clog2(NUM_WAYS)-1:0]   rb_way_sel_r;

  always_ff @(posedge clk) begin
    rd_tag_cmp_r <= rd_tag_cmp;
    rb_way_sel_r <= rb_way_sel;
  end

  always_comb begin
    rd_hit   = 1'b0;
    rd_way   = '0;
    rd_dirty = 1'b0;
    for (int i = 0; i < NUM_WAYS; i++) begin
      if (sram_rdata[i][WORD_VALID_BIT] &&
          (sram_rdata[i][WORD_TAG_HI:WORD_TAG_LO] == rd_tag_cmp_r)) begin
        rd_hit   = 1'b1;
        rd_way   = $clog2(NUM_WAYS)'(i);
        rd_dirty = sram_rdata[i][WORD_DIRTY_BIT];
      end
    end
  end

  // -------------------------------------------------------------------------
  // Read-back output — mux registered SRAM output by way
  // -------------------------------------------------------------------------
  assign rb_tag   = sram_rdata[rb_way_sel_r][WORD_TAG_HI:WORD_TAG_LO];
  assign rb_dirty = sram_rdata[rb_way_sel_r][WORD_DIRTY_BIT];
  assign rb_valid = sram_rdata[rb_way_sel_r][WORD_VALID_BIT];

endmodule
