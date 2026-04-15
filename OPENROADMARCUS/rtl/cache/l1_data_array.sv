// =============================================================================
// l1_data_array.sv — L1 Data Cache Data Storage (SRAM-backed)
// =============================================================================
// 16 sets × 4 ways × 64 bytes per line = 4KB total.
// Each way is backed by one sram_1rw_wrapper (256 × 32-bit words).
// Address mapping: addr[7:0] = {set[3:0], word_beat[3:0]}.
//
// Line interface is word-granular: one 32-bit word per cycle driven by
// line_beat (0..WORDS_PER_LINE-1).  The cache controller passes beat_cnt
// directly so no internal sequencer is needed.
//
// SRAM read latency is 1 cycle (registered output).  The cache controller
// accounts for this with the S_HIT_RETURN state on word reads and by
// pre-issuing line reads one beat ahead during write-back.

/* verilator lint_off UNUSEDSIGNAL */

module l1_data_array #(
  parameter int NUM_SETS   = 16,
  parameter int NUM_WAYS   = 4,
  parameter int LINE_BYTES = 64,
  parameter int WORD_WIDTH = 32
) (
  input  logic                                  clk,

  // -------------------------------------------------------------------------
  // Word-level access (CPU load/store) — 1-cycle registered read
  // -------------------------------------------------------------------------
  input  logic                                  word_en,
  input  logic                                  word_we,
  input  logic [$clog2(NUM_SETS)-1:0]           word_set,
  input  logic [$clog2(NUM_WAYS)-1:0]           word_way,
  input  logic [$clog2(LINE_BYTES)-1:0]         word_offset,   // byte offset in line
  input  logic [WORD_WIDTH/8-1:0]               word_be,       // byte enables
  input  logic [WORD_WIDTH-1:0]                 word_wdata,
  output logic [WORD_WIDTH-1:0]                 word_rdata,    // valid 1 cycle after word_en

  // -------------------------------------------------------------------------
  // Line-level access — word-granular (one 32-bit word per cycle)
  // line_beat selects which word within the cache line (0..WORDS_PER_LINE-1)
  // -------------------------------------------------------------------------
  input  logic                                  line_en,
  input  logic                                  line_we,
  input  logic [$clog2(NUM_SETS)-1:0]           line_set,
  input  logic [$clog2(NUM_WAYS)-1:0]           line_way,
  input  logic [$clog2(LINE_BYTES/4)-1:0]       line_beat,     // word index 0..15
  input  logic [WORD_WIDTH-1:0]                 line_wdata,
  output logic [WORD_WIDTH-1:0]                 line_rdata     // valid 1 cycle after line_en
);

  // -------------------------------------------------------------------------
  // Derived parameters
  // -------------------------------------------------------------------------
  localparam int WORDS_PER_LINE = LINE_BYTES / (WORD_WIDTH / 8);  // 16
  localparam int SRAM_DEPTH     = NUM_SETS * WORDS_PER_LINE;       // 256
  localparam int SRAM_ADDR_W    = $clog2(SRAM_DEPTH);              // 8

  // -------------------------------------------------------------------------
  // Per-way SRAM port signals
  // -------------------------------------------------------------------------
  logic [11:0] sram_addr  [0:NUM_WAYS-1];
  logic [31:0] sram_wdata [0:NUM_WAYS-1];
  logic        sram_en    [0:NUM_WAYS-1];
  logic        sram_we    [0:NUM_WAYS-1];
  logic [31:0] sram_rdata [0:NUM_WAYS-1];

  // -------------------------------------------------------------------------
  // Address computation
  // word access: {set, word_offset[OFFSET_W-1:2]}
  // line access: {set, line_beat}
  // -------------------------------------------------------------------------
  wire [SRAM_ADDR_W-1:0] word_sram_addr =
      {word_set, word_offset[$clog2(LINE_BYTES)-1:2]};

  wire [SRAM_ADDR_W-1:0] line_sram_addr =
      {line_set, line_beat};

  // -------------------------------------------------------------------------
  // Byte-enable masking for word writes
  // Sub-word stores: bytes not enabled are written as 0 (no read-modify-write).
  // Cache line fills always use word_be=4'hF so correctness is preserved there.
  // -------------------------------------------------------------------------
  logic [WORD_WIDTH-1:0] word_wdata_be;
  genvar bb;
  generate
    for (bb = 0; bb < WORD_WIDTH/8; bb++) begin : gen_be
      assign word_wdata_be[bb*8 +: 8] =
          word_be[bb] ? word_wdata[bb*8 +: 8] : 8'h00;
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Per-way instantiation: arbitrate word vs. line, then instantiate SRAM
  // Word access takes priority over line access (they are mutually exclusive
  // by construction in the cache controller, but priority is stated explicitly).
  // -------------------------------------------------------------------------
  genvar w;
  generate
    for (w = 0; w < NUM_WAYS; w++) begin : gen_ways
      always_comb begin
        if (word_en && (word_way == w[1:0])) begin
          sram_en   [w] = 1'b1;
          sram_we   [w] = word_we;
          sram_addr [w] = {{(12-SRAM_ADDR_W){1'b0}}, word_sram_addr};
          sram_wdata[w] = word_wdata_be;
        end else if (line_en && (line_way == w[1:0])) begin
          sram_en   [w] = 1'b1;
          sram_we   [w] = line_we;
          sram_addr [w] = {{(12-SRAM_ADDR_W){1'b0}}, line_sram_addr};
          sram_wdata[w] = line_wdata;
        end else begin
          sram_en   [w] = 1'b0;
          sram_we   [w] = 1'b0;
          sram_addr [w] = '0;
          sram_wdata[w] = '0;
        end
      end

      sram_1rw_wrapper u_sram (
        .clk   (clk),
        .rst_n (1'b1),
        .en    (sram_en   [w]),
        .we    (sram_we   [w]),
        .addr  (sram_addr [w]),
        .wdata (sram_wdata[w]),
        .rdata (sram_rdata[w])
      );
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Output mux: SRAM read is registered (1-cycle latency).
  // Register the way select so it aligns with the registered SRAM output.
  // -------------------------------------------------------------------------
  logic [$clog2(NUM_WAYS)-1:0] word_way_r;
  logic [$clog2(NUM_WAYS)-1:0] line_way_r;

  always_ff @(posedge clk) begin
    word_way_r <= word_way;
    line_way_r <= line_way;
  end

  // Variable-index unpacked-array mux (sv2v converts to case statement)
  assign word_rdata = sram_rdata[word_way_r];
  assign line_rdata = sram_rdata[line_way_r];

endmodule
