// =============================================================================
// l1_data_array.sv — L1 Data Cache Data Storage
// =============================================================================
// 16 sets × 4 ways × 64 bytes per line = 4KB total.
// Supports word-granularity read/write with byte enables.
// Also supports full-line read/write for fills and writebacks.

/* verilator lint_off UNUSEDSIGNAL */

module l1_data_array #(
  parameter int NUM_SETS   = 16,
  parameter int NUM_WAYS   = 4,
  parameter int LINE_BYTES = 64,
  parameter int WORD_WIDTH = 32
) (
  input  logic                                    clk,

  // -----------------------------------------------------------------------
  // Word-level access (for CPU load/store)
  // -----------------------------------------------------------------------
  input  logic                                    word_en,
  input  logic                                    word_we,
  input  logic [$clog2(NUM_SETS)-1:0]             word_set,
  input  logic [$clog2(NUM_WAYS)-1:0]             word_way,
  input  logic [$clog2(LINE_BYTES)-1:0]           word_offset,  // byte offset within line
  input  logic [WORD_WIDTH/8-1:0]                 word_be,      // byte enables
  input  logic [WORD_WIDTH-1:0]                   word_wdata,
  output logic [WORD_WIDTH-1:0]                   word_rdata,

  // -----------------------------------------------------------------------
  // Line-level access (for cache fill / writeback)
  // -----------------------------------------------------------------------
  input  logic                                    line_en,
  input  logic                                    line_we,
  input  logic [$clog2(NUM_SETS)-1:0]             line_set,
  input  logic [$clog2(NUM_WAYS)-1:0]             line_way,
  input  logic [LINE_BYTES*8-1:0]                 line_wdata,
  output logic [LINE_BYTES*8-1:0]                 line_rdata
);

  // -----------------------------------------------------------------------
  // Storage: flattened byte array per set per way
  // -----------------------------------------------------------------------
  localparam int OFFSET_W = $clog2(LINE_BYTES);

  logic [7:0] data [NUM_SETS][NUM_WAYS][LINE_BYTES];

  // -----------------------------------------------------------------------
  // Word read (combinational)
  // -----------------------------------------------------------------------
  logic [OFFSET_W-1:0] word_base;
  assign word_base = {word_offset[OFFSET_W-1:2], 2'b00}; // word-aligned

  always_comb begin
    word_rdata = '0;
    for (int b = 0; b < WORD_WIDTH/8; b++) begin
      word_rdata[b*8 +: 8] = data[word_set][word_way][word_base + OFFSET_W'(b)];
    end
  end

  // -----------------------------------------------------------------------
  // Line read (combinational)
  // -----------------------------------------------------------------------
  always_comb begin
    line_rdata = '0;
    for (int b = 0; b < LINE_BYTES; b++) begin
      line_rdata[b*8 +: 8] = data[line_set][line_way][b];
    end
  end

  // -----------------------------------------------------------------------
  // Writes (sequential)
  // -----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    // Word write with byte enables
    if (word_en && word_we) begin
      for (int b = 0; b < WORD_WIDTH/8; b++) begin
        if (word_be[b]) begin
          data[word_set][word_way][word_base + OFFSET_W'(b)] <= word_wdata[b*8 +: 8];
        end
      end
    end

    // Line write (full cache line fill)
    if (line_en && line_we) begin
      for (int b = 0; b < LINE_BYTES; b++) begin
        data[line_set][line_way][b] <= line_wdata[b*8 +: 8];
      end
    end
  end

endmodule
