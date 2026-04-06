// =============================================================================
// l2_data_array.sv — L2 Cache Data Storage
// =============================================================================
// Single-ported SRAM-style data storage for L2 cache.
// Supports word-granularity reads and full-line writes (for fills/evictions).
// Organized as NUM_SETS × NUM_WAYS × LINE_BYTES.

/* verilator lint_off UNUSEDPARAM */

module l2_data_array #(
  parameter int ADDR_WIDTH  = 32,
  parameter int DATA_WIDTH  = 32,
  parameter int NUM_SETS    = 256,
  parameter int NUM_WAYS    = 8,
  parameter int LINE_BYTES  = 64,
  localparam int WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8),
  localparam int SET_BITS       = $clog2(NUM_SETS),
  localparam int WAY_BITS       = $clog2(NUM_WAYS),
  localparam int WORD_BITS      = $clog2(WORDS_PER_LINE)
) (
  input  logic              clk,

  // --- Word read port ---
  input  logic              rd_en,
  input  logic [SET_BITS-1:0]            rd_set,
  input  logic [WAY_BITS-1:0]            rd_way,
  input  logic [WORD_BITS-1:0]           rd_word,
  output logic [DATA_WIDTH-1:0]          rd_data,

  // --- Word write port (for store-hit) ---
  input  logic              wr_en,
  input  logic [SET_BITS-1:0]            wr_set,
  input  logic [WAY_BITS-1:0]            wr_way,
  input  logic [WORD_BITS-1:0]           wr_word,
  input  logic [DATA_WIDTH-1:0]          wr_data,
  input  logic [DATA_WIDTH/8-1:0]        wr_be,

  // --- Full-line read port (for eviction writeback) ---
  input  logic              line_rd_en,
  input  logic [SET_BITS-1:0]            line_rd_set,
  input  logic [WAY_BITS-1:0]            line_rd_way,
  input  logic [WORD_BITS-1:0]           line_rd_word,
  output logic [DATA_WIDTH-1:0]          line_rd_data,

  // --- Full-line write port (for line fill from DRAM) ---
  input  logic              line_wr_en,
  input  logic [SET_BITS-1:0]            line_wr_set,
  input  logic [WAY_BITS-1:0]            line_wr_way,
  input  logic [WORD_BITS-1:0]           line_wr_word,
  input  logic [DATA_WIDTH-1:0]          line_wr_data
);

  // =========================================================================
  // Data storage  — large, will map to SRAM macros in synthesis
  // =========================================================================
  logic [DATA_WIDTH-1:0] data [NUM_SETS][NUM_WAYS][WORDS_PER_LINE];

  // =========================================================================
  // Word-level read
  // =========================================================================
  always_ff @(posedge clk) begin
    if (rd_en)
      rd_data <= data[rd_set][rd_way][rd_word];
  end

  // =========================================================================
  // Word-level write (byte-enable for partial stores)
  // =========================================================================
  always_ff @(posedge clk) begin
    if (wr_en) begin
      for (int b = 0; b < DATA_WIDTH/8; b++) begin
        if (wr_be[b])
          data[wr_set][wr_way][wr_word][b*8 +: 8] <= wr_data[b*8 +: 8];
      end
    end
  end

  // =========================================================================
  // Line-level read (one word per cycle during eviction)
  // =========================================================================
  always_ff @(posedge clk) begin
    if (line_rd_en)
      line_rd_data <= data[line_rd_set][line_rd_way][line_rd_word];
  end

  // =========================================================================
  // Line-level write (one word per cycle during fill)
  // =========================================================================
  always_ff @(posedge clk) begin
    if (line_wr_en)
      data[line_wr_set][line_wr_way][line_wr_word] <= line_wr_data;
  end

endmodule
