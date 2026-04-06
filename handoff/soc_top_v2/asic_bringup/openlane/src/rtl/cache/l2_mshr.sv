// =============================================================================
// l2_mshr.sv — Miss Status Holding Registers for Non-Blocking L2
// =============================================================================
// Tracks outstanding cache misses to allow multiple requests in flight.
// Each MSHR entry stores: address, requesting ID, state.
// Supports secondary miss merging (hit-under-miss for same line).

/* verilator lint_off UNUSEDSIGNAL */

module l2_mshr #(
  parameter int ADDR_WIDTH   = 32,
  parameter int NUM_ENTRIES  = 4,
  parameter int ID_WIDTH     = 4,
  parameter int LINE_BYTES   = 64
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- Allocate a new MSHR entry (on L2 miss) ---
  input  logic              alloc_valid,
  output logic              alloc_ready,       // 0 if all entries full
  input  logic [ADDR_WIDTH-1:0] alloc_addr,
  input  logic [ID_WIDTH-1:0]   alloc_id,
  input  logic              alloc_is_write,
  output logic [$clog2(NUM_ENTRIES)-1:0] alloc_idx,

  // --- Lookup: check if address already tracked (secondary miss) ---
  input  logic              lookup_valid,
  input  logic [ADDR_WIDTH-1:0] lookup_addr,
  output logic              lookup_hit,
  output logic [$clog2(NUM_ENTRIES)-1:0] lookup_idx,

  // --- Complete / deallocate (when fill arrives from DRAM) ---
  input  logic              complete_valid,
  input  logic [$clog2(NUM_ENTRIES)-1:0] complete_idx,
  output logic [ADDR_WIDTH-1:0] complete_addr,
  output logic [ID_WIDTH-1:0]   complete_id,
  output logic              complete_is_write,

  // --- Status ---
  output logic              full,
  output logic              empty,
  output logic [$clog2(NUM_ENTRIES):0] count
);

  // =========================================================================
  // MSHR entry
  // =========================================================================
  localparam int LINE_ADDR_WIDTH = ADDR_WIDTH - $clog2(LINE_BYTES);

  logic                        entry_valid     [NUM_ENTRIES];
  logic [LINE_ADDR_WIDTH-1:0]  entry_line_addr [NUM_ENTRIES];
  logic [ID_WIDTH-1:0]         entry_req_id    [NUM_ENTRIES];
  logic                        entry_is_write  [NUM_ENTRIES];

  // =========================================================================
  // Free-list tracking
  // =========================================================================
  logic [NUM_ENTRIES-1:0] active_bits;

  always_comb begin
    for (int i = 0; i < NUM_ENTRIES; i++)
      active_bits[i] = entry_valid[i];
  end

  assign full  = &active_bits;
  assign empty = ~|active_bits;

  // Count active entries
  always_comb begin
    count = '0;
    for (int i = 0; i < NUM_ENTRIES; i++)
      count = count + {{($clog2(NUM_ENTRIES)){1'b0}}, active_bits[i]};
  end

  // Find first free entry
  logic                   has_free;

  always_comb begin
    has_free  = 1'b0;
    alloc_idx = '0;
    for (int i = NUM_ENTRIES - 1; i >= 0; i--) begin
      if (!entry_valid[i]) begin
        has_free      = 1'b1;
        alloc_idx     = i[$clog2(NUM_ENTRIES)-1:0];
      end
    end
  end

  assign alloc_ready = has_free;

  // =========================================================================
  // Line address comparison for secondary miss detection
  // =========================================================================
  logic [LINE_ADDR_WIDTH-1:0] lookup_line_addr;
  assign lookup_line_addr = lookup_addr[ADDR_WIDTH-1:$clog2(LINE_BYTES)];

  always_comb begin
    lookup_hit = 1'b0;
    lookup_idx = '0;
    if (lookup_valid) begin
      for (int i = 0; i < NUM_ENTRIES; i++) begin
        if (entry_valid[i] && entry_line_addr[i] == lookup_line_addr) begin
          lookup_hit = 1'b1;
          lookup_idx = i[$clog2(NUM_ENTRIES)-1:0];
        end
      end
    end
  end

  // =========================================================================
  // Complete readback
  // =========================================================================
  assign complete_addr     = {entry_line_addr[complete_idx], {$clog2(LINE_BYTES){1'b0}}};
  assign complete_id       = entry_req_id[complete_idx];
  assign complete_is_write = entry_is_write[complete_idx];

  // =========================================================================
  // Allocate / Complete
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_ENTRIES; i++) begin
        entry_valid[i]     <= 1'b0;
        entry_line_addr[i] <= '0;
        entry_req_id[i]    <= '0;
        entry_is_write[i]  <= 1'b0;
      end
    end else begin
      // Deallocate on complete
      if (complete_valid)
        entry_valid[complete_idx] <= 1'b0;

      // Allocate new entry
      if (alloc_valid && alloc_ready) begin
        entry_valid[alloc_idx]     <= 1'b1;
        entry_line_addr[alloc_idx] <= alloc_addr[ADDR_WIDTH-1:$clog2(LINE_BYTES)];
        entry_req_id[alloc_idx]    <= alloc_id;
        entry_is_write[alloc_idx]  <= alloc_is_write;
      end
    end
  end

endmodule
