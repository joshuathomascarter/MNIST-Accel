// =============================================================================
// stride_prefetcher.sv — L2 Stride Prefetcher
// =============================================================================
// Monitors L2 miss addresses and detects constant-stride patterns.
// When a stable stride is detected, issues prefetch requests to the L2.
//
// Algorithm:
//   - Table of recent miss PCs (or addresses) → stride entry
//   - Each entry: last_addr, stride, confidence counter (2-bit)
//   - Prefetch when confidence saturates

/* verilator lint_off UNUSEDSIGNAL */

module stride_prefetcher #(
  parameter int ADDR_WIDTH    = 32,
  parameter int TABLE_ENTRIES = 16,
  parameter int LINE_BYTES    = 64,
  parameter int CONF_MAX      = 3    // 2-bit saturation
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- Miss notification from L2 controller ---
  input  logic              miss_valid,
  input  logic [ADDR_WIDTH-1:0] miss_addr,

  // --- Prefetch request output ---
  output logic              pf_req_valid,
  input  logic              pf_req_ready,
  output logic [ADDR_WIDTH-1:0] pf_req_addr,

  // --- Enable / configuration ---
  input  logic              pf_enable
);

  // =========================================================================
  // Derived constants
  // =========================================================================
  localparam int LINE_ADDR_BITS = ADDR_WIDTH - $clog2(LINE_BYTES);
  localparam int IDX_BITS       = $clog2(TABLE_ENTRIES);

  // =========================================================================
  // Stride table storage
  // =========================================================================
  logic                              table_valid_r [TABLE_ENTRIES];
  logic [LINE_ADDR_BITS-1:0]         table_last_line_r [TABLE_ENTRIES];
  logic signed [LINE_ADDR_BITS-1:0]  table_stride_r [TABLE_ENTRIES];
  logic [1:0]                        table_confidence_r [TABLE_ENTRIES];

  // =========================================================================
  // Miss line address
  // =========================================================================
  logic [LINE_ADDR_BITS-1:0] miss_line;
  assign miss_line = miss_addr[ADDR_WIDTH-1:$clog2(LINE_BYTES)];

  // =========================================================================
  // Table index — hash the miss address
  // =========================================================================
  logic [IDX_BITS-1:0] tbl_idx;
  assign tbl_idx = miss_line[IDX_BITS-1:0];  // simple modulo hash

  // =========================================================================
  // Training logic
  // =========================================================================
  logic                              cur_valid;
  logic [LINE_ADDR_BITS-1:0]         cur_last_line;
  logic signed [LINE_ADDR_BITS-1:0]  cur_stride;
  logic [1:0]                        cur_confidence;

  assign cur_valid = table_valid_r[tbl_idx];
  assign cur_last_line = table_last_line_r[tbl_idx];
  assign cur_stride = table_stride_r[tbl_idx];
  assign cur_confidence = table_confidence_r[tbl_idx];

  logic signed [LINE_ADDR_BITS-1:0] observed_stride;
  assign observed_stride = miss_line - cur_last_line;

  logic stride_match;
  assign stride_match = (observed_stride == cur_stride) && cur_valid;

  // Prefetch candidate
  logic [LINE_ADDR_BITS-1:0] pf_line;
  assign pf_line = miss_line + cur_stride;

  // =========================================================================
  // State machine
  // =========================================================================
  typedef enum logic [1:0] {
    PF_IDLE,
    PF_TRAIN,
    PF_ISSUE
  } pf_state_e;

  pf_state_e state, state_next;

  logic should_prefetch;
  assign should_prefetch = pf_enable && stride_match &&
                           (cur_confidence == CONF_MAX[1:0]);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= PF_IDLE;
    else
      state <= state_next;
  end

  always_comb begin
    state_next = state;
    case (state)
      PF_IDLE: begin
        if (miss_valid)
          state_next = PF_TRAIN;
      end

      PF_TRAIN: begin
        if (should_prefetch)
          state_next = PF_ISSUE;
        else
          state_next = PF_IDLE;
      end

      PF_ISSUE: begin
        if (pf_req_ready)
          state_next = PF_IDLE;
      end

      default: state_next = PF_IDLE;
    endcase
  end

  // =========================================================================
  // Table update
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < TABLE_ENTRIES; i++)
        table_valid_r[i] <= 1'b0;
      for (int i = 0; i < TABLE_ENTRIES; i++)
        table_last_line_r[i] <= '0;
      for (int i = 0; i < TABLE_ENTRIES; i++)
        table_stride_r[i] <= '0;
      for (int i = 0; i < TABLE_ENTRIES; i++)
        table_confidence_r[i] <= '0;
    end else if (state == PF_TRAIN) begin
      // Always update last_line
      table_last_line_r[tbl_idx] <= miss_line;

      if (!cur_valid) begin
        // First access — initialize entry
        table_valid_r[tbl_idx] <= 1'b1;
        table_stride_r[tbl_idx] <= '0;
        table_confidence_r[tbl_idx] <= '0;
      end else if (stride_match) begin
        // Stride confirmed — increase confidence
        if (cur_confidence < CONF_MAX[1:0])
          table_confidence_r[tbl_idx] <= cur_confidence + 2'd1;
      end else begin
        // New stride — reset confidence, record new stride
        table_stride_r[tbl_idx] <= observed_stride;
        table_confidence_r[tbl_idx] <= 2'd0;
      end
    end
  end

  // =========================================================================
  // Prefetch request
  // =========================================================================
  logic [ADDR_WIDTH-1:0] pf_addr_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      pf_addr_reg <= '0;
    else if (state == PF_TRAIN && should_prefetch)
      pf_addr_reg <= {pf_line, {$clog2(LINE_BYTES){1'b0}}};
  end

  assign pf_req_valid = (state == PF_ISSUE);
  assign pf_req_addr  = pf_addr_reg;

endmodule
