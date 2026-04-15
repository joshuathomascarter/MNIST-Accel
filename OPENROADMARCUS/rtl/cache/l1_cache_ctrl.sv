// =============================================================================
// l1_cache_ctrl.sv — L1 Data Cache Controller FSM
// =============================================================================
// Blocking cache controller.  States:
//   IDLE         — wait for CPU request
//   TAG_READ     — issue tag SRAM read (1 cycle)   ← NEW
//   COMPARE_TAG  — evaluate SRAM result (hit/miss)
//   HIT_RETURN   — wait 1 cycle for data SRAM registered read on hit
//   WRITE_BACK   — evict dirty line to memory (WORDS_PER_LINE beats)
//   ALLOCATE     — fetch new line from memory (WORDS_PER_LINE beats),
//                  writing each word directly to SRAM as it arrives
//   REFILL_DONE  — update tags, return hit word to CPU, back to IDLE
//
// Tag array: now uses sram_1rw_wrapper (registered 1-cycle read).
//   TAG_READ issues the SRAM read; COMPARE_TAG sees the result.
// Data array: 1-cycle latency (unchanged) absorbed by HIT_RETURN.
//
// CPU-side: OBI-like (req/gnt/rvalid/rdata)
// Memory-side: simple req/ack burst interface (not full AXI — wrapped by top)

module l1_cache_ctrl #(
  parameter int ADDR_WIDTH  = 32,
  parameter int DATA_WIDTH  = 32,
  parameter int NUM_SETS    = 16,
  parameter int NUM_WAYS    = 4,
  parameter int LINE_BYTES  = 64
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // -----------------------------------------------------------------------
  // CPU-side interface (OBI-like)
  // -----------------------------------------------------------------------
  input  logic                          cpu_req,
  output logic                          cpu_gnt,
  input  logic [ADDR_WIDTH-1:0]         cpu_addr,
  input  logic                          cpu_we,
  input  logic [DATA_WIDTH/8-1:0]       cpu_be,
  input  logic [DATA_WIDTH-1:0]         cpu_wdata,
  output logic                          cpu_rvalid,
  output logic [DATA_WIDTH-1:0]         cpu_rdata,

  // -----------------------------------------------------------------------
  // Memory-side interface (burst)
  // -----------------------------------------------------------------------
  output logic                          mem_req,
  input  logic                          mem_gnt,
  output logic [ADDR_WIDTH-1:0]         mem_addr,
  output logic                          mem_we,
  output logic [DATA_WIDTH-1:0]         mem_wdata,
  input  logic                          mem_rvalid,
  input  logic [DATA_WIDTH-1:0]         mem_rdata,
  output logic                          mem_last,

  // -----------------------------------------------------------------------
  // Cache status
  // -----------------------------------------------------------------------
  output logic                          cache_busy
);

  // -----------------------------------------------------------------------
  // Derived parameters
  // -----------------------------------------------------------------------
  localparam int OFFSET_BITS    = $clog2(LINE_BYTES);
  localparam int SET_BITS       = $clog2(NUM_SETS);
  localparam int TAG_BITS       = ADDR_WIDTH - SET_BITS - OFFSET_BITS;
  localparam int WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);   // 16
  localparam int WORD_IDX_BITS  = $clog2(WORDS_PER_LINE);           // 4
  localparam logic [WORD_IDX_BITS-1:0] LAST_BEAT =
      WORD_IDX_BITS'(WORDS_PER_LINE - 1);

  // -----------------------------------------------------------------------
  // FSM states
  // -----------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE,
    S_TAG_READ,      // new: issue tag SRAM read
    S_COMPARE_TAG,   // evaluate tag SRAM result
    S_HIT_RETURN,    // 1-cycle stall for data SRAM registered read
    S_WRITE_BACK,
    S_ALLOCATE,
    S_REFILL_DONE
  } state_e;

  state_e state, state_next;

  // -----------------------------------------------------------------------
  // Registered request
  // -----------------------------------------------------------------------
  logic [ADDR_WIDTH-1:0]     req_addr_r;
  logic                      req_we_r;
  logic [DATA_WIDTH/8-1:0]   req_be_r;
  logic [DATA_WIDTH-1:0]     req_wdata_r;

  // Address decomposition
  logic [TAG_BITS-1:0]       req_tag;
  logic [SET_BITS-1:0]       req_set;
  logic [OFFSET_BITS-1:0]    req_offset;

  assign req_tag    = req_addr_r[ADDR_WIDTH-1 -: TAG_BITS];
  assign req_set    = req_addr_r[OFFSET_BITS +: SET_BITS];
  assign req_offset = req_addr_r[OFFSET_BITS-1:0];

  // Word index of the requested word within the cache line
  logic [WORD_IDX_BITS-1:0]  req_word_idx;
  assign req_word_idx = req_offset[OFFSET_BITS-1:2];

  // -----------------------------------------------------------------------
  // Tag array interface (new SRAM-backed, 1-cycle read latency)
  // -----------------------------------------------------------------------
  logic                          tag_rd_en;
  logic [SET_BITS-1:0]           tag_rd_set;
  logic [TAG_BITS-1:0]           tag_rd_cmp;
  logic                          tag_rd_hit;
  logic [$clog2(NUM_WAYS)-1:0]   tag_rd_way;
  logic                          tag_rd_dirty;

  logic                          tag_rb_en;
  logic [SET_BITS-1:0]           tag_rb_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_rb_way_sel;
  logic [TAG_BITS-1:0]           tag_rb_tag;
  logic                          tag_rb_dirty;
  logic                          tag_rb_valid;

  logic                          tag_write_en;
  logic [SET_BITS-1:0]           tag_write_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_write_way;
  logic [TAG_BITS-1:0]           tag_write_tag;
  logic                          tag_write_valid;
  logic                          tag_write_dirty;

  // -----------------------------------------------------------------------
  // Data array interface
  // -----------------------------------------------------------------------
  logic                          data_word_en;
  logic                          data_word_we;
  logic [DATA_WIDTH-1:0]         data_word_rdata;

  logic                          data_line_en;
  logic                          data_line_we;
  logic [$clog2(NUM_WAYS)-1:0]   data_line_way;
  logic [WORD_IDX_BITS-1:0]      data_line_beat;
  logic [DATA_WIDTH-1:0]         data_line_wdata;
  logic [DATA_WIDTH-1:0]         data_line_rdata;

  // -----------------------------------------------------------------------
  // LRU interface
  // -----------------------------------------------------------------------
  logic                          lru_access_valid;
  logic [$clog2(NUM_WAYS)-1:0]   lru_victim_way;

  // -----------------------------------------------------------------------
  // Burst counter and victim tracking
  // -----------------------------------------------------------------------
  logic [WORD_IDX_BITS-1:0]      beat_cnt;
  logic [$clog2(NUM_WAYS)-1:0]   victim_way_r;

  // Hit word captured during ALLOCATE (avoids 512-bit fill_buffer)
  logic [DATA_WIDTH-1:0]         hit_word_r;

  // -----------------------------------------------------------------------
  // Tag array instantiation
  // -----------------------------------------------------------------------
  l1_tag_array #(
    .ADDR_WIDTH (ADDR_WIDTH),
    .NUM_SETS   (NUM_SETS),
    .NUM_WAYS   (NUM_WAYS),
    .LINE_BYTES (LINE_BYTES)
  ) u_tags (
    .clk          (clk),
    .rst_n        (rst_n),
    .rd_en        (tag_rd_en),
    .rd_set       (tag_rd_set),
    .rd_tag_cmp   (tag_rd_cmp),
    .rd_hit       (tag_rd_hit),
    .rd_way       (tag_rd_way),
    .rd_dirty     (tag_rd_dirty),
    .rb_en        (tag_rb_en),
    .rb_set       (tag_rb_set),
    .rb_way_sel   (tag_rb_way_sel),
    .rb_tag       (tag_rb_tag),
    .rb_dirty     (tag_rb_dirty),
    .rb_valid     (tag_rb_valid),
    .write_en     (tag_write_en),
    .write_set    (tag_write_set),
    .write_way    (tag_write_way),
    .write_tag    (tag_write_tag),
    .write_valid  (tag_write_valid),
    .write_dirty  (tag_write_dirty),
    .inv_en       (1'b0),
    .inv_set      ('0),
    .inv_way      ('0)
  );

  // -----------------------------------------------------------------------
  // Data array instantiation
  // -----------------------------------------------------------------------
  l1_data_array #(
    .NUM_SETS   (NUM_SETS),
    .NUM_WAYS   (NUM_WAYS),
    .LINE_BYTES (LINE_BYTES),
    .WORD_WIDTH (DATA_WIDTH)
  ) u_data (
    .clk         (clk),
    .word_en     (data_word_en),
    .word_we     (data_word_we),
    .word_set    (req_set),
    .word_way    (tag_rd_hit ? tag_rd_way : victim_way_r),
    .word_offset (req_offset),
    .word_be     (req_be_r),
    .word_wdata  (req_wdata_r),
    .word_rdata  (data_word_rdata),
    .line_en     (data_line_en),
    .line_we     (data_line_we),
    .line_set    (req_set),
    .line_way    (data_line_way),
    .line_beat   (data_line_beat),
    .line_wdata  (data_line_wdata),
    .line_rdata  (data_line_rdata)
  );

  // -----------------------------------------------------------------------
  // LRU instantiation
  // -----------------------------------------------------------------------
  l1_lru #(
    .NUM_SETS (NUM_SETS),
    .NUM_WAYS (NUM_WAYS)
  ) u_lru (
    .clk          (clk),
    .rst_n        (rst_n),
    .access_valid (lru_access_valid),
    .access_set   (req_set),
    .access_way   (tag_rd_hit ? tag_rd_way : victim_way_r),
    .query_set    (req_set),
    .victim_way   (lru_victim_way)
  );

  // -----------------------------------------------------------------------
  // FSM next-state logic
  // -----------------------------------------------------------------------
  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:
        if (cpu_req)
          state_next = S_TAG_READ;

      S_TAG_READ:
        state_next = S_COMPARE_TAG;   // always 1-cycle for SRAM read

      S_COMPARE_TAG:
        if (tag_rd_hit)
          state_next = S_HIT_RETURN;
        else if (tag_rb_valid && tag_rb_dirty)
          state_next = S_WRITE_BACK;
        else
          state_next = S_ALLOCATE;

      S_HIT_RETURN:
        state_next = S_IDLE;

      S_WRITE_BACK:
        if (mem_gnt && (beat_cnt == LAST_BEAT))
          state_next = S_ALLOCATE;

      S_ALLOCATE:
        if (mem_rvalid && (beat_cnt == LAST_BEAT))
          state_next = S_REFILL_DONE;

      S_REFILL_DONE:
        state_next = S_IDLE;

      default:
        state_next = S_IDLE;
    endcase
  end

  // -----------------------------------------------------------------------
  // Registered state and datapath
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      req_addr_r   <= '0;
      req_we_r     <= 1'b0;
      req_be_r     <= '0;
      req_wdata_r  <= '0;
      beat_cnt     <= '0;
      victim_way_r <= '0;
      hit_word_r   <= '0;
    end else begin
      state <= state_next;
      case (state)
        S_IDLE: begin
          if (cpu_req) begin
            req_addr_r  <= cpu_addr;
            req_we_r    <= cpu_we;
            req_be_r    <= cpu_be;
            req_wdata_r <= cpu_wdata;
          end
        end

        // TAG_READ just waits — the SRAM read is issued combinationally
        // via tag_rd_en and tag_rb_en (see below).

        S_COMPARE_TAG: begin
          if (!tag_rd_hit) begin
            // Latch the LRU victim for use in WRITE_BACK / ALLOCATE
            victim_way_r <= lru_victim_way;
            beat_cnt     <= '0;
          end
        end

        S_WRITE_BACK: begin
          if (mem_gnt)
            beat_cnt <= beat_cnt + 1;
        end

        S_ALLOCATE: begin
          if (mem_rvalid) begin
            if (beat_cnt == req_word_idx)
              hit_word_r <= data_line_wdata;  // alloc_wdata (store-miss merged)
            beat_cnt <= beat_cnt + 1;
          end
        end

        S_REFILL_DONE: begin
          beat_cnt <= '0;
        end

        default: ;
      endcase
    end
  end

  // -----------------------------------------------------------------------
  // Store-miss merge
  // -----------------------------------------------------------------------
  logic [DATA_WIDTH-1:0] alloc_wdata;
  always_comb begin
    alloc_wdata = mem_rdata;
    if (req_we_r && (beat_cnt == req_word_idx)) begin
      for (int b = 0; b < DATA_WIDTH/8; b++) begin
        if (req_be_r[b])
          alloc_wdata[b*8 +: 8] = req_wdata_r[b*8 +: 8];
      end
    end
  end

  // -----------------------------------------------------------------------
  // Tag array control
  // TAG_READ: issue read for all 4 ways (rd_en) and read-back (rb_en)
  //   using the LRU victim way as the rb selector (available combinationally).
  // -----------------------------------------------------------------------
  assign tag_rd_en      = (state == S_TAG_READ);
  assign tag_rd_set     = req_set;
  assign tag_rd_cmp     = req_tag;

  // rb read: same set, use lru_victim_way as selector
  assign tag_rb_en      = (state == S_TAG_READ);
  assign tag_rb_set     = req_set;
  assign tag_rb_way_sel = lru_victim_way;

  // Tag write: on hit+store (set dirty) or on fill completion
  assign tag_write_en    = ((state == S_COMPARE_TAG) && tag_rd_hit && req_we_r) ||
                           (state == S_REFILL_DONE);
  assign tag_write_set   = req_set;
  assign tag_write_way   = (state == S_REFILL_DONE) ? victim_way_r : tag_rd_way;
  assign tag_write_tag   = req_tag;
  assign tag_write_valid = 1'b1;
  assign tag_write_dirty = (state == S_REFILL_DONE) ? req_we_r :
                           ((state == S_COMPARE_TAG) && req_we_r) ? 1'b1 : tag_rd_dirty;

  // -----------------------------------------------------------------------
  // Data array control (unchanged logic, updated signal names)
  // -----------------------------------------------------------------------

  // Word port: used for cache hits (load or store)
  assign data_word_en = (state == S_COMPARE_TAG) && tag_rd_hit;
  assign data_word_we = (state == S_COMPARE_TAG) && tag_rd_hit && req_we_r;

  // Line port: same logic as before, updated state/signal references
  assign data_line_en =
      ((state == S_COMPARE_TAG) && !tag_rd_hit) ||
      (state == S_WRITE_BACK) ||
      ((state == S_ALLOCATE) && mem_rvalid);

  assign data_line_we =
      (state == S_ALLOCATE) && mem_rvalid;

  assign data_line_way =
      (state == S_COMPARE_TAG) ? lru_victim_way : victim_way_r;

  assign data_line_beat =
      (state == S_COMPARE_TAG) ? WORD_IDX_BITS'(0) :
      (state == S_WRITE_BACK)  ? (mem_gnt ? beat_cnt + WORD_IDX_BITS'(1) : beat_cnt) :
      beat_cnt;

  assign data_line_wdata = alloc_wdata;

  // -----------------------------------------------------------------------
  // Output assignments
  // -----------------------------------------------------------------------
  assign cpu_gnt    = (state == S_IDLE) && cpu_req;
  assign cpu_rvalid = (state == S_HIT_RETURN) || (state == S_REFILL_DONE);
  assign cpu_rdata  = (state == S_REFILL_DONE) ? hit_word_r : data_word_rdata;
  assign cache_busy = (state != S_IDLE);

  // LRU update on hit or fill completion
  assign lru_access_valid = ((state == S_COMPARE_TAG) && tag_rd_hit) ||
                            (state == S_REFILL_DONE);

  // Memory-side
  logic [ADDR_WIDTH-1:0] wb_addr;
  assign wb_addr = {tag_rb_tag, req_set, {OFFSET_BITS{1'b0}}};

  assign mem_req   = (state == S_WRITE_BACK) || (state == S_ALLOCATE);
  assign mem_we    = (state == S_WRITE_BACK);
  assign mem_addr  = (state == S_WRITE_BACK) ?
                     (wb_addr + {{(ADDR_WIDTH-WORD_IDX_BITS-2){1'b0}}, beat_cnt, 2'b00}) :
                     ({req_addr_r[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}} +
                      {{(ADDR_WIDTH-WORD_IDX_BITS-2){1'b0}}, beat_cnt, 2'b00});
  assign mem_wdata = data_line_rdata;
  assign mem_last  = (beat_cnt == LAST_BEAT);

endmodule
