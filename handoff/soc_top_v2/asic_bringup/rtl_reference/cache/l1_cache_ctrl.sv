// =============================================================================
// l1_cache_ctrl.sv — L1 Data Cache Controller FSM
// =============================================================================
// Blocking cache controller. States:
//   IDLE        — wait for CPU request
//   COMPARE_TAG — check tag array (1-cycle latency, combinational hit)
//   WRITE_BACK  — evict dirty line to memory (LINE_BYTES/4 beats)
//   ALLOCATE    — fetch new line from memory (LINE_BYTES/4 beats)
//   REFILL_DONE — write filled line to data array, return to IDLE
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
  localparam int WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
  localparam int WORD_IDX_BITS  = $clog2(WORDS_PER_LINE);
  localparam logic [WORD_IDX_BITS-1:0] LAST_BEAT = WORD_IDX_BITS'(WORDS_PER_LINE - 1);

  // -----------------------------------------------------------------------
  // FSM states
  // -----------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE,
    S_COMPARE_TAG,
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

  // -----------------------------------------------------------------------
  // Tag array interface
  // -----------------------------------------------------------------------
  logic                          tag_lookup_hit;
  logic [$clog2(NUM_WAYS)-1:0]   tag_lookup_way;
  logic                          tag_lookup_dirty;

  logic                          tag_write_en;
  logic [$clog2(NUM_SETS)-1:0]   tag_write_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_write_way;
  logic [TAG_BITS-1:0]           tag_write_tag;
  logic                          tag_write_valid;
  logic                          tag_write_dirty;

  logic [TAG_BITS-1:0]           tag_rb_tag;
  logic                          tag_rb_dirty;
  logic                          tag_rb_valid;

  // -----------------------------------------------------------------------
  // Data array interface
  // -----------------------------------------------------------------------
  logic                          data_word_en;
  logic                          data_word_we;
  logic [DATA_WIDTH-1:0]         data_word_rdata;

  logic                          data_line_en;
  logic                          data_line_we;
  logic [LINE_BYTES*8-1:0]       data_line_wdata;
  logic [LINE_BYTES*8-1:0]       data_line_rdata;

  // -----------------------------------------------------------------------
  // LRU interface
  // -----------------------------------------------------------------------
  logic                          lru_access_valid;
  logic [$clog2(NUM_WAYS)-1:0]   lru_victim_way;

  // -----------------------------------------------------------------------
  // Burst counter for writeback / allocate
  // -----------------------------------------------------------------------
  logic [WORD_IDX_BITS-1:0]      beat_cnt;
  logic [LINE_BYTES*8-1:0]       fill_buffer;
  logic [$clog2(NUM_WAYS)-1:0]   victim_way_r;

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
    .lookup_addr  (req_addr_r),
    .lookup_hit   (tag_lookup_hit),
    .lookup_way   (tag_lookup_way),
    .lookup_dirty (tag_lookup_dirty),
    .write_en     (tag_write_en),
    .write_set    (tag_write_set),
    .write_way    (tag_write_way),
    .write_tag    (tag_write_tag),
    .write_valid  (tag_write_valid),
    .write_dirty  (tag_write_dirty),
    .inv_en       (1'b0),
    .inv_set      ('0),
    .inv_way      ('0),
    .rb_set       (req_set),
    .rb_way       (victim_way_r),
    .rb_tag       (tag_rb_tag),
    .rb_dirty     (tag_rb_dirty),
    .rb_valid     (tag_rb_valid)
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
    .word_way    (tag_lookup_hit ? tag_lookup_way : victim_way_r),
    .word_offset (req_offset),
    .word_be     (req_be_r),
    .word_wdata  (req_wdata_r),
    .word_rdata  (data_word_rdata),
    .line_en     (data_line_en),
    .line_we     (data_line_we),
    .line_set    (req_set),
    .line_way    (victim_way_r),
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
    .access_way   (tag_lookup_hit ? tag_lookup_way : victim_way_r),
    .query_set    (req_set),
    .victim_way   (lru_victim_way)
  );

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= S_IDLE;
    else
      state <= state_next;
  end

  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:
        if (cpu_req)
          state_next = S_COMPARE_TAG;

      S_COMPARE_TAG:
        if (tag_lookup_hit)
          state_next = S_IDLE;           // hit — done in 1 cycle
        else if (tag_rb_valid && tag_rb_dirty)
          state_next = S_WRITE_BACK;     // dirty eviction first
        else
          state_next = S_ALLOCATE;       // clean miss — go fill

      S_WRITE_BACK:
        if (mem_gnt && (beat_cnt == LAST_BEAT))
          state_next = S_ALLOCATE;

      S_ALLOCATE:
        if (mem_rvalid && (beat_cnt == LAST_BEAT))
          state_next = S_REFILL_DONE;

      S_REFILL_DONE:
        state_next = S_IDLE;             // write line + respond to CPU

      default:
        state_next = S_IDLE;
    endcase
  end

  // -----------------------------------------------------------------------
  // Datapath control
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      req_addr_r   <= '0;
      req_we_r     <= 1'b0;
      req_be_r     <= '0;
      req_wdata_r  <= '0;
      beat_cnt     <= '0;
      fill_buffer  <= '0;
      victim_way_r <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (cpu_req) begin
            req_addr_r   <= cpu_addr;
            req_we_r     <= cpu_we;
            req_be_r     <= cpu_be;
            req_wdata_r  <= cpu_wdata;
          end
        end

        S_COMPARE_TAG: begin
          if (!tag_lookup_hit) begin
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
            // Pack incoming words into fill buffer
            fill_buffer[beat_cnt * DATA_WIDTH +: DATA_WIDTH] <= mem_rdata;
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
  // Output assignments
  // -----------------------------------------------------------------------

  // CPU-side
  assign cpu_gnt    = (state == S_IDLE) && cpu_req;
  assign cpu_rvalid = ((state == S_COMPARE_TAG) && tag_lookup_hit) ||
                      (state == S_REFILL_DONE);
  assign cpu_rdata  = (state == S_REFILL_DONE) ?
                      fill_buffer[req_offset[$clog2(LINE_BYTES)-1:2] * DATA_WIDTH +: DATA_WIDTH] :
                      data_word_rdata;
  assign cache_busy = (state != S_IDLE);

  // Data array control
  assign data_word_en = (state == S_COMPARE_TAG) && tag_lookup_hit;
  assign data_word_we = (state == S_COMPARE_TAG) && tag_lookup_hit && req_we_r;
  assign data_line_en = (state == S_WRITE_BACK) || (state == S_REFILL_DONE);
  assign data_line_we = (state == S_REFILL_DONE);

  always_comb begin
    data_line_wdata = fill_buffer;
    if (state == S_REFILL_DONE && req_we_r) begin
      for (int b = 0; b < DATA_WIDTH/8; b++) begin
        if (req_be_r[b]) begin
          data_line_wdata[{(req_offset[$clog2(LINE_BYTES)-1:2] * DATA_WIDTH) + (b * 8)} +: 8] =
            req_wdata_r[b*8 +: 8];
        end
      end
    end
  end

  // Tag array control
  assign tag_write_en    = ((state == S_COMPARE_TAG) && tag_lookup_hit && req_we_r) ||
                           (state == S_REFILL_DONE);
  assign tag_write_set   = req_set;
  assign tag_write_way   = (state == S_REFILL_DONE) ? victim_way_r : tag_lookup_way;
  assign tag_write_tag   = req_tag;
  assign tag_write_valid = 1'b1;
  assign tag_write_dirty = (state == S_REFILL_DONE) ? req_we_r :
                           ((state == S_COMPARE_TAG) && req_we_r) ? 1'b1 : tag_lookup_dirty;

  // LRU update on hit or fill
  assign lru_access_valid = ((state == S_COMPARE_TAG) && tag_lookup_hit) ||
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
  assign mem_wdata = data_line_rdata[beat_cnt * DATA_WIDTH +: DATA_WIDTH];
  assign mem_last  = (beat_cnt == LAST_BEAT);

endmodule
