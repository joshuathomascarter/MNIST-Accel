// =============================================================================
// l2_cache_top.sv — L2 Unified Cache Top-Level
// =============================================================================
// Non-blocking L2 cache sitting between AXI crossbar (upstream) and
// DRAM controller (downstream).
//
// Features:
//   - 8-way set-associative, 256 sets × 64B lines = 128KB
//   - MSHRs for hit-under-miss (up to 4 outstanding misses)
//   - Stride prefetcher
//   - Tree-PLRU replacement
//   - Write-back, write-allocate policy
//
// Upstream: AXI4 slave (from crossbar)
// Downstream: AXI4 master (to DRAM controller)

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */
import soc_pkg::*;

module l2_cache_top #(
  parameter int ADDR_WIDTH   = 32,
  parameter int DATA_WIDTH   = 32,
  parameter int ID_WIDTH     = 4,
  parameter int NUM_SETS     = 256,
  parameter int NUM_WAYS     = 8,
  parameter int LINE_BYTES   = 64,
  parameter int NUM_MSHR     = 4
) (
  input  logic              clk,
  input  logic              rst_n,

  // =====================================================================
  // AXI4 Slave — upstream from crossbar
  // =====================================================================
  // AW
  input  logic              s_axi_awvalid,
  output logic              s_axi_awready,
  input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
  input  logic [ID_WIDTH-1:0]   s_axi_awid,
  input  logic [7:0]        s_axi_awlen,
  input  logic [2:0]        s_axi_awsize,
  input  logic [1:0]        s_axi_awburst,
  // W
  input  logic              s_axi_wvalid,
  output logic              s_axi_wready,
  input  logic [DATA_WIDTH-1:0] s_axi_wdata,
  input  logic [DATA_WIDTH/8-1:0] s_axi_wstrb,
  input  logic              s_axi_wlast,
  // B
  output logic              s_axi_bvalid,
  input  logic              s_axi_bready,
  output logic [1:0]        s_axi_bresp,
  output logic [ID_WIDTH-1:0] s_axi_bid,
  // AR
  input  logic              s_axi_arvalid,
  output logic              s_axi_arready,
  input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
  input  logic [ID_WIDTH-1:0]   s_axi_arid,
  input  logic [7:0]        s_axi_arlen,
  input  logic [2:0]        s_axi_arsize,
  input  logic [1:0]        s_axi_arburst,
  // R
  output logic              s_axi_rvalid,
  input  logic              s_axi_rready,
  output logic [DATA_WIDTH-1:0] s_axi_rdata,
  output logic [1:0]        s_axi_rresp,
  output logic [ID_WIDTH-1:0] s_axi_rid,
  output logic              s_axi_rlast,

  // =====================================================================
  // AXI4 Master — downstream to DRAM
  // =====================================================================
  // AW
  output logic              m_axi_awvalid,
  input  logic              m_axi_awready,
  output logic [ADDR_WIDTH-1:0] m_axi_awaddr,
  output logic [ID_WIDTH-1:0]   m_axi_awid,
  output logic [7:0]        m_axi_awlen,
  output logic [2:0]        m_axi_awsize,
  output logic [1:0]        m_axi_awburst,
  // W
  output logic              m_axi_wvalid,
  input  logic              m_axi_wready,
  output logic [DATA_WIDTH-1:0] m_axi_wdata,
  output logic [DATA_WIDTH/8-1:0] m_axi_wstrb,
  output logic              m_axi_wlast,
  // B
  input  logic              m_axi_bvalid,
  output logic              m_axi_bready,
  input  logic [1:0]        m_axi_bresp,
  input  logic [ID_WIDTH-1:0] m_axi_bid,
  // AR
  output logic              m_axi_arvalid,
  input  logic              m_axi_arready,
  output logic [ADDR_WIDTH-1:0] m_axi_araddr,
  output logic [ID_WIDTH-1:0]   m_axi_arid,
  output logic [7:0]        m_axi_arlen,
  output logic [2:0]        m_axi_arsize,
  output logic [1:0]        m_axi_arburst,
  // R
  input  logic              m_axi_rvalid,
  output logic              m_axi_rready,
  input  logic [DATA_WIDTH-1:0] m_axi_rdata,
  input  logic [1:0]        m_axi_rresp,
  input  logic [ID_WIDTH-1:0] m_axi_rid,
  input  logic              m_axi_rlast,

  // =====================================================================
  // Configuration / Status
  // =====================================================================
  input  logic              pf_enable,
  output logic              cache_busy
);

  // =========================================================================
  // Derived parameters
  // =========================================================================
  localparam int OFFSET_BITS    = $clog2(LINE_BYTES);
  localparam int INDEX_BITS     = $clog2(NUM_SETS);
  localparam int TAG_WIDTH      = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
  localparam int WAY_BITS       = $clog2(NUM_WAYS);
  localparam int WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);
  localparam int WORD_SEL_BITS  = $clog2(WORDS_PER_LINE);

  // =========================================================================
  // Address decomposition
  // =========================================================================
  function automatic [TAG_WIDTH-1:0] addr_tag;
    input [ADDR_WIDTH-1:0] addr;
    begin
      addr_tag = addr[ADDR_WIDTH-1 -: TAG_WIDTH];
    end
  endfunction

  function automatic [INDEX_BITS-1:0] addr_set;
    input [ADDR_WIDTH-1:0] addr;
    begin
      addr_set = addr[OFFSET_BITS +: INDEX_BITS];
    end
  endfunction

  function automatic [WORD_SEL_BITS-1:0] addr_word;
    input [ADDR_WIDTH-1:0] addr;
    begin
      addr_word = addr[$clog2(DATA_WIDTH/8) +: WORD_SEL_BITS];
    end
  endfunction

  // =========================================================================
  // Controller FSM
  // =========================================================================
  typedef enum logic [3:0] {
    C_IDLE,
    C_TAG_RD,          // Read tag array
    C_HIT_RD,          // Read hit — fetch data
    C_HIT_WR,          // Write hit — update data + dirty
    C_MISS_ALLOC,      // Allocate MSHR
    C_EVICT_CHECK,     // Check if victim is dirty
    C_EVICT_WB,        // Write-back dirty victim to DRAM
    C_FILL_REQ,        // Issue fill request to DRAM
    C_FILL_WAIT,       // Wait for fill data from DRAM
    C_FILL_UPDATE,     // Write fill data into data array + update tag
    C_RESP_RD,         // Send read response upstream
    C_RESP_WR,         // Send write response upstream
    C_PF_CHECK         // Process prefetch request
  } ctrl_state_e;

  ctrl_state_e ctrl_state, ctrl_state_next;

  // =========================================================================
  // Request pipeline registers
  // =========================================================================
  logic [ADDR_WIDTH-1:0]  req_addr;
  logic [ID_WIDTH-1:0]    req_id;
  logic                   req_is_write;
  logic [DATA_WIDTH-1:0]  req_wdata;
  logic [DATA_WIDTH/8-1:0] req_wstrb;
  logic [7:0]             req_len;
  logic                   wr_aw_pending;
  logic                   wr_w_pending;
  logic [ADDR_WIDTH-1:0]  wr_awaddr_r;
  logic [ID_WIDTH-1:0]    wr_awid_r;
  logic [7:0]             wr_awlen_r;
  logic [DATA_WIDTH-1:0]  wr_wdata_r;
  logic [DATA_WIDTH/8-1:0] wr_wstrb_r;
  logic                   s_axi_aw_fire;
  logic                   s_axi_w_fire;
  logic                   write_req_ready;

  // =========================================================================
  // Tag array interface signals
  // =========================================================================
  logic                          tag_lookup_valid;
  logic [INDEX_BITS-1:0]         tag_lookup_set;
  logic [TAG_WIDTH-1:0]          tag_lookup_tag;
  logic                          tag_lookup_hit;
  logic [$clog2(NUM_WAYS)-1:0]   tag_lookup_way;
  logic                          tag_lookup_dirty;

  logic                          tag_write_valid;
  logic [INDEX_BITS-1:0]         tag_write_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_write_way;
  logic [TAG_WIDTH-1:0]          tag_write_tag;
  logic                          tag_write_dirty;

  logic                          tag_inv_valid;
  logic [INDEX_BITS-1:0]         tag_inv_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_inv_way;

  logic                          tag_dc_valid;
  logic [INDEX_BITS-1:0]         tag_dc_set;
  logic [$clog2(NUM_WAYS)-1:0]   tag_dc_way;
  logic                          tag_dc_is_dirty;
  logic [TAG_WIDTH-1:0]          tag_dc_tag;

  // =========================================================================
  // Data array interface signals
  // =========================================================================
  logic                          data_rd_en;
  logic [INDEX_BITS-1:0]         data_rd_set;
  logic [$clog2(NUM_WAYS)-1:0]   data_rd_way;
  logic [WORD_SEL_BITS-1:0]      data_rd_word;
  logic [DATA_WIDTH-1:0]         data_rd_data;

  logic                          data_wr_en;
  logic [INDEX_BITS-1:0]         data_wr_set;
  logic [$clog2(NUM_WAYS)-1:0]   data_wr_way;
  logic [WORD_SEL_BITS-1:0]      data_wr_word;
  logic [DATA_WIDTH-1:0]         data_wr_data;
  logic [DATA_WIDTH/8-1:0]       data_wr_be;

  logic                          data_line_rd_en;
  logic [INDEX_BITS-1:0]         data_line_rd_set;
  logic [$clog2(NUM_WAYS)-1:0]   data_line_rd_way;
  logic [WORD_SEL_BITS-1:0]      data_line_rd_word;
  logic [DATA_WIDTH-1:0]         data_line_rd_data;

  logic                          data_line_wr_en;
  logic [INDEX_BITS-1:0]         data_line_wr_set;
  logic [$clog2(NUM_WAYS)-1:0]   data_line_wr_way;
  logic [WORD_SEL_BITS-1:0]      data_line_wr_word;
  logic [DATA_WIDTH-1:0]         data_line_wr_data;

  // =========================================================================
  // MSHR interface signals
  // =========================================================================
  logic                          mshr_alloc_valid;
  logic                          mshr_alloc_ready;
  logic [ADDR_WIDTH-1:0]         mshr_alloc_addr;
  logic [ID_WIDTH-1:0]           mshr_alloc_id;
  logic                          mshr_alloc_is_write;
  logic [$clog2(NUM_MSHR)-1:0]   mshr_alloc_idx;

  logic                          mshr_lookup_valid;
  logic [ADDR_WIDTH-1:0]         mshr_lookup_addr;
  logic                          mshr_lookup_hit;
  logic [$clog2(NUM_MSHR)-1:0]   mshr_lookup_idx;

  logic                          mshr_complete_valid;
  logic [$clog2(NUM_MSHR)-1:0]   mshr_complete_idx;
  logic [ADDR_WIDTH-1:0]         mshr_complete_addr;
  logic [ID_WIDTH-1:0]           mshr_complete_id;
  logic                          mshr_complete_is_write;

  logic                          mshr_full, mshr_empty;

  // =========================================================================
  // LRU (Tree-PLRU) — per-set replacement
  // =========================================================================
  logic [NUM_WAYS-2:0] plru_bits [NUM_SETS];  // Tree-PLRU state per set
  logic [$clog2(NUM_WAYS)-1:0] victim_way;
  logic [NUM_WAYS-2:0]          plru_hit_bits_next;
  logic [NUM_WAYS-2:0]          plru_fill_bits_next;
  integer                       plru_victim_idx_c;
  integer                       plru_victim_level_c;
  integer                       plru_hit_idx_c;
  integer                       plru_hit_parent_c;
  integer                       plru_hit_level_c;
  integer                       plru_fill_idx_c;
  integer                       plru_fill_parent_c;
  integer                       plru_fill_level_c;
  integer                       plru_reset_set_i;

  // Find victim via Tree-PLRU traversal
  always_comb begin
    plru_victim_idx_c = 0;
    for (plru_victim_level_c = 0; plru_victim_level_c < WAY_BITS; plru_victim_level_c = plru_victim_level_c + 1) begin
      if (plru_bits[addr_set(req_addr)][plru_victim_idx_c])
        plru_victim_idx_c = 2 * plru_victim_idx_c + 2;
      else
        plru_victim_idx_c = 2 * plru_victim_idx_c + 1;
    end
    victim_way = WAY_BITS'(plru_victim_idx_c - (NUM_WAYS - 1));
  end

  always_comb begin
    plru_hit_bits_next = plru_bits[addr_set(req_addr)];
    plru_hit_idx_c = int'(tag_lookup_way) + (NUM_WAYS - 1);
    for (plru_hit_level_c = 0; plru_hit_level_c < WAY_BITS; plru_hit_level_c = plru_hit_level_c + 1) begin
      plru_hit_parent_c = (plru_hit_idx_c - 1) / 2;
      if (plru_hit_idx_c == 2 * plru_hit_parent_c + 1)
        plru_hit_bits_next[plru_hit_parent_c] = 1'b1;
      else
        plru_hit_bits_next[plru_hit_parent_c] = 1'b0;
      plru_hit_idx_c = plru_hit_parent_c;
    end
  end

  always_comb begin
    plru_fill_bits_next = plru_bits[addr_set(req_addr)];
    plru_fill_idx_c = int'(tag_write_way) + (NUM_WAYS - 1);
    for (plru_fill_level_c = 0; plru_fill_level_c < WAY_BITS; plru_fill_level_c = plru_fill_level_c + 1) begin
      plru_fill_parent_c = (plru_fill_idx_c - 1) / 2;
      if (plru_fill_idx_c == 2 * plru_fill_parent_c + 1)
        plru_fill_bits_next[plru_fill_parent_c] = 1'b1;
      else
        plru_fill_bits_next[plru_fill_parent_c] = 1'b0;
      plru_fill_idx_c = plru_fill_parent_c;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (plru_reset_set_i = 0; plru_reset_set_i < NUM_SETS; plru_reset_set_i = plru_reset_set_i + 1)
        plru_bits[plru_reset_set_i] <= '0;
    end else begin
      if ((ctrl_state == C_HIT_RD || ctrl_state == C_HIT_WR) && tag_lookup_hit)
        plru_bits[addr_set(req_addr)] <= plru_hit_bits_next;
      if (ctrl_state == C_FILL_UPDATE)
        plru_bits[addr_set(req_addr)] <= plru_fill_bits_next;
    end
  end

  // =========================================================================
  // Prefetcher interface
  // =========================================================================
  logic              pf_miss_valid;
  logic [ADDR_WIDTH-1:0] pf_miss_addr;
  logic              pf_req_valid;
  logic              pf_req_ready;
  logic [ADDR_WIDTH-1:0] pf_req_addr;

  // =========================================================================
  // Beat counter for burst fill/evict
  // =========================================================================
  logic [WORD_SEL_BITS-1:0] beat_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      beat_cnt <= '0;
    else begin
      case (ctrl_state)
        C_FILL_REQ:  beat_cnt <= '0;
        C_EVICT_WB:  if (m_axi_wvalid && m_axi_wready) beat_cnt <= beat_cnt + 1;
        C_FILL_WAIT: if (m_axi_rvalid && m_axi_rready) beat_cnt <= beat_cnt + 1;
        default: ;
      endcase
    end
  end

  // =========================================================================
  // FSM — State register
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      ctrl_state <= C_IDLE;
    else
      ctrl_state <= ctrl_state_next;
  end

  // =========================================================================
  // FSM — Next state logic
  // =========================================================================
  assign s_axi_aw_fire  = s_axi_awvalid && s_axi_awready;
  assign s_axi_w_fire   = s_axi_wvalid && s_axi_wready;
  assign write_req_ready = (wr_aw_pending || s_axi_aw_fire) &&
                           (wr_w_pending  || s_axi_w_fire);

  always_comb begin
    ctrl_state_next = ctrl_state;

    case (ctrl_state)
      C_IDLE: begin
        if (s_axi_arvalid && s_axi_arready)
          ctrl_state_next = C_TAG_RD;
        else if (!s_axi_arvalid && write_req_ready)
          ctrl_state_next = C_TAG_RD;
        else if (pf_req_valid && !wr_aw_pending && !wr_w_pending)
          ctrl_state_next = C_PF_CHECK;
      end

      C_TAG_RD:
        ctrl_state_next = tag_lookup_hit ?
          (req_is_write ? C_HIT_WR : C_HIT_RD) :
          C_MISS_ALLOC;

      C_HIT_RD:
        ctrl_state_next = C_RESP_RD;

      C_HIT_WR:
        ctrl_state_next = C_RESP_WR;

      C_MISS_ALLOC: begin
        if (mshr_lookup_hit)
          ctrl_state_next = C_IDLE;  // Secondary miss — merged
        else if (mshr_alloc_ready)
          ctrl_state_next = C_EVICT_CHECK;
        else
          ctrl_state_next = C_IDLE;  // MSHR full — retry later
      end

      C_EVICT_CHECK:
        ctrl_state_next = tag_dc_is_dirty ? C_EVICT_WB : C_FILL_REQ;

      C_EVICT_WB: begin
        if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
          if (m_axi_bvalid)
            ctrl_state_next = C_FILL_REQ;
        end
      end

      C_FILL_REQ:
        if (m_axi_arready)
          ctrl_state_next = C_FILL_WAIT;

      C_FILL_WAIT:
        if (m_axi_rvalid && m_axi_rlast)
          ctrl_state_next = C_FILL_UPDATE;

      C_FILL_UPDATE:
        ctrl_state_next = req_is_write ? C_RESP_WR : C_RESP_RD;

      C_RESP_RD:
        if (s_axi_rvalid && s_axi_rready)
          ctrl_state_next = C_IDLE;

      C_RESP_WR:
        if (s_axi_bvalid && s_axi_bready)
          ctrl_state_next = C_IDLE;

      C_PF_CHECK: begin
        // Tag lookup for prefetch — if miss, allocate MSHR
        if (tag_lookup_hit || mshr_full)
          ctrl_state_next = C_IDLE;  // Already cached or no MSHR
        else
          ctrl_state_next = C_EVICT_CHECK;
      end

      default: ctrl_state_next = C_IDLE;
    endcase
  end

  // =========================================================================
  // Request capture
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      req_addr     <= '0;
      req_id       <= '0;
      req_is_write <= 1'b0;
      req_wdata    <= '0;
      req_wstrb    <= '0;
      req_len      <= '0;
      wr_aw_pending <= 1'b0;
      wr_w_pending  <= 1'b0;
      wr_awaddr_r   <= '0;
      wr_awid_r     <= '0;
      wr_awlen_r    <= '0;
      wr_wdata_r    <= '0;
      wr_wstrb_r    <= '0;
    end else if (ctrl_state == C_IDLE) begin
      if (s_axi_arvalid && s_axi_arready) begin
        req_addr     <= s_axi_araddr;
        req_id       <= s_axi_arid;
        req_is_write <= 1'b0;
        req_wdata    <= '0;
        req_wstrb    <= '0;
        req_len      <= s_axi_arlen;
      end else begin
        if (s_axi_aw_fire) begin
          wr_aw_pending <= 1'b1;
          wr_awaddr_r   <= s_axi_awaddr;
          wr_awid_r     <= s_axi_awid;
          wr_awlen_r    <= s_axi_awlen;
        end

        if (s_axi_w_fire) begin
          wr_w_pending <= 1'b1;
          wr_wdata_r   <= s_axi_wdata;
          wr_wstrb_r   <= s_axi_wstrb;
        end

        if (!s_axi_arvalid && write_req_ready) begin
          req_addr      <= s_axi_aw_fire ? s_axi_awaddr : wr_awaddr_r;
          req_id        <= s_axi_aw_fire ? s_axi_awid   : wr_awid_r;
          req_is_write  <= 1'b1;
          req_wdata     <= s_axi_w_fire  ? s_axi_wdata  : wr_wdata_r;
          req_wstrb     <= s_axi_w_fire  ? s_axi_wstrb  : wr_wstrb_r;
          req_len       <= s_axi_aw_fire ? s_axi_awlen  : wr_awlen_r;
          wr_aw_pending <= 1'b0;
          wr_w_pending  <= 1'b0;
        end else if (pf_req_valid && !wr_aw_pending && !wr_w_pending) begin
          req_addr     <= pf_req_addr;
          req_id       <= '0;
          req_is_write <= 1'b0;
          req_wdata    <= '0;
          req_wstrb    <= '0;
          req_len      <= '0;
        end
      end
    end
  end

  // =========================================================================
  // Upstream handshake
  // =========================================================================
  assign s_axi_arready = (ctrl_state == C_IDLE) && !wr_aw_pending && !wr_w_pending;
  assign s_axi_awready = (ctrl_state == C_IDLE) && !s_axi_arvalid && !wr_aw_pending;
  assign s_axi_wready  = (ctrl_state == C_IDLE) && !s_axi_arvalid && !wr_w_pending;

  // =========================================================================
  // Tag array connections
  // =========================================================================
  assign tag_lookup_valid = (ctrl_state == C_TAG_RD) || (ctrl_state == C_PF_CHECK);
  assign tag_lookup_set   = addr_set(req_addr);
  assign tag_lookup_tag   = addr_tag(req_addr);

  assign tag_write_valid  = (ctrl_state == C_FILL_UPDATE);
  assign tag_write_set    = addr_set(req_addr);
  assign tag_write_way    = victim_way;
  assign tag_write_tag    = addr_tag(req_addr);
  assign tag_write_dirty  = req_is_write;

  assign tag_inv_valid    = 1'b0;  // No coherence invalidation in this version
  assign tag_inv_set      = '0;
  assign tag_inv_way      = '0;

  assign tag_dc_valid     = (ctrl_state == C_EVICT_CHECK);
  assign tag_dc_set       = addr_set(req_addr);
  assign tag_dc_way       = victim_way;

  // =========================================================================
  // Data array connections
  // =========================================================================
  // Read hit
  assign data_rd_en   = (ctrl_state == C_HIT_RD);
  assign data_rd_set  = addr_set(req_addr);
  assign data_rd_way  = tag_lookup_way;
  assign data_rd_word = addr_word(req_addr);

  // Write hit
  assign data_wr_en   = (ctrl_state == C_HIT_WR);
  assign data_wr_set  = addr_set(req_addr);
  assign data_wr_way  = tag_lookup_way;
  assign data_wr_word = addr_word(req_addr);
  assign data_wr_data = req_wdata;
  assign data_wr_be   = req_wstrb;

  // Eviction read (one word per cycle)
  assign data_line_rd_en   = (ctrl_state == C_EVICT_WB);
  assign data_line_rd_set  = addr_set(req_addr);
  assign data_line_rd_way  = victim_way;
  assign data_line_rd_word = beat_cnt;

  // Fill write (one word per cycle)
  assign data_line_wr_en   = (ctrl_state == C_FILL_WAIT) && m_axi_rvalid;
  assign data_line_wr_set  = addr_set(req_addr);
  assign data_line_wr_way  = victim_way;
  assign data_line_wr_word = beat_cnt;
  assign data_line_wr_data = m_axi_rdata;

  // =========================================================================
  // MSHR connections
  // =========================================================================
  assign mshr_lookup_valid  = (ctrl_state == C_MISS_ALLOC);
  assign mshr_lookup_addr   = req_addr;

  assign mshr_alloc_valid   = (ctrl_state == C_MISS_ALLOC) && !mshr_lookup_hit;
  assign mshr_alloc_addr    = req_addr;
  assign mshr_alloc_id      = req_id;
  assign mshr_alloc_is_write = req_is_write;

  assign mshr_complete_valid = (ctrl_state == C_FILL_UPDATE);
  assign mshr_complete_idx   = mshr_alloc_idx;  // Entry that was just filled

  // =========================================================================
  // Downstream AXI4 master — fill reads + eviction writes
  // =========================================================================
  // AR channel (fill read)
  assign m_axi_arvalid = (ctrl_state == C_FILL_REQ);
  assign m_axi_araddr  = {req_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
  assign m_axi_arid    = req_id;
  assign m_axi_arlen   = 8'(WORDS_PER_LINE - 1);
  assign m_axi_arsize  = 3'($clog2(DATA_WIDTH / 8));
  assign m_axi_arburst = 2'b01;  // INCR

  // R channel
  assign m_axi_rready  = (ctrl_state == C_FILL_WAIT);

  // AW channel (eviction writeback)
  assign m_axi_awvalid = (ctrl_state == C_EVICT_WB) && (beat_cnt == '0);
  assign m_axi_awaddr  = {tag_dc_tag, addr_set(req_addr), {OFFSET_BITS{1'b0}}};
  assign m_axi_awid    = req_id;
  assign m_axi_awlen   = 8'(WORDS_PER_LINE - 1);
  assign m_axi_awsize  = 3'($clog2(DATA_WIDTH / 8));
  assign m_axi_awburst = 2'b01;  // INCR

  // W channel
  assign m_axi_wvalid  = (ctrl_state == C_EVICT_WB);
  assign m_axi_wdata   = data_line_rd_data;
  assign m_axi_wstrb   = '1;
  assign m_axi_wlast   = (beat_cnt == WORD_SEL_BITS'(WORDS_PER_LINE - 1));

  // B channel
  assign m_axi_bready  = 1'b1;

  // =========================================================================
  // Upstream read response
  // =========================================================================
  assign s_axi_rvalid = (ctrl_state == C_RESP_RD);
  assign s_axi_rdata  = data_rd_data;
  assign s_axi_rresp  = 2'b00;  // OKAY
  assign s_axi_rid    = req_id;
  assign s_axi_rlast  = 1'b1;   // Single-beat responses for now

  // Upstream write response
  assign s_axi_bvalid = (ctrl_state == C_RESP_WR);
  assign s_axi_bresp  = 2'b00;  // OKAY
  assign s_axi_bid    = req_id;

  // =========================================================================
  // Prefetcher
  // =========================================================================
  assign pf_miss_valid = (ctrl_state == C_MISS_ALLOC) && !mshr_lookup_hit;
  assign pf_miss_addr  = req_addr;
  assign pf_req_ready  = (ctrl_state == C_IDLE) && !s_axi_arvalid && !s_axi_awvalid &&
                         !s_axi_wvalid && !wr_aw_pending && !wr_w_pending;

  assign cache_busy = (ctrl_state != C_IDLE);

  // =========================================================================
  // Sub-module instantiations
  // =========================================================================
  l2_tag_array #(
    .ADDR_WIDTH (ADDR_WIDTH),
    .NUM_SETS   (NUM_SETS),
    .NUM_WAYS   (NUM_WAYS),
    .LINE_BYTES (LINE_BYTES)
  ) u_tag (
    .clk              (clk),
    .rst_n            (rst_n),
    .lookup_valid     (tag_lookup_valid),
    .lookup_set       (tag_lookup_set),
    .lookup_tag       (tag_lookup_tag),
    .lookup_hit       (tag_lookup_hit),
    .lookup_way       (tag_lookup_way),
    .lookup_dirty     (tag_lookup_dirty),
    .write_valid      (tag_write_valid),
    .write_set        (tag_write_set),
    .write_way        (tag_write_way),
    .write_tag        (tag_write_tag),
    .write_dirty      (tag_write_dirty),
    .inv_valid        (tag_inv_valid),
    .inv_set          (tag_inv_set),
    .inv_way          (tag_inv_way),
    .dirty_check_valid   (tag_dc_valid),
    .dirty_check_set     (tag_dc_set),
    .dirty_check_way     (tag_dc_way),
    .dirty_check_is_dirty(tag_dc_is_dirty),
    .dirty_check_tag     (tag_dc_tag)
  );

  l2_data_array #(
    .ADDR_WIDTH (ADDR_WIDTH),
    .DATA_WIDTH (DATA_WIDTH),
    .NUM_SETS   (NUM_SETS),
    .NUM_WAYS   (NUM_WAYS),
    .LINE_BYTES (LINE_BYTES)
  ) u_data (
    .clk            (clk),
    .rd_en          (data_rd_en),
    .rd_set         (data_rd_set),
    .rd_way         (data_rd_way),
    .rd_word        (data_rd_word),
    .rd_data        (data_rd_data),
    .wr_en          (data_wr_en),
    .wr_set         (data_wr_set),
    .wr_way         (data_wr_way),
    .wr_word        (data_wr_word),
    .wr_data        (data_wr_data),
    .wr_be          (data_wr_be),
    .line_rd_en     (data_line_rd_en),
    .line_rd_set    (data_line_rd_set),
    .line_rd_way    (data_line_rd_way),
    .line_rd_word   (data_line_rd_word),
    .line_rd_data   (data_line_rd_data),
    .line_wr_en     (data_line_wr_en),
    .line_wr_set    (data_line_wr_set),
    .line_wr_way    (data_line_wr_way),
    .line_wr_word   (data_line_wr_word),
    .line_wr_data   (data_line_wr_data)
  );

  l2_mshr #(
    .ADDR_WIDTH  (ADDR_WIDTH),
    .NUM_ENTRIES (NUM_MSHR),
    .ID_WIDTH    (ID_WIDTH),
    .LINE_BYTES  (LINE_BYTES)
  ) u_mshr (
    .clk              (clk),
    .rst_n            (rst_n),
    .alloc_valid      (mshr_alloc_valid),
    .alloc_ready      (mshr_alloc_ready),
    .alloc_addr       (mshr_alloc_addr),
    .alloc_id         (mshr_alloc_id),
    .alloc_is_write   (mshr_alloc_is_write),
    .alloc_idx        (mshr_alloc_idx),
    .lookup_valid     (mshr_lookup_valid),
    .lookup_addr      (mshr_lookup_addr),
    .lookup_hit       (mshr_lookup_hit),
    .lookup_idx       (mshr_lookup_idx),
    .complete_valid   (mshr_complete_valid),
    .complete_idx     (mshr_complete_idx),
    .complete_addr    (mshr_complete_addr),
    .complete_id      (mshr_complete_id),
    .complete_is_write(mshr_complete_is_write),
    .full             (mshr_full),
    .empty            (mshr_empty),
    .count            ()
  );

  stride_prefetcher #(
    .ADDR_WIDTH    (ADDR_WIDTH),
    .TABLE_ENTRIES (16),
    .LINE_BYTES    (LINE_BYTES)
  ) u_prefetcher (
    .clk          (clk),
    .rst_n        (rst_n),
    .miss_valid   (pf_miss_valid),
    .miss_addr    (pf_miss_addr),
    .pf_req_valid (pf_req_valid),
    .pf_req_ready (pf_req_ready),
    .pf_req_addr  (pf_req_addr),
    .pf_enable    (pf_enable)
  );

endmodule
