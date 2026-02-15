// bsr_scheduler.sv — BSR sparse scheduler for 14×14 weight-stationary systolic array
// Iterates BSR row_ptr/col_idx metadata, skips zero blocks, drives weight
// loading and activation streaming into the systolic array.
//
// FSM: IDLE → FETCH_PTR1/2 → CALC_LEN → FETCH_COL → LOAD_WGT → WAIT_WGT
//      → STREAM_ACT → NEXT_BLK (loop) → NEXT_K (loop) → IDLE
//
// col_idx stored at meta_raddr offset COL_IDX_BASE in metadata BRAM.
// load_weight_r delayed 1 cycle to match BRAM read latency.

`default_nettype none

module bsr_scheduler #(
    parameter M_W        = 10,
    parameter K_W        = 12,
    parameter ADDR_W     = 32,
    parameter BLOCK_SIZE     = 14,  // Must match systolic array (14×14)
    parameter COL_IDX_BASE  = 256   // BRAM offset where col_idx array starts
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 start,
    input  wire                 abort,
    output reg                  busy,
    output reg                  done,

    input  wire [M_W-1:0]       MT,       // M tile count
    input  wire [K_W-1:0]       KT,       // K tile count (BSR rows)

    // Metadata BRAM interface (row_ptr + col_idx)
    output reg  [31:0]          meta_raddr,
    output reg                  meta_ren,
    input  wire                 meta_req_ready,
    input  wire [31:0]          meta_rdata,
    input  wire                 meta_rvalid,
    output wire                 meta_ready,

    // Buffer interfaces
    output reg                  wgt_rd_en,
    output reg  [ADDR_W-1:0]    wgt_addr,
    output reg                  act_rd_en,
    output reg  [ADDR_W-1:0]    act_addr,

    // Systolic array control
    output wire                 load_weight,  // 1-cycle delayed to match BRAM latency
    output reg                  pe_en,        // block_valid to systolic array
    output reg                  accum_en,
    output reg                  pe_clr        // Pulse to clear PE accumulators
);

    // ---------- FSM States (one-hot) ----------
    localparam [9:0]
        S_IDLE        = 10'b0000000001,
        S_FETCH_PTR1  = 10'b0000000010,
        S_FETCH_PTR2  = 10'b0000000100,
        S_CALC_LEN    = 10'b0000001000,
        S_FETCH_COL   = 10'b0000010000,
        S_LOAD_WGT    = 10'b0000100000,
        S_WAIT_WGT    = 10'b0001000000,
        S_STREAM_ACT  = 10'b0010000000,
        S_NEXT_BLK    = 10'b0100000000,
        S_NEXT_K      = 10'b1000000000;

    (* fsm_encoding = "one_hot" *) reg [9:0] state, state_n;

    // ---------- Internal Registers ----------
    reg [K_W-1:0] k_idx;
    reg [M_W-1:0] m_idx;
    reg [$clog2(2*BLOCK_SIZE):0] load_cnt;
    reg [$clog2(2*BLOCK_SIZE):0] wait_cnt;    // R11: dedicated counter for S_WAIT_WGT
    reg [$clog2(2*BLOCK_SIZE):0] stream_cnt;

    localparam LOAD_CNT_MAX   = BLOCK_SIZE - 1;           // 13
    localparam STREAM_CNT_MAX = 2 * BLOCK_SIZE - 2;       // 26 (27 cycles: 14 feed + 13 drain)

    reg [31:0]    blk_ptr;       // Current block index
    reg [31:0]    blk_end;       // End index for current k-row
    reg [31:0]    n_idx;         // Column tile from col_idx[blk_ptr]

    reg load_weight_r;
    assign load_weight = load_weight_r;

    reg [31:0] ptr_start_reg;    // Latched row_ptr[k]
    reg meta_req_sent;

  // ---------- Next State Logic (One-Hot Direct Assign) ----------
  always @(*) begin
    state_n = 10'b0;
    casez (state)
      S_IDLE:       state_n = (start && KT != 0) ? S_FETCH_PTR1 : S_IDLE;  // Q26: KT=0 stays idle
      S_FETCH_PTR1: state_n = meta_rvalid ? S_FETCH_PTR2 : S_FETCH_PTR1;
      S_FETCH_PTR2: state_n = meta_rvalid ? S_CALC_LEN : S_FETCH_PTR2;
      S_CALC_LEN:   state_n = (ptr_start_reg == meta_rdata) ? S_NEXT_K : S_FETCH_COL;
      S_FETCH_COL:  state_n = meta_rvalid ? S_LOAD_WGT : S_FETCH_COL;
      S_LOAD_WGT:   state_n = (load_cnt == BLOCK_SIZE) ? S_WAIT_WGT : S_LOAD_WGT;
      S_WAIT_WGT:   state_n = (wait_cnt == LOAD_CNT_MAX) ? S_STREAM_ACT : S_WAIT_WGT;
      S_STREAM_ACT: state_n = (stream_cnt == STREAM_CNT_MAX) ? S_NEXT_BLK : S_STREAM_ACT;
      S_NEXT_BLK:   state_n = (blk_ptr < blk_end) ? S_FETCH_COL : S_NEXT_K;
      S_NEXT_K:     state_n = (k_idx < KT - 1) ? S_FETCH_PTR1 : S_IDLE;
      default:      state_n = S_IDLE;
    endcase
  end

  // ---------- Datapath & Output Logic ----------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= S_IDLE;
      k_idx          <= 0;
      m_idx          <= 0;
      load_cnt       <= 0;
      wait_cnt       <= 0;
      stream_cnt     <= 0;
      blk_ptr        <= 0;
      blk_end        <= 0;
      meta_ren       <= 0;
      wgt_rd_en      <= 0;
      act_rd_en      <= 0;
      load_weight_r  <= 0;
      pe_en          <= 0;
      pe_clr         <= 0;
      busy           <= 0;
      done           <= 0;
      meta_req_sent  <= 0;
    end else begin
      state <= state_n;

      if (state != state_n) meta_req_sent <= 0;
      if (meta_ren && meta_req_ready) meta_req_sent <= 1;

      // synthesis translate_off
      if (state != state_n)
        $display("[SCHED] @%0t state: %10b -> %10b  k=%0d m=%0d blk=%0d",
                 $time, state, state_n, k_idx, m_idx, blk_ptr);
      // synthesis translate_on

      // Default pulse outputs
      meta_ren  <= 0;
      wgt_rd_en <= 0;
      act_rd_en <= 0;
      pe_en     <= 0;
      accum_en  <= 0;
      pe_clr    <= 0;
      done      <= 0;

      case (state)
        S_IDLE: begin
          busy <= 0;
          k_idx <= 0;
          load_cnt <= 0;
          wait_cnt <= 0;
          stream_cnt <= 0;
          load_weight_r <= 0;
          if (start && KT != 0) busy <= 1;  // Q26: don't start if nothing to do
          if (start && KT == 0) done <= 1;  // Q26: immediately signal done
        end

        S_FETCH_PTR1: begin
          meta_ren   <= !meta_req_sent;
          meta_raddr <= k_idx;
          // synthesis translate_off
          if (meta_rvalid)
            $display("[SCHED] S_FETCH_PTR1: addr=%0d rdata=%0d",
                     k_idx, meta_rdata);
          // synthesis translate_on
          if (meta_rvalid) ptr_start_reg <= meta_rdata;
        end

        S_FETCH_PTR2: begin
          meta_ren   <= !meta_req_sent;
          meta_raddr <= k_idx + 1;
          // synthesis translate_off
          if (meta_rvalid)
            $display("[SCHED] S_FETCH_PTR2: addr=%0d rdata=%0d",
                     k_idx + 1, meta_rdata);
          // synthesis translate_on
        end

        S_CALC_LEN: begin
          blk_ptr <= ptr_start_reg;
          blk_end <= meta_rdata;
        end

        S_FETCH_COL: begin
          meta_ren   <= !meta_req_sent;
          meta_raddr <= COL_IDX_BASE + blk_ptr;  // Use parameter, not magic 128
          if (meta_rvalid) n_idx <= meta_rdata;
        end

        S_LOAD_WGT: begin
          wgt_rd_en <= 1;
          if (load_cnt <= LOAD_CNT_MAX)
            wgt_addr <= (blk_ptr * BLOCK_SIZE) + load_cnt;

          load_weight_r <= (load_cnt > 0);
          load_cnt <= load_cnt + 1;

          if (load_cnt == BLOCK_SIZE) begin
            m_idx    <= 0;
            load_cnt <= 0;
            wait_cnt <= 0;
          end
        end

        S_WAIT_WGT: begin
          load_weight_r <= 0;
          wait_cnt <= wait_cnt + 1;

          if (wait_cnt == LOAD_CNT_MAX) begin
            act_rd_en  <= 1;
            act_addr   <= (k_idx * BLOCK_SIZE);
            wait_cnt   <= 0;
            stream_cnt <= 0;
          end
        end

        S_STREAM_ACT: begin
          pe_en    <= 1;
          accum_en <= 1;  // Enable accumulation during streaming
          stream_cnt <= stream_cnt + 1;

          if (stream_cnt < BLOCK_SIZE) begin
            act_rd_en <= 1;
            act_addr  <= (k_idx * BLOCK_SIZE) + stream_cnt;
          end

          if (stream_cnt == STREAM_CNT_MAX)
            stream_cnt <= 0;
        end

        S_NEXT_BLK: begin
          blk_ptr <= blk_ptr + 1;
        end

        S_NEXT_K: begin
          k_idx <= k_idx + 1;
          if (k_idx == KT - 1) begin
            done <= 1;
            busy <= 0;
          end
        end
      endcase

      if (abort) begin
        state <= S_IDLE;
        busy  <= 0;
        pe_clr <= 1;  // Q27: clear PE accumulators on abort
      end
    end
  end

  // Q29: only accept metadata responses when in a fetch state
  assign meta_ready = (state == S_FETCH_PTR1) || (state == S_FETCH_PTR2) || (state == S_FETCH_COL);

endmodule

`default_nettype wire
