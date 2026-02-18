// bsr_scheduler.sv — BSR sparse scheduler for 14×14 weight-stationary systolic array
// Iterates BSR row_ptr/col_idx metadata, skips zero blocks, drives weight
// loading and activation streaming into the systolic array.
//
// FSM: IDLE → FETCH_PTR1/2 → CALC_LEN → FETCH_COL → LOAD_WGT → WAIT_WGT
//      → STREAM_ACT → NEXT_BLK (loop) → NEXT_K (loop) → IDLE
//
// Direct BRAM wires — no meta decoder. Scheduler has separate read ports
// for row_ptr and col_idx BRAMs (combinational reads, 1-cycle addr→data).
// load_weight_r delayed 1 cycle to match BRAM read latency.

`default_nettype none

module bsr_scheduler #(
    parameter M_W           = 10,
    parameter K_W           = 12,
    parameter ADDR_W        = 32,
    parameter BRAM_ADDR_W   = 10,
    parameter BLOCK_SIZE    = 14    // Must match systolic array (14×14)
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 start,
    input  wire                 abort,
    output reg                  busy,
    output reg                  done,

    input  wire [M_W-1:0]       MT,       // M tile count
    input  wire [K_W-1:0]       KT,       // K tile count (BSR rows)

    // Direct BRAM read: row_ptr (combinational read, 1-cycle latency)
    output reg  [BRAM_ADDR_W-1:0] row_ptr_rd_addr,
    input  wire [31:0]            row_ptr_rd_data,

    // Direct BRAM read: col_idx (combinational read, 1-cycle latency)
    output reg  [BRAM_ADDR_W-1:0] col_idx_rd_addr,
    input  wire [15:0]            col_idx_rd_data,

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

    // MT input is part of the configuration interface but unused by this
    // scheduler variant (BSR iterates K-rows via row_ptr, not M-rows).
    wire _unused_MT = &{1'b0, MT};

    // ---------- Internal Registers ----------
    reg [K_W-1:0] k_idx;
    reg [M_W-1:0] m_idx;
    reg [$clog2(2*BLOCK_SIZE):0] load_cnt;
    reg [$clog2(2*BLOCK_SIZE):0] wait_cnt;
    reg [$clog2(2*BLOCK_SIZE):0] stream_cnt;

    localparam LOAD_CNT_MAX   = BLOCK_SIZE - 1;           // 13
    localparam STREAM_CNT_MAX = 2 * BLOCK_SIZE - 2;       // 26

    reg [31:0]    blk_ptr;       // Current block index
    reg [31:0]    blk_end;       // End index for current k-row

    reg load_weight_r;
    assign load_weight = load_weight_r;

    reg [31:0] ptr_start_reg;    // Latched row_ptr[k]

    // 1-cycle wait for BRAM read latency (addr registered → data combinational)
    reg fetch_wait;

    // col_idx_rd_data is read in S_FETCH_COL but currently unused in the
    // address computation (activation address uses k_idx). Suppress lint.
    wire _unused_col_idx = &{1'b0, col_idx_rd_data};

  // ---------- Next State Logic (One-Hot Direct Assign) ----------
  always @(*) begin
    state_n = 10'b0;
    casez (state)
      S_IDLE:       state_n = (start && KT != 0) ? S_FETCH_PTR1 : S_IDLE;
      S_FETCH_PTR1: state_n = fetch_wait ? S_FETCH_PTR2 : S_FETCH_PTR1;
      S_FETCH_PTR2: state_n = fetch_wait ? S_CALC_LEN   : S_FETCH_PTR2;
      S_CALC_LEN:   state_n = (ptr_start_reg == blk_end) ? S_NEXT_K : S_FETCH_COL;
      S_FETCH_COL:  state_n = fetch_wait ? S_LOAD_WGT   : S_FETCH_COL;
      S_LOAD_WGT:   state_n = (load_cnt == BLOCK_SIZE) ? S_WAIT_WGT : S_LOAD_WGT;
      S_WAIT_WGT:   state_n = (wait_cnt == LOAD_CNT_MAX) ? S_STREAM_ACT : S_WAIT_WGT;
      S_STREAM_ACT: state_n = (stream_cnt == STREAM_CNT_MAX) ? S_NEXT_BLK : S_STREAM_ACT;
      S_NEXT_BLK:   state_n = (blk_ptr + 32'd1 < blk_end) ? S_FETCH_COL : S_NEXT_K;
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
      wgt_rd_en      <= 0;
      act_rd_en      <= 0;
      load_weight_r  <= 0;
      pe_en          <= 0;
      pe_clr         <= 0;
      busy           <= 0;
      done           <= 0;
      fetch_wait     <= 0;
      row_ptr_rd_addr <= 0;
      col_idx_rd_addr <= 0;
      ptr_start_reg  <= 0;
    end else begin
      state <= state_n;

      // Clear fetch_wait on any state transition so each new fetch
      // state gets exactly 1 cycle of address setup before reading.
      if (state != state_n) fetch_wait <= 0;

      // synthesis translate_off
      if (state != state_n)
        $display("[SCHED] @%0t state: %10b -> %10b  k=%0d m=%0d blk=%0d",
                 $time, state, state_n, k_idx, m_idx, blk_ptr);
      // synthesis translate_on

      // Default pulse outputs
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
          if (start && KT != 0) busy <= 1;
          if (start && KT == 0) done <= 1;
        end

        // ---- Fetch row_ptr[k] ----
        // Cycle 0: present address → BRAM reads combinationally
        // Cycle 1: fetch_wait=1, data valid on row_ptr_rd_data → capture
        S_FETCH_PTR1: begin
          row_ptr_rd_addr <= k_idx[BRAM_ADDR_W-1:0];
          if (fetch_wait) begin
            ptr_start_reg <= row_ptr_rd_data;
            // synthesis translate_off
            $display("[SCHED] S_FETCH_PTR1: addr=%0d rdata=%0d",
                     k_idx, row_ptr_rd_data);
            // synthesis translate_on
          end else begin
            fetch_wait <= 1;
          end
        end

        // ---- Fetch row_ptr[k+1] ----
        S_FETCH_PTR2: begin
          row_ptr_rd_addr <= k_idx[BRAM_ADDR_W-1:0] + {{(BRAM_ADDR_W-1){1'b0}}, 1'b1};
          if (fetch_wait) begin
            blk_end <= row_ptr_rd_data;
            // synthesis translate_off
            $display("[SCHED] S_FETCH_PTR2: addr=%0d rdata=%0d",
                     k_idx + 1, row_ptr_rd_data);
            // synthesis translate_on
          end else begin
            fetch_wait <= 1;
          end
        end

        S_CALC_LEN: begin
          blk_ptr <= ptr_start_reg;
          // blk_end already set in S_FETCH_PTR2
        end

        // ---- Fetch col_idx[blk_ptr] ----
        S_FETCH_COL: begin
          col_idx_rd_addr <= blk_ptr[BRAM_ADDR_W-1:0];
          if (!fetch_wait)
            fetch_wait <= 1;
          // col_idx_rd_data available when fetch_wait=1
        end

        S_LOAD_WGT: begin
          wgt_rd_en <= 1;
          if (load_cnt <= LOAD_CNT_MAX)
            wgt_addr <= (blk_ptr * BLOCK_SIZE) + {{(32-$clog2(2*BLOCK_SIZE)-1){1'b0}}, load_cnt};

          // Fix: Start load_weight at T0 so Data0 (arriving T1) latches into Row 0
          load_weight_r <= (load_cnt < BLOCK_SIZE);
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

          // Fix: Remove preemptive read - S_STREAM_ACT will issue first read
          // The preemptive read was causing address 0 to be read twice
          if (wait_cnt == LOAD_CNT_MAX) begin
            wait_cnt   <= 0;
            stream_cnt <= 0;
          end
        end

        S_STREAM_ACT: begin
          pe_en    <= 1;
          accum_en <= 1;
          stream_cnt <= stream_cnt + 1;

          if (stream_cnt < BLOCK_SIZE) begin
            act_rd_en <= 1;
            act_addr  <= ({{(32-K_W){1'b0}}, k_idx} * BLOCK_SIZE) + {{(32-$clog2(2*BLOCK_SIZE)-1){1'b0}}, stream_cnt};
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

        default: begin
          busy <= 0;
        end
      endcase

      if (abort) begin
        state <= S_IDLE;
        busy  <= 0;
        pe_clr <= 1;
      end
    end
  end

endmodule

`default_nettype wire
