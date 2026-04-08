// =============================================================================
// accel_tile.sv — Single Accelerator Tile (Wraps Array + Scratchpad + NI)
// =============================================================================
// A tile is the fundamental compute unit in the multi-tile system.
// Each tile contains:
//   - 16×16 sparse INT8 systolic array (reuses existing accel_top core)
//   - Scratchpad SRAM (activation/weight/output banks)
//   - Tile controller FSM
//   - Network Interface (packetizer/depacketizer)
//   - Root-side reduce consumer for local MSG_REDUCE traffic
//
// The NoC local port connects to the mesh router at this tile's position.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */
import noc_pkg::*;

module accel_tile #(
  parameter int TILE_ID    = 0,
  parameter int N_ROWS     = 16,
  parameter int N_COLS     = 16,
  parameter int DATA_W     = 8,
  parameter int ACC_W      = 32,
  parameter int SP_DEPTH   = 4096,
  parameter int SP_DATA_W  = 32,
  parameter int SP_ADDR_W  = $clog2(SP_DEPTH),
  parameter int NUM_VCS    = noc_pkg::NUM_VCS
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- NoC local port ---
  output flit_t             noc_flit_out,
  output logic              noc_valid_out,
  input  logic [NUM_VCS-1:0] noc_credit_in,

  input  flit_t             noc_flit_in,
  input  logic              noc_valid_in,
  output logic [NUM_VCS-1:0] noc_credit_out,

  // --- AXI slave for host CSR access (optional, for tile 0) ---
  input  logic [31:0]       csr_wdata,
  input  logic [7:0]        csr_addr,
  input  logic              csr_wen,
  output logic [31:0]       csr_rdata,

  // --- Barrier (directly wired to barrier_sync) ---
  output logic              barrier_req,
  input  logic              barrier_done,

  // --- Tile status ---
  output logic              tile_busy,
  output logic              tile_done
);

  // =========================================================================
  // Internal wires
  // =========================================================================
  // Tile controller ↔ scratchpad
  logic              sp_a_en, sp_a_we;
  logic [SP_ADDR_W-1:0] sp_a_addr;
  logic [SP_DATA_W-1:0] sp_a_wdata, sp_a_rdata;
  logic              sp_a_mux_en, sp_a_mux_we;
  logic [SP_ADDR_W-1:0] sp_a_mux_addr;
  logic [SP_DATA_W-1:0] sp_a_mux_wdata;

  // Systolic array ↔ scratchpad (port B)
  logic              sp_b_en;
  logic [SP_ADDR_W-1:0] sp_b_addr;
  logic [SP_DATA_W-1:0] sp_b_rdata;

  // Tile controller ↔ systolic
  logic sa_start, ctrl_sa_load_weight, ctrl_sa_pe_en, ctrl_sa_accum_en, sa_done;

  // Tile controller → NI
  logic sparse_hint;

  // Local reduce-consumer state
  logic [NUM_VCS-1:0] ni_credit_out_int;
  logic [NUM_VCS-1:0] reduce_credit_out;
  logic reduce_local_valid;
  logic reduce_commit_valid;
  logic [7:0] reduce_commit_id;
  logic [31:0] reduce_commit_value;
  logic [15:0] reduce_packets_consumed;
  logic [15:0] reduce_groups_completed;
  logic [7:0] reduce_last_id;
  logic [31:0] reduce_last_value;
  logic [31:0] reduce_results [256];
  logic [255:0] reduce_result_valid;

  // DMA path
  logic dma_req_valid, dma_req_ready, dma_req_write;
  logic dma_req_aw_ready, dma_req_ar_ready;
  logic [31:0] dma_req_addr;
  logic [15:0] dma_req_len;
  logic dma_data_valid, dma_data_ready;
  logic [SP_DATA_W-1:0] dma_data_in;
  logic [31:0] dma_data_word;
  logic dma_wdata_valid, dma_wdata_ready;
  logic dma_wdata_last;
  logic [SP_DATA_W-1:0] dma_wdata_out;
  logic dma_store_done;

  // Reduce-inject path (tile_controller → NI)
  logic        reduce_inj_valid;
  logic        reduce_inj_ready;
  logic [7:0]  reduce_inj_id;
  logic [3:0]  reduce_inj_expect;
  logic [3:0]  reduce_inj_dst;
  logic [31:0] reduce_inj_val;

  // Tile-local clock enables. Only the fully local compute array is moved onto
  // a gated clock today; scratchpad and NI stay on the root clock until their
  // interfaces grow explicit FIFO/handshake boundaries.
  logic compute_clk_en;
  logic scratchpad_clk_en;
  logic ni_clk_en;
  logic compute_clk;

  // Command interface (from CSR decode)
  logic cmd_valid, cmd_ready;
  logic [7:0] cmd_opcode;
  logic [31:0] cmd_arg0, cmd_arg1, cmd_arg2;
  logic cmd_issue;

  localparam int ACT_VEC_W            = N_ROWS * DATA_W;
  localparam int WGT_VEC_W            = N_COLS * DATA_W;
  localparam int ACT_VEC_WORDS        = ACT_VEC_W / SP_DATA_W;
  localparam int WGT_VEC_WORDS        = WGT_VEC_W / SP_DATA_W;
  localparam int RESULT_WORDS         = N_ROWS * N_COLS;
  localparam int ROW_IDX_W            = (N_ROWS <= 1) ? 1 : $clog2(N_ROWS);
  localparam int ACT_WORD_IDX_W       = (ACT_VEC_WORDS <= 1) ? 1 : $clog2(ACT_VEC_WORDS);
  localparam int WGT_WORD_IDX_W       = (WGT_VEC_WORDS <= 1) ? 1 : $clog2(WGT_VEC_WORDS);
  localparam int RESULT_WORD_IDX_W    = (RESULT_WORDS <= 1) ? 1 : $clog2(RESULT_WORDS);
  localparam int WEIGHT_SETTLE_CYCLES = (N_ROWS > 0) ? (N_ROWS - 1) : 0;
  localparam int WEIGHT_WAIT_W        = (WEIGHT_SETTLE_CYCLES <= 1) ? 1 : $clog2(WEIGHT_SETTLE_CYCLES + 1);
  localparam int DRAIN_CYCLES         = (2 * N_COLS) - 2;
  localparam int DRAIN_CNT_W          = (DRAIN_CYCLES <= 1) ? 1 : $clog2(DRAIN_CYCLES + 1);

  typedef enum logic [3:0] {
    C_IDLE,
    C_PRELOAD_WGT_REQ,
    C_PRELOAD_WGT_CAP,
    C_PRELOAD_ACT_REQ,
    C_PRELOAD_ACT_CAP,
    C_CLR,
    C_WGT_PLAY,
    C_WGT_WAIT,
    C_ACT_PLAY,
    C_ACT_DRAIN,
    C_STORE,
    C_DONE
  } compute_state_e;

  compute_state_e compute_state;

  logic [SP_ADDR_W-1:0] compute_act_base;
  logic [SP_ADDR_W-1:0] compute_wgt_base;
  logic [SP_ADDR_W-1:0] compute_out_base;
  logic [ROW_IDX_W-1:0] wgt_row_idx;
  logic [ROW_IDX_W-1:0] act_vec_idx;
  logic [WGT_WORD_IDX_W-1:0] wgt_word_idx;
  logic [ACT_WORD_IDX_W-1:0] act_word_idx;
  logic [WEIGHT_WAIT_W-1:0] weight_wait_ctr;
  logic [DRAIN_CNT_W-1:0] drain_ctr;
  logic [RESULT_WORD_IDX_W-1:0] store_word_idx;
  logic [WGT_VEC_W-1:0] wgt_block_buf [0:N_ROWS-1];
  logic [ACT_VEC_W-1:0] act_block_buf [0:N_ROWS-1];
  logic                  compute_sa_clr;
  logic                  compute_sa_load_weight;
  logic                  compute_sa_block_valid;
  logic [ACT_VEC_W-1:0]  compute_sa_a_vec;
  logic [WGT_VEC_W-1:0]  compute_sa_b_vec;
  logic [N_ROWS*N_COLS*ACC_W-1:0] systolic_out_flat;
  logic                  compute_sp_a_en;
  logic                  compute_sp_a_we;
  logic [SP_ADDR_W-1:0]  compute_sp_a_addr;
  logic [SP_DATA_W-1:0]  compute_sp_a_wdata;


  // =========================================================================
  // CSR decode → command FIFO (simple)
  // =========================================================================
  // CSR registers:
  //   0x00: CMD_OPCODE (write triggers command)
  //   0x04: CMD_ARG0
  //   0x08: CMD_ARG1
  //   0x0C: CMD_ARG2
  //   0x10: STATUS (read: busy/done)
  //   0x30: RESULT[0]  (first accumulator)
  //   0x34: RESULT[1]
  //   0x38: RESULT[2]
  //   0x3C: RESULT[3]
  //
  // OP_COMPUTE command arguments are interpreted as scratchpad word addresses:
  //   CMD_ARG0 = activation block base
  //   CMD_ARG1 = weight block base
  //   CMD_ARG2 = output block base

  logic [31:0] arg0_reg, arg1_reg, arg2_reg;
  logic        cmd_pending;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      arg0_reg    <= '0;
      arg1_reg    <= '0;
      arg2_reg    <= '0;
      cmd_pending <= 1'b0;
      cmd_issue   <= 1'b0;
      cmd_opcode  <= '0;
    end else begin
      cmd_issue <= 1'b0;

      // Launch a queued command as a single-cycle pulse once the controller
      // is ready. Keeping cmd_valid level-high caused stale commands to replay
      // when the controller returned to IDLE.
      if (cmd_pending && cmd_ready) begin
        cmd_issue   <= 1'b1;
        cmd_pending <= 1'b0;
      end

      if (csr_wen) begin
        case (csr_addr)
          8'h04: arg0_reg <= csr_wdata;
          8'h08: arg1_reg <= csr_wdata;
          8'h0C: arg2_reg <= csr_wdata;
          8'h00: begin
            cmd_opcode  <= csr_wdata[7:0];
            cmd_pending <= 1'b1;
          end
          default: ;
        endcase
      end

`ifndef SYNTHESIS
                 TILE_ID, cmd_opcode, arg0_reg, arg1_reg, arg2_reg);
`endif
    end
  end

  assign cmd_valid = cmd_issue;
  assign cmd_arg0  = arg0_reg;
  assign cmd_arg1  = arg1_reg;
  assign cmd_arg2  = arg2_reg;

  // CSR read
  always_comb begin
    case (csr_addr)
      8'h00: csr_rdata = {24'h0, cmd_opcode};
      8'h04: csr_rdata = arg0_reg;
      8'h08: csr_rdata = arg1_reg;
      8'h0C: csr_rdata = arg2_reg;
      8'h10: csr_rdata = {30'h0, tile_done, tile_busy};
      8'h20: csr_rdata = {16'h0, reduce_groups_completed};
      8'h24: csr_rdata = {16'h0, reduce_packets_consumed};
      8'h28: csr_rdata = {24'h0, reduce_last_id};
      8'h2C: csr_rdata = reduce_last_value;
      8'h30: csr_rdata = systolic_out_flat[0*ACC_W +: 32];
      8'h34: csr_rdata = systolic_out_flat[1*ACC_W +: 32];
      8'h38: csr_rdata = systolic_out_flat[2*ACC_W +: 32];
      8'h3C: csr_rdata = systolic_out_flat[3*ACC_W +: 32];
      default: csr_rdata = '0;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      reduce_last_id    <= '0;
      reduce_last_value <= '0;
      reduce_result_valid <= '0;
      for (int rid = 0; rid < 256; rid++)
        reduce_results[rid] <= '0;
    end else if (reduce_commit_valid) begin
      reduce_results[reduce_commit_id]      <= reduce_commit_value;
      reduce_result_valid[reduce_commit_id] <= 1'b1;
      reduce_last_id                        <= reduce_commit_id;
      reduce_last_value                     <= reduce_commit_value;
    end
  end

  // =========================================================================
  // Tile Controller
  // =========================================================================
  tile_controller #(
    .TILE_ID   (TILE_ID),
    .SP_ADDR_W (SP_ADDR_W),
    .SP_DATA_W (SP_DATA_W),
    .N_ROWS    (N_ROWS),
    .N_COLS    (N_COLS)
  ) u_ctrl (
    .clk            (clk),
    .rst_n          (rst_n),
    .cmd_valid      (cmd_valid),
    .cmd_ready      (cmd_ready),
    .cmd_opcode     (cmd_opcode),
    .cmd_arg0       (cmd_arg0),
    .cmd_arg1       (cmd_arg1),
    .cmd_arg2       (cmd_arg2),
    .busy           (tile_busy),
    .done           (tile_done),
    .error          (),
    .sparse_hint    (sparse_hint),
    .sp_a_en        (sp_a_en),
    .sp_a_we        (sp_a_we),
    .sp_a_addr      (sp_a_addr),
    .sp_a_wdata     (sp_a_wdata),
    .sp_a_rdata     (sp_a_rdata),
    .sa_start       (sa_start),
    .sa_load_weight (ctrl_sa_load_weight),
    .sa_pe_en       (ctrl_sa_pe_en),
    .sa_accum_en    (ctrl_sa_accum_en),
    .sa_done        (sa_done),
    .dma_req_valid  (dma_req_valid),
    .dma_req_ready  (dma_req_ready),
    .dma_req_write  (dma_req_write),
    .dma_req_addr   (dma_req_addr),
    .dma_req_len    (dma_req_len),
    .dma_data_valid (dma_data_valid),
    .dma_data_ready (dma_data_ready),
    .dma_data_in    (dma_data_in),
    .dma_wdata_valid(dma_wdata_valid),
    .dma_wdata_ready(dma_wdata_ready),
    .dma_wdata_out  (dma_wdata_out),
    .dma_wdata_last (dma_wdata_last),
    .dma_done_valid (dma_store_done),
    .barrier_req    (barrier_req),
    .barrier_done   (barrier_done),
    .reduce_inj_valid  (reduce_inj_valid),
    .reduce_inj_id     (reduce_inj_id),
    .reduce_inj_expect (reduce_inj_expect),
    .reduce_inj_dst    (reduce_inj_dst),
    .reduce_inj_val    (reduce_inj_val),
    .reduce_inj_ready  (reduce_inj_ready)
  );

  assign compute_clk_en    = sa_start || (compute_state != C_IDLE);
  assign scratchpad_clk_en = sp_a_mux_en || sp_b_en;
  assign ni_clk_en         = tile_busy || noc_valid_in || noc_valid_out;

  clock_gate_cell u_compute_clk_gate (
    .clk_i     (clk),
    .en_i      (compute_clk_en),
    .test_en_i (1'b0),
    .clk_o     (compute_clk)
  );

  // =========================================================================
  // Scratchpad
  // =========================================================================
  accel_scratchpad #(
    .DEPTH      (SP_DEPTH),
    .DATA_WIDTH (SP_DATA_W),
    .ADDR_WIDTH (SP_ADDR_W)
  ) u_sp (
    .clk     (clk),
    .rst_n   (rst_n),
    .clk_en  (scratchpad_clk_en),
    .a_en    (sp_a_mux_en),
    .a_we    (sp_a_mux_we),
    .a_addr  (sp_a_mux_addr),
    .a_wdata (sp_a_mux_wdata),
    .a_rdata (sp_a_rdata),
    .b_en    (sp_b_en),
    .b_addr  (sp_b_addr),
    .b_rdata (sp_b_rdata)
  );

  // Scratchpad port A is shared between the controller (DMA load/store) and
  // the compute engine when spilling accumulators back into tile-local memory.
  assign sp_a_mux_en    = compute_sp_a_en ? 1'b1           : sp_a_en;
  assign sp_a_mux_we    = compute_sp_a_en ? 1'b1           : sp_a_we;
  assign sp_a_mux_addr  = compute_sp_a_en ? compute_sp_a_addr  : sp_a_addr;
  assign sp_a_mux_wdata = compute_sp_a_en ? compute_sp_a_wdata : sp_a_wdata;

  // Tile-local compute engine:
  //   1. Clear the array.
  //   2. Read one 16x16 weight block from scratchpad into the PE rows.
  //   3. Stream one 16x16 activation block.
  //   4. Drain the skewed wavefront.
  //   5. Spill the 256 INT32 accumulators back to scratchpad.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      compute_state    <= C_IDLE;
      compute_act_base <= '0;
      compute_wgt_base <= '0;
      compute_out_base <= '0;
      wgt_row_idx      <= '0;
      act_vec_idx      <= '0;
      wgt_word_idx     <= '0;
      act_word_idx     <= '0;
      weight_wait_ctr  <= '0;
      drain_ctr        <= '0;
      store_word_idx   <= '0;
      for (int row = 0; row < N_ROWS; row++) begin
        wgt_block_buf[row] <= '0;
        act_block_buf[row] <= '0;
      end
    end else if (compute_clk_en) begin
      case (compute_state)
        C_IDLE: begin
          if (sa_start) begin
            compute_act_base <= cmd_arg0[SP_ADDR_W-1:0];
            compute_wgt_base <= cmd_arg1[SP_ADDR_W-1:0];
            compute_out_base <= cmd_arg2[SP_ADDR_W-1:0];
            wgt_row_idx      <= '0;
            act_vec_idx      <= '0;
            wgt_word_idx     <= '0;
            act_word_idx     <= '0;
            weight_wait_ctr  <= '0;
            drain_ctr        <= '0;
            store_word_idx   <= '0;
            compute_state    <= C_PRELOAD_WGT_REQ;
          end
        end

        C_PRELOAD_WGT_REQ: begin
          compute_state <= C_PRELOAD_WGT_CAP;
        end

        C_PRELOAD_WGT_CAP: begin
          wgt_block_buf[wgt_row_idx][wgt_word_idx*SP_DATA_W +: SP_DATA_W] <= sp_b_rdata;
          `ifndef SYNTHESIS
          `endif
          if (wgt_word_idx == WGT_WORD_IDX_W'(WGT_VEC_WORDS - 1)) begin
            wgt_word_idx  <= '0;
            if (wgt_row_idx == ROW_IDX_W'(N_ROWS - 1)) begin
              wgt_row_idx   <= '0;
              compute_state <= C_PRELOAD_ACT_REQ;
            end else begin
              wgt_row_idx   <= wgt_row_idx + 1'b1;
              compute_state <= C_PRELOAD_WGT_REQ;
            end
          end else begin
            wgt_word_idx  <= wgt_word_idx + 1'b1;
            compute_state <= C_PRELOAD_WGT_REQ;
          end
        end

        C_PRELOAD_ACT_REQ: begin
          compute_state <= C_PRELOAD_ACT_CAP;
        end

        C_PRELOAD_ACT_CAP: begin
          act_block_buf[act_vec_idx][act_word_idx*SP_DATA_W +: SP_DATA_W] <= sp_b_rdata;
          if (act_word_idx == ACT_WORD_IDX_W'(ACT_VEC_WORDS - 1)) begin
            act_word_idx <= '0;
            if (act_vec_idx == ROW_IDX_W'(N_ROWS - 1)) begin
              act_vec_idx    <= '0;
              weight_wait_ctr <= '0;
              compute_state  <= C_CLR;
            end else begin
              act_vec_idx   <= act_vec_idx + 1'b1;
              compute_state <= C_PRELOAD_ACT_REQ;
            end
          end else begin
            act_word_idx  <= act_word_idx + 1'b1;
            compute_state <= C_PRELOAD_ACT_REQ;
          end
        end

        C_CLR: begin
          wgt_row_idx   <= '0;
          compute_state <= C_WGT_PLAY;
        end

        C_WGT_PLAY: begin
          if (wgt_row_idx == ROW_IDX_W'(N_ROWS - 1)) begin
            wgt_row_idx <= '0;
            if (WEIGHT_SETTLE_CYCLES == 0) begin
              act_vec_idx   <= '0;
              compute_state <= C_ACT_PLAY;
            end else begin
              weight_wait_ctr <= '0;
              compute_state   <= C_WGT_WAIT;
            end
          end else begin
            wgt_row_idx <= wgt_row_idx + 1'b1;
          end
        end

        C_WGT_WAIT: begin
          if (weight_wait_ctr == WEIGHT_WAIT_W'(WEIGHT_SETTLE_CYCLES - 1)) begin
            weight_wait_ctr <= '0;
            act_vec_idx     <= '0;
            compute_state   <= C_ACT_PLAY;
          end else begin
            weight_wait_ctr <= weight_wait_ctr + 1'b1;
          end
        end

        C_ACT_PLAY: begin
          if (act_vec_idx == ROW_IDX_W'(N_ROWS - 1)) begin
            act_vec_idx <= '0;
            if (DRAIN_CYCLES == 0) begin
              store_word_idx <= '0;
              compute_state  <= C_STORE;
            end else begin
              drain_ctr     <= '0;
              compute_state <= C_ACT_DRAIN;
            end
          end else begin
            act_vec_idx   <= act_vec_idx + 1'b1;
          end
        end

        C_ACT_DRAIN: begin
          if (drain_ctr == DRAIN_CNT_W'(DRAIN_CYCLES - 1)) begin
            drain_ctr      <= '0;
            store_word_idx <= '0;
            compute_state  <= C_STORE;
          end else begin
            drain_ctr <= drain_ctr + 1'b1;
          end
        end

        C_STORE: begin
          if (store_word_idx == RESULT_WORD_IDX_W'(RESULT_WORDS - 1)) begin
            compute_state <= C_DONE;
          end else begin
            store_word_idx <= store_word_idx + 1'b1;
          end
        end

        C_DONE: begin
          compute_state <= C_IDLE;
        end

        default: begin
          compute_state <= C_IDLE;
        end
      endcase
    end
  end

  always_comb begin
    compute_sa_clr         = 1'b0;
    compute_sa_load_weight = 1'b0;
    compute_sa_block_valid = 1'b0;
    compute_sa_a_vec       = '0;
    compute_sa_b_vec       = '0;
    compute_sp_a_en        = 1'b0;
    compute_sp_a_we        = 1'b0;
    compute_sp_a_addr      = '0;
    compute_sp_a_wdata     = '0;
    sp_b_en                = 1'b0;
    sp_b_addr              = '0;

    case (compute_state)
      C_CLR: begin
        compute_sa_clr = 1'b1;
      end

      C_PRELOAD_WGT_REQ: begin
        sp_b_en   = 1'b1;
        sp_b_addr = compute_wgt_base +
                    SP_ADDR_W'(wgt_row_idx * WGT_VEC_WORDS) +
                    SP_ADDR_W'(wgt_word_idx);
      end

      C_PRELOAD_ACT_REQ: begin
        sp_b_en   = 1'b1;
        sp_b_addr = compute_act_base +
                    SP_ADDR_W'(act_vec_idx * ACT_VEC_WORDS) +
                    SP_ADDR_W'(act_word_idx);
      end

      C_WGT_PLAY: begin
        compute_sa_load_weight = 1'b1;
        compute_sa_b_vec       = wgt_block_buf[wgt_row_idx];
        `ifndef SYNTHESIS
        `endif
      end

      C_ACT_PLAY: begin
        compute_sa_block_valid = ctrl_sa_pe_en && ctrl_sa_accum_en;
        compute_sa_a_vec       = act_block_buf[act_vec_idx];
      end

      C_ACT_DRAIN: begin
        compute_sa_block_valid = ctrl_sa_pe_en && ctrl_sa_accum_en;
      end

      C_STORE: begin
        compute_sp_a_en    = 1'b1;
        compute_sp_a_we    = 1'b1;
        compute_sp_a_addr  = compute_out_base + SP_ADDR_W'(store_word_idx);
        compute_sp_a_wdata = systolic_out_flat[store_word_idx*ACC_W +: SP_DATA_W];
      end

      default: ;
    endcase
  end

  always @(posedge clk) begin
    if (compute_sp_a_en && sp_a_en)
      $error("accel_tile: compute spill collided with controller scratchpad access");
    if (sp_b_en)
      assert (int'(sp_b_addr) < SP_DEPTH)
        else $error("accel_tile: compute read address %0d out of range", sp_b_addr);
    if (compute_sp_a_en)
      assert (int'(compute_sp_a_addr) < SP_DEPTH)
        else $error("accel_tile: compute write address %0d out of range", compute_sp_a_addr);
  end

  systolic_array_sparse #(
    .N_ROWS (N_ROWS),
    .N_COLS (N_COLS),
    .DATA_W (DATA_W),
    .ACC_W  (ACC_W)
  ) u_systolic (
    .clk         (compute_clk),
    .rst_n       (rst_n),
    .clk_en      (1'b1),
    .block_valid (compute_sa_block_valid),
    .load_weight (compute_sa_load_weight),
    .clr         (compute_sa_clr),
    .a_in_flat   (compute_sa_a_vec),
    .b_in_flat   (compute_sa_b_vec),
    .c_out_flat  (systolic_out_flat)
  );

  assign sa_done        = (compute_state == C_DONE);
  assign dma_req_ready  = dma_req_write ? dma_req_aw_ready : dma_req_ar_ready;
  assign dma_data_in    = SP_DATA_W'(dma_data_word);

  wire _unused_ctrl_compute = &{1'b0, ctrl_sa_load_weight};

  assign reduce_local_valid = noc_valid_in &&
                              (noc_flit_in.msg_type == MSG_REDUCE) &&
                              (noc_flit_in.flit_type == FLIT_HEADTAIL);

  tile_reduce_consumer #(
    .NUM_VCS     (NUM_VCS),
    .ENTRY_DEPTH (noc_pkg::INNET_SP_DEPTH)
  ) u_reduce_sink (
    .clk              (clk),
    .rst_n            (rst_n),
    .enable           (ni_clk_en),
    .flit_in          (noc_flit_in),
    .valid_in         (reduce_local_valid),
    .credit_out       (reduce_credit_out),
    .commit_valid     (reduce_commit_valid),
    .commit_id        (reduce_commit_id),
    .commit_value     (reduce_commit_value),
    .packets_consumed (reduce_packets_consumed),
    .groups_completed (reduce_groups_completed)
  );

  // =========================================================================
  // Network Interface
  // =========================================================================
  noc_network_interface #(
    .NODE_ID    (TILE_ID),
    .ADDR_WIDTH (32),
    .DATA_WIDTH (32),
    .NUM_VCS    (NUM_VCS)
  ) u_ni (
    .clk            (clk),
    .rst_n          (rst_n),
    .clk_en         (ni_clk_en),
    // AXI-like (connected to DMA path)
    .aw_valid       (dma_req_valid && dma_req_write),
    .aw_ready       (dma_req_aw_ready),
    .aw_addr        (dma_req_addr),
    .aw_len         (dma_req_len[7:0]),
    .aw_id          (4'(TILE_ID)),
    .w_valid        (dma_wdata_valid),
    .w_ready        (dma_wdata_ready),
    .w_data         (dma_wdata_out[31:0]),
    .w_last         (dma_wdata_last),
    .b_valid        (dma_store_done),
    .b_ready        (1'b1),
    .b_id           (),
    .b_resp         (),
    .ar_valid       (dma_req_valid && !dma_req_write),
    .ar_ready       (dma_req_ar_ready),
    .ar_addr        (dma_req_addr),
    .ar_id          (4'(TILE_ID)),
    .ar_len         (dma_req_len[7:0]),
    .r_valid        (dma_data_valid),
    .r_ready        (dma_data_ready),
    .r_data         (dma_data_word),
    .r_id           (),
    .r_resp         (),
    .r_last         (),
    // Sparsity
    .sparse_hint    (sparse_hint),
    // Reduce inject (from tile controller OP_REDUCE)
    .reduce_inj_valid  (reduce_inj_valid),
    .reduce_inj_ready  (reduce_inj_ready),
    .reduce_inj_id     (reduce_inj_id),
    .reduce_inj_expect (reduce_inj_expect),
    .reduce_inj_dst    (reduce_inj_dst),
    .reduce_inj_val    (reduce_inj_val),
    // NoC
    .noc_flit_out   (noc_flit_out),
    .noc_valid_out  (noc_valid_out),
    .noc_credit_in  (noc_credit_in),
    .noc_flit_in    (noc_flit_in),
    .noc_valid_in   (noc_valid_in && !reduce_local_valid),
    .noc_credit_out (ni_credit_out_int)
  );

  assign noc_credit_out = ni_credit_out_int | reduce_credit_out;

endmodule
