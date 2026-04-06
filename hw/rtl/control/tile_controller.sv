// =============================================================================
// tile_controller.sv — Per-Tile FSM Controller
// =============================================================================
// Manages the lifecycle of a single accelerator tile:
//   1. IDLE → wait for command from host (via NoC or CSR)
//   2. LOAD → DMA activations + weights from DRAM/other tiles into scratchpad
//   3. COMPUTE → drive systolic array, stream data from scratchpad
//   4. STORE → write results back via NoC / DMA
//   5. SYNC → barrier with other tiles (for multi-tile operations)
//
// Commands arrive via a simple FIFO interface (filled by NI or CSR).

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module tile_controller #(
  parameter int TILE_ID      = 0,
  parameter int SP_ADDR_W    = 12,       // Scratchpad address width
  parameter int SP_DATA_W    = 64,       // Scratchpad data width
  parameter int N_ROWS       = 16,       // Systolic array rows
  parameter int N_COLS       = 16,       // Systolic array cols
  parameter int DMA_MAX_BURST = 16       // Max beats per NoC/AXI burst
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Command interface (from NI or CSR) ---
  input  logic                   cmd_valid,
  output logic                   cmd_ready,
  input  logic [7:0]             cmd_opcode,
  input  logic [31:0]            cmd_arg0,     // e.g., DMA source address
  input  logic [31:0]            cmd_arg1,     // e.g., DMA length / dest
  input  logic [31:0]            cmd_arg2,     // e.g., matrix M dimension

  // --- Status ---
  output logic                   busy,
  output logic                   done,
  output logic                   error,
  output logic                   sparse_hint,  // Feeds NI for VC allocation

  // --- Scratchpad port A (DMA fill/drain) ---
  output logic                   sp_a_en,
  output logic                   sp_a_we,
  output logic [SP_ADDR_W-1:0]  sp_a_addr,
  output logic [SP_DATA_W-1:0]  sp_a_wdata,
  input  logic [SP_DATA_W-1:0]  sp_a_rdata,

  // --- Systolic array control ---
  output logic                   sa_start,
  output logic                   sa_load_weight,
  output logic                   sa_pe_en,
  output logic                   sa_accum_en,
  input  logic                   sa_done,

  // --- DMA request to NI (simplified) ---
  output logic                   dma_req_valid,
  input  logic                   dma_req_ready,
  output logic                   dma_req_write,  // 0=read, 1=write
  output logic [31:0]            dma_req_addr,
  output logic [15:0]            dma_req_len,

  // --- DMA data to/from NI ---
  input  logic                   dma_data_valid,
  output logic                   dma_data_ready,
  input  logic [SP_DATA_W-1:0]  dma_data_in,

  output logic                   dma_wdata_valid,
  input  logic                   dma_wdata_ready,
  output logic [SP_DATA_W-1:0]  dma_wdata_out,
  output logic                   dma_wdata_last,
  input  logic                   dma_done_valid,

  // --- Barrier interface ---
  output logic                   barrier_req,
  input  logic                   barrier_done,

  // --- Reduce-inject interface (OP_REDUCE → NI MSG_REDUCE) ---
  output logic                   reduce_inj_valid,  // one flit per output row
  output logic [7:0]             reduce_inj_id,     // reduce_id for this flit
  output logic [3:0]             reduce_inj_expect, // expected contributors
  output logic [3:0]             reduce_inj_dst,    // root tile NoC node id
  output logic [31:0]            reduce_inj_val,    // INT32 partial sum
  input  logic                   reduce_inj_ready   // NI accepted the flit
);

  // =========================================================================
  // Opcodes
  // =========================================================================
  localparam logic [7:0] OP_NOP     = 8'h00;
  localparam logic [7:0] OP_LOAD    = 8'h01;  // DMA read → scratchpad
  localparam logic [7:0] OP_STORE   = 8'h02;  // scratchpad → DMA write
  localparam logic [7:0] OP_COMPUTE = 8'h03;  // Run systolic array
  localparam logic [7:0] OP_BARRIER = 8'h04;  // Synchronize with other tiles
  localparam logic [7:0] OP_SPARSE  = 8'h05;  // Set sparse hint mode
  localparam logic [7:0] OP_REDUCE  = 8'h06;  // Emit MSG_REDUCE flits to root
  localparam int BYTES_PER_WORD = SP_DATA_W / 8;

  function automatic logic [15:0] burst_word_count(input logic [15:0] words_remaining);
    begin
      if (words_remaining > 16'(DMA_MAX_BURST))
        burst_word_count = 16'(DMA_MAX_BURST);
      else
        burst_word_count = words_remaining;
    end
  endfunction

  // =========================================================================
  // State machine
  // =========================================================================
  typedef enum logic [4:0] {
    S_IDLE,
    S_LOAD_REQ,
    S_LOAD_DATA,
    S_STORE_REQ,
    S_STORE_READ,
    S_STORE_DATA,
    S_STORE_RESP,
    S_COMPUTE_START,
    S_COMPUTE_WAIT,
    S_BARRIER_WAIT,
    S_REDUCE_READ,   // request SP read for current reduce row
    S_REDUCE_EMIT,   // wait for NI to accept MSG_REDUCE flit
    S_DONE
  } state_e;

  state_e state;
  logic [SP_ADDR_W-1:0] sp_ptr;
  logic [15:0]          xfer_remaining;
  logic [15:0]          burst_cnt;
  logic [15:0]          burst_words;
  logic [31:0]          saved_addr;

  // OP_REDUCE state
  logic [3:0]           reduce_row_idx;    // current row being emitted (0..N_ROWS-1)
  logic [7:0]           reduce_id_base;    // reduce_id for row 0
  logic [3:0]           reduce_expect_r;   // expected contributor count
  logic [3:0]           reduce_dst_r;      // root tile node id
  logic [SP_ADDR_W-1:0] reduce_sp_base;    // scratchpad word base
  logic [31:0]          reduce_val_latch;  // latched SP readout for current row

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      sp_ptr      <= '0;
      xfer_remaining <= '0;
      burst_cnt   <= '0;
      burst_words <= '0;
      saved_addr  <= '0;
      sparse_hint <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          if (cmd_valid) begin
            case (cmd_opcode)
              OP_LOAD: begin
                if (cmd_arg1[15:0] == '0) begin
                  state <= S_DONE;
                end else begin
                  saved_addr     <= cmd_arg0;
                  xfer_remaining <= cmd_arg1[15:0];
                  burst_words    <= burst_word_count(cmd_arg1[15:0]);
                  sp_ptr         <= SP_ADDR_W'(cmd_arg2);
                  burst_cnt      <= '0;
                  state          <= S_LOAD_REQ;
                end
              end
              OP_STORE: begin
                if (cmd_arg1[15:0] == '0) begin
                  state <= S_DONE;
                end else begin
                  saved_addr     <= cmd_arg0;
                  xfer_remaining <= cmd_arg1[15:0];
                  burst_words    <= burst_word_count(cmd_arg1[15:0]);
                  sp_ptr         <= SP_ADDR_W'(cmd_arg2);
                  burst_cnt      <= '0;
                  state          <= S_STORE_REQ;
                end
              end
              OP_COMPUTE: begin
                state <= S_COMPUTE_START;
              end
              OP_BARRIER: begin
                state <= S_BARRIER_WAIT;
              end
              OP_SPARSE: begin
                sparse_hint <= cmd_arg0[0];
                state       <= S_DONE;
              end
              OP_REDUCE: begin
                // arg0 = SP result base word offset
                // arg1[3:0]  = root tile dst id
                // arg1[11:4] = reduce_id for row 0
                // arg1[15:12]= reduce_expect count
                reduce_sp_base    <= SP_ADDR_W'(cmd_arg0);
                reduce_dst_r      <= cmd_arg1[3:0];
                reduce_id_base    <= cmd_arg1[11:4];
                reduce_expect_r   <= cmd_arg1[15:12];
                reduce_row_idx    <= '0;
                state             <= S_REDUCE_READ;
              end
              default: state <= S_DONE;
            endcase
          end
        end

        // --- LOAD path ---
        S_LOAD_REQ: begin
          if (dma_req_ready) begin
            burst_cnt <= '0;
            state <= S_LOAD_DATA;
          end
        end

        S_LOAD_DATA: begin
          if (dma_data_valid && dma_data_ready) begin
            saved_addr     <= saved_addr + BYTES_PER_WORD;
            sp_ptr         <= sp_ptr + 1'b1;
            xfer_remaining <= xfer_remaining - 1'b1;

            if (burst_cnt == (burst_words - 1'b1)) begin
              burst_cnt <= '0;
              if (xfer_remaining == 16'd1) begin
                burst_words <= '0;
                state       <= S_DONE;
              end else begin
                burst_words <= burst_word_count(xfer_remaining - 16'd1);
                state       <= S_LOAD_REQ;
              end
            end else begin
              burst_cnt <= burst_cnt + 1'b1;
            end
          end
        end

        // --- STORE path ---
        S_STORE_REQ: begin
          if (dma_req_ready) begin
            burst_cnt <= '0;
            state     <= S_STORE_READ;
          end
        end

        S_STORE_READ: begin
          state <= S_STORE_DATA;
        end

        S_STORE_DATA: begin
          if (dma_wdata_valid && dma_wdata_ready) begin
            saved_addr     <= saved_addr + BYTES_PER_WORD;
            sp_ptr         <= sp_ptr + 1'b1;
            xfer_remaining <= xfer_remaining - 1'b1;

            if (burst_cnt == (burst_words - 1'b1)) begin
              burst_cnt <= '0;
              if (xfer_remaining == 16'd1)
                burst_words <= '0;
              else
                burst_words <= burst_word_count(xfer_remaining - 16'd1);
              state <= S_STORE_RESP;
            end else begin
              burst_cnt <= burst_cnt + 1'b1;
              state     <= S_STORE_READ;
            end
          end
        end

        S_STORE_RESP: begin
          if (dma_done_valid) begin
            if (xfer_remaining == '0)
              state <= S_DONE;
            else
              state <= S_STORE_REQ;
          end
        end

        // --- COMPUTE path ---
        S_COMPUTE_START: begin
          state <= S_COMPUTE_WAIT;
        end

        S_COMPUTE_WAIT: begin
          if (sa_done)
            state <= S_DONE;
        end

        // --- BARRIER ---
        S_BARRIER_WAIT: begin
          if (barrier_done)
            state <= S_DONE;
        end

        // --- REDUCE ---
        // S_REDUCE_READ: prime scratchpad port A read for current row.
        // We use sp_a_en (read, no write) and wait one cycle for SRAM to respond.
        S_REDUCE_READ: begin
          // Latch the SP read result on the cycle after requesting it.
          // (SRAM has 1-cycle read latency; we'll capture sp_a_rdata in
          //  S_REDUCE_EMIT on the first cycle after this state.)
          state <= S_REDUCE_EMIT;
        end

        // S_REDUCE_EMIT: hold reduce_inj_valid until NI acks (reduce_inj_ready).
        S_REDUCE_EMIT: begin
          reduce_val_latch <= sp_a_rdata;  // capture scratchpad read on entry cycle
          if (reduce_inj_ready) begin
            if (reduce_row_idx == 4'(N_ROWS - 1)) begin
              state <= S_DONE;
            end else begin
              reduce_row_idx <= reduce_row_idx + 1'b1;
              state          <= S_REDUCE_READ;
            end
          end
        end

        S_DONE: begin
          state <= S_IDLE;
        end

        default: begin
          state <= S_IDLE;
        end
      endcase
    end
  end

  // =========================================================================
  // Output assignments
  // =========================================================================
  assign busy      = (state != S_IDLE);
  assign done      = (state == S_DONE);
  assign error     = 1'b0;
  assign cmd_ready = (state == S_IDLE);

  // Scratchpad port A
  assign sp_a_en    = (state == S_LOAD_DATA && dma_data_valid) ||
                      (state == S_STORE_READ) ||
                      (state == S_REDUCE_READ);
  assign sp_a_we    = (state == S_LOAD_DATA && dma_data_valid);
  assign sp_a_addr  = (state == S_REDUCE_READ || state == S_REDUCE_EMIT)
                      ? SP_ADDR_W'(reduce_sp_base + SP_ADDR_W'(reduce_row_idx))
                      : sp_ptr;
  assign sp_a_wdata = dma_data_in;

  // DMA data path
  assign dma_data_ready  = (state == S_LOAD_DATA);
  assign dma_wdata_valid = (state == S_STORE_DATA);
  assign dma_wdata_out   = sp_a_rdata;
  assign dma_wdata_last  = (state == S_STORE_DATA) && (burst_cnt == (burst_words - 1'b1));

  // DMA request
  assign dma_req_valid = (state == S_LOAD_REQ || state == S_STORE_REQ);
  assign dma_req_write = (state == S_STORE_REQ);
  assign dma_req_addr  = saved_addr;
  assign dma_req_len   = (burst_words == '0) ? '0 : (burst_words - 1'b1);

  // Systolic array control
  assign sa_start       = (state == S_COMPUTE_START);
  assign sa_load_weight = 1'b0; // Controlled internally by BSR scheduler
  assign sa_pe_en       = (state == S_COMPUTE_WAIT);
  assign sa_accum_en    = (state == S_COMPUTE_WAIT);

  // Barrier
  assign barrier_req = (state == S_BARRIER_WAIT);

  // Reduce inject
  assign reduce_inj_valid  = (state == S_REDUCE_EMIT);
  assign reduce_inj_id     = reduce_id_base + {4'h0, reduce_row_idx};
  assign reduce_inj_expect = reduce_expect_r;
  assign reduce_inj_dst    = reduce_dst_r;
  assign reduce_inj_val    = (state == S_REDUCE_EMIT) ? sp_a_rdata : reduce_val_latch;

endmodule
