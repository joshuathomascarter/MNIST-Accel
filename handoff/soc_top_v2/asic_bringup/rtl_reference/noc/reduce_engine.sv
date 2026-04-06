// =============================================================================
// reduce_engine.sv — Tree Reduction Engine
// =============================================================================
// Collects partial sums from multiple tiles and reduces them via addition.
// Used for accumulating outputs when a matrix multiply is partitioned across
// tiles along the reduction (K) dimension.
//
// Two modes:
//   1. LOCAL: Accumulate incoming data into local scratchpad
//   2. FORWARD: Add local partial + incoming, forward result upstream

/* verilator lint_off IMPORTSTAR */
import noc_pkg::*;

module reduce_engine #(
  parameter int NODE_ID   = 0,
  parameter int SP_ADDR_W = 12,
  parameter int SP_DATA_W = 64,
  parameter int ACC_W     = 32    // Accumulator width (matches PE accumulator)
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Configuration ---
  input  logic                   enable,
  input  logic [NODE_BITS-1:0]   reduce_root,      // Node that collects final result
  input  logic [SP_ADDR_W-1:0]   accum_base_addr,  // Where to accumulate in scratchpad
  input  logic [15:0]            reduce_length,     // Number of words to reduce

  // --- Incoming reduce flits from NoC ---
  input  flit_t                  rx_flit,
  input  logic                   rx_valid,
  output logic                   rx_ready,

  // --- Scratchpad R/W for accumulation ---
  output logic                   sp_en,
  output logic                   sp_we,
  output logic [SP_ADDR_W-1:0]  sp_addr,
  output logic [SP_DATA_W-1:0]  sp_wdata,
  input  logic [SP_DATA_W-1:0]  sp_rdata,

  // --- NoC output (for forwarding reduced result) ---
  output flit_t                  tx_flit,
  output logic                   tx_valid,
  input  logic [noc_pkg::NUM_VCS-1:0] tx_credit,

  // --- Status ---
  output logic                   reduce_done,
  output logic                   busy
);

  typedef enum logic [2:0] {
    RD_IDLE,
    RD_RECV,
    RD_READ_SP,
    RD_ACCUMULATE,
    RD_WRITE_SP,
    RD_FORWARD,
    RD_DONE
  } rd_state_e;

  rd_state_e state;
  logic [SP_ADDR_W-1:0] acc_ptr;
  logic [15:0]          recv_cnt;
  logic [SP_DATA_W-1:0] incoming;
  logic [SP_DATA_W-1:0] accumulated;

  // Accumulate: treat as packed INT32 values within 64-bit word
  // Two INT32 accumulators per 64-bit word
  logic [ACC_W-1:0] in_lo, in_hi, sp_lo, sp_hi, sum_lo, sum_hi;

  assign in_lo = incoming[ACC_W-1:0];
  assign in_hi = incoming[SP_DATA_W-1:ACC_W];
  assign sp_lo = sp_rdata[ACC_W-1:0];
  assign sp_hi = sp_rdata[SP_DATA_W-1:ACC_W];
  assign sum_lo = sp_lo + in_lo;
  assign sum_hi = sp_hi + in_hi;
  assign accumulated = {sum_hi, sum_lo};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= RD_IDLE;
      acc_ptr  <= '0;
      recv_cnt <= '0;
      incoming <= '0;
    end else begin
      case (state)
        RD_IDLE: begin
          if (enable && rx_valid &&
              (rx_flit.msg_type == MSG_REDUCE) &&
              (rx_flit.flit_type == FLIT_HEAD ||
               rx_flit.flit_type == FLIT_HEADTAIL)) begin
            acc_ptr  <= accum_base_addr;
            recv_cnt <= '0;
            state    <= RD_RECV;
          end
        end

        RD_RECV: begin
          if (rx_valid) begin
            incoming <= SP_DATA_W'(rx_flit.payload);
            state    <= RD_READ_SP;
          end
        end

        RD_READ_SP: begin
          // Wait one cycle for scratchpad read
          state <= RD_ACCUMULATE;
        end

        RD_ACCUMULATE: begin
          state <= RD_WRITE_SP;
        end

        RD_WRITE_SP: begin
          acc_ptr  <= acc_ptr + 1;
          recv_cnt <= recv_cnt + 1;

          if (rx_flit.flit_type == FLIT_TAIL ||
              rx_flit.flit_type == FLIT_HEADTAIL ||
              recv_cnt == reduce_length - 1) begin
            // Check if we're root or need to forward
            if (NODE_BITS'(NODE_ID) == reduce_root)
              state <= RD_DONE;
            else
              state <= RD_FORWARD;
          end else begin
            state <= RD_RECV;
          end
        end

        RD_FORWARD: begin
          // Forward accumulated result towards root
          if (tx_credit[0]) begin
            state <= RD_DONE;
          end
        end

        RD_DONE: begin
          state <= RD_IDLE;
        end
      endcase
    end
  end

  // Scratchpad interface
  assign sp_en   = (state == RD_READ_SP || state == RD_WRITE_SP);
  assign sp_we   = (state == RD_WRITE_SP);
  assign sp_addr = acc_ptr;
  assign sp_wdata = accumulated;

  // RX handshake
  assign rx_ready = (state == RD_RECV);

  // TX forward
  always_comb begin
    tx_flit  = '0;
    tx_valid = 1'b0;

    if (state == RD_FORWARD && tx_credit[0]) begin
      tx_valid = 1'b1;
      tx_flit  = make_reduce_flit(
        NODE_BITS'(NODE_ID),          // src = this node
        reduce_root,                   // dst = reduction root
        2'h0,                          // VC 0
        8'h00,                         // reduce_id (carry through as 0 for leaf reduce)
        4'h1,                          // expect=1 (already accumulated, single result)
        accumulated[ACC_W-1:0],        // lower 32 bits of accumulated value
        4'h1                           // represents one completed reduced result
      );
    end
  end

  assign reduce_done = (state == RD_DONE);
  assign busy        = (state != RD_IDLE);

endmodule
