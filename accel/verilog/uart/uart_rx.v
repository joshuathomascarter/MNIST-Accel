// -----------------------------------------------------------------------------
// uart_rx.v — UART Receiver (Accel v1 style)
//  - Async serial -> synchronous bytes with oversampling
//  - Start-bit qualification at mid, LSB-first data, optional parity, stop check
//  - Integer divider or fractional NCO for oversample tick
//  - Optional 3-point majority vote around mid-bit
// -----------------------------------------------------------------------------
// Parameters:
//   CLK_HZ       : core clock frequency in Hz (e.g., 50_000_000)
//   BAUD         : UART baud rate (e.g., 115_200)
//   OVERSAMPLE   : oversample ratio (typ 16)
//   PARITY       : 0:none, 1:even, 2:odd
//   STOP_BITS    : number of stop bits (1 or 2)
//   USE_NCO      : 0=integer divider, 1=fractional NCO for ce_ovr
//   ACCW         : NCO accumulator width when USE_NCO=1 (typ 32)
//   MAJORITY3    : 0=single mid sample, 1=3-point majority at MID-1, MID, MID+1
//
// Ports:
//   i_clk        : clock
//   i_rst_n      : synchronous active-low reset
//   i_rx         : async serial RX line (idle high)
//   o_data[7:0]  : received byte (valid on o_valid)
//   o_valid      : 1-cycle pulse when o_data is presented
//   o_frm_err    : 1-cycle flag: stop-bit sampled low
//   o_par_err    : 1-cycle flag: parity mismatch (if PARITY!=0)
//
// Notes:
//  - Add an RX FIFO externally if the consumer can stall.
//  - For synthesis, ensure proper timing constraints; mark sync flops ASYNC_REG.
// -----------------------------------------------------------------------------

`timescale 1ns/1ps
`default_nettype none

module uart_rx #(
  parameter DATA_BITS    = 8,
  parameter CLK_HZ       = 50_000_000,
  parameter BAUD         = 115_200,
  parameter OVERSAMPLE   = 16,
  parameter PARITY       = 0,   // 0:none, 1:even, 2:odd
  parameter STOP_BITS    = 1,   // 1 or 2
  parameter USE_NCO      = 0,   // 0:int divider, 1:NCO
  parameter ACCW         = 32,  // NCO accumulator width
  parameter MAJORITY3    = 0    // 0:single mid, 1:3-point vote
)(
  input  wire        i_clk,
  input  wire        i_rst_n,
  input  wire        i_rx,
  output reg [DATA_BITS-1:0] o_data,
  output reg         o_valid,
  output reg         o_frm_err,
  output reg         o_par_err
);

  // ----------------------------
  // 0) Basic sanity
  // ----------------------------
  initial begin
    // Synthesis tools ignore $error; helpful in sim
    if (OVERSAMPLE < 4)  $error("OVERSAMPLE must be >= 4");
    if (STOP_BITS  < 1 || STOP_BITS > 2) $error("STOP_BITS must be 1 or 2");
    if (PARITY > 2) $error("PARITY must be 0,1,2");
  end

  // ----------------------------
  // 1) CDC: 2-FF synchronizer
  // ----------------------------
  (* ASYNC_REG = "TRUE" *) reg rx_meta;
  (* ASYNC_REG = "TRUE" *) reg rx_sync;

  always_ff @(posedge i_clk) begin
    rx_meta <= i_rx;
    rx_sync <= rx_meta;
  end

  // ----------------------------
  // 2) Oversample tick (ce_ovr)
  // ----------------------------
  // We want ce_ovr to pulse OVERSAMPLE times per UART bit.
  // - Integer divider path uses TICKS_PER_OSR = CLK_HZ/(BAUD*OVERSAMPLE)
  // - NCO path uses accumulator with increment ~= (BAUD*OVERSAMPLE)/CLK_HZ * 2^ACCW
  localparam int unsigned BAUDxOSR = BAUD * OVERSAMPLE;

  reg ce_ovr;

  generate
    if (!USE_NCO) begin : g_div
      localparam int unsigned TICKS_PER_OSR = (CLK_HZ / BAUDxOSR > 0) ? (CLK_HZ / BAUDxOSR) : 1;
      reg [$clog2(TICKS_PER_OSR)-1:0] cnt_osr;

      always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
          cnt_osr <= '0;
          ce_ovr  <= 1'b0;
        end else begin
          if (cnt_osr == TICKS_PER_OSR-1) begin
            cnt_osr <= '0;
            ce_ovr  <= 1'b1;
          end else begin
            cnt_osr <= cnt_osr + 1'b1;
            ce_ovr  <= 1'b0;
          end
        end
      end
    end else begin : g_nco
      // Fractional NCO
      // INC = round( (BAUD*OVERSAMPLE / CLK_HZ) * 2^ACCW )
      localparam int unsigned NCO_INC = ( ( (BAUDxOSR << ACCW) + (CLK_HZ/2) ) / CLK_HZ );
      reg [ACCW-1:0] acc;

      always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
          acc   <= '0;
          ce_ovr<= 1'b0;
        end else begin
          // add inc and detect carry
          {ce_ovr, acc} <= acc + NCO_INC;
        end
      end
    end
  endgenerate

  // ----------------------------
  // 3) Phase / indices / regs
  // ----------------------------
  localparam int unsigned MID = OVERSAMPLE/2;

  reg [$clog2(OVERSAMPLE)-1:0] phase;
  reg [$clog2(DATA_BITS)-1:0]  bit_idx;    // 0..DATA_BITS-1
  reg [DATA_BITS-1:0]          shift_reg;
  reg [0:0]                    stop_cnt;   // 0 or 1 to count stop bits

  // For majority-3 around MID
  reg samp_m1, samp_0, samp_p1;
  reg use_sample; // final sampled bit

  // ----------------------------
  // 4) FSM
  // ----------------------------
  typedef enum reg [2:0] { IDLE, START, DATA, PARITY_S, STOP, DONE } state_t;
  state_t state;

  // ----------------------------
  // 5) Parity calc helper
  // ----------------------------
  function automatic reg expected_parity_even(input reg [DATA_BITS-1:0] d);
    expected_parity_even = ^d; // even parity bit should make total ones even
  endfunction
  function automatic reg expected_parity_odd(input reg [DATA_BITS-1:0] d);
    expected_parity_odd = ~(^d);
  endfunction

  // ----------------------------
  // 6) Main FSM
  // ----------------------------
  always_ff @(posedge i_clk) begin
    if (!i_rst_n) begin
      state     <= IDLE;
      phase     <= '0;
      bit_idx   <= '0;
      shift_reg <= '0;
      stop_cnt  <= '0;
      o_data    <= '0;
      o_valid   <= 1'b0;
      o_frm_err <= 1'b0;
      o_par_err <= 1'b0;
      samp_m1   <= 1'b1;
      samp_0    <= 1'b1;
      samp_p1   <= 1'b1;
    end else begin
      // default outputs are 1-cycle pulses
      o_valid   <= 1'b0;
      o_frm_err <= 1'b0;
      o_par_err <= 1'b0;

      if (ce_ovr) begin
        // Track samples around MID if MAJORITY3 enabled
        // We update the three taps on the corresponding phases.
        if (MAJORITY3) begin
          if (phase == (MID-1)) samp_m1 <= rx_sync;
          if (phase == MID)     samp_0  <= rx_sync;
          if (phase == (MID+1)) samp_p1 <= rx_sync;
        end

        unique case (state)
          // -------------------------------------------------
          IDLE: begin
            phase   <= '0;
            bit_idx <= '0;
            // detect start edge (high->low). A simple detect:
            // If line low, move to START and verify at mid.
            if (rx_sync == 1'b0) begin
              state <= START;
            end
          end

          // -------------------------------------------------
          START: begin
            // advance phase; at MID, confirm start still low
            phase <= phase + 1'b1;

            if (phase == MID) begin
              use_sample <= (MAJORITY3) ? ((samp_m1 + samp_0 + samp_p1) >= 2) : rx_sync;
              if (use_sample == 1'b0) begin
                // valid start detected
                phase   <= '0;
                bit_idx <= '0;
                state   <= DATA;
              end else begin
                // false start -> back to IDLE
                state <= IDLE;
              end
            end
          end

          // -------------------------------------------------
          DATA: begin
            phase <= phase + 1'b1;

            if (phase == MID) begin
              use_sample <= (MAJORITY3) ? ((samp_m1 + samp_0 + samp_p1) >= 2) : rx_sync;

              // shift LSB-first: newest bit enters MSB of shift_reg shift-right
              // (store LSB-first -> we want incoming bit as MSB of concat)
              shift_reg <= {use_sample, shift_reg[DATA_BITS-1:1]};

              if (bit_idx == (DATA_BITS-1)) begin
                phase <= '0;
                if (PARITY != 0) begin
                  state <= PARITY_S;
                end else begin
                  stop_cnt <= '0;
                  state    <= STOP;
                end
              end else begin
                bit_idx <= bit_idx + 1'b1;
              end
            end
          end

          // -------------------------------------------------
          PARITY_S: begin
            phase <= phase + 1'b1;

            if (phase == MID) begin
              use_sample <= (MAJORITY3) ? ((samp_m1 + samp_0 + samp_p1) >= 2) : rx_sync;

              if (PARITY == 1) begin
                // even parity expected
                o_par_err <= (use_sample != expected_parity_even(shift_reg));
              end else if (PARITY == 2) begin
                // odd parity expected
                o_par_err <= (use_sample != expected_parity_odd(shift_reg));
              end
              phase    <= '0;
              stop_cnt <= '0;
              state    <= STOP;
            end
          end

          // -------------------------------------------------
          STOP: begin
            phase <= phase + 1'b1;

            if (phase == MID) begin
              use_sample <= (MAJORITY3) ? ((samp_m1 + samp_0 + samp_p1) >= 2) : rx_sync;

              if (use_sample == 1'b0) begin
                // stop must be high
                o_frm_err <= 1'b1;
              end

              if (stop_cnt == (STOP_BITS-1)) begin
                // present byte now
                o_data  <= shift_reg;
                o_valid <= 1'b1;
                state   <= DONE;
              end else begin
                stop_cnt <= stop_cnt + 1'b1;
                phase    <= '0; // sample next stop at its mid
              end
            end
          end

          // -------------------------------------------------
          DONE: begin
            // Ready for next frame immediately
            state <= IDLE;
          end

          default: state <= IDLE;
        endcase
      end // if ce_ovr
    end
  end

  // ---------------------------------------------------------------------------
  // Optional simple assertions (ignored by synthesis; helpful in formal/sim)
  // ---------------------------------------------------------------------------
`ifndef SYNTHESIS
  // o_valid only in DONE cycle
  // (soft check: on o_valid, state was DONE in previous ce_ovr sample)
`endif

endmodule

`default_nettype wire
