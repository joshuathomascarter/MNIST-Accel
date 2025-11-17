// -----------------------------------------------------------------------------
// uart_tx.v — UART Transmitter (Accel v1 style)
//  - Parallel byte -> async serial (idle=1), start(0), 8 data (LSB first),
//    optional parity, STOP_BITS (1/2)
//  - Integer divider or fractional NCO for oversample tick (same as RX)
//  - Clean ready/valid handshake: i_valid when i_ready=1 latches a byte
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module uart_tx #(
  parameter int unsigned DATA_BITS    = 8,
  parameter int unsigned CLK_HZ       = 50_000_000,
  parameter int unsigned BAUD         = 115_200,
  parameter int unsigned OVERSAMPLE   = 16,
  parameter int unsigned PARITY       = 0,   // 0:none, 1:even, 2:odd
  parameter int unsigned STOP_BITS    = 1,   // 1 or 2
  parameter bit           USE_NCO     = 0,   // 0:int divider, 1:NCO
  parameter int unsigned  ACCW        = 32
)(
  input  wire        i_clk,
  input  wire        i_rst_n,

  // Byte input handshake
  input  wire [DATA_BITS-1:0]  i_data,
  input  wire        i_valid,      // assert with i_data to request send
  output reg       i_ready,      // high when TX can accept a new byte

  // Serial line out (idle=1)
  output reg       o_tx
);

  // ----------------------------
  // Oversample clock-enable
  // ----------------------------
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
      localparam int unsigned NCO_INC = (((BAUDxOSR << ACCW) + (CLK_HZ/2)) / CLK_HZ);
      reg [ACCW-1:0] acc;
      always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
          acc   <= '0;
          ce_ovr<= 1'b0;
        end else begin
          {ce_ovr, acc} <= acc + NCO_INC;
        end
      end
    end
  endgenerate

  // ----------------------------
  // Bit timing/phase
  // ----------------------------
  localparam int unsigned MID = OVERSAMPLE/2;
  reg [$clog2(OVERSAMPLE)-1:0] phase;

  // ----------------------------
  // Shift register + control
  // ----------------------------
  typedef enum reg [2:0] { IDLE, START, DATA, PARITY_S, STOP } state_t;
  state_t state;

  reg [DATA_BITS-1:0]  shifter;
  reg [$clog2(DATA_BITS)-1:0]  bit_idx;     // 0..DATA_BITS-1
  reg [0:0]  stop_cnt;    // 0 or 1
  reg        parity_bit;  // computed if PARITY!=0
  reg        load_byte;   // handshake fire

  // Expected parity helpers
  function automatic reg exp_even(input reg [DATA_BITS-1:0] d); exp_even = ^d; endfunction
  function automatic reg exp_odd (input reg [DATA_BITS-1:0] d); exp_odd  = ~(^d); endfunction

  // Ready/valid handshake
  assign load_byte = i_valid & i_ready;

  always_ff @(posedge i_clk) begin
    if (!i_rst_n) begin
      state    <= IDLE;
      o_tx     <= 1'b1;  // idle high
      phase    <= '0;
      bit_idx  <= '0;
      stop_cnt <= '0;
      shifter  <= '0;
      i_ready  <= 1'b1;
      parity_bit <= 1'b0;
    end else begin
      if (state == IDLE) begin
        i_ready <= 1'b1;
      end else begin
        i_ready <= 1'b0;
      end

      if (ce_ovr) begin
        unique case (state)
          IDLE: begin
            // Wait for a byte to send
            if (load_byte) begin
              shifter <= i_data;              // capture
              bit_idx <= '0;
              phase   <= '0;
              // precompute parity if used
              if (PARITY == 1)      parity_bit <= exp_even(i_data);
              else if (PARITY == 2) parity_bit <= exp_odd(i_data);
              // drive start bit low
              o_tx  <= 1'b0;
              state <= START;
            end else begin
              o_tx  <= 1'b1; // keep idle high
            end
          end

          START: begin
            // hold start for one bit; advance phase
            phase <= phase + 1'b1;
            if (phase == OVERSAMPLE-1) begin
              phase <= '0;
              state <= DATA;
              // first data bit (LSB) will be driven next cycles
              o_tx  <= shifter[0];
            end
          end

          DATA: begin
            phase <= phase + 1'b1;
            if (phase == MID-1) begin
              // drive current bit stable around mid-bit
              o_tx <= shifter[0];
            end
            if (phase == OVERSAMPLE-1) begin
              phase   <= '0;
              shifter <= {1'b0, shifter[DATA_BITS-1:1]}; // shift right, next bit to LSB
              if (bit_idx == (DATA_BITS-1)) begin
                if (PARITY != 0) begin
                  state <= PARITY_S;
                  o_tx  <= parity_bit;
                end else begin
                  state    <= STOP;
                  stop_cnt <= '0;
                  o_tx     <= 1'b1; // stop is high
                end
              end else begin
                bit_idx <= bit_idx + 1'b1;
                o_tx    <= shifter[1]; // next bit preview
              end
            end
          end

          PARITY_S: begin
            phase <= phase + 1'b1;
            if (phase == OVERSAMPLE-1) begin
              phase    <= '0;
              state    <= STOP;
              stop_cnt <= '0;
              o_tx     <= 1'b1; // stop high
            end
          end

          STOP: begin
            phase <= phase + 1'b1;
            if (phase == OVERSAMPLE-1) begin
              phase <= '0;
              if (stop_cnt == (STOP_BITS-1)) begin
                state <= IDLE;   // done
                o_tx  <= 1'b1;   // back to idle
              end else begin
                stop_cnt <= stop_cnt + 1'b1;
                o_tx     <= 1'b1; // keep high for next stop
              end
            end
          end

          default: state <= IDLE;
        endcase
      end // ce_ovr
    end
  end

endmodule
`default_nettype wire
