// systolic_array.sv — Weight-Stationary INT8 systolic array (dense path)
// N_ROWS × N_COLS PE grid with triangular activation skew, per-row clock gating,
// and systolic load_weight propagation. Used by accel_top for dense GEMM tiles.

`ifndef SYSTOLIC_ARRAY_V
`define SYSTOLIC_ARRAY_V
`default_nettype none

module systolic_array #(
  parameter N_ROWS = 14,
  parameter N_COLS = 14,
  parameter PIPE   = 1,
  parameter ENABLE_CLOCK_GATING = 1
)(
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  input  wire clr,
  input  wire load_weight,
  input  wire [N_ROWS-1:0] row_en,
  input  wire [N_ROWS*8-1:0] a_in_flat,
  input  wire [N_COLS*8-1:0] b_in_flat,
  output wire [N_ROWS*N_COLS*32-1:0] c_out_flat
);

  // ---------- 1. Input Unpacking + Triangular Activation Skew ----------
  wire signed [7:0] a_in_raw [0:N_ROWS-1];
  wire signed [7:0] a_in     [0:N_ROWS-1];
  wire signed [7:0] b_in     [0:N_COLS-1];

  genvar ui;
  generate
    for (ui = 0; ui < N_ROWS; ui = ui + 1) begin : UNPACK_A_RAW
      assign a_in_raw[ui] = a_in_flat[ui*8 +: 8];
    end
    for (ui = 0; ui < N_COLS; ui = ui + 1) begin : UNPACK_B
      assign b_in[ui] = b_in_flat[ui*8 +: 8];
    end

    // Triangular skew: row i gets i delay stages
    for (ui = 0; ui < N_ROWS; ui = ui + 1)
      if (ui == 0) begin : gen_no_delay
        assign a_in[ui] = a_in_raw[ui];
      end else begin : gen_delay
        reg signed [7:0] delay_regs [0:ui-1];
        integer k;
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (k=0; k<ui; k=k+1) delay_regs[k] <= 8'sd0;
          end else if (en) begin
            delay_regs[0] <= a_in_raw[ui];
            for (k=1; k<ui; k=k+1)
               delay_regs[k] <= delay_regs[k-1];
          end
        end
        assign a_in[ui] = delay_regs[ui-1];
      end
  endgenerate

  // ---------- 2. Per-Row Clock Gating ----------
  wire [N_ROWS-1:0] row_clk;

  generate
    if (ENABLE_CLOCK_GATING) begin : gen_clock_gating
      for (ui = 0; ui < N_ROWS; ui = ui + 1) begin : gen_row_gates
        wire row_clk_en = row_en[ui] & (en | load_weight | clr);
        `ifdef XILINX_FPGA
          BUFGCE row_clk_gate (.I(clk), .CE(row_clk_en), .O(row_clk[ui]));
        `else
          reg row_clk_en_latched;
          always @(clk or row_clk_en)
            if (!clk) row_clk_en_latched <= row_clk_en;
          assign row_clk[ui] = clk & row_clk_en_latched;
        `endif
      end
    end else begin : gen_no_clock_gating
      for (ui = 0; ui < N_ROWS; ui = ui + 1) begin : gen_row_direct_clk
        assign row_clk[ui] = clk;
      end
    end
  endgenerate

  // ---------- 3. PE Grid ----------
  wire signed [7:0]  a_fwd [0:N_ROWS-1][0:N_COLS-1];
  wire signed [31:0] acc_mat [0:N_ROWS-1][0:N_COLS-1];
  wire load_weight_fwd [0:N_ROWS-1][0:N_COLS-1];

  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL
        wire signed [7:0] a_src = (c == 0) ? a_in[r] : a_fwd[r][c-1];
        wire signed [7:0] b_src = b_in[c];
        wire load_weight_src = (c == 0) ? load_weight : load_weight_fwd[r][c-1];

        pe #(.PIPE(PIPE)) u_pe (
          .clk(row_clk[r]),
          .rst_n(rst_n),
          .a_in(a_src),
          .b_in(b_src),
          .en(en),
          .clr(clr),
          .load_weight(load_weight_src),
          .a_out(a_fwd[r][c]),
          .load_weight_out(load_weight_fwd[r][c]),
          .acc(acc_mat[r][c])
        );
      end
    end
  endgenerate

  // ---------- 4. Output Packing ----------
  // Pack 2D accumulator matrix into flat output vector (row-major order).
  // Equivalent to: assign c_out_flat = { >> {acc_mat} };
  // (streaming operator not supported by all synthesis tools)
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : PACK_ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : PACK_COL
        assign c_out_flat[(r * N_COLS + c) * 32 +: 32] = acc_mat[r][c];
      end
    end
  endgenerate

  // ---------- 5. Assertions ----------

endmodule
`endif
`default_nettype wire
