`ifndef SYSTOLIC_ARRAY_V
`define SYSTOLIC_ARRAY_V
`default_nettype none
// -----------------------------------------------------------------------------
// Title      : systolic_array
// File       : systolic_array.v
// Description: TRUE ROW-STATIONARY INT8 systolic array with Pk=1.
//              Verilog-2001 compliant; uses generate loops and flattened ports.
//
//              ROW-STATIONARY DATAFLOW:
//              - Weights loaded ONCE and stored stationary in each PE
//              - Activations stream horizontally (west to east)
//              - No vertical weight flow (weights broadcast to all PEs in same column)
//              - Maximum weight reuse efficiency
//
// Requirements Trace:
//   REQ-ACCEL-SYST-01: Provide N_ROWS x N_COLS grid of PEs accumulating INT32 psums.
//   REQ-ACCEL-SYST-02: Stream activations (A) from west edge, broadcast weights (B) to columns.
//   REQ-ACCEL-SYST-03: Support synchronous clear of all partial sums via clr.
//   REQ-ACCEL-SYST-04: Deterministic hold when en=0.
//   REQ-ACCEL-SYST-05: Support weight preloading via load_weight control.
//   REQ-ACCEL-SYST-06: Optional single-cycle pipeline inside each PE (PIPE param).
// -----------------------------------------------------------------------------
// Parameters:
//   N_ROWS : number of rows (default 2)
//   N_COLS : number of columns (default 2)
//   PIPE   : pass to each PE (1 inserts internal pipeline)
//   SAT    : pass to each mac8 instance inside PE
// -----------------------------------------------------------------------------
// Port Flattening:
//   a_in_flat  [N_ROWS*8-1:0] contains N_ROWS signed 8-bit lanes (row-major)
//   b_in_flat  [N_COLS*8-1:0] contains N_COLS signed 8-bit lanes (col-major)
//   c_out_flat [N_ROWS*N_COLS*32-1:0] concatenated row-major
//   load_weight: when high, PEs capture b_in and store stationary
// -----------------------------------------------------------------------------
module systolic_array #(
  parameter N_ROWS = 2,
  parameter N_COLS = 2,
  parameter PIPE   = 1,
  parameter SAT    = 0
)(
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  input  wire clr,
  input  wire load_weight,  // NEW: weight loading control
  input  wire [N_ROWS*8-1:0] a_in_flat,
  input  wire [N_COLS*8-1:0] b_in_flat,
  output wire [N_ROWS*N_COLS*32-1:0] c_out_flat
);

  // Unpack input activation and weight vectors
  wire signed [7:0] a_in [0:N_ROWS-1];
  wire signed [7:0] b_in [0:N_COLS-1];
  genvar ui;
  generate
    for (ui = 0; ui < N_ROWS; ui = ui + 1) begin : UNPACK_A
      assign a_in[ui] = a_in_flat[ui*8 +: 8];
    end
    for (ui = 0; ui < N_COLS; ui = ui + 1) begin : UNPACK_B
      assign b_in[ui] = b_in_flat[ui*8 +: 8];
    end
  endgenerate

  // Forwarding nets between PEs (ONLY for activations, NOT weights!)
  wire signed [7:0] a_fwd [0:N_ROWS-1][0:N_COLS-1];
  wire signed [31:0] acc_mat [0:N_ROWS-1][0:N_COLS-1];

  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL
        // Activation source: from west edge or left neighbor
        wire signed [7:0] a_src = (c == 0) ? a_in[r] : a_fwd[r][c-1];
        
        // Weight source: BROADCAST from top edge to ALL rows in this column
        // NO vertical weight flow - same weight to all PEs in a column!
        wire signed [7:0] b_src = b_in[c];  // Direct from input, no forwarding
        
        pe #(.PIPE(PIPE), .SAT(SAT)) u_pe (
          .clk(clk), .rst_n(rst_n),
          .a_in(a_src), 
          .b_in(b_src),  // Weight broadcast (not forwarded from neighbor)
          .en(en), 
          .clr(clr),
          .load_weight(load_weight),  // NEW: weight loading control
          .a_out(a_fwd[r][c]),
          // b_out removed - no weight forwarding!
          .acc(acc_mat[r][c])
        );
      end
    end
  endgenerate

  // Pack outputs row-major into flat bus
  integer pi, pj;
  reg [N_ROWS*N_COLS*32-1:0] c_pack;
  always @(*) begin
    c_pack = { (N_ROWS*N_COLS*32){1'b0} };
    for (pi = 0; pi < N_ROWS; pi = pi + 1) begin
      for (pj = 0; pj < N_COLS; pj = pj + 1) begin
        c_pack[(pi*N_COLS+pj)*32 +: 32] = acc_mat[pi][pj];
      end
    end
  end
  assign c_out_flat = c_pack;

endmodule
`default_nettype wire
`endif
