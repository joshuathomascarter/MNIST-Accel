/*
  pe.v - Row-Stationary Processing Element (PE)
  ---------------------------------------------
  TRUE ROW-STATIONARY DATAFLOW:
   - Weights are loaded ONCE and stored stationary in registers
   - Activations flow horizontally through the PE (left to right)
   - Partial sums accumulate locally using mac8
   - Pk = 1 lane (single MAC per PE)
   
  Features:
   - STATIONARY weight storage (does NOT flow to neighbors)
   - Pass-through of activations with configurable 1-stage pipeline
   - Local partial-sum accumulation
   
  Parameters:
   PIPE = 1 -> enable internal pipeline for activation forwarding
   SAT  = 0 -> forwarded to mac8 for saturation behaviour
   
  Controls:
   clk, rst_n       : clock / active-low reset
   clr              : synchronous clear of local partial-sum
   en               : enable for MAC accumulation this cycle
   load_weight      : load b_in into stationary weight register
   
  IO:
   a_in             : incoming INT8 activation (flows horizontally)
   b_in             : incoming INT8 weight (loaded once via load_weight)
   a_out            : forwarded activation to neighbor PE (right)
   acc              : local 32-bit partial-sum (from mac8)
   
  NOTE: b_out removed - weights do NOT flow in row-stationary!
*/
`ifndef PE_V
`define PE_V
`default_nettype none
// -----------------------------------------------------------------------------
// Title      : pe
// File       : pe.v
// Description: Row-Stationary Processing Element (single MAC lane).
//              Verilog-2001 compliant; weights stored stationary,
//              activations flow horizontally with optional pipeline skew.
//
// Requirements Trace:
//   REQ-ACCEL-PE-01: Store weight stationary, forward activation only.
//   REQ-ACCEL-PE-02: Accumulate partial sum locally via mac8 (Pk=1).
//   REQ-ACCEL-PE-03: Support synchronous clear of partial sum (clr).
//   REQ-ACCEL-PE-04: Provide deterministic hold when en=0.
//   REQ-ACCEL-PE-05: Support weight preloading before computation phase.
// -----------------------------------------------------------------------------
// Parameters:
//   PIPE (0/1): 1 inserts a pipeline register stage for activation forwarding.
//   SAT  (0/1): passed to mac8 for saturation behavior.
// -----------------------------------------------------------------------------
module pe #(parameter PIPE = 1, parameter SAT = 0)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [7:0] a_in,
    input  wire signed [7:0] b_in,
    input  wire              en,
    input  wire              clr,
    input  wire              load_weight,  // NEW: weight load control
    output wire signed [7:0] a_out,
    // b_out REMOVED - weights are stationary!
    output wire signed [31:0] acc
);

    // Stationary weight register (THE KEY CHANGE!)
    reg signed [7:0] weight_reg;
    
    // Internal registers for activation pipeline
    reg signed [7:0] a_reg;
    reg signed [7:0] a_del;
    wire sat_internal;

    // Weight loading - capture and hold
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'sd0;
        end else begin
            if (load_weight) begin
                weight_reg <= b_in;  // Load weight and hold stationary
            end
            // Weight stays constant until next load_weight pulse
        end
    end

    // Activation pipeline (Verilog-2001 style)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 8'sd0;
            a_del <= 8'sd0;
        end else begin
            if (clr) begin
                a_reg <= 8'sd0; // zero activation on clear
            end else begin
                a_reg <= a_in;  // Activation flows through
            end
            a_del <= a_reg; // forward chain for skew
        end
    end

    // Select signals depending on PIPE parameter
    wire signed [7:0] mac_a = (PIPE) ? a_reg : a_in;
    wire signed [7:0] mac_b = weight_reg;  // ALWAYS use stationary weight!
    assign a_out = (PIPE) ? a_del : a_in;
    // b_out removed - no weight forwarding in row-stationary

    mac8 #( .SAT(SAT) ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .a(mac_a), .b(mac_b),  // b uses stored weight
        .clr(clr), .en(en),
        .acc(acc), .sat_flag(sat_internal)
    );

endmodule
`default_nettype wire
`endif

