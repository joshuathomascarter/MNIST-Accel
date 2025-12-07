/*
  mac8.sv - Signed 8x8 -> 32 multiply-accumulate with DSP optimizations
  --------------------------------------------------------------------
  Optimized for Xilinx DSP48E1 Slices.
  
  Pipeline Stages:
  1. Multiply (A * B)
  2. Accumulate (Acc + Prod) -> Registered
  3. Saturate (Flag Only) -> Output
  
  Latency: 1 cycle for Accumulation.
*/

`default_nettype none

module mac8 #(              
    parameter ENABLE_ZERO_BYPASS = 1  // Power optimization (Skip 0*0)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [7:0] a,
    input  wire signed [7:0] b,
    input  wire              bypass,  // ← NEW: 0=MAC, 1=Adder
    input  wire              en,      // Clock Enable / Power Gate
    input  wire              clr,     // Synchronous Clear
    output logic signed [31:0] acc    // Accumulator Output
);

    // -------------------------------------------------------------------------
    // 1. DSP Inference Hints & Signals
    // -------------------------------------------------------------------------
    // Force Vivado to use DSP slices, not LUTs
    (* use_dsp = "yes" *)
    logic signed [15:0] prod;
    logic signed [31:0] sum_comb;
    logic signed [31:0] acc_reg;
    
    // -------------------------------------------------------------------------
    // 2. Multiply Stage (with True Operand Isolation)
    // -------------------------------------------------------------------------
    // Zero Bypass: Force inputs to 0 if either input is 0 (saves multiplier power)
    logic mult_active;
    logic signed [7:0] a_gated, b_gated;
    
    // UPDATE: Force multiplier off if we are in bypass mode (saving power)
    assign mult_active = bypass ? 1'b0 : (ENABLE_ZERO_BYPASS ? (|a && |b) : 1'b1);
    /*
    Added the bypass in this conditional statement as if we have a bypass a * b may 
    have run in the backgroud
    */

    assign a_gated = mult_active ? a : 8'sd0;
    assign b_gated = mult_active ? b : 8'sd0;

    always_comb begin
        prod = a_gated * b_gated;  // Multiplier now sees 0*0 when inputs are zero
    end

    // -------------------------------------------------------------------------
    // 3. Accumulate Stage (The Critical Path)
    // -------------------------------------------------------------------------
    // We calculate the sum combinatorially, then register it.
    // Sign extension: {{16{prod[15]}}, prod} converts 16-bit prod to 32-bit.
    always_comb begin
        if (bypass) begin
            // Residual Mode: Skip multiply, just add activation to accumulator
            sum_comb = acc_reg + {{24{a[7]}}, a};
        end else begin
            // MAC Mode: Normal multiply-accumulate
            sum_comb = acc_reg + {{16{prod[15]}}, prod};
        end
    end     

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 32'sd0;
        end else begin
            if (clr) begin
                acc_reg <= 32'sd0;
            end else if (en) begin // removed the multiactive as with bypass b is zero so this 
                // Standard wrap-around accumulation.
                // We rely on sat_flag to detect issues rather than clamping logic
                // which would slow down the critical path.
                acc_reg <= sum_comb;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 4. Saturation & Overflow Logic (Pipelined) - REMOVED
    // -------------------------------------------------------------------------
    
    // Output Assignment
    assign acc = acc_reg;

    // -------------------------------------------------------------------------
    // 5. Assertions (Design by Contract)
    // -------------------------------------------------------------------------
    // These run only in simulation.
    
    // Property 1: Inputs should not be X when enabled
    property p_no_unknowns;
        @(posedge clk) disable iff (!rst_n) (en |-> (!$isunknown(a) && !$isunknown(b)));
    endproperty
    
    assert property (p_no_unknowns) 
        else $warning("MAC8 Warning: Inputs 'a' or 'b' are X/Z while enabled!");

    // Property 2: Clear takes priority over Enable
    // If clr is high, next cycle acc must be 0, regardless of en
    property p_clr_priority;
        @(posedge clk) disable iff (!rst_n) (clr |=> (acc == 0));
    endproperty
    
    assert property (p_clr_priority)
        else $error("MAC8 Error: Clear did not reset accumulator!");

endmodule

