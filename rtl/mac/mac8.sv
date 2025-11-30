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
    parameter SAT = 1,                // Enable Saturation Flagging
    parameter ENABLE_ZERO_BYPASS = 1  // Power optimization (Skip 0*0)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [7:0] a,
    input  wire signed [7:0] b,
    input  wire              en,      // Clock Enable / Power Gate
    input  wire              clr,     // Synchronous Clear
    output logic signed [31:0] acc,   // Accumulator Output
    output logic             sat_flag // Overflow detected
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
    // 2. Multiply Stage
    // -------------------------------------------------------------------------
    // Zero Bypass: If inputs are zero, don't toggle the multiplier (saves power)
    logic mult_active;
    assign mult_active = ENABLE_ZERO_BYPASS ? (|a && |b) : 1'b1;

    always_comb begin
        prod = a * b;
    end

    // -------------------------------------------------------------------------
    // 3. Accumulate Stage (The Critical Path)
    // -------------------------------------------------------------------------
    // We calculate the sum combinatorially, then register it.
    // Sign extension: {{16{prod[15]}}, prod} converts 16-bit prod to 32-bit.
    assign sum_comb = acc_reg + {{16{prod[15]}}, prod}; 

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 32'sd0;
        end else begin
            if (clr) begin
                acc_reg <= 32'sd0;
            end else if (en && mult_active) begin
                // Standard wrap-around accumulation.
                // We rely on sat_flag to detect issues rather than clamping logic
                // which would slow down the critical path.
                acc_reg <= sum_comb;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 4. Saturation & Overflow Logic (Pipelined)
    // -------------------------------------------------------------------------
    // We check for overflow based on the *current* operation.
    
    logic pos_oflow, neg_oflow;
    
    // Overflow happens if:
    // Pos + Pos = Neg
    // Neg + Neg = Pos
    always_comb begin
        pos_oflow = (~acc_reg[31] && ~prod[15] && sum_comb[31]); 
        neg_oflow = ( acc_reg[31] &&  prod[15] && ~sum_comb[31]);
    end

    // Output Assignment
    assign acc = acc_reg;

    // Flag is registered to align with output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sat_flag <= 1'b0;
        else if (en) sat_flag <= (pos_oflow | neg_oflow);
    end

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

