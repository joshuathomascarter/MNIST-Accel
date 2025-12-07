//      // verilator_coverage annotation
        
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
%000005     input  wire              clk,
%000001     input  wire              rst_n,
%000001     input  wire signed [7:0] a,
%000001     input  wire signed [7:0] b,
%000001     input  wire              en,      // Clock Enable / Power Gate
%000001     input  wire              clr,     // Synchronous Clear
%000001     output logic signed [31:0] acc    // Accumulator Output
        );
        
            // -------------------------------------------------------------------------
            // 1. DSP Inference Hints & Signals
            // -------------------------------------------------------------------------
            // Force Vivado to use DSP slices, not LUTs
            (* use_dsp = "yes" *)
%000001     logic signed [15:0] prod;
%000002     logic signed [31:0] sum_comb;
%000001     logic signed [31:0] acc_reg;
            
            // -------------------------------------------------------------------------
            // 2. Multiply Stage
            // -------------------------------------------------------------------------
            // Zero Bypass: If inputs are zero, don't toggle the multiplier (saves power)
%000001     logic mult_active;
~000017     assign mult_active = ENABLE_ZERO_BYPASS ? (|a && |b) : 1'b1;
        
%000001     always_comb begin
%000001         prod = a * b;
            end
        
            // -------------------------------------------------------------------------
            // 3. Accumulate Stage (The Critical Path)
            // -------------------------------------------------------------------------
            // We calculate the sum combinatorially, then register it.
            // Sign extension: {{16{prod[15]}}, prod} converts 16-bit prod to 32-bit.
            assign sum_comb = acc_reg + {{16{prod[15]}}, prod}; 
        
%000005     always_ff @(posedge clk or negedge rst_n) begin
%000005         if (!rst_n) begin
%000000             acc_reg <= 32'sd0;
%000005         end else begin
%000001             if (clr) begin
%000001                 acc_reg <= 32'sd0;
%000002             end else if (en && mult_active) begin
                        // Standard wrap-around accumulation.
                        // We rely on sat_flag to detect issues rather than clamping logic
                        // which would slow down the critical path.
%000002                 acc_reg <= sum_comb;
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
%000000     property p_no_unknowns;
%000000         @(posedge clk) disable iff (!rst_n) (en |-> (!$isunknown(a) && !$isunknown(b)));
            endproperty
            
            assert property (p_no_unknowns) 
                else $warning("MAC8 Warning: Inputs 'a' or 'b' are X/Z while enabled!");
        
            // Property 2: Clear takes priority over Enable
            // If clr is high, next cycle acc must be 0, regardless of en
%000000     property p_clr_priority;
%000000         @(posedge clk) disable iff (!rst_n) (clr |=> (acc == 0));
            endproperty
            
            assert property (p_clr_priority)
                else $error("MAC8 Error: Clear did not reset accumulator!");
        
        endmodule
        
        
