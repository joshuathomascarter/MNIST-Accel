/*
  pe.sv - Weight-Stationary Processing Element
  --------------------------------------------
  - Weights are loaded once and held (Stationary).
  - Activations flow horizontally (Streaming).
  - Partial Sums accumulate locally.
*/

`ifndef PE_V
`define PE_V
`default_nettype none

module pe #(
    parameter PIPE = 1,  // 1 = Pipeline activation (better Fmax), 0 = Combinational
    parameter SAT  = 1   // Enable Saturation in MAC
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [7:0] a_in,        // Activation In (Left)
    input  wire signed [7:0] b_in,        // Weight Load Data (Top)
    input  wire              en,          // Mac Enable
    input  wire              clr,         // Accumulator Clear
    input  wire              load_weight, // Control: Load b_in into internal register
    
    output logic signed [7:0] a_out,      // Activation Out (Right)
    output logic signed [31:0] acc        // Accumulator Result
);

    // -------------------------------------------------------------------------
    // 1. Internal State
    // -------------------------------------------------------------------------
    logic signed [7:0] weight_reg;
    logic signed [7:0] a_reg;

    // -------------------------------------------------------------------------
    // 2. Weight Stationary Logic
    // -------------------------------------------------------------------------
    // The weight is loaded only when load_weight is high, otherwise it holds.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'sd0;
        end else if (load_weight) begin
            weight_reg <= b_in;
        end
    end

    // -------------------------------------------------------------------------
    // 3. Activation Pipeline (Horizontal Forwarding)
    // -------------------------------------------------------------------------
    generate
        if (PIPE) begin : gen_pipe
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) a_out <= 8'sd0;
                else        a_out <= a_in;
            end
        end else begin : gen_comb
            // Combinational pass-through (Not recommended for large arrays)
            always_comb a_out = a_in;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 4. MAC Unit Instance
    // -------------------------------------------------------------------------
    // We use the optimized mac8 module we just created.
    wire sat_flag_unused; // We can expose this if needed later
    
    mac8 #(
        .SAT(SAT),
        .ENABLE_ZERO_BYPASS(1)
    ) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .a(a_in),        // Use current activation
        .b(weight_reg),  // Use STATIONARY weight
        .en(en),
        .clr(clr),
        .acc(acc),
        .sat_flag(sat_flag_unused)
    );

    // -------------------------------------------------------------------------
    // 5. Assertions (Design by Contract)
    // -------------------------------------------------------------------------
    // These run only in simulation to catch logic bugs.
    
    // Property 1: Never load weight and enable MAC at the same time
    // (This would cause a race condition or undefined math behavior)
    property p_no_load_and_compute;
        @(posedge clk) disable iff (!rst_n) (load_weight |-> !en);
    endproperty
    
    assert property (p_no_load_and_compute) 
        else $error("PE Error: Attempted to Load Weight and Compute simultaneously!");

    // Property 2: If clear is high, accumulator should be 0 next cycle
    // (Note: This depends on mac8 internal latency, so we check after 1 cycle)
    property p_clear_works;
        @(posedge clk) disable iff (!rst_n) (clr |=> (acc == 0));
    endproperty
    
    // assert property (p_clear_works); // Uncomment after verifying mac8 latency

endmodule
`default_nettype wire
`endif

