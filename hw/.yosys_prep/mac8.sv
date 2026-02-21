// mac8.sv — Signed INT8 MAC unit for MNIST CNN inference
// Maps to one DSP48E1 slice on Zynq-7020 (PYNQ-Z2)
// Operation: acc = acc + (a × b)   |   1-cycle latency, 1 MAC/cycle

`default_nettype none

module mac8 (
    input  wire              clk,       // System clock (100 MHz target)
    input  wire              rst_n,     // Active-low async reset
    input  wire signed [7:0] a,         // Activation (signed INT8)
    input  wire signed [7:0] b,         // Weight (signed INT8, weight-stationary)
    input  wire              en,        // MAC enable (hold when 0)
    input  wire              clr,       // Sync clear (priority over en)
    output logic signed [31:0] acc      // 32-bit accumulator output
);

    // Force DSP48E1 inference instead of LUT-based multiply
    (* use_dsp = "yes" *)
    logic signed [15:0] prod;
    logic signed [31:0] sum_comb;
    logic signed [31:0] acc_reg;

    // Multiply: combinational, maps to DSP48E1 multiplier
    always_comb begin
        prod = a * b;
    end

    // Accumulate: sign-extend 16-bit product to 32 bits, add to running sum
    always_comb begin
        sum_comb = acc_reg + {{16{prod[15]}}, prod};
    end

    // Registered accumulator
    // Priority: rst_n (async) > clr (sync) > en > hold
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= 32'sd0;
        else if (clr)
            acc_reg <= 32'sd0;
        else if (en)
            acc_reg <= sum_comb;
    end

    assign acc = acc_reg;

    // Simulation-only assertions (skipped during synthesis)

endmodule

