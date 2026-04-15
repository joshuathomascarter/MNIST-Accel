// pe.sv — Weight-stationary Processing Element for 16×16 INT8 systolic array
// Holds one weight stationary, streams activations West→East, accumulates INT32.
// Resources: 1 DSP48E1 + 48 FFs (weight_reg + act pipeline + acc) + ~20 LUTs

`ifndef PE_V
`define PE_V
`default_nettype none

module pe #(
    parameter PIPE = 1   // 1 = register activation pipeline (required for timing)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              clk_en,
    input  wire signed [7:0] a_in,          // Activation from left PE or buffer
    input  wire signed [7:0] b_in,          // Weight data (sampled on load_weight)
    input  wire              en,            // MAC enable (mutually exclusive with load_weight)
    input  wire              clr,           // Sync clear accumulator
    input  wire              load_weight,   // Capture b_in into weight register
    output logic signed [7:0] a_out,        // Activation forwarded to right PE
    output logic              load_weight_out, // Systolic load_weight propagation
    output logic signed [31:0] acc          // 32-bit accumulated result
);

    logic signed [7:0] weight_reg;

    // Weight register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                   weight_reg <= 8'sd0;
        else if (clk_en && load_weight) weight_reg <= b_in;
    end

    // Systolic load_weight propagation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        load_weight_out <= 1'b0;
        else if (clk_en)   load_weight_out <= load_weight;
    end

    // Activation pipeline (horizontal forwarding) — PIPE=1 always (PIPE=0 unused)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      a_out <= 8'sd0;
        else if (clk_en) a_out <= a_in;
    end

    // MAC unit
    mac8 u_mac (
        .clk    (clk),
        .rst_n  (rst_n),
        .clk_en (clk_en),
        .a      (a_in),
        .b      (weight_reg),
        .en     (en),
        .clr    (clr),
        .acc    (acc)
    );

    // Simulation-only assertions (skipped during synthesis)
`ifndef SYNTHESIS
    /* verilator lint_off SYNCASYNCNET */
    property p_no_load_and_compute;
        @(posedge clk) disable iff (!rst_n) (load_weight |-> !en);
    endproperty
    // R8 fix: bsr_scheduler S_WAIT_WGT guarantees BLOCK_SIZE-cycle gap
    assert property (p_no_load_and_compute)
        else $error("PE: load_weight and en asserted simultaneously");

    property p_clear_works;
        @(posedge clk) disable iff (!rst_n) (clr |=> (acc == 0));
    endproperty
    // cover property (p_clear_works);
    /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
`default_nettype wire
`endif

