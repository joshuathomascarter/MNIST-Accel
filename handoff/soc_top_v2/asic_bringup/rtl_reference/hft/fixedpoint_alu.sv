// ===========================================================================
// fixedpoint_alu.sv — 32-bit Q16.16 Fixed-Point ALU
// ===========================================================================
// Operations: ADD, SUB, MUL, ABS, CMP (compare)
//
// Q16.16 format: 16-bit integer + 16-bit fraction, signed (2's complement)
// Multiplication uses DSP48: (A × B) >> 16  (keep Q16.16 result)
//
// Combinational result with 1-cycle registered output option (REG_OUT=1)
//
// Resource estimate: 1 DSP48 (for MUL), ~60 LUTs
// ===========================================================================

module fixedpoint_alu #(
    parameter int WIDTH   = 32,   // total bit width (Q16.16)
    parameter int FRAC    = 16,   // fractional bits
    parameter bit REG_OUT = 1'b1  // register output for timing closure
)(
    input  logic               clk,
    input  logic               rst_n,

    // Operands
    input  logic [WIDTH-1:0]   a,
    input  logic [WIDTH-1:0]   b,

    // Operation select
    input  logic [2:0]         op,
    //   3'b000 = ADD    : result = a + b
    //   3'b001 = SUB    : result = a - b
    //   3'b010 = MUL    : result = (a * b) >> FRAC
    //   3'b011 = ABS    : result = |a|
    //   3'b100 = CMP    : result = {30'b0, a > b, a == b}

    input  logic               valid_in,
    output logic               valid_out,
    output logic [WIDTH-1:0]   result,
    output logic               overflow     // set on ADD/SUB/MUL overflow
);

    // -----------------------------------------------------------------------
    // Combinational ALU
    // -----------------------------------------------------------------------
    logic signed [WIDTH-1:0]   sa, sb;
    // (* use_dsp = "yes" *) tells Vivado to map the 32×32 multiply to a
    // DSP48E1 cascade (~2.5 ns) instead of ~7K LUTs.  The REG_OUT=1 register
    // at the output of this always_comb block provides the required pipeline
    // stage, keeping the DSP-to-FF path within the 5 ns (200 MHz) budget.
    (* use_dsp = "yes" *) logic signed [2*WIDTH-1:0] mul_full;
    logic        [WIDTH-1:0]   res_comb;
    logic                      ovf_comb;
    logic                      valid_comb;

    assign sa = $signed(a);
    assign sb = $signed(b);

    always_comb begin
        res_comb   = '0;
        ovf_comb   = 1'b0;
        valid_comb = valid_in;
        mul_full   = sa * sb;

        case (op)
            3'b000: begin  // ADD
                logic signed [WIDTH:0] sum;
                sum      = {sa[WIDTH-1], sa} + {sb[WIDTH-1], sb};
                res_comb = sum[WIDTH-1:0];
                ovf_comb = (sum[WIDTH] != sum[WIDTH-1]);
            end

            3'b001: begin  // SUB
                logic signed [WIDTH:0] diff;
                diff     = {sa[WIDTH-1], sa} - {sb[WIDTH-1], sb};
                res_comb = diff[WIDTH-1:0];
                ovf_comb = (diff[WIDTH] != diff[WIDTH-1]);
            end

            3'b010: begin  // MUL (Q16.16 × Q16.16 → Q16.16)
                // Take bits [WIDTH+FRAC-1 : FRAC] for proper Q-format shift
                res_comb = mul_full[WIDTH+FRAC-1 : FRAC];
                // Overflow: check if upper bits beyond result are sign extension
                ovf_comb = (mul_full[2*WIDTH-1 : WIDTH+FRAC] !=
                            {(WIDTH-FRAC){mul_full[WIDTH+FRAC-1]}});
            end

            3'b011: begin  // ABS
                res_comb = (sa < 0) ? (~a + 1) : a;
            end

            3'b100: begin  // CMP
                res_comb = {
                    {(WIDTH-2){1'b0}},
                    (sa > sb)  ? 1'b1 : 1'b0,   // bit 1: GT
                    (sa == sb) ? 1'b1 : 1'b0     // bit 0: EQ
                };
            end

            default: begin
                res_comb = '0;
            end
        endcase
    end

    // -----------------------------------------------------------------------
    // Registered output
    // -----------------------------------------------------------------------
    generate
        if (REG_OUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    result    <= '0;
                    overflow  <= 1'b0;
                    valid_out <= 1'b0;
                end else begin
                    result    <= res_comb;
                    overflow  <= ovf_comb;
                    valid_out <= valid_comb;
                end
            end
        end else begin : gen_comb
            assign result    = res_comb;
            assign overflow  = ovf_comb;
            assign valid_out = valid_comb;
        end
    endgenerate

endmodule
