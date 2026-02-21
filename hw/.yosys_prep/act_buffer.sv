`ifndef ACT_BUFFER_V
`define ACT_BUFFER_V
`default_nettype none
// act_buffer.sv — Double-buffered activation SRAM with ping-pong banks
// 1-cycle read latency, clock-gated when idle

module act_buffer #(
    parameter TM                  = 14,
    parameter ADDR_WIDTH          = 7,
    parameter ENABLE_CLOCK_GATING = 1
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // DMA write port
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [TM*8-1:0]       wdata,
    input  wire                  bank_sel_wr,
    // Systolic array read port
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] k_idx,
    input  wire                  bank_sel_rd,
    output reg  [TM*8-1:0]       a_vec
);

    // Clock gating
    wire buf_clk_en, buf_gated_clk;
    assign buf_clk_en = we | rd_en;
    
    generate
        if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
            `ifdef XILINX_FPGA
                BUFGCE buf_clk_gate (
                    .I  (clk),
                    .CE (buf_clk_en),
                    .O  (buf_gated_clk)
                );
            `else
                reg buf_clk_en_latched;
                always @(clk or buf_clk_en) begin
                    if (!clk) buf_clk_en_latched <= buf_clk_en;
                end
                assign buf_gated_clk = clk & buf_clk_en_latched;
            `endif
        end else begin : gen_no_gate
            assign buf_gated_clk = clk;
        end
    endgenerate

    // Memory banks (ping-pong, inferred as BRAM)
    reg [TM*8-1:0] mem0 [0:(1<<ADDR_WIDTH)-1] /* verilator public */;
    reg [TM*8-1:0] mem1 [0:(1<<ADDR_WIDTH)-1] /* verilator public */;

    // Assertions

    // Write (uses gated clock - only toggles when we=1)
    always @(posedge buf_gated_clk) begin
        if (we) begin
            if (bank_sel_wr == 1'b0)
                mem0[waddr] <= wdata;
            else
                mem1[waddr] <= wdata;
        end
    end

    // Read (uses MAIN clock - must output zeros when rd_en=0 for proper drain)
    // Clock gating would prevent the zero-output from happening
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            a_vec <= {TM*8{1'b0}};
        else if (rd_en) begin
            if (bank_sel_rd == 1'b0)
                a_vec <= mem0[k_idx];
            else
                a_vec <= mem1[k_idx];
        end else begin
            a_vec <= {TM*8{1'b0}};  // Zero output during drain cycles
        end
    end

endmodule
`default_nettype wire
`endif
