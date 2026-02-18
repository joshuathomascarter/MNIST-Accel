`ifndef WGT_BUFFER_V
`define WGT_BUFFER_V
`default_nettype none
// wgt_buffer.sv — Double-buffered weight SRAM with ping-pong banks
// Structurally identical to act_buffer, TN-wide output

module wgt_buffer #(
    parameter TN                  = 14,  // typically overridden to 14
    parameter ADDR_WIDTH          = 7,
    parameter ENABLE_CLOCK_GATING = 1
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // DMA write port
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [TN*8-1:0]       wdata,
    input  wire                  bank_sel_wr,
    // Systolic array read port
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] k_idx,
    input  wire                  bank_sel_rd,
    output reg  [TN*8-1:0]       b_vec
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
    reg [TN*8-1:0] mem0 [0:(1<<ADDR_WIDTH)-1];
    reg [TN*8-1:0] mem1 [0:(1<<ADDR_WIDTH)-1];

    // Assertions
    initial begin
        assert (TN > 0 && TN <= 1024)
            else $fatal("wgt_buffer: TN=%0d out of range (1-1024)", TN);
        assert (ADDR_WIDTH > 0 && ADDR_WIDTH <= 12)
            else $fatal("wgt_buffer: ADDR_WIDTH=%0d out of range (1-12)", ADDR_WIDTH);
    end

    always @(posedge clk) begin
        if (we) assert (waddr < (1<<ADDR_WIDTH))
            else $error("wgt_buffer: waddr %0d >= depth %0d", waddr, (1<<ADDR_WIDTH));
        if (rd_en) assert (k_idx < (1<<ADDR_WIDTH))
            else $error("wgt_buffer: k_idx %0d >= depth %0d", k_idx, (1<<ADDR_WIDTH));
    end

    always @(posedge clk) begin
        assert (bank_sel_wr == 1'b0 || bank_sel_wr == 1'b1)
            else $error("wgt_buffer: bank_sel_wr=%b invalid", bank_sel_wr);
        assert (bank_sel_rd == 1'b0 || bank_sel_rd == 1'b1)
            else $error("wgt_buffer: bank_sel_rd=%b invalid", bank_sel_rd);
    end

    // Write
    always @(posedge buf_gated_clk) begin
        if (we) begin
            if (bank_sel_wr == 1'b0)
                mem0[waddr] <= wdata;
            else
                mem1[waddr] <= wdata;
        end
    end

    // Read (1-cycle latency: address registered -> data out)
    always @(posedge buf_gated_clk or negedge rst_n) begin
        if (!rst_n)
            b_vec <= {TN*8{1'b0}};
        else if (rd_en) begin
            if (bank_sel_rd == 1'b0)
                b_vec <= mem0[k_idx];
            else
                b_vec <= mem1[k_idx];
        end
    end

endmodule
`default_nettype wire
`endif
