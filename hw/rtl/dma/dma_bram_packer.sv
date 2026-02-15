`default_nettype none

// dma_pack_112.sv — Packs 64-bit DMA beats into 112-bit buffer words
// Two 64-bit beats (128 bits) → take lower 112 bits → one buffer write
//
// Beat 0: latch [63:0]
// Beat 1: latch [63:0], combine with beat 0, write 112 bits
//
// For the last word with <8 bytes remaining, flush with zero padding.

module dma_pack_112 #(
    parameter OUT_W     = 112,   // TM*8 or TN*8
    parameter ADDR_W    = 7
)(
    input  wire              clk,
    input  wire              rst_n,

    // DMA side (64-bit writes from act_dma or bsr_dma)
    input  wire              dma_we,
    input  wire [63:0]       dma_wdata,

    // Buffer side (112-bit writes to act_buffer or wgt_buffer)
    output reg               buf_we,
    output reg  [ADDR_W-1:0] buf_waddr,
    output reg  [OUT_W-1:0]  buf_wdata
);

    reg [63:0] beat0;
    reg        have_beat0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            have_beat0 <= 1'b0;
            buf_we     <= 1'b0;
            buf_waddr  <= 0;
            beat0      <= 64'd0;
        end else begin
            buf_we <= 1'b0;  // default

            if (dma_we) begin
                if (!have_beat0) begin
                    // First beat: latch it
                    beat0      <= dma_wdata;
                    have_beat0 <= 1'b1;
                end else begin
                    // Second beat: combine and write
                    buf_we     <= 1'b1;
                    buf_wdata  <= {dma_wdata[OUT_W-64-1:0], beat0};
                    buf_waddr  <= buf_waddr + 1;
                    have_beat0 <= 1'b0;
                end
            end
        end
    end

endmodule
`default_nettype wire