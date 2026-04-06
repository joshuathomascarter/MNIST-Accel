// dma_pack_112.sv — Pack 64-bit DMA beats into 128-bit buffer words
//
// Simple packer: 16-byte rows align perfectly to 8-byte beats (16 = 2×8).
// Every 2 beats produce one 128-bit row. No residue, no block-boundary reset.
//
//   beat 0: accumulate low 64 bits
//   beat 1: emit {dma_wdata, sr[63:0]} = 128 bits → 1 row
//
// 2 beats → 1 row.  32 beats → 16 rows (one block).

`default_nettype none

module dma_pack_112 #(
    parameter OUT_W          = 128,   // 16 × 8
    parameter ADDR_W         = 10,
    parameter ROWS_PER_BLOCK = 16     // 16 rows per block (unused — naturally aligned)
)(
    input  wire              clk,
    input  wire              rst_n,

    // DMA side — 64-bit beats
    input  wire              dma_we,
    input  wire [63:0]       dma_wdata,

    // Buffer side — 128-bit rows
    output reg               buf_we,
    output reg  [ADDR_W-1:0] buf_waddr,
    output reg  [OUT_W-1:0]  buf_wdata
);

    // Hold register for first beat of each pair
    reg [63:0] sr;
    reg        phase;       // 0 = accumulate, 1 = emit

    // ROWS_PER_BLOCK unused — kept for port compatibility
    wire _unused_rpb = &{1'b0, ROWS_PER_BLOCK[0]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sr        <= 64'd0;
            phase     <= 1'b0;
            buf_we    <= 1'b0;
            buf_waddr <= {ADDR_W{1'b1}};   // wraps to 0 on first emit
            buf_wdata <= {OUT_W{1'b0}};
        end else begin
            buf_we <= 1'b0;

            if (dma_we) begin
                if (!phase) begin
                    // Beat 0: accumulate low 64 bits
                    sr    <= dma_wdata;
                    phase <= 1'b1;
                end else begin
                    // Beat 1: emit 128-bit row {high, low}
                    buf_we    <= 1'b1;
                    buf_waddr <= buf_waddr + {{(ADDR_W-1){1'b0}}, 1'b1};
                    buf_wdata <= {dma_wdata, sr};
                    phase     <= 1'b0;
                end
            end
        end
    end

endmodule

`default_nettype wire
