// dma_pack_112.sv — Pack 64-bit DMA beats into 112-bit buffer words
//
// Shift-register packer: 14-byte rows don't align to 8-byte beats (LCM=56).
// Accumulate incoming bytes; emit one 112-bit word each time ≥14 bytes present.
//
//   cnt    +64 bits  emit?  new cnt   residue
//   0      → 64      no     64        —
//   64     → 128     YES    16        dma[63:48]
//   16     → 80      no     80        —
//   80     → 144     YES    32        dma[63:32]
//   32     → 96      no     96        —
//   96     → 160     YES    48        dma[63:16]
//   48     → 112     YES    0         (none)
//
// 7 beats → 4 rows.  25 beats → 14 rows (one block).
// After 14 emits (ROWS_PER_BLOCK), residue is auto-cleared so the next
// block starts clean even though 196 bytes ≠ 25×8 = 200 bytes.

`default_nettype none

module dma_pack_112 #(
    parameter OUT_W          = 112,   // 14 × 8
    parameter ADDR_W         = 10,
    parameter ROWS_PER_BLOCK = 14     // auto-reset after this many emits
)(
    input  wire              clk,
    input  wire              rst_n,

    // DMA side — 64-bit beats
    input  wire              dma_we,
    input  wire [63:0]       dma_wdata,

    // Buffer side — 112-bit rows
    output reg               buf_we,
    output reg  [ADDR_W-1:0] buf_waddr,
    output reg  [OUT_W-1:0]  buf_wdata
);

    // Shift register: max residue = 96 bits (when cnt=96).
    reg [95:0] sr;
    reg [7:0]  cnt;         // valid bits in sr: one of {0,16,32,48,64,80,96}
    reg [7:0]  row_cnt;     // emits since last block-boundary reset

    localparam [7:0] LAST_ROW = ROWS_PER_BLOCK[7:0] - 8'd1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sr        <= 96'd0;
            cnt       <= 8'd0;
            row_cnt   <= 8'd0;
            buf_we    <= 1'b0;
            buf_waddr <= {ADDR_W{1'b1}};   // wraps to 0 on first emit
            buf_wdata <= {OUT_W{1'b0}};
        end else begin
            buf_we <= 1'b0;

            if (dma_we) begin
                case (cnt)
                    // — accumulate-only (total < 112) —
                    8'd0: begin
                        sr[63:0]  <= dma_wdata;
                        cnt       <= 8'd64;
                    end
                    8'd16: begin
                        sr[79:16] <= dma_wdata;
                        cnt       <= 8'd80;
                    end
                    8'd32: begin
                        sr[95:32] <= dma_wdata;
                        cnt       <= 8'd96;
                    end

                    // — emit cases (total ≥ 112) —
                    8'd48: begin                           // 48+64 = 112 → exact
                        buf_we    <= 1'b1;
                        buf_waddr <= buf_waddr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        buf_wdata <= {dma_wdata, sr[47:0]};
                        cnt       <= 8'd0;
                        row_cnt   <= row_cnt + 8'd1;
                    end
                    8'd64: begin                           // 64+64 = 128 → emit 112, keep 16
                        buf_we    <= 1'b1;
                        buf_waddr <= buf_waddr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        buf_wdata <= {dma_wdata[47:0], sr[63:0]};
                        sr[15:0]  <= dma_wdata[63:48];
                        cnt       <= 8'd16;
                        row_cnt   <= row_cnt + 8'd1;
                    end
                    8'd80: begin                           // 80+64 = 144 → emit 112, keep 32
                        buf_we    <= 1'b1;
                        buf_waddr <= buf_waddr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        buf_wdata <= {dma_wdata[31:0], sr[79:0]};
                        sr[31:0]  <= dma_wdata[63:32];
                        cnt       <= 8'd32;
                        row_cnt   <= row_cnt + 8'd1;
                    end
                    8'd96: begin                           // 96+64 = 160 → emit 112, keep 48
                        buf_we    <= 1'b1;
                        buf_waddr <= buf_waddr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        buf_wdata <= {dma_wdata[15:0], sr[95:0]};
                        sr[47:0]  <= dma_wdata[63:16];
                        cnt       <= 8'd48;
                        row_cnt   <= row_cnt + 8'd1;
                    end

                    default: begin                         // should never happen
                        sr  <= 96'd0;
                        cnt <= 8'd0;
                    end
                endcase

                // Auto-reset at block boundary (last NBA wins)
                if (cnt >= 8'd48 && row_cnt == LAST_ROW) begin
                    cnt     <= 8'd0;
                    sr      <= 96'd0;
                    row_cnt <= 8'd0;
                end
            end
        end
    end

endmodule

`default_nettype wire
