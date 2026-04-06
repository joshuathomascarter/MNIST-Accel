// output_bram_buffer.sv — Double-buffered output BRAM for multi-layer pipeline
// =============================================================================
//
// TRUE DUAL-PORT BRAM with two 8KB banks for ping-pong buffering.
// Sits between the output_accumulator and either:
//   (a) feedback path → act_buffer (intermediate layers)
//   (b) out_dma → DDR (last layer)
//
// Port A: Write from output_accumulator (quantized INT8 data, 64-bit packed)
// Port B: Read by output_bram_ctrl (for feedback or DMA drain)
//
// Bank sizing:
//   MNIST max output: conv1 = 6272B, conv2 = 1600B, fc1 = 128B, fc2 = 10B
//   8KB per bank handles all layers with margin.
//   Total: 2 × 8KB = 16KB = 4 BRAM36 on Zynq-7020 (2.9% of 140 available)
//
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module output_bram_buffer #(
    parameter DATA_W     = 64,       // 64-bit words (8 × INT8)
    parameter ADDR_W     = 10,       // 2^10 = 1024 words per bank (8KB)
    parameter DEPTH      = 1024
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // =========================================================================
    // Bank Select
    // =========================================================================
    input  wire                 bank_sel,   // 0 = write bank0/read bank1, 1 = vice versa

    // =========================================================================
    // Port A — Write (from output_accumulator via ctrl)
    // =========================================================================
    input  wire                 wr_en,
    input  wire [ADDR_W-1:0]   wr_addr,
    input  wire [DATA_W-1:0]   wr_data,

    // =========================================================================
    // Port B — Read (from output_bram_ctrl for feedback or DMA)
    // =========================================================================
    input  wire                 rd_en,
    input  wire [ADDR_W-1:0]   rd_addr,
    output reg  [DATA_W-1:0]   rd_data,
    output wire                 rd_valid    // 1-cycle delayed from rd_en
);

    // =========================================================================
    // Memory Banks (inferred as BRAM36 on Xilinx)
    // =========================================================================
    (* ram_style = "block" *) reg [DATA_W-1:0] bank0 [0:DEPTH-1];
    (* ram_style = "block" *) reg [DATA_W-1:0] bank1 [0:DEPTH-1];

    // Read valid pipeline (1-cycle BRAM latency)
    reg rd_valid_r;
    assign rd_valid = rd_valid_r;

    // =========================================================================
    // Write Port (to active write bank)
    // =========================================================================
    // bank_sel=0: write to bank0, read from bank1
    // bank_sel=1: write to bank1, read from bank0
    always @(posedge clk) begin
        if (wr_en) begin
            if (!bank_sel)
                bank0[wr_addr] <= wr_data;
            else
                bank1[wr_addr] <= wr_data;
        end
    end

    // =========================================================================
    // Read Port (from inactive/completed bank)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data    <= {DATA_W{1'b0}};
            rd_valid_r <= 1'b0;
        end else begin
            rd_valid_r <= rd_en;
            if (rd_en) begin
                if (bank_sel)
                    rd_data <= bank0[rd_addr];  // bank_sel=1: read bank0
                else
                    rd_data <= bank1[rd_addr];  // bank_sel=0: read bank1
            end
        end
    end

endmodule

`default_nettype wire
