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
    output wire [DATA_W-1:0]   rd_data,
    output wire                 rd_valid    // 1-cycle delayed from rd_en
);

    // =========================================================================
    // Memory Banks (two 1RW SRAM banks)
    // =========================================================================
    wire                bank0_en;
    wire                bank0_we;
    wire [ADDR_W-1:0]   bank0_addr;
    wire [DATA_W-1:0]   bank0_rdata;

    wire                bank1_en;
    wire                bank1_we;
    wire [ADDR_W-1:0]   bank1_addr;
    wire [DATA_W-1:0]   bank1_rdata;

    assign bank0_we   = wr_en && !bank_sel;
    assign bank1_we   = wr_en &&  bank_sel;
    assign bank0_en   = bank0_we || (rd_en && bank_sel);
    assign bank1_en   = bank1_we || (rd_en && !bank_sel);
    assign bank0_addr = bank0_we ? wr_addr : rd_addr;
    assign bank1_addr = bank1_we ? wr_addr : rd_addr;

    // Read valid pipeline (1-cycle BRAM latency)
    reg rd_valid_r;
    assign rd_valid = rd_valid_r;
    assign rd_data  = bank_sel ? bank0_rdata : bank1_rdata;

    sram_1rw_wrapper #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W),
        .DEPTH  (DEPTH)
    ) u_bank0_mem (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (bank0_en),
        .we    (bank0_we),
        .addr  (bank0_addr),
        .wdata (wr_data),
        .rdata (bank0_rdata)
    );

    sram_1rw_wrapper #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W),
        .DEPTH  (DEPTH)
    ) u_bank1_mem (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (bank1_en),
        .we    (bank1_we),
        .addr  (bank1_addr),
        .wdata (wr_data),
        .rdata (bank1_rdata)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_r <= 1'b0;
        end else begin
            rd_valid_r <= rd_en;
        end
    end

endmodule

`default_nettype wire
