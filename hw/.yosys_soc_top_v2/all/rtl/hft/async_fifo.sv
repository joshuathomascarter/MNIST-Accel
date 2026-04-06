// ===========================================================================
// async_fifo.sv — Asynchronous FIFO with Gray-code CDC
// ===========================================================================
// Dual-clock domain FIFO for crossing between ETH-RX clock (125 MHz) and
// system clock (100 MHz).
//
// Parameters:
//   DEPTH  — FIFO depth (must be power of 2), default 16
//   WIDTH  — data width, default 64
//
// Uses Gray-code pointer synchronization (2-FF) for safe CDC.
// Resource estimate: ~100 LUTs + DEPTH×WIDTH flip-flops (or BRAM if large)
// ===========================================================================

module async_fifo #(
    parameter int DEPTH = 16,
    parameter int WIDTH = 64
)(
    // Write port (producer clock domain)
    input  logic             wr_clk,
    input  logic             wr_rst_n,
    input  logic [WIDTH-1:0] wr_data,
    input  logic             wr_en,
    output logic             full,

    // Read port (consumer clock domain)
    input  logic             rd_clk,
    input  logic             rd_rst_n,
    output logic [WIDTH-1:0] rd_data,
    input  logic             rd_en,
    output logic             empty
);

    // -----------------------------------------------------------------------
    // Derived parameters
    // -----------------------------------------------------------------------
    localparam int ADDR_W = $clog2(DEPTH);

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    // -----------------------------------------------------------------------
    // Write-domain pointers (binary + Gray)
    // -----------------------------------------------------------------------
    logic [ADDR_W:0] wr_ptr_bin;       // extra MSB for full/empty detection
    logic [ADDR_W:0] wr_ptr_gray;
    logic [ADDR_W:0] wr_ptr_gray_next;
    logic [ADDR_W:0] rd_ptr_gray_sync; // read pointer synced to write domain

    // -----------------------------------------------------------------------
    // Read-domain pointers (binary + Gray)
    // -----------------------------------------------------------------------
    logic [ADDR_W:0] rd_ptr_bin;
    logic [ADDR_W:0] rd_ptr_gray;
    logic [ADDR_W:0] rd_ptr_gray_next;
    logic [ADDR_W:0] wr_ptr_gray_sync; // write pointer synced to read domain

    // -----------------------------------------------------------------------
    // Gray-code conversion helpers
    // -----------------------------------------------------------------------
    function [ADDR_W:0] bin2gray;
        input [ADDR_W:0] b;
        bin2gray = b ^ (b >> 1);
    endfunction

    // -----------------------------------------------------------------------
    // Write domain
    // -----------------------------------------------------------------------
    assign wr_ptr_gray_next = bin2gray(wr_ptr_bin + 1);

    always_ff @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            wr_ptr_bin  <= '0;
            wr_ptr_gray <= '0;
        end else if (wr_en && !full) begin
            mem[wr_ptr_bin[ADDR_W-1:0]] <= wr_data;
            wr_ptr_bin  <= wr_ptr_bin + 1;
            wr_ptr_gray <= wr_ptr_gray_next;
        end
    end

    // -----------------------------------------------------------------------
    // Read domain
    // -----------------------------------------------------------------------
    assign rd_ptr_gray_next = bin2gray(rd_ptr_bin + 1);

    always_ff @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            rd_ptr_bin  <= '0;
            rd_ptr_gray <= '0;
        end else if (rd_en && !empty) begin
            rd_ptr_bin  <= rd_ptr_bin + 1;
            rd_ptr_gray <= rd_ptr_gray_next;
        end
    end

    assign rd_data = mem[rd_ptr_bin[ADDR_W-1:0]];

    // -----------------------------------------------------------------------
    // 2-FF synchronizers
    // -----------------------------------------------------------------------
    // Sync read pointer → write domain
    logic [ADDR_W:0] rd_gray_meta, rd_gray_sync;
    always_ff @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            rd_gray_meta <= '0;
            rd_gray_sync <= '0;
        end else begin
            rd_gray_meta <= rd_ptr_gray;
            rd_gray_sync <= rd_gray_meta;
        end
    end
    assign rd_ptr_gray_sync = rd_gray_sync;

    // Sync write pointer → read domain
    logic [ADDR_W:0] wr_gray_meta, wr_gray_sync;
    always_ff @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            wr_gray_meta <= '0;
            wr_gray_sync <= '0;
        end else begin
            wr_gray_meta <= wr_ptr_gray;
            wr_gray_sync <= wr_gray_meta;
        end
    end
    assign wr_ptr_gray_sync = wr_gray_sync;

    // -----------------------------------------------------------------------
    // Full / Empty flags
    // -----------------------------------------------------------------------
    // Full:  Gray pointers match but top 2 bits are inverted
    assign full  = (wr_ptr_gray_next == {~rd_ptr_gray_sync[ADDR_W:ADDR_W-1],
                                          rd_ptr_gray_sync[ADDR_W-2:0]}) && wr_en ? 1'b1 :
                   (wr_ptr_gray      == {~rd_ptr_gray_sync[ADDR_W:ADDR_W-1],
                                          rd_ptr_gray_sync[ADDR_W-2:0]});
    // Empty: Gray pointers are identical
    assign empty = (rd_ptr_gray == wr_ptr_gray_sync);

    // -----------------------------------------------------------------------
    // CDC assertions (simulation only)
    // -----------------------------------------------------------------------
    // synthesis translate_off
`ifndef VERILATOR
    async_fifo_sva #(
        .DEPTH (DEPTH),
        .WIDTH (WIDTH)
    ) u_sva (
        .wr_clk      (wr_clk),
        .wr_rst_n    (wr_rst_n),
        .wr_en       (wr_en),
        .full        (full),
        .wr_ptr_gray (wr_ptr_gray),
        .rd_clk      (rd_clk),
        .rd_rst_n    (rd_rst_n),
        .rd_en       (rd_en),
        .empty       (empty),
        .rd_ptr_gray (rd_ptr_gray)
    );
`endif
    // synthesis translate_on

endmodule
