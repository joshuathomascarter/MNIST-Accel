// =============================================================================
// dram_phy_simple_mem.sv — Lightweight backing store behind dram_phy_* signals
// =============================================================================
// Interprets the custom DRAM PHY command set from dram_ctrl_top and stores
// 32-bit words in a simple internal memory array. This is intended for FPGA
// bringup and ASIC pre-silicon simulation, where a persistent memory image is
// more useful than a constant read-data stub.

`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module dram_phy_simple_mem #(
    parameter int NUM_BANKS = 8,
    parameter int ROW_BITS = 14,
    parameter int COL_BITS = 10,
    parameter int DATA_W = 32,
    parameter int MEM_WORDS = 16384,
    parameter INIT_FILE = ""
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [NUM_BANKS-1:0]  dram_phy_act,
    input  logic [NUM_BANKS-1:0]  dram_phy_read,
    input  logic [NUM_BANKS-1:0]  dram_phy_write,
    input  logic [NUM_BANKS-1:0]  dram_phy_pre,
    input  logic [ROW_BITS-1:0]   dram_phy_row,
    input  logic [COL_BITS-1:0]   dram_phy_col,
    input  logic                  dram_phy_ref,
    input  logic [DATA_W-1:0]     dram_phy_wdata,
    input  logic [DATA_W/8-1:0]   dram_phy_wstrb,
    output logic [DATA_W-1:0]     dram_phy_rdata,
    output logic                  dram_phy_rdata_valid
);

    localparam int BANK_BITS = (NUM_BANKS <= 1) ? 1 : $clog2(NUM_BANKS);
    localparam int COL_WORD_BITS = (COL_BITS <= 1) ? 1 : (COL_BITS - 1);
    localparam int MEM_ADDR_W = (MEM_WORDS <= 1) ? 1 : $clog2(MEM_WORDS);

    logic [ROW_BITS-1:0] open_row [0:NUM_BANKS-1];
    logic                row_open [0:NUM_BANKS-1];
    logic [DATA_W-1:0]   mem [0:MEM_WORDS-1];
    logic                read_pending;
    logic [MEM_ADDR_W-1:0] read_addr_pending;

    function automatic logic [MEM_ADDR_W-1:0] phy_to_mem_addr(
        input logic [BANK_BITS-1:0] bank,
        input logic [ROW_BITS-1:0] row,
        input logic [COL_BITS-1:0] col
    );
        logic [ROW_BITS + BANK_BITS + COL_WORD_BITS - 1:0] linear_word_addr;
        begin
            linear_word_addr = {row, bank, col[COL_BITS-1:1]};
            phy_to_mem_addr = linear_word_addr[MEM_ADDR_W-1:0];
        end
    endfunction

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dram_phy_rdata       <= '0;
            dram_phy_rdata_valid <= 1'b0;
            read_pending         <= 1'b0;
            read_addr_pending    <= '0;
            for (int bank = 0; bank < NUM_BANKS; bank++) begin
                open_row[bank] <= '0;
                row_open[bank] <= 1'b0;
            end
        end else begin
            dram_phy_rdata_valid <= read_pending;
            if (read_pending)
                dram_phy_rdata <= mem[read_addr_pending];
            read_pending <= 1'b0;

            if (dram_phy_ref) begin
                // Refresh is ignored in the simple backing store.
            end

            for (int bank = 0; bank < NUM_BANKS; bank++) begin
                logic [MEM_ADDR_W-1:0] mem_addr;

                if (dram_phy_act[bank]) begin
                    open_row[bank] <= dram_phy_row;
                    row_open[bank] <= 1'b1;
                end

                if (dram_phy_pre[bank])
                    row_open[bank] <= 1'b0;

                if (row_open[bank] && dram_phy_write[bank]) begin
                    mem_addr = phy_to_mem_addr(BANK_BITS'(bank), open_row[bank], dram_phy_col);
                    for (int byte_idx = 0; byte_idx < (DATA_W / 8); byte_idx++) begin
                        if (dram_phy_wstrb[byte_idx])
                            mem[mem_addr][8*byte_idx +: 8] <= dram_phy_wdata[8*byte_idx +: 8];
                    end
                end

                if (row_open[bank] && dram_phy_read[bank]) begin
                    read_pending      <= 1'b1;
                    read_addr_pending <= phy_to_mem_addr(BANK_BITS'(bank), open_row[bank], dram_phy_col);
                end
            end
        end
    end

endmodule

`default_nettype wire
