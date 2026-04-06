// ===========================================================================
// dram_addr_decoder.sv — DRAM Address Decoder (AXI addr → bank/row/col)
// ===========================================================================
// Decodes a linear AXI byte address into DRAM bank, row, and column fields.
//
// Supports two interleaving schemes (configurable via parameter):
//   MODE 0: Row-Bank-Column (RBC) — maximizes row buffer hits for sequential
//   MODE 1: Bank-Row-Column (BRC) — maximizes bank-level parallelism
//
// For DDR3/DDR4 on Zynq-7020 PS:
//   8 banks, 14-bit row, 10-bit column, 16-bit data width (1-byte column)
//
// Resource estimate: ~30 LUTs (pure combinational)
// ===========================================================================

module dram_addr_decoder #(
    parameter int AXI_ADDR_W  = 32,
    parameter int BANK_BITS   = 3,      // 8 banks
    parameter int ROW_BITS    = 14,
    parameter int COL_BITS    = 10,
    parameter int BUS_BYTES   = 2,      // 16-bit DDR data bus → 1 byte-select bit
    parameter int MODE        = 0       // 0=RBC, 1=BRC
)(
    input  logic [AXI_ADDR_W-1:0]  axi_addr,

    output logic [BANK_BITS-1:0]   bank,
    output logic [ROW_BITS-1:0]    row,
    output logic [COL_BITS-1:0]    col
);

    localparam int BYTE_OFF = $clog2(BUS_BYTES);  // 1 for 16-bit bus

    generate
        if (MODE == 0) begin : gen_rbc
            // Row-Bank-Column layout: [ROW | BANK | COL | BYTE_OFF]
            assign col  = axi_addr[BYTE_OFF +: COL_BITS];
            assign bank = axi_addr[BYTE_OFF + COL_BITS +: BANK_BITS];
            assign row  = axi_addr[BYTE_OFF + COL_BITS + BANK_BITS +: ROW_BITS];
        end else begin : gen_brc
            // Bank-Row-Column layout: [BANK | ROW | COL | BYTE_OFF]
            assign col  = axi_addr[BYTE_OFF +: COL_BITS];
            assign row  = axi_addr[BYTE_OFF + COL_BITS +: ROW_BITS];
            assign bank = axi_addr[BYTE_OFF + COL_BITS + ROW_BITS +: BANK_BITS];
        end
    endgenerate

endmodule
