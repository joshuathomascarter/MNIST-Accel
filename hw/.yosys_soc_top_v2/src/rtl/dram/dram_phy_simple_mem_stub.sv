// Synthesis stub for dram_phy_simple_mem — blackboxed for area estimation
// The actual memory model is excluded from synthesis
`default_nettype none

module dram_phy_simple_mem #(
    parameter int NUM_BANKS = 8,
    parameter int ROW_BITS  = 14,
    parameter int COL_BITS  = 10,
    parameter int DATA_W    = 32,
    parameter int MEM_WORDS = 524288,
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
    // Blackbox stub — outputs driven to 0 for synthesis area estimation
    assign dram_phy_rdata       = '0;
    assign dram_phy_rdata_valid = 1'b0;
endmodule
`default_nettype wire
