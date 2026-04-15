// Synthesis stub — replaces the 524K-word DRAM backing store.
// The memory is not synthesized; use BRAM inference in real
// implementation. This stub gives Yosys a proper black-box.
module dram_phy_simple_mem #(
    parameter NUM_BANKS  = 8,
    parameter ROW_BITS   = 14,
    parameter COL_BITS   = 10,
    parameter DATA_W     = 32,
    parameter MEM_WORDS  = 524288,
    parameter INIT_FILE  = ""
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [NUM_BANKS-1:0]  dram_phy_act,
    input  wire [NUM_BANKS-1:0]  dram_phy_read,
    input  wire [NUM_BANKS-1:0]  dram_phy_write,
    input  wire [NUM_BANKS-1:0]  dram_phy_pre,
    input  wire [ROW_BITS-1:0]   dram_phy_row,
    input  wire [COL_BITS-1:0]   dram_phy_col,
    input  wire                  dram_phy_ref,
    input  wire [DATA_W-1:0]     dram_phy_wdata,
    input  wire [DATA_W/8-1:0]   dram_phy_wstrb,
    output reg  [DATA_W-1:0]     dram_phy_rdata,
    output reg                   dram_phy_rdata_valid
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dram_phy_rdata       <= {DATA_W{1'b0}};
            dram_phy_rdata_valid <= 1'b0;
        end else begin
            dram_phy_rdata_valid <= |dram_phy_read;
            dram_phy_rdata       <= {DATA_W{1'b0}};
        end
    end
endmodule
