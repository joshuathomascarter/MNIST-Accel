`ifndef WGT_BUFFER_V
`define WGT_BUFFER_V
`default_nettype none
// wgt_buffer.sv — Double-buffered weight SRAM with ping-pong banks
// Structurally identical to act_buffer, TN-wide output

module wgt_buffer #(
    parameter TN                  = 16,  // typically overridden to 16
    parameter ADDR_WIDTH          = 7,
    parameter ENABLE_CLOCK_GATING = 1
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // DMA write port
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [TN*8-1:0]       wdata,
    input  wire                  bank_sel_wr,
    // Systolic array read port
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] k_idx,
    input  wire                  bank_sel_rd,
    output wire [TN*8-1:0]       b_vec
);

    // Clock gating
    wire buf_clk_en, buf_gated_clk;
    assign buf_clk_en = we | rd_en;
    
    generate
        if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
            `ifdef XILINX_FPGA
                BUFGCE buf_clk_gate (
                    .I  (clk),
                    .CE (buf_clk_en),
                    .O  (buf_gated_clk)
                );
            `elsif ASIC_SYNTHESIS
                // OpenLane bring-up uses a single ungated clock. Re-introduce
                // ICG cells later once CTS/power architecture is defined.
                assign buf_gated_clk = clk;
            `else
                reg buf_clk_en_latched;
                always @(clk or buf_clk_en) begin
                    if (!clk) buf_clk_en_latched <= buf_clk_en;
                end
                assign buf_gated_clk = clk & buf_clk_en_latched;
            `endif
        end else begin : gen_no_gate
            assign buf_gated_clk = clk;
        end
    endgenerate

    // Two single-port SRAM banks form the ping-pong buffer.
    wire                  bank0_en;
    wire                  bank0_we;
    wire [ADDR_WIDTH-1:0] bank0_addr;
    wire [TN*8-1:0]       bank0_rdata;

    wire                  bank1_en;
    wire                  bank1_we;
    wire [ADDR_WIDTH-1:0] bank1_addr;
    wire [TN*8-1:0]       bank1_rdata;

    assign bank0_we   = we && (bank_sel_wr == 1'b0);
    assign bank1_we   = we && (bank_sel_wr == 1'b1);
    assign bank0_en   = bank0_we || (rd_en && (bank_sel_rd == 1'b0));
    assign bank1_en   = bank1_we || (rd_en && (bank_sel_rd == 1'b1));
    assign bank0_addr = bank0_we ? waddr : k_idx;
    assign bank1_addr = bank1_we ? waddr : k_idx;

    // Assertions



    sram_1rw_wrapper #(
        .DATA_W (TN * 8),
        .ADDR_W (ADDR_WIDTH),
        .DEPTH  (1 << ADDR_WIDTH)
    ) u_bank0_mem (
        .clk   (buf_gated_clk),
        .rst_n (rst_n),
        .en    (bank0_en),
        .we    (bank0_we),
        .addr  (bank0_addr),
        .wdata (wdata),
        .rdata (bank0_rdata)
    );

    sram_1rw_wrapper #(
        .DATA_W (TN * 8),
        .ADDR_W (ADDR_WIDTH),
        .DEPTH  (1 << ADDR_WIDTH)
    ) u_bank1_mem (
        .clk   (buf_gated_clk),
        .rst_n (rst_n),
        .en    (bank1_en),
        .we    (bank1_we),
        .addr  (bank1_addr),
        .wdata (wdata),
        .rdata (bank1_rdata)
    );

    assign b_vec = (bank_sel_rd == 1'b0) ? bank0_rdata : bank1_rdata;

endmodule
`default_nettype wire
`endif
