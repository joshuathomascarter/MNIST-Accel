// =============================================================================
// axi_dma_bridge.sv — Unified AXI Write-Burst Router for Activations, Weights, Metadata
// =============================================================================
// Purpose:
//   Routes AXI4-Full write bursts to act_buffer, wgt_buffer, or bsr_dma based on address.
//   Address decoding:
//     [31:30] = 00 → activations (act_buffer)
//     [31:30] = 01 → weights (wgt_buffer)
//     [31:30] = 10 → metadata/BSR blocks (to FIFO for bsr_dma)
//
// Features:
//   - AXI4 write address and data channels
//   - Burst length support (WLEN up to 256)
//   - Address-based routing to three destinations
//   - Write strobe (WSTRB) handling
//   - Flow control and error reporting
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module axi_dma_bridge #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter ADDR_WIDTH_LOCAL = 6  // Local address width for buffers
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // AXI4 Write Address Channel
    input  wire [ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [1:0]              s_axi_awburst,
    input  wire [7:0]              s_axi_awlen,
    input  wire [2:0]              s_axi_awsize,
    input  wire                    s_axi_awvalid,
    output reg                     s_axi_awready,
    
    // AXI4 Write Data Channel
    input  wire [DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [(DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                    s_axi_wlast,
    input  wire                    s_axi_wvalid,
    output reg                     s_axi_wready,
    
    // AXI4 Write Response Channel
    output reg [1:0]               s_axi_bresp,
    output reg                     s_axi_bvalid,
    input  wire                    s_axi_bready,
    
    // OUTPUT: Activation Buffer Write Port
    output reg [DATA_WIDTH-1:0]    act_buf_wdata,
    output reg [ADDR_WIDTH_LOCAL-1:0] act_buf_waddr,
    output reg                     act_buf_wen,
    
    // OUTPUT: Weight Buffer Write Port
    output reg [DATA_WIDTH-1:0]    wgt_buf_wdata,
    output reg [ADDR_WIDTH_LOCAL-1:0] wgt_buf_waddr,
    output reg                     wgt_buf_wen,
    
    // OUTPUT: BSR DMA FIFO Write Port (metadata/blocks)
    output reg [DATA_WIDTH-1:0]    bsr_fifo_wdata,
    output reg                     bsr_fifo_wen,
    input  wire                    bsr_fifo_full,
    
    // Status & error
    output reg                     axi_error,
    output reg [31:0]              words_written
);

    // State machine for burst handling
    localparam [1:0] IDLE       = 2'd0,
                     BURST_DATA = 2'd1,
                     WAIT_RESP  = 2'd2;
    
    reg [1:0] state;
    reg [7:0] beat_count;
    reg [7:0] burst_len;
    reg [1:0] burst_type;
    reg [31:0] burst_addr;  // Captured address to determine route
    reg [1:0] target_type;  // Which buffer: 00=act, 01=wgt, 10=bsr
    
    // Address decode
    wire [1:0] addr_target = s_axi_awaddr[31:30];
    
    // ========================================================================
    // Write Address Latch
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            burst_len <= 8'd0;
            burst_type <= 2'd0;
            burst_addr <= 32'd0;
            target_type <= 2'd0;
        end else begin
            case (state)
                IDLE: begin
                    if (s_axi_awvalid) begin
                        burst_len <= s_axi_awlen;
                        burst_type <= s_axi_awburst;
                        burst_addr <= s_axi_awaddr;
                        target_type <= addr_target;  // Latch which buffer
                        s_axi_awready <= 1'b0;
                        // Transition to receive data
                    end
                end
                default: s_axi_awready <= 1'b0;
            endcase
        end
    end
    
    // ========================================================================
    // Write Data Path & Address-Based Routing
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
            act_buf_wen <= 1'b0;
            wgt_buf_wen <= 1'b0;
            bsr_fifo_wen <= 1'b0;
            beat_count <= 8'd0;
            axi_error <= 1'b0;
            words_written <= 32'd0;
        end else begin
            // Default: clear write enables (pulse outputs)
            act_buf_wen <= 1'b0;
            wgt_buf_wen <= 1'b0;
            bsr_fifo_wen <= 1'b0;
            
            case (state)
                IDLE: begin
                    s_axi_wready <= 1'b1;
                    if (s_axi_awvalid && s_axi_awready) begin
                        state <= BURST_DATA;
                        beat_count <= 8'd0;
                    end
                end
                
                BURST_DATA: begin
                    // Accept write data and route based on target_type
                    if (s_axi_wvalid && s_axi_wready) begin
                        // Route to appropriate buffer based on address
                        case (target_type)
                            2'b00: begin  // Activation buffer
                                act_buf_wdata <= s_axi_wdata;
                                act_buf_waddr <= (burst_addr[ADDR_WIDTH_LOCAL-1:0] + beat_count);
                                act_buf_wen <= 1'b1;
                            end
                            2'b01: begin  // Weight buffer
                                wgt_buf_wdata <= s_axi_wdata;
                                wgt_buf_waddr <= (burst_addr[ADDR_WIDTH_LOCAL-1:0] + beat_count);
                                wgt_buf_wen <= 1'b1;
                            end
                            2'b10: begin  // BSR metadata/blocks FIFO
                                if (!bsr_fifo_full) begin
                                    bsr_fifo_wdata <= s_axi_wdata;
                                    bsr_fifo_wen <= 1'b1;
                                end else begin
                                    // FIFO full: stall
                                    s_axi_wready <= 1'b0;
                                    axi_error <= 1'b1;
                                end
                            end
                            default: axi_error <= 1'b1;  // Invalid address
                        endcase
                        
                        if (!axi_error) begin
                            words_written <= words_written + 1;
                            beat_count <= beat_count + 1;
                            
                            // Check if burst complete
                            if (s_axi_wlast || (beat_count == burst_len)) begin
                                state <= WAIT_RESP;
                                s_axi_wready <= 1'b0;
                            end
                        end
                    end else begin
                        s_axi_wready <= 1'b1;
                    end
                end
                
                WAIT_RESP: begin
                    // Send write response
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp <= axi_error ? 2'b11 : 2'b00;  // SLVERR or OKAY
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 1'b0;
                        state <= IDLE;
                        s_axi_awready <= 1'b1;
                        axi_error <= 1'b0;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of axi_dma_bridge.sv (Unified Router)
// ============================================================================={
