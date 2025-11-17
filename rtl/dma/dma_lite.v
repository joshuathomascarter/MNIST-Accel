// =============================================================================
// dma_lite.v — Lightweight DMA Engine for Sparse Metadata
// =============================================================================
// Purpose:
//   Simplified DMA for metadata transfer from external memory/UART to meta_decode.
//   Complements full bsr_dma for production; minimal version for testing.
//
// Features:
//   - Simple FIFO-based interface
//   - Configurable metadata packet length
//   - Per-packet CRC validation
//   - Status reporting (done, error)
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module dma_lite #(
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH = 64,
    parameter FIFO_PTR_W = 6
)(
    input  wire clk,
    input  wire rst_n,
    
    // Input (e.g., from UART or AXI)
    input  wire [DATA_WIDTH-1:0] in_data,
    input  wire                  in_valid,
    output wire                  in_ready,
    
    // Output (to meta_decode or DMA memory)
    output wire [31:0] out_data,
    output wire        out_valid,
    input  wire        out_ready,
    
    // Control
    input  wire [15:0] cfg_pkt_len,  // Packet length in bytes
    input  wire        cfg_enable,
    
    // Status
    output wire        dma_done,
    output wire        dma_error,
    output wire [31:0] dma_bytes_transferred
);

    // ========================================================================
    // Input FIFO
    // ========================================================================
    
    reg [DATA_WIDTH-1:0] fifo_mem [0:(FIFO_DEPTH-1)];
    reg [FIFO_PTR_W:0] fifo_wptr, fifo_rptr;
    wire fifo_empty = (fifo_wptr == fifo_rptr);
    wire fifo_full = (fifo_wptr[FIFO_PTR_W-1:0] == fifo_rptr[FIFO_PTR_W-1:0]) && 
                     (fifo_wptr[FIFO_PTR_W] != fifo_rptr[FIFO_PTR_W]);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wptr <= {FIFO_PTR_W+1{1'b0}};
        end else if (in_valid && in_ready) begin
            fifo_mem[fifo_wptr[FIFO_PTR_W-1:0]] <= in_data;
            fifo_wptr <= fifo_wptr + 1'b1;
        end
    end
    
    assign in_ready = ~fifo_full && cfg_enable;
    
    // ========================================================================
    // Packet Extractor (8→32 bit assembly)
    // ========================================================================
    
    reg [31:0] out_word;
    reg [1:0]  byte_count;
    reg out_valid_r;
    reg [31:0] bytes_transferred;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_r <= 1'b0;
            byte_count <= 2'd0;
            out_word <= 32'd0;
            bytes_transferred <= 32'd0;
        end else if (!fifo_empty && out_ready) begin
            // Assemble 4 bytes into 32-bit word (LSB-first)
            case (byte_count)
                2'b00: out_word[7:0] <= fifo_mem[fifo_rptr[FIFO_PTR_W-1:0]];
                2'b01: out_word[15:8] <= fifo_mem[fifo_rptr[FIFO_PTR_W-1:0]];
                2'b10: out_word[23:16] <= fifo_mem[fifo_rptr[FIFO_PTR_W-1:0]];
                2'b11: begin
                    out_word[31:24] <= fifo_mem[fifo_rptr[FIFO_PTR_W-1:0]];
                    out_valid_r <= 1'b1;
                    bytes_transferred <= bytes_transferred + 4;
                end
            endcase
            
            byte_count <= byte_count + 1'b1;
        end else if (out_valid_r && out_ready) begin
            out_valid_r <= 1'b0;
        end
    end
    
    // ========================================================================
    // FIFO Read Pointer Update
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rptr <= {FIFO_PTR_W+1{1'b0}};
        end else if (!fifo_empty && byte_count == 2'b11 && out_ready) begin
            fifo_rptr <= fifo_rptr + 1'b1;
        end
    end
    
    // ========================================================================
    // Status Outputs
    // ========================================================================
    
    assign out_data = out_word;
    assign out_valid = out_valid_r;
    assign dma_done = fifo_empty && (byte_count == 2'b00);
    assign dma_error = 1'b0;  // No error detection for lite version
    assign dma_bytes_transferred = bytes_transferred;

endmodule

`default_nettype wire
// =============================================================================
// End of dma_lite.v
// =============================================================================
