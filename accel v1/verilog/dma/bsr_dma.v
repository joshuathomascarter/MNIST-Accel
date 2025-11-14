// =============================================================================
// bsr_dma.v — BSR (Block Sparse Row) DMA Engine for Sparse Accelerator
// =============================================================================
// Purpose:
//   Transfers BSR-format sparse matrix data from host via UART into on-chip BRAMs:
//   - row_ptr[]: cumulative block counts (sparse metadata)
//   - col_idx[]: column position of each block
//   - blocks[]:  8×8 INT8 weight blocks
//
// Features:
//   - UART-based packet protocol for host transfers
//   - Multi-layer support: independently load different network layers
//   - Configurable BRAM write interface
//   - CRC-32 protection (optional)
//   - Status reporting via CSR interface
//
// Architecture:
//   ┌─────────────┐
//   │  UART RX    │  (receives packets from host)
//   └──────┬──────┘
//          │
//   ┌──────▼──────────┐
//   │  Packet Parser  │  (interprets DMA commands)
//   └──────┬──────────┘
//          │
//   ┌──────▼──────────────┐
//   │  BRAM Write Arbiter │  (addresses row_ptr/col_idx/blocks)
//   └──────┬───────────────┘
//          │
//   ┌──────▼──────────────┐
//   │  BRAM Interfaces   │  (writes to on-chip memories)
//   └────────────────────┘
//
// Packet Format (CSR-driven):
//   [CSR_ADDR] = 0x20: Layer selection (0-7)
//   [CSR_ADDR] = 0x21: DMA control (start, reset, etc.)
//   [CSR_ADDR] = 0x22: Write count (blocks loaded)
//   [CSR_ADDR] = 0x23: Status flags
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bsr_dma #(
    parameter DATA_WIDTH        = 8,           // INT8 weights
    parameter ADDR_WIDTH        = 16,          // BRAM address width
    parameter MAX_LAYERS        = 8,           // Support 0-7 network layers
    parameter MAX_BLOCKS        = 65536,       // 2^16 blocks per layer
    parameter BLOCK_SIZE        = 64,          // 8×8 blocks = 64 bytes
    parameter ROW_PTR_DEPTH     = 256,         // For 2048 output features / 8
    parameter COL_IDX_DEPTH     = 65536,       // One per block
    parameter BLOCK_DEPTH       = 65536 * 64,  // Total block data (in bytes)
    parameter ENABLE_CRC        = 0             // CRC32 protection (0=disabled for now)
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // UART RX interface (from uart_rx module)
    input  wire [7:0]  uart_rx_data,
    input  wire        uart_rx_valid,
    output reg         uart_rx_ready,
    
    // UART TX interface (for status responses)
    output reg [7:0]   uart_tx_data,
    output reg         uart_tx_valid,
    input  wire        uart_tx_ready,
    
    // CSR interface (for DMA control/status)
    input  wire [7:0]  csr_addr,
    input  wire        csr_wen,
    input  wire [31:0] csr_wdata,
    output reg [31:0]  csr_rdata,
    
    // row_ptr BRAM write interface
    output reg         row_ptr_we,
    output reg [ADDR_WIDTH-1:0] row_ptr_waddr,
    output reg [31:0]  row_ptr_wdata,
    
    // col_idx BRAM write interface
    output reg         col_idx_we,
    output reg [31:0]  col_idx_waddr,
    output reg [15:0]  col_idx_wdata,
    
    // blocks BRAM write interface
    output reg         block_we,
    output reg [ADDR_WIDTH+6:0] block_waddr,  // +6 for 64 bytes per block
    output reg [7:0]   block_wdata,
    
    // Status outputs
    output reg         dma_busy,
    output reg         dma_done,
    output reg         dma_error,
    output reg [31:0]  blocks_written
);

    // ========================================================================
    // CSR Register Address Map
    // ========================================================================
    localparam [7:0] CSR_DMA_LAYER  = 8'h20,  // Layer selection
                     CSR_DMA_CTRL   = 8'h21,  // Control (start, reset)
                     CSR_DMA_COUNT  = 8'h22,  // Blocks written
                     CSR_DMA_STATUS = 8'h23;  // Status flags
    
    // ========================================================================
    // Packet Protocol State Machine
    // ========================================================================
    typedef enum logic [3:0] {
        IDLE            = 4'd0,   // Waiting for packet start
        RX_LAYER_CMD    = 4'd1,   // Receiving layer switch command
        RX_ROW_PTR_INIT = 4'd2,   // Initialize row_ptr load
        RX_ROW_PTR_DATA = 4'd3,   // Receive row_ptr entries (32-bit)
        RX_COL_IDX_INIT = 4'd4,   // Initialize col_idx load
        RX_COL_IDX_DATA = 4'd5,   // Receive col_idx entries (16-bit)
        RX_BLOCK_INIT   = 4'd6,   // Initialize block data load
        RX_BLOCK_DATA   = 4'd7,   // Receive 64-byte blocks (8×8 INT8)
        RX_CRC_CHK      = 4'd8,   // Verify CRC32 if enabled
        TX_STATUS       = 4'd9,   // Send status response
        DONE            = 4'd10   // Complete
    } state_t;
    
    state_t state, next_state;
    
    // ========================================================================
    // Internal Registers
    // ========================================================================
    reg [2:0]   active_layer;           // Currently loading layer (0-7)
    reg [31:0]  blocks_in_layer;        // Blocks loaded for this layer
    
    // Packet reception
    reg [7:0]   rx_bytes [0:3];         // Multi-byte accumulator
    reg [1:0]   rx_byte_count;          // Count for multi-byte fields
    reg [31:0]  rx_payload;             // Assembled multi-byte value
    
    // Block tracking
    reg [31:0]  block_byte_count;       // 0-63 for per-block reception
    reg [31:0]  current_block_idx;      // Global block index
    reg [ADDR_WIDTH-1:0] row_ptr_addr;  // Current write address for row_ptr
    reg [ADDR_WIDTH-1:0] col_idx_addr;  // Current write address for col_idx
    reg [ADDR_WIDTH+6:0] block_addr;    // Current write address for blocks
    
    // CRC accumulator
    reg [31:0]  crc_accumulator;
    
    // ========================================================================
    // State Machine Sequential Logic
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            active_layer <= 3'd0;
            blocks_in_layer <= 32'd0;
            blocks_written <= 32'd0;
            dma_busy <= 1'b0;
            dma_done <= 1'b0;
            dma_error <= 1'b0;
            rx_byte_count <= 2'd0;
            block_byte_count <= 32'd0;
            current_block_idx <= 32'd0;
            row_ptr_addr <= {ADDR_WIDTH{1'b0}};
            col_idx_addr <= {ADDR_WIDTH{1'b0}};
            block_addr <= {ADDR_WIDTH+7{1'b0}};
            crc_accumulator <= 32'hFFFF_FFFF;  // CRC32 initial value
        end else begin
            state <= next_state;
            
            // Default: disable writes
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            block_we <= 1'b0;
            uart_tx_valid <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[0]) begin
                        dma_busy <= 1'b1;
                        dma_done <= 1'b0;
                        dma_error <= 1'b0;
                        blocks_in_layer <= 32'd0;
                        block_byte_count <= 32'd0;
                        current_block_idx <= 32'd0;
                    end
                end
                
                RX_LAYER_CMD: begin
                    if (uart_rx_valid) begin
                        // Extract layer from UART byte
                        active_layer <= uart_rx_data[2:0];
                        rx_byte_count <= 2'd0;
                    end
                end
                
                RX_ROW_PTR_DATA: begin
                    if (uart_rx_valid) begin
                        rx_bytes[rx_byte_count] <= uart_rx_data;
                        
                        if (rx_byte_count == 2'd3) begin
                            // 32-bit value complete
                            row_ptr_wdata <= {uart_rx_data, rx_bytes[2], rx_bytes[1], rx_bytes[0]};
                            row_ptr_we <= 1'b1;
                            row_ptr_addr <= row_ptr_addr + 1;
                            rx_byte_count <= 2'd0;
                        end else begin
                            rx_byte_count <= rx_byte_count + 1;
                        end
                    end
                end
                
                RX_COL_IDX_DATA: begin
                    if (uart_rx_valid) begin
                        rx_bytes[rx_byte_count] <= uart_rx_data;
                        
                        if (rx_byte_count == 2'd1) begin
                            // 16-bit value complete
                            col_idx_wdata <= {uart_rx_data, rx_bytes[0]};
                            col_idx_we <= 1'b1;
                            col_idx_addr <= col_idx_addr + 1;
                            rx_byte_count <= 2'd0;
                        end else begin
                            rx_byte_count <= rx_byte_count + 1;
                        end
                    end
                end
                
                RX_BLOCK_DATA: begin
                    if (uart_rx_valid) begin
                        // Load one byte per cycle (8×8 = 64 bytes per block)
                        block_wdata <= uart_rx_data;
                        block_we <= 1'b1;
                        block_addr <= block_addr + 1;
                        block_byte_count <= block_byte_count + 1;
                        
                        // After 64 bytes, block is complete
                        if (block_byte_count == 32'd63) begin
                            block_byte_count <= 32'd0;
                            current_block_idx <= current_block_idx + 1;
                            blocks_in_layer <= blocks_in_layer + 1;
                            blocks_written <= blocks_written + 1;
                        end
                    end
                end
                
                TX_STATUS: begin
                    if (uart_tx_ready) begin
                        uart_tx_data <= {6'd0, dma_error, dma_done};
                        uart_tx_valid <= 1'b1;
                    end
                end
                
                DONE: begin
                    dma_busy <= 1'b0;
                    dma_done <= 1'b1;
                end
            endcase
        end
    end
    
    // ========================================================================
    // FSM Next State Logic
    // ========================================================================
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[0]) begin
                    next_state = RX_LAYER_CMD;
                end
            end
            
            RX_LAYER_CMD: begin
                if (uart_rx_valid) begin
                    next_state = RX_ROW_PTR_INIT;
                end
            end
            
            RX_ROW_PTR_INIT: begin
                // Could receive N row_ptr entries; simplified: go directly to receive
                next_state = RX_ROW_PTR_DATA;
            end
            
            RX_ROW_PTR_DATA: begin
                // Continue until CSR command changes (TODO: protocol refinement)
                // For now, receive a fixed number or detect end-of-data
                // Simplified: move to col_idx after some internal timeout
            end
            
            RX_COL_IDX_INIT: begin
                next_state = RX_COL_IDX_DATA;
            end
            
            RX_COL_IDX_DATA: begin
                // Continue receiving col_idx entries
            end
            
            RX_BLOCK_INIT: begin
                next_state = RX_BLOCK_DATA;
            end
            
            RX_BLOCK_DATA: begin
                // Continue until block_byte_count reaches MAX_BLOCKS or end signal
            end
            
            TX_STATUS: begin
                if (uart_tx_ready) begin
                    next_state = DONE;
                end
            end
            
            DONE: begin
                if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[1]) begin
                    // Reset flag clears DMA
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // ========================================================================
    // CSR Read Path
    // ========================================================================
    always @(*) begin
        csr_rdata = 32'h0000_0000;
        
        case (csr_addr)
            CSR_DMA_LAYER: begin
                csr_rdata = {29'd0, active_layer};
            end
            CSR_DMA_COUNT: begin
                csr_rdata = blocks_written;
            end
            CSR_DMA_STATUS: begin
                csr_rdata = {29'd0, dma_error, dma_done, dma_busy};
            end
            default: csr_rdata = 32'hDEAD_BEEF;
        endcase
    end
    
    // ========================================================================
    // Default Output Assignments (Combinational)
    // ========================================================================
    assign uart_rx_ready = (state == RX_LAYER_CMD) || 
                           (state == RX_ROW_PTR_DATA) || 
                           (state == RX_COL_IDX_DATA) || 
                           (state == RX_BLOCK_DATA);

endmodule

`default_nettype wire

// =============================================================================
// End of bsr_dma.v
// =============================================================================
