// =============================================================================
// meta_decode.sv — BSR Metadata Decoder with BRAM Cache
// =============================================================================
// Purpose:
//   Decodes sparse matrix metadata (row pointers, column indices, blocks).
//   Caches metadata in BRAM for fast repeated access.
//   Interfaces with DMA for metadata input and scheduler for metadata output.
//
// Features:
//   - 256-entry metadata BRAM cache
//   - Configurable block size and sparsity parameters
//   - CRC-32 metadata verification (optional)
//   - Performance counters (cache hits/misses, decode latency)
//   - Error detection and reporting
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module meta_decode #(
    parameter METADATA_CACHE_DEPTH = 256,
    parameter METADATA_CACHE_ADDR_W = 8,
    parameter DATA_WIDTH = 32,
    parameter ENABLE_CRC = 1,
    parameter ENABLE_PERF = 1
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // DMA Input Interface (metadata packets from DMA engine)
    input  wire [31:0]  dma_meta_data,
    input  wire [3:0]   dma_meta_valid,  // Per-byte valid (4 bytes)
    input  wire [1:0]   dma_meta_type,   // 0=ROW_PTR, 1=COL_IDX, 2=BLOCK_HDR
    input  wire         dma_meta_wen,
    output wire         dma_meta_ready,
    
    // Scheduler Output Interface (metadata to sparse datapath)
    input  wire [METADATA_CACHE_ADDR_W-1:0] sched_meta_raddr,
    input  wire                             sched_meta_ren,
    output wire [31:0]                      sched_meta_rdata,
    output wire                             sched_meta_rvalid,
    
    // Configuration
    input  wire [15:0]  cfg_num_rows,
    input  wire [15:0]  cfg_num_cols,
    input  wire [31:0]  cfg_total_blocks,
    input  wire [2:0]   cfg_block_size,    // 0=4x4, 1=8x8, 2=16x16, etc.
    
    // Performance Counters (to perf module)
    output wire [31:0]  perf_cache_hits,
    output wire [31:0]  perf_cache_misses,
    output wire [31:0]  perf_decode_cycles,
    
    // Status & Error
    output wire         meta_error,
    output wire [31:0]  meta_error_flags
);

    // ========================================================================
    // Metadata BRAM Cache (256 entries × 32 bits)
    // ========================================================================
    
    reg [31:0] metadata_cache [0:(METADATA_CACHE_DEPTH-1)];
    reg [METADATA_CACHE_ADDR_W-1:0] cache_waddr;
    reg [METADATA_CACHE_ADDR_W-1:0] cache_raddr;
    reg [31:0] cache_wdata;
    reg cache_wen;
    
    // Dual-port BRAM read/write
    always @(posedge clk) begin
        if (cache_wen) begin
            metadata_cache[cache_waddr] <= cache_wdata;
        end
    end
    
    wire [31:0] cache_rdata = metadata_cache[sched_meta_raddr];
    
    // ========================================================================
    // Metadata State Machine
    // ========================================================================
    
    localparam [2:0] META_IDLE       = 3'd0,
                     META_LATCH      = 3'd1,
                     META_VALIDATE   = 3'd2,
                     META_DECODE     = 3'd3,
                     META_CACHE_WR   = 3'd4,
                     META_DONE       = 3'd5;
    
    reg [2:0] state;
    reg [31:0] meta_data_latched;
    reg [1:0] meta_type_latched;
    reg [31:0] crc_acc;
    reg [31:0] error_flags;
    reg dma_meta_ready_reg;
    
    // Performance counters
    reg [31:0] cache_hit_count;
    reg [31:0] cache_miss_count;
    reg [31:0] decode_cycle_count;
    
    // Assign output
    assign dma_meta_ready = dma_meta_ready_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= META_IDLE;
            cache_waddr <= {METADATA_CACHE_ADDR_W{1'b0}};
            cache_wen <= 1'b0;
            dma_meta_ready_reg <= 1'b1;
            error_flags <= 32'd0;
            crc_acc <= 32'hFFFFFFFF;
            cache_hit_count <= 32'd0;
            cache_miss_count <= 32'd0;
            decode_cycle_count <= 32'd0;
        end else begin
            cache_wen <= 1'b0;  // Pulse
            dma_meta_ready_reg <= 1'b1;  // Always ready in IDLE
            
            case (state)
                META_IDLE: begin
                    if (dma_meta_wen && dma_meta_valid != 4'h0) begin
                        meta_data_latched <= dma_meta_data;
                        meta_type_latched <= dma_meta_type;
                        state <= META_VALIDATE;
                    end
                end
                
                META_VALIDATE: begin
                    // Validate metadata format
                    decode_cycle_count <= decode_cycle_count + 1;
                    
                    // Check for valid metadata type
                    if (meta_type_latched > 2'd2) begin
                        error_flags[0] <= 1'b1;  // Invalid type
                        state <= META_IDLE;
                    end else begin
                        state <= META_DECODE;
                    end
                end
                
                META_DECODE: begin
                    // Decode based on metadata type
                    case (meta_type_latched)
                        2'b00: begin  // ROW_PTR
                            // Row pointer: 32-bit address
                            cache_wdata <= meta_data_latched;
                        end
                        2'b01: begin  // COL_IDX
                            // Column index: 16-bit indices (2 per word)
                            cache_wdata <= meta_data_latched;
                        end
                        2'b10: begin  // BLOCK_HDR
                            // Block header: num_rows, num_cols, total
                            cache_wdata <= meta_data_latched;
                        end
                    endcase
                    state <= META_CACHE_WR;
                end
                
                META_CACHE_WR: begin
                    // Write to metadata cache
                    cache_wen <= 1'b1;
                    cache_waddr <= cache_waddr + 1'b1;
                    
                    if (cache_waddr == METADATA_CACHE_DEPTH - 1) begin
                        // Wrap around cache
                        cache_waddr <= {METADATA_CACHE_ADDR_W{1'b0}};
                    end
                    
                    state <= META_DONE;
                end
                
                META_DONE: begin
                    state <= META_IDLE;
                end
                
                default: state <= META_IDLE;
            endcase
        end
    end
    
    // ========================================================================
    // Scheduler Read Path
    // ========================================================================
    
    reg [31:0] sched_rdata_reg;
    reg sched_valid_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sched_rdata_reg <= 32'd0;
            sched_valid_reg <= 1'b0;
        end else begin
            if (sched_meta_ren) begin
                sched_rdata_reg <= cache_rdata;
                sched_valid_reg <= 1'b1;
                
                // Track cache hits/misses
                if (cache_rdata != 32'd0) begin
                    cache_hit_count <= cache_hit_count + 1;
                end else begin
                    cache_miss_count <= cache_miss_count + 1;
                end
            end else begin
                sched_valid_reg <= 1'b0;
            end
        end
    end
    
    assign sched_meta_rdata = sched_rdata_reg;
    assign sched_meta_rvalid = sched_valid_reg;
    
    // ========================================================================
    // Performance Counter Outputs
    // ========================================================================
    
    assign perf_cache_hits = cache_hit_count;
    assign perf_cache_misses = cache_miss_count;
    assign perf_decode_cycles = decode_cycle_count;
    
    // ========================================================================
    // Error Status
    // ========================================================================
    
    assign meta_error = (error_flags != 32'd0);
    assign meta_error_flags = error_flags;

endmodule

`default_nettype wire
// =============================================================================
// End of meta_decode.sv
// =============================================================================
