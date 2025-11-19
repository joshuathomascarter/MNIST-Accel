// =============================================================================
// top_sparse.v — Top-Level Sparse Accelerator Integration (Phase 5)
// =============================================================================
// Purpose:
//   Integrates UART RX, DMA (metadata assembly), meta_decode (BRAM cache),
//   bsr_scheduler (workload generation), systolic_array_sparse (2×2 PE compute),
//   and output BRAM for complete sparse pipeline.
//
// Architecture:
//   UART/AXI → DMA → meta_decode → bsr_scheduler → systolic_sparse → output BRAM
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module top_sparse #(
    parameter N_ROWS = 2,
    parameter N_COLS = 2,
    parameter BLOCK_SIZE = 8,
    parameter CLK_HZ = 50_000_000,
    parameter BAUD = 115_200
)(
    input  wire clk,
    input  wire rst_n,
    
    // UART Interface (primary data path for sparse metadata + blocks)
    input  wire uart_rx,
    output wire uart_tx,
    
    // AXI4-Lite Interface (CSR configuration only)
    input  wire [31:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    
    // Status
    output wire busy,
    output wire done_pulse,
    output wire error
);

    // ========================================================================
    // Internal Signals
    // ========================================================================
    
    // UART RX (8-bit stream)
    wire [7:0] uart_rx_data;
    wire       uart_rx_valid;
    wire       uart_rx_ready;
    
    // DMA Lite output (32-bit metadata words)
    wire [31:0] dma_meta_data;
    wire        dma_meta_wen;
    wire        dma_meta_ready;
    
    // Metadata decoder → scheduler (BRAM read)
    wire [7:0] sched_meta_raddr;
    wire        sched_meta_ren;
    wire [31:0] sched_meta_rdata;
    wire        sched_meta_rvalid;
    
    // Metadata decoder status
    wire meta_error;
    wire [31:0] meta_error_flags;
    wire [31:0] perf_cache_hits;
    wire [31:0] perf_cache_misses;
    wire [31:0] perf_decode_cycles;
    
    // Scheduler → Systolic (block data)
    wire systolic_valid;
    wire [7:0] systolic_block_data [0:63];  // 8×8 INT8
    wire [15:0] systolic_block_row;
    wire [15:0] systolic_block_col;
    wire systolic_ready;
    wire systolic_done;
    
    // Systolic output
    wire systolic_result_valid;
    wire [31:0] systolic_result_data [0:1][0:7];  // 2×8 INT32 results
    wire [15:0] systolic_result_row;
    wire [15:0] systolic_result_col;
    wire systolic_result_ready;
    
    // ========================================================================
    // UART RX (8-bit input stream)
    // ========================================================================
    // NOTE: Placeholder for external UART receiver
    // In production, instantiate uart_rx module from verilog/uart/uart_rx.v
    
    // ========================================================================
    // DMA Lite (Metadata Assembly: 8→32 bit)
    // ========================================================================
    
    dma_lite #(
        .DATA_WIDTH(8),
        .FIFO_DEPTH(64),
        .FIFO_PTR_W(6)
    ) dma_lite_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(uart_rx_data),
        .in_valid(uart_rx_valid),
        .in_ready(uart_rx_ready),
        .out_data(dma_meta_data),
        .out_valid(dma_meta_wen),
        .out_ready(dma_meta_ready),
        .cfg_pkt_len(16'd4),
        .cfg_enable(1'b1),
        .dma_done(),
        .dma_error(),
        .dma_bytes_transferred()
    );
    
    // ========================================================================
    // Metadata Decoder with BRAM Cache (256 entries)
    // ========================================================================
    
    meta_decode #(
        .METADATA_CACHE_DEPTH(256),
        .METADATA_CACHE_ADDR_W(8),
        .DATA_WIDTH(32),
        .ENABLE_CRC(1),
        .ENABLE_PERF(1)
    ) meta_decode_inst (
        .clk(clk),
        .rst_n(rst_n),
        .dma_meta_data(dma_meta_data),
        .dma_meta_valid({4{dma_meta_wen}}),
        .dma_meta_type(2'b00),
        .dma_meta_wen(dma_meta_wen),
        .dma_meta_ready(dma_meta_ready),
        .sched_meta_raddr(sched_meta_raddr),
        .sched_meta_ren(sched_meta_ren),
        .sched_meta_rdata(sched_meta_rdata),
        .sched_meta_rvalid(sched_meta_rvalid),
        .cfg_num_rows(16'd32),
        .cfg_num_cols(16'd32),
        .cfg_total_blocks(32'd512),
        .cfg_block_size(3'd1),
        .perf_cache_hits(perf_cache_hits),
        .perf_cache_misses(perf_cache_misses),
        .perf_decode_cycles(perf_decode_cycles),
        .meta_error(meta_error),
        .meta_error_flags(meta_error_flags)
    );
    
    // ========================================================================
    // BSR Scheduler (Metadata BRAM Cache → Block Workload)
    // ========================================================================
    
    bsr_scheduler #(
        .BLOCK_H(BLOCK_SIZE),
        .BLOCK_W(BLOCK_SIZE),
        .BLOCK_SIZE(BLOCK_SIZE * BLOCK_SIZE),
        .MAX_BLOCK_ROWS(256),
        .MAX_BLOCKS(65536),
        .DATA_WIDTH(8)
    ) bsr_scheduler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(dma_meta_wen),
        .cfg_num_block_rows(16'd32),
        .cfg_num_block_cols(16'd32),
        .cfg_total_blocks(perf_cache_hits + perf_cache_misses),
        .cfg_layer_switch(1'b0),
        .cfg_active_layer(3'd0),
        .cfg_layer_ready(),
        
        // Metadata BRAM read interfaces
        .row_ptr_rd_en(sched_meta_ren),
        .row_ptr_rd_addr(sched_meta_raddr),
        .row_ptr_rd_data(sched_meta_rdata),
        .col_idx_rd_en(),
        .col_idx_rd_addr(),
        .col_idx_rd_data(),
        
        // Block data BRAM (TODO: connect to block storage)
        .block_rd_en(),
        .block_rd_addr(),
        .block_rd_data(),
        
        // Systolic array interface
        .systolic_valid(systolic_valid),
        .systolic_block(systolic_block_data),
        .systolic_block_row(systolic_block_row),
        .systolic_block_col(systolic_block_col),
        .systolic_ready(systolic_ready),
        .systolic_done(systolic_done),
        
        // Status
        .done(),
        .busy(),
        .blocks_processed()
    );
    
    // ========================================================================
    // Sparse Systolic Array (2×2 PE, 8×8 block compute)
    // ========================================================================
    
    systolic_array_sparse #(
        .PE_ROWS(N_ROWS),
        .PE_COLS(N_COLS),
        .DATA_WIDTH(8),
        .ACC_WIDTH(32),
        .BLOCK_H(BLOCK_SIZE),
        .BLOCK_W(BLOCK_SIZE)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(systolic_valid),
        .block_data(systolic_block_data),
        .block_row(systolic_block_row),
        .block_col(systolic_block_col),
        .ready(systolic_ready),
        .act_data(),  // TODO: Connect to activation stream
        .act_valid(1'b0),
        .result_valid(systolic_result_valid),
        .result_data(systolic_result_data),
        .result_block_row(systolic_result_row),
        .result_block_col(systolic_result_col),
        .result_ready(systolic_result_ready),
        .done(systolic_done),
        .busy()
    );
    
    // ========================================================================
    // Output BRAM Buffer (4KB: 1K entries × 32-bit INT32)
    // ========================================================================
    
    reg [31:0] output_bram [0:1023];
    reg [9:0] output_wr_addr;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_wr_addr <= 10'd0;
        end else begin
            if (systolic_result_valid) begin
                // Store first result column from 2×8 block
                output_bram[output_wr_addr] <= systolic_result_data[0][0];
                output_wr_addr <= output_wr_addr + 1'b1;
            end
        end
    end
    
    // ========================================================================
    // Status Outputs
    // ========================================================================
    
    assign busy = dma_meta_wen | sched_meta_rvalid | systolic_valid | systolic_result_valid;
    assign done_pulse = (output_wr_addr == 10'h3FF) & systolic_result_valid;
    assign error = meta_error;
    
    // UART TX: Loop back RX for now (or stream results)
    assign uart_tx = uart_rx;

endmodule

`default_nettype wire
// =============================================================================
// End of top_sparse.v
// =============================================================================
