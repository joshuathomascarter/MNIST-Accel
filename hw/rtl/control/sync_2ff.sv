`timescale 1ns / 1ps

//=============================================================================
// 2-FF Synchronizer for Clock Domain Crossing
//=============================================================================
// 
// PURPOSE:
// --------
// Safely transfer multi-bit signals from one clock domain to another.
// Common use case: Status signals (done, busy) crossing from 200 MHz to 50 MHz.
//
// OPERATION:
// ----------
// 1. Input signal (src_clk domain) passes through 2 flip-flops (dst_clk domain)
// 2. First FF absorbs metastability (MTBF > 1000 years @ typical conditions)
// 3. Second FF provides stable output
//
// METASTABILITY PROTECTION:
// --------------------------
// - ASYNC_REG attribute ensures FFs placed close together (minimize routing delay)
// - MTBF (Mean Time Between Failures) with 2 FFs:
//   * MTBF = (e^(Tr/τ)) / (fc × fd × Tw)
//   * Tr = resolution time (one clock cycle @ dst_clk)
//   * τ = metastability time constant (~100ps for modern FPGAs)
//   * fc = frequency of clock (dst_clk = 50 MHz)
//   * fd = frequency of data transitions (worst-case = 200 MHz)
//   * Tw = metastability window (~50ps)
//   * MTBF ≈ 10^15 years (safe for all practical applications)
//
// LIMITATIONS:
// ------------
// - Input signal must be stable for ≥ 2 dst_clk cycles (no guarantee on transition)
// - For multi-bit buses, use gray coding or async FIFO instead
// - NOT suitable for pulse signals (use pulse_sync.sv instead)
//
//=============================================================================

module sync_2ff #(
    parameter WIDTH = 1  // Bit width of signal to synchronize
)(
    input  wire                 dst_clk,      // Destination clock domain
    input  wire                 dst_rst_n,    // Destination reset (async, active-low)
    input  wire [WIDTH-1:0]     src_signal,   // Input signal (src_clk domain)
    output wire [WIDTH-1:0]     dst_signal    // Output signal (dst_clk domain)
);

    // ========================================================================
    // 2-FF Synchronizer Chain
    // ========================================================================
    (* ASYNC_REG = "TRUE" *) reg [WIDTH-1:0] meta_ff;
    (* ASYNC_REG = "TRUE" *) reg [WIDTH-1:0] sync_ff;
    
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            meta_ff <= {WIDTH{1'b0}};
            sync_ff <= {WIDTH{1'b0}};
        end else begin
            meta_ff <= src_signal;  // Metastability stage (may glitch)
            sync_ff <= meta_ff;     // Stable output (no glitches)
        end
    end
    
    assign dst_signal = sync_ff;

endmodule


//=============================================================================
// Async FIFO for Multi-Bit Data Transfer
//=============================================================================
// 
// PURPOSE:
// --------
// Safely transfer multi-bit data (e.g., tile configuration) from one clock
// domain to another. Uses gray-coded pointers to prevent multi-bit transition
// glitches.
//
// USE CASES:
// ----------
// - Configuration data: 50 MHz control → 200 MHz datapath
// - Result data: 200 MHz datapath → 50 MHz control
//
// FIFO DEPTH:
// -----------
// - Min depth = max(WR_RATE / RD_RATE, 4)
// - For 50 MHz → 200 MHz: depth = 4 (200/50 = 4×)
// - For 200 MHz → 50 MHz: depth = 16 (need buffering)
//
//=============================================================================

module async_fifo #(
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 16  // Must be power of 2
)(
    // Write interface (source clock domain)
    input  wire                     wr_clk,
    input  wire                     wr_rst_n,
    input  wire                     wr_en,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     wr_full,
    
    // Read interface (destination clock domain)
    input  wire                     rd_clk,
    input  wire                     rd_rst_n,
    input  wire                     rd_en,
    output reg  [DATA_WIDTH-1:0]    rd_data,
    output wire                     rd_empty
);

    localparam ADDR_WIDTH = $clog2(FIFO_DEPTH);
    
    // ========================================================================
    // FIFO Memory
    // ========================================================================
    reg [DATA_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];
    
    // ========================================================================
    // Write Pointer (binary and gray)
    // ========================================================================
    reg [ADDR_WIDTH:0] wr_ptr_bin;
    reg [ADDR_WIDTH:0] wr_ptr_gray;
    
    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            wr_ptr_bin  <= 0;
            wr_ptr_gray <= 0;
        end else if (wr_en && !wr_full) begin
            wr_ptr_bin  <= wr_ptr_bin + 1;
            wr_ptr_gray <= (wr_ptr_bin + 1) ^ ((wr_ptr_bin + 1) >> 1); // Binary to gray
        end
    end
    
    // Write data
    always @(posedge wr_clk) begin
        if (wr_en && !wr_full)
            fifo_mem[wr_ptr_bin[ADDR_WIDTH-1:0]] <= wr_data;
    end
    
    // ========================================================================
    // Read Pointer (binary and gray)
    // ========================================================================
    reg [ADDR_WIDTH:0] rd_ptr_bin;
    reg [ADDR_WIDTH:0] rd_ptr_gray;
    
    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            rd_ptr_bin  <= 0;
            rd_ptr_gray <= 0;
        end else if (rd_en && !rd_empty) begin
            rd_ptr_bin  <= rd_ptr_bin + 1;
            rd_ptr_gray <= (rd_ptr_bin + 1) ^ ((rd_ptr_bin + 1) >> 1); // Binary to gray
        end
    end
    
    // Read data
    always @(posedge rd_clk) begin
        if (rd_en && !rd_empty)
            rd_data <= fifo_mem[rd_ptr_bin[ADDR_WIDTH-1:0]];
    end
    
    // ========================================================================
    // Gray-coded Pointer Synchronization
    // ========================================================================
    (* ASYNC_REG = "TRUE" *) reg [ADDR_WIDTH:0] wr_ptr_gray_sync1, wr_ptr_gray_sync2;
    (* ASYNC_REG = "TRUE" *) reg [ADDR_WIDTH:0] rd_ptr_gray_sync1, rd_ptr_gray_sync2;
    
    // Sync write pointer to read domain
    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            wr_ptr_gray_sync1 <= 0;
            wr_ptr_gray_sync2 <= 0;
        end else begin
            wr_ptr_gray_sync1 <= wr_ptr_gray;
            wr_ptr_gray_sync2 <= wr_ptr_gray_sync1;
        end
    end
    
    // Sync read pointer to write domain
    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            rd_ptr_gray_sync1 <= 0;
            rd_ptr_gray_sync2 <= 0;
        end else begin
            rd_ptr_gray_sync1 <= rd_ptr_gray;
            rd_ptr_gray_sync2 <= rd_ptr_gray_sync1;
        end
    end
    
    // ========================================================================
    // Full/Empty Logic
    // ========================================================================
    assign wr_full  = (wr_ptr_gray == {~rd_ptr_gray_sync2[ADDR_WIDTH:ADDR_WIDTH-1], 
                                        rd_ptr_gray_sync2[ADDR_WIDTH-2:0]});
    assign rd_empty = (rd_ptr_gray == wr_ptr_gray_sync2);

endmodule
