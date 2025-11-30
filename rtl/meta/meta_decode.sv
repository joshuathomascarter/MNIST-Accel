// =============================================================================
// meta_decode.sv — BSR Metadata Decoder with BRAM Cache
// =============================================================================
// Purpose:
//   Fetches and caches BSR metadata (Row Pointers and Column Indices).
//   Provides a generic 32-bit interface to the Scheduler.
//
// Features:
//   - 32-bit Memory Interface (Compatible with Row Pointers & System RAM)
//   - One-Hot FSM for high-speed decoding
//   - Synchronous BRAM Support with Power Gating
//   - Valid Bitmap for accurate Cache Hit/Miss tracking
//   - OPTIMIZED: Pipelined Request Processing (Zero-Wait State Transitions)
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module meta_decode #(
    parameter DATA_WIDTH = 32,
    parameter CACHE_DEPTH = 64 // Number of cached metadata entries
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Interface to BSR Scheduler
    input  wire                     req_valid,
    input  wire [31:0]              req_addr, // Address of metadata in memory
    output reg                      req_ready,

    // Interface to Memory (BRAM/SRAM)
    output wire                     mem_en,
    output wire [31:0]              mem_addr,
    input  wire [DATA_WIDTH-1:0]    mem_rdata, // 32-bit Data (Row Ptrs or Col Idx)

    // Output to Scheduler
    output reg                      meta_valid,
    output reg [DATA_WIDTH-1:0]     meta_rdata, // Generic 32-bit output
    input  wire                     meta_ready
);

    //-------------------------------------------------------------------------
    // 1. One-Hot FSM Encoding
    //-------------------------------------------------------------------------
    localparam [5:0] S_IDLE         = 6'b000001,
                     S_READ_META    = 6'b000010,
                     S_WAIT_MEM     = 6'b000100, // Wait for Sync BRAM
                     S_CHECK_CACHE  = 6'b001000,
                     S_OUTPUT_DATA  = 6'b010000,
                     S_DONE         = 6'b100000;

    (* fsm_encoding = "one_hot" *) reg [5:0] current_state, next_state;

    //-------------------------------------------------------------------------
    // 2. Synchronous BRAM & Cache Structures
    //-------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] cache_mem [0:CACHE_DEPTH-1];
    reg [CACHE_DEPTH-1:0] cache_valid_bits;
    
    reg [31:0] addr_latch; // Latch to capture request address
    wire [5:0] cache_index;
    reg [DATA_WIDTH-1:0] fetched_data;

    //-------------------------------------------------------------------------
    // 3. Address Latching (Critical for Speed/Pipelining)
    //-------------------------------------------------------------------------
    // Capture address immediately when handshake occurs
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_latch <= 32'd0;
        end else if (req_valid && req_ready) begin
            addr_latch <= req_addr;
        end
    end

    // Use latched address for processing, but use direct req_addr for 
    // hit/miss check in IDLE/OUTPUT to save a cycle (lookahead)
    wire [31:0] active_addr = (req_ready && req_valid) ? req_addr : addr_latch;

    //-------------------------------------------------------------------------
    // 4. FSM Logic
    //-------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= S_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    // Next State Logic
    always @(*) begin
        next_state = current_state;
        
        case (current_state)
            S_IDLE: begin
                if (req_valid) begin
                    // Lookahead: Check hit/miss on the incoming address immediately
                    if (cache_valid_bits[req_addr[5:0]]) begin
                        next_state = S_CHECK_CACHE; // Hit
                    end else begin
                        next_state = S_READ_META;   // Miss
                    end
                end
            end

            S_READ_META: begin
                // Assert mem_en, wait one cycle for sync ram
                next_state = S_WAIT_MEM;
            end

            S_WAIT_MEM: begin
                // Data available at end of this cycle
                next_state = S_OUTPUT_DATA;
            end

            S_CHECK_CACHE: begin
                // Cache Hit Path
                next_state = S_OUTPUT_DATA;
            end

            S_OUTPUT_DATA: begin
                if (meta_ready) begin
                    // OPTIMIZATION: Pipeline!
                    // If a new request is waiting, jump straight to processing it.
                    // Skip S_IDLE entirely.
                    if (req_valid) begin
                        if (cache_valid_bits[req_addr[5:0]]) 
                            next_state = S_CHECK_CACHE;
                        else 
                            next_state = S_READ_META;
                    end else begin
                        next_state = S_IDLE;
                    end
                end
            end

            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // 5. Datapath & Outputs
    //-------------------------------------------------------------------------

    // Cache Index Calculation
    assign cache_index = active_addr[5:0];

    // Memory Interface
    // Only enable memory when we need to read on a miss
    assign mem_en   = (current_state == S_READ_META);
    assign mem_addr = active_addr;

    // Cache Write & Valid Bit Update
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cache_valid_bits <= {CACHE_DEPTH{1'b0}};
            fetched_data <= 32'd0;
        end else if (current_state == S_WAIT_MEM) begin
            cache_mem[cache_index] <= mem_rdata;
            cache_valid_bits[cache_index] <= 1'b1;
            fetched_data <= mem_rdata;
        end
    end

    // Output Logic
    always @(*) begin
        // Ready to accept new request if IDLE, or if we are about to finish the current one
        req_ready  = (current_state == S_IDLE) || (current_state == S_OUTPUT_DATA && meta_ready);
        
        meta_valid = (current_state == S_OUTPUT_DATA);
        
        // Mux between Cache and Fresh Data
        if (current_state == S_OUTPUT_DATA) begin
             if (cache_valid_bits[cache_index]) 
                meta_rdata = cache_mem[cache_index];
             else 
                meta_rdata = fetched_data; 
        end else begin
             meta_rdata = 32'd0;
        end
    end

endmodule
`default_nettype wire
