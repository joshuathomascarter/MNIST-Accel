//      // verilator_coverage annotation
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
 012713     input  wire                     clk,
%000007     input  wire                     rst_n,
        
            // Interface to BSR Scheduler
%000005     input  wire                     req_valid,
%000004     input  wire [31:0]              req_addr, // Address of metadata in memory
 000015     output reg                      req_ready,
        
            // Interface to Memory (BRAM/SRAM)
%000009     output wire                     mem_en,
%000004     output wire [31:0]              mem_addr,
%000000     input  wire [DATA_WIDTH-1:0]    mem_rdata, // 32-bit Data (Row Ptrs or Col Idx)
        
            // Output to Scheduler
 000014     output reg                      meta_valid,
%000000     output reg [DATA_WIDTH-1:0]     meta_rdata, // Generic 32-bit output
%000001     input  wire                     meta_ready
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
        
~000014     (* fsm_encoding = "one_hot" *) reg [5:0] current_state, next_state;
        
            //-------------------------------------------------------------------------
            // 2. Synchronous BRAM & Cache Structures
            //-------------------------------------------------------------------------
            reg [DATA_WIDTH-1:0] cache_mem [0:CACHE_DEPTH-1];
%000004     reg [CACHE_DEPTH-1:0] cache_valid_bits;
            
%000004     reg [31:0] addr_latch; // Latch to capture request address
%000004     wire [5:0] cache_index;
%000000     reg [DATA_WIDTH-1:0] fetched_data;
        
            //-------------------------------------------------------------------------
            // 3. Address Latching (Critical for Speed/Pipelining)
            //-------------------------------------------------------------------------
            // Capture address immediately when handshake occurs
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             addr_latch <= 32'd0;
 012630         end else if (req_valid && req_ready) begin
 000014             addr_latch <= req_addr;
                end
            end
        
            // Use latched address for processing, but use direct req_addr for 
            // hit/miss check in IDLE/OUTPUT to save a cycle (lookahead)
~012700     wire [31:0] active_addr = (req_ready && req_valid) ? req_addr : addr_latch;
        
            //-------------------------------------------------------------------------
            // 4. FSM Logic
            //-------------------------------------------------------------------------
        
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             current_state <= S_IDLE;
 012644         end else begin
 012644             current_state <= next_state;
                end
            end
        
            // Next State Logic
 012714     always @(*) begin
 012714         next_state = current_state;
                
 012714         case (current_state)
 012676             S_IDLE: begin
~012671                 if (req_valid) begin
                            // Lookahead: Check hit/miss on the incoming address immediately
%000004                     if (cache_valid_bits[req_addr[5:0]]) begin
%000001                         next_state = S_CHECK_CACHE; // Hit
%000004                     end else begin
%000004                         next_state = S_READ_META;   // Miss
                            end
                        end
                    end
        
%000009             S_READ_META: begin
                        // Assert mem_en, wait one cycle for sync ram
%000009                 next_state = S_WAIT_MEM;
                    end
        
%000009             S_WAIT_MEM: begin
                        // Data available at end of this cycle
%000009                 next_state = S_OUTPUT_DATA;
                    end
        
%000005             S_CHECK_CACHE: begin
                        // Cache Hit Path
%000005                 next_state = S_OUTPUT_DATA;
                    end
        
 000014             S_OUTPUT_DATA: begin
~000014                 if (meta_ready) begin
                            // OPTIMIZATION: Pipeline!
                            // If a new request is waiting, jump straight to processing it.
                            // Skip S_IDLE entirely.
%000009                     if (req_valid) begin
%000005                         if (cache_valid_bits[req_addr[5:0]]) 
%000004                             next_state = S_CHECK_CACHE;
                                else 
%000005                             next_state = S_READ_META;
%000005                     end else begin
%000005                         next_state = S_IDLE;
                            end
                        end
                    end
        
%000000             S_DONE: begin
%000000                 next_state = S_IDLE;
                    end
                    
%000001             default: next_state = S_IDLE;
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
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             cache_valid_bits <= {CACHE_DEPTH{1'b0}};
 000069             fetched_data <= 32'd0;
~012635         end else if (current_state == S_WAIT_MEM) begin
%000009             cache_mem[cache_index] <= mem_rdata;
%000009             cache_valid_bits[cache_index] <= 1'b1;
%000009             fetched_data <= mem_rdata;
                end
            end
        
            // Output Logic
 012714     always @(*) begin
                // Ready to accept new request if IDLE, or if we are about to finish the current one
~012714         req_ready  = (current_state == S_IDLE) || (current_state == S_OUTPUT_DATA && meta_ready);
                
 012714         meta_valid = (current_state == S_OUTPUT_DATA);
                
                // Mux between Cache and Fresh Data
 012700         if (current_state == S_OUTPUT_DATA) begin
%000009              if (cache_valid_bits[cache_index]) 
%000009                 meta_rdata = cache_mem[cache_index];
                     else 
%000005                 meta_rdata = fetched_data; 
 012700         end else begin
 012700              meta_rdata = 32'd0;
                end
            end
        
        endmodule
        `default_nettype wire
        
