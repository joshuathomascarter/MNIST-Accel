`timescale 1ns / 1ps

//================================================================================
// Sparse Systolic Array for BSR Sparse GEMM
//================================================================================
//
// Author: Hardware Acceleration Team
// Purpose: 2×2 PE systolic array optimized for 8×8 sparse block processing
//          Integrates with BSR scheduler for 8-10× speedup on sparse networks
//
// ARCHITECTURE:
// ============
//
//     ┌─────────────────────────────────────────────────────┐
//     │         SYSTOLIC ARRAY (2×2 PEs)                    │
//     │                                                      │
//     │     A input (activations)                           │
//     │          ↓                                          │
//     │     ┌────────┐      B input (weights)               │
//     │     │ PE(0,0)│←──────                               │
//     │     └───┬─→──┘                                      │
//     │         ↓  ↘                                        │
//     │     ┌────────┐                                      │
//     │     │ PE(1,0)│                                      │
//     │     └───┬────┘                                      │
//     │         ↓                                           │
//     │    Accumulator                                      │
//     └─────────────────────────────────────────────────────┘
//
// OPERATION:
// ==========
// 1. Receives 8×8 INT8 block from BSR scheduler
// 2. Processes 2 rows at a time (2×2 PE array)
// 3. INT8 × INT8 → INT32 MAC operations
// 4. Accumulates partial sums
// 5. Outputs INT32 results for dequantization
//
// TIMING (8×8 block):
// ===================
// - Block load: 1 cycle (parallel from scheduler)
// - Computation: 8 cycles (stream through PEs)
// - Total latency: ~10 cycles per block
// - Throughput: 4 MACs/cycle (2×2 PEs)
//
// SPARSITY BENEFIT:
// =================
// Dense: Process 18,432 blocks (MNIST FC1)
// 90% Sparse: Process 1,843 blocks
// Speedup: 10× (skip 90% of blocks!)
//
//================================================================================

module systolic_array_sparse #(
    parameter PE_ROWS = 2,          // 2×2 PE array
    parameter PE_COLS = 2,
    parameter DATA_WIDTH = 8,       // INT8 input
    parameter ACC_WIDTH = 32,       // INT32 accumulator
    parameter BLOCK_H = 8,          // Process 8×8 blocks
    parameter BLOCK_W = 8
)(
    // Clock and reset
    input  logic clk,
    input  logic rst_n,
    
    // Control interface from BSR scheduler
    input  logic                    valid_in,       // Block data valid
    input  logic [DATA_WIDTH-1:0]   block_data [0:63], // 8×8 INT8 block
    input  logic [15:0]             block_row,      // Block row index
    input  logic [15:0]             block_col,      // Block column index
    output logic                    ready,          // Ready for new block
    
    // Activation input (dense INT8 matrix A)
    input  logic [DATA_WIDTH-1:0]   act_data [0:BLOCK_H-1], // 8 activation values
    input  logic                    act_valid,
    
    // Output interface
    output logic                    result_valid,
    output logic [ACC_WIDTH-1:0]    result_data [0:PE_ROWS-1][0:BLOCK_W-1], // 2×8 INT32 results
    output logic [15:0]             result_block_row,
    output logic [15:0]             result_block_col,
    input  logic                    result_ready,   // Downstream ready
    
    // Status
    output logic                    done,           // Block computation done
    output logic                    busy            // Array is computing
);

    //========================================================================
    // PE Array Internal State
    //========================================================================
    typedef enum logic [2:0] {
        IDLE,
        LOAD_BLOCK,
        COMPUTE,
        OUTPUT
    } state_t;
    
    state_t state, next_state;
    
    //========================================================================
    // PE Array Registers
    //========================================================================
    // Weight buffer (8×8 block stored row-major)
    logic [DATA_WIDTH-1:0] weight_buffer [0:BLOCK_H-1][0:BLOCK_W-1];
    
    // Activation buffer (sliding window)
    logic [DATA_WIDTH-1:0] act_buffer [0:BLOCK_H-1];
    
    // PE accumulators (2 PEs × 8 outputs each = 16 accumulators)
    logic signed [ACC_WIDTH-1:0] pe_acc [0:PE_ROWS-1][0:BLOCK_W-1];
    
    // Computation counters
    logic [3:0] compute_cycle;      // 0-7 for 8 cycles
    logic [2:0] pe_row_idx;         // Which PE row (0-1)
    
    // Block position tracking
    logic [15:0] current_block_row;
    logic [15:0] current_block_col;
    
    //========================================================================
    // FSM State Machine
    //========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_cycle <= '0;
            pe_row_idx <= '0;
            busy <= 1'b0;
            done <= 1'b0;
            current_block_row <= '0;
            current_block_col <= '0;
            
            // Clear accumulators
            for (int i = 0; i < PE_ROWS; i++) begin
                for (int j = 0; j < BLOCK_W; j++) begin
                    pe_acc[i][j] <= '0;
                end
            end
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (valid_in) begin
                        busy <= 1'b1;
                        done <= 1'b0;
                        current_block_row <= block_row;
                        current_block_col <= block_col;
                        compute_cycle <= '0;
                        pe_row_idx <= '0;
                        
                        // Clear accumulators for new block
                        for (int i = 0; i < PE_ROWS; i++) begin
                            for (int j = 0; j < BLOCK_W; j++) begin
                                pe_acc[i][j] <= '0;
                            end
                        end
                    end
                end
                
                LOAD_BLOCK: begin
                    // Load 8×8 block into weight buffer
                    for (int i = 0; i < BLOCK_H; i++) begin
                        for (int j = 0; j < BLOCK_W; j++) begin
                            weight_buffer[i][j] <= block_data[i * BLOCK_W + j];
                        end
                    end
                end
                
                COMPUTE: begin
                    // MAC operations: PE[i][j] += act[k] * weight[i][k]
                    // Process 2 rows (PE_ROWS) in parallel
                    
                    if (act_valid) begin
                        // Load activation slice
                        for (int k = 0; k < BLOCK_H; k++) begin
                            act_buffer[k] <= act_data[k];
                        end
                        
                        // Compute for 2 PE rows
                        for (int pe_r = 0; pe_r < PE_ROWS; pe_r++) begin
                            // Each PE computes dot product with 8 weights
                            for (int out_col = 0; out_col < BLOCK_W; out_col++) begin
                                // MAC: accumulator += activation * weight
                                pe_acc[pe_r][out_col] <= pe_acc[pe_r][out_col] + 
                                    ($signed(act_buffer[compute_cycle]) * 
                                     $signed(weight_buffer[pe_r * 4 + compute_cycle][out_col]));
                            end
                        end
                        
                        compute_cycle <= compute_cycle + 1;
                    end
                end
                
                OUTPUT: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    //========================================================================
    // FSM Next State Logic
    //========================================================================
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (valid_in) next_state = LOAD_BLOCK;
            end
            
            LOAD_BLOCK: begin
                next_state = COMPUTE;
            end
            
            COMPUTE: begin
                if (compute_cycle == BLOCK_H - 1) begin
                    next_state = OUTPUT;
                end
            end
            
            OUTPUT: begin
                if (result_ready) begin
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    //========================================================================
    // Output Interface
    //========================================================================
    assign ready = (state == IDLE);
    assign result_valid = (state == OUTPUT);
    assign result_block_row = current_block_row;
    assign result_block_col = current_block_col;
    
    // Connect accumulators to output
    always_comb begin
        for (int i = 0; i < PE_ROWS; i++) begin
            for (int j = 0; j < BLOCK_W; j++) begin
                result_data[i][j] = pe_acc[i][j];
            end
        end
    end
    
    //========================================================================
    // Performance Counters (Optional)
    //========================================================================
    logic [31:0] blocks_computed;
    logic [63:0] total_macs;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            blocks_computed <= '0;
            total_macs <= '0;
        end else begin
            if (state == OUTPUT && result_ready) begin
                blocks_computed <= blocks_computed + 1;
                total_macs <= total_macs + (PE_ROWS * BLOCK_W * BLOCK_H);
            end
        end
    end
    
    //========================================================================
    // Assertions for Verification
    //========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (state == LOAD_BLOCK) begin
            assert (valid_in) else $error("LOAD_BLOCK without valid_in");
        end
        
        if (state == COMPUTE && compute_cycle >= BLOCK_H) begin
            $error("compute_cycle overflow: %0d", compute_cycle);
        end
        
        if (result_valid && !result_ready) begin
            // Result output backpressure
            $warning("Systolic array stalled - downstream not ready");
        end
    end
    // synthesis translate_on

endmodule
