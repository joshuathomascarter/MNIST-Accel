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
    
    // Computation counters
    logic [3:0] compute_cycle;      // 0-7 for 8 cycles
    logic [1:0] pe_tile_row;        // Which 2-row tile (0-3 for 8 rows)
    logic [1:0] pe_tile_col;        // Which 2-col tile (0-3 for 8 cols)
    
    // Block position tracking
    logic [15:0] current_block_row;
    logic [15:0] current_block_col;
    
    //========================================================================
    // Systolic Array Interface Signals
    //========================================================================
    logic                    systolic_en;
    logic                    systolic_clr;
    logic                    systolic_load_weight;
    logic [PE_ROWS*8-1:0]    systolic_a_in_flat;    // 2 rows × 8 bits
    logic [PE_COLS*8-1:0]    systolic_b_in_flat;    // 2 cols × 8 bits
    logic [PE_ROWS*PE_COLS*32-1:0] systolic_c_out_flat; // 4 × 32 bits
    
    // Unpack systolic output
    logic signed [ACC_WIDTH-1:0] systolic_acc [0:PE_ROWS-1][0:PE_COLS-1];
    
    // Final accumulated results (multiple tiles → full 2×8 output)
    logic signed [ACC_WIDTH-1:0] pe_acc [0:PE_ROWS-1][0:BLOCK_W-1];
    
    //========================================================================
    // Instantiate Base Systolic Array
    //========================================================================
    systolic_array #(
        .N_ROWS(PE_ROWS),
        .N_COLS(PE_COLS),
        .PIPE(1),
        .SAT(0)
    ) u_systolic (
        .clk(clk),
        .rst_n(rst_n),
        .en(systolic_en),
        .clr(systolic_clr),
        .load_weight(systolic_load_weight),
        .a_in_flat(systolic_a_in_flat),
        .b_in_flat(systolic_b_in_flat),
        .c_out_flat(systolic_c_out_flat)
    );
    
    // Unpack systolic array outputs
    always_comb begin
        for (int i = 0; i < PE_ROWS; i++) begin
            for (int j = 0; j < PE_COLS; j++) begin
                systolic_acc[i][j] = systolic_c_out_flat[(i*PE_COLS+j)*32 +: 32];
            end
        end
    end
    
    //========================================================================
    // FSM State Machine
    //========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_cycle <= '0;
            pe_tile_row <= '0;
            pe_tile_col <= '0;
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
                        pe_tile_row <= '0;
                        pe_tile_col <= '0;
                        
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
                    
                    // Also load activations
                    if (act_valid) begin
                        for (int k = 0; k < BLOCK_H; k++) begin
                            act_buffer[k] <= act_data[k];
                        end
                    end
                end
                
                COMPUTE: begin
                    // Feed systolic array cycle-by-cycle
                    // Process 2×2 tile at a time, need 16 iterations for 8×8 block
                    
                    if (compute_cycle == BLOCK_H - 1) begin
                        // Accumulate results from systolic array
                        for (int i = 0; i < PE_ROWS; i++) begin
                            for (int j = 0; j < PE_COLS; j++) begin
                                pe_acc[i][pe_tile_col * PE_COLS + j] <= 
                                    pe_acc[i][pe_tile_col * PE_COLS + j] + systolic_acc[i][j];
                            end
                        end
                        
                        // Move to next tile
                        compute_cycle <= '0;
                        if (pe_tile_col == (BLOCK_W / PE_COLS) - 1) begin
                            pe_tile_col <= '0;
                            // Done with this row tile
                        end else begin
                            pe_tile_col <= pe_tile_col + 1;
                        end
                    end else begin
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
    // Systolic Array Control Logic
    //========================================================================
    always_comb begin
        systolic_en = (state == COMPUTE);
        systolic_clr = (state == IDLE);
        systolic_load_weight = (state == LOAD_BLOCK);
        
        // Pack activation inputs (different activation for each PE row)
        // Cycle 0: act[0], act[1]
        // Cycle 1: act[2], act[3]
        // Cycle 2: act[4], act[5]
        // Cycle 3: act[6], act[7]
        systolic_a_in_flat[0*8 +: 8] = act_buffer[compute_cycle * PE_ROWS + 0];
        systolic_a_in_flat[1*8 +: 8] = act_buffer[compute_cycle * PE_ROWS + 1];
        
        // Pack weight inputs (indexed by output row, K dimension)
        // weight_buffer[output_row][k_idx]
        // For tile processing output rows [tile_row*2, tile_row*2+1]
        // Each column gets weights from its corresponding output row
        // Cycle advances through K dimension in pairs (0-1, 2-3, 4-5, 6-7)
        for (int j = 0; j < PE_COLS; j++) begin
            systolic_b_in_flat[j*8 +: 8] = weight_buffer[pe_tile_col * PE_COLS + j][compute_cycle * PE_ROWS + 0];
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
                // Check if we've completed all tiles
                if (compute_cycle == BLOCK_H - 1 && 
                    pe_tile_col == (BLOCK_W / PE_COLS) - 1) begin
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
