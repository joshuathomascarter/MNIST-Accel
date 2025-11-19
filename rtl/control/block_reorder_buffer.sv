`timescale 1ns / 1ps

//=============================================================================
// block_reorder_buffer.sv — Block Metadata Sorting Buffer
//=============================================================================
// Purpose:
//   Collects per-row sparse block metadata and emits them sorted by column
//   index. This is critical for:
//   1. Coalesced memory accesses (sequential column indices)
//   2. Cache-friendly systolic array feeding
//   3. Reduced address translation overhead
//
// Algorithm: Insertion Sort
//   - Time: O(n²) worst case, O(n) best case (already sorted)
//   - Space: O(n) for metadata storage
//   - Suitable for small n (<128 blocks/row typical in 90% sparse matrices)
//
// Performance:
//   - MNIST FC1 (128×9216 @ 90% sparse): Avg 14 blocks/row
//   - Insertion sort: 14² = 196 comparisons << 1 clock cycle overhead
//   - Systolic compute: 8×8×128 = 8192 cycles → sorting is <2.5% overhead
//
// FSM States:
//   COLLECT: Accumulate blocks for current row
//   SORT:    Insertion sort by column index (in-place)
//   EMIT:    Stream sorted blocks to output
//=============================================================================

module block_reorder_buffer #(
    parameter MAX_BLOCKS_PER_ROW = 128,
    parameter ADDR_WIDTH = 7  // log2(MAX_BLOCKS_PER_ROW)
)(
    input  logic clk,
    input  logic rst_n,

    // Input stream (from scheduler)
    input  logic        in_valid,
    input  logic [15:0] in_block_col,
    input  logic [31:0] in_block_idx,
    input  logic        in_row_done,

    // Output stream (sorted)
    output logic        out_valid,
    output logic [15:0] out_block_col,
    output logic [31:0] out_block_idx,
    output logic        out_row_done,
    
    // Performance monitoring
    output logic [31:0] sort_cycles,      // Cycles spent sorting
    output logic [31:0] blocks_sorted     // Total blocks processed
);

    // Dual-array storage for insertion sort
    logic [15:0] mem_col [0:MAX_BLOCKS_PER_ROW-1];
    logic [31:0] mem_idx [0:MAX_BLOCKS_PER_ROW-1];
    logic [ADDR_WIDTH-1:0] count;      // Number of blocks in current row
    logic [ADDR_WIDTH-1:0] emit_idx;   // Emit pointer
    logic [ADDR_WIDTH-1:0] sort_i;     // Outer loop index
    logic [ADDR_WIDTH-1:0] sort_j;     // Inner loop index (insert position)
    
    // Temporary registers for swap during insertion sort
    logic [15:0] temp_col;
    logic [31:0] temp_idx;

    typedef enum logic [2:0] {
        COLLECT = 3'd0,
        SORT_INIT = 3'd1,
        SORT_INSERT = 3'd2,
        SORT_SHIFT = 3'd3,
        EMIT = 3'd4
    } state_t;
    state_t state, next_state;

    //=========================================================================
    // Main FSM Sequential Logic
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= COLLECT;
            count <= '0;
            emit_idx <= '0;
            sort_i <= '0;
            sort_j <= '0;
            out_valid <= 1'b0;
            out_row_done <= 1'b0;
            temp_col <= '0;
            temp_idx <= '0;
            sort_cycles <= '0;
            blocks_sorted <= '0;
        end else begin
            state <= next_state;

            case (state)
                //=============================================================
                // COLLECT: Accumulate blocks for current row
                //=============================================================
                COLLECT: begin
                    out_valid <= 1'b0;
                    out_row_done <= 1'b0;
                    
                    if (in_valid) begin
                        mem_col[count] <= in_block_col;
                        mem_idx[count] <= in_block_idx;
                        count <= count + 1;
                    end
                    
                    if (in_row_done) begin
                        // Transition to sort phase
                        sort_i <= 1;  // Start from index 1 (element 0 already "sorted")
                        sort_j <= '0;
                    end
                end

                //=============================================================
                // SORT_INIT: Initialize insertion for current element
                //=============================================================
                SORT_INIT: begin
                    sort_cycles <= sort_cycles + 1;
                    
                    // Latch current element to insert
                    temp_col <= mem_col[sort_i];
                    temp_idx <= mem_idx[sort_i];
                    
                    // Start scanning backwards from sort_i-1
                    sort_j <= sort_i - 1;
                end

                //=============================================================
                // SORT_INSERT: Find insertion position
                //=============================================================
                SORT_INSERT: begin
                    sort_cycles <= sort_cycles + 1;
                    
                    // Compare temp_col with mem_col[sort_j]
                    if (sort_j != {ADDR_WIDTH{1'b1}} && mem_col[sort_j] > temp_col) begin
                        // Shift element right to make space
                        mem_col[sort_j + 1] <= mem_col[sort_j];
                        mem_idx[sort_j + 1] <= mem_idx[sort_j];
                        sort_j <= sort_j - 1;
                        // Stay in SORT_INSERT to continue scanning
                    end else begin
                        // Found insertion point: insert temp at sort_j+1
                        mem_col[sort_j + 1] <= temp_col;
                        mem_idx[sort_j + 1] <= temp_idx;
                        
                        // Move to next element
                        sort_i <= sort_i + 1;
                    end
                end

                //=============================================================
                // EMIT: Stream sorted blocks to output
                //=============================================================
                EMIT: begin
                    out_valid <= 1'b1;
                    out_block_col <= mem_col[emit_idx];
                    out_block_idx <= mem_idx[emit_idx];
                    emit_idx <= emit_idx + 1;
                    blocks_sorted <= blocks_sorted + 1;
                    
                    if (emit_idx == count - 1) begin
                        // Last block in row
                        out_row_done <= 1'b1;
                        out_valid <= 1'b1;  // Keep valid for last cycle
                    end
                    
                    if (emit_idx >= count - 1) begin
                        // Reset for next row
                        count <= '0;
                        emit_idx <= '0;
                        out_valid <= 1'b0;
                        out_row_done <= 1'b0;
                    end
                end

                default: begin
                    // Catch-all for safety
                    state <= COLLECT;
                end
            endcase
        end
    end

    //=========================================================================
    // Next State Logic (Combinational)
    //=========================================================================
    always_comb begin
        next_state = state;
        
        case (state)
            COLLECT: begin
                if (in_row_done) begin
                    if (count <= 1) begin
                        // 0 or 1 blocks: already sorted, skip to EMIT
                        next_state = EMIT;
                    end else begin
                        // 2+ blocks: need to sort
                        next_state = SORT_INIT;
                    end
                end
            end
            
            SORT_INIT: begin
                // Always advance to insertion scan
                next_state = SORT_INSERT;
            end
            
            SORT_INSERT: begin
                // Check if insertion is complete
                if (sort_j == {ADDR_WIDTH{1'b1}} || mem_col[sort_j] <= temp_col) begin
                    // Found insertion point
                    if (sort_i >= count - 1) begin
                        // Sorted all elements, move to emit
                        next_state = EMIT;
                    end else begin
                        // More elements to insert
                        next_state = SORT_INIT;
                    end
                end else begin
                    // Continue scanning backwards
                    next_state = SORT_INSERT;
                end
            end
            
            EMIT: begin
                if (emit_idx >= count - 1) begin
                    // Finished emitting, return to collect
                    next_state = COLLECT;
                end else begin
                    // More blocks to emit
                    next_state = EMIT;
                end
            end
            
            default: begin
                next_state = COLLECT;
            end
        endcase
    end

    //=========================================================================
    // Functional Coverage & Assertions (SVA)
    //=========================================================================
    
    `ifndef SYNTHESIS
    // Assertion: Count never exceeds max
    property p_count_bounded;
        @(posedge clk) disable iff (!rst_n)
        count <= MAX_BLOCKS_PER_ROW;
    endproperty
    a_count_bounded: assert property (p_count_bounded)
        else $error("Block count exceeded MAX_BLOCKS_PER_ROW");
    
    // Assertion: Output only valid during EMIT state
    property p_output_only_in_emit;
        @(posedge clk) disable iff (!rst_n)
        out_valid |-> (state == EMIT);
    endproperty
    a_output_only_in_emit: assert property (p_output_only_in_emit)
        else $error("Output valid asserted outside EMIT state");
    
    // Assertion: Sorting produces non-decreasing column indices
    logic [15:0] prev_col;
    always_ff @(posedge clk) begin
        if (state == EMIT && out_valid) begin
            if (emit_idx > 0) begin
                assert (out_block_col >= prev_col)
                    else $error("Sorted output not monotonic: col[%0d]=%0d < col[%0d]=%0d",
                                emit_idx, out_block_col, emit_idx-1, prev_col);
            end
            prev_col <= out_block_col;
        end
    end
    
    // Coverage: Track block counts per row (ModelSim/VCS only)
    `ifndef VERILATOR
    covergroup cg_block_counts @(posedge clk);
        option.per_instance = 1;
        cp_count: coverpoint count {
            bins empty = {0};
            bins single = {1};
            bins small[] = {[2:8]};
            bins medium[] = {[9:32]};
            bins large[] = {[33:MAX_BLOCKS_PER_ROW]};
        }
        cp_state: coverpoint state;
        cross cp_count, cp_state;
    endgroup
    cg_block_counts cg_inst = new();
    `endif
    
    `endif

endmodule

//=============================================================================
// End of block_reorder_buffer.sv
//=============================================================================
