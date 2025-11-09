`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// BSR Scheduler for Sparse Matrix Multiplication
//
// This module implements the BSR (Block Sparse Row) format traversal FSM.
// It reads row pointers, column indices, and schedules 8x8 block loads
// for the systolic array, skipping empty rows for sparsity speedup.
//
// BSR Format:
// - row_ptr: [num_block_rows+1] - cumulative block counts per row
// - col_idx: [num_blocks] - column position of each block
// - data: [num_blocks, 8, 8] - the actual INT8 weight blocks
//
// FSM States:
// 1. IDLE: Wait for start signal
// 2. READ_ROW_PTR: Read row_ptr[i] and row_ptr[i+1]
// 3. CHECK_EMPTY: If row_ptr[i+1] == row_ptr[i], skip row
// 4. READ_COL_IDX: Read col_idx[block_idx]
// 5. LOAD_BLOCK: DMA 64 bytes from BRAM to systolic array
// 6. COMPUTE: Wait for systolic array to finish
// 7. NEXT_BLOCK: Increment block_idx, check if done with row
// 8. NEXT_ROW: Increment block_row, check if done with all rows
// 9. DONE: Signal completion
//////////////////////////////////////////////////////////////////////////////////

module bsr_scheduler #(
    parameter BLOCK_SIZE = 8,  // 8x8 blocks
    parameter BLOCK_BYTES = 64, // 8*8 = 64 bytes per block
    parameter MAX_BLOCK_ROWS = 16, // For FC1: 128/8 = 16
    parameter MAX_BLOCK_COLS = 1152, // For FC1: 9216/8 = 1152
    parameter MAX_BLOCKS = 18432 // 16*1152 = 18432 theoretical max
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,           // Start computation

    // BSR Metadata BRAM interfaces
    output logic [15:0] row_ptr_addr,  // Address for row_ptr BRAM
    input  logic [31:0] row_ptr_data,  // Data from row_ptr BRAM (4 bytes per entry)

    output logic [15:0] col_idx_addr,  // Address for col_idx BRAM
    input  logic [31:0] col_idx_data,  // Data from col_idx BRAM (4 bytes per entry)

    // Block data BRAM interface
    output logic [15:0] block_addr,     // Address for block data BRAM
    input  logic [7:0]  block_data[0:63], // 64 bytes (8x8 INT8 block)

    // Systolic array interface
    output logic        systolic_start,     // Start systolic computation
    output logic [7:0]  systolic_block[0:63], // 8x8 INT8 block to systolic array
    output logic [15:0] systolic_col,       // Which column this block affects
    input  logic        systolic_done,      // Systolic array finished

    // Control signals
    output logic        done,               // Scheduler finished
    output logic        busy                // Scheduler is active
);

    // FSM States
    typedef enum logic [3:0] {
        IDLE,
        READ_ROW_PTR,
        CHECK_EMPTY,
        READ_COL_IDX,
        LOAD_BLOCK,
        COMPUTE,
        NEXT_BLOCK,
        NEXT_ROW,
        DONE
    } state_t;

    state_t state, next_state;

    // Counters and registers
    logic [15:0] block_row_counter;     // Current block row (0 to num_block_rows-1)
    logic [31:0] block_start_reg;       // row_ptr[block_row]
    logic [31:0] block_end_reg;         // row_ptr[block_row+1]
    logic [31:0] block_idx_counter;     // Current block index within row
    logic [31:0] num_blocks_in_row;     // block_end - block_start

    logic [31:0] current_col_idx;       // col_idx[block_idx]

    // Configuration (set at start)
    logic [15:0] num_block_rows;        // Total block rows
    logic [15:0] num_block_cols;        // Total block columns

    // Block data buffer
    logic [7:0] block_buffer[0:63];

    // FSM Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            block_row_counter <= 0;
            block_idx_counter <= 0;
            block_start_reg <= 0;
            block_end_reg <= 0;
            num_blocks_in_row <= 0;
            current_col_idx <= 0;
            done <= 0;
            busy <= 0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy <= 1;
                        done <= 0;
                        block_row_counter <= 0;
                        // Configuration should be set here or via parameters
                        num_block_rows <= 16;  // FC1: 128/8 = 16
                        num_block_cols <= 1152; // FC1: 9216/8 = 1152
                    end
                end

                READ_ROW_PTR: begin
                    // Read row_ptr[block_row] and row_ptr[block_row+1]
                    // This happens in the same cycle via BRAM interface
                end

                CHECK_EMPTY: begin
                    num_blocks_in_row <= block_end_reg - block_start_reg;
                end

                READ_COL_IDX: begin
                    // Read col_idx[block_idx]
                    // This happens in the same cycle via BRAM interface
                end

                LOAD_BLOCK: begin
                    // Block data is read via BRAM interface
                    // Copy to buffer for systolic array
                    for (int i = 0; i < 64; i++) begin
                        block_buffer[i] <= block_data[i];
                    end
                end

                COMPUTE: begin
                    // Wait for systolic array
                end

                NEXT_BLOCK: begin
                    block_idx_counter <= block_idx_counter + 1;
                end

                NEXT_ROW: begin
                    block_row_counter <= block_row_counter + 1;
                    block_idx_counter <= 0; // Reset for next row
                end

                DONE: begin
                    busy <= 0;
                    done <= 1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;

        case (state)
            IDLE: begin
                if (start) next_state = READ_ROW_PTR;
            end

            READ_ROW_PTR: begin
                next_state = CHECK_EMPTY;
            end

            CHECK_EMPTY: begin
                if (num_blocks_in_row == 0) begin
                    // Empty row - skip to next row
                    if (block_row_counter >= num_block_rows - 1) begin
                        next_state = DONE;
                    end else begin
                        next_state = NEXT_ROW;
                    end
                end else begin
                    // Has blocks - start processing
                    next_state = READ_COL_IDX;
                end
            end

            READ_COL_IDX: begin
                next_state = LOAD_BLOCK;
            end

            LOAD_BLOCK: begin
                next_state = COMPUTE;
            end

            COMPUTE: begin
                if (systolic_done) begin
                    next_state = NEXT_BLOCK;
                end
            end

            NEXT_BLOCK: begin
                if (block_idx_counter >= block_end_reg - 1) begin
                    // Finished this row
                    if (block_row_counter >= num_block_rows - 1) begin
                        next_state = DONE;
                    end else begin
                        next_state = NEXT_ROW;
                    end
                end else begin
                    // More blocks in this row
                    next_state = READ_COL_IDX;
                end
            end

            NEXT_ROW: begin
                next_state = READ_ROW_PTR;
            end

            DONE: begin
                next_state = IDLE;
            end
        endcase
    end

    // BRAM Address Generation
    always_comb begin
        // Row pointer BRAM addresses
        case (state)
            READ_ROW_PTR: begin
                row_ptr_addr = block_row_counter;        // row_ptr[i]
                // Note: In real hardware, you'd need two reads or pipelined access
                // For simplicity, assuming synchronous BRAM with two ports
            end
            default: row_ptr_addr = 0;
        endcase

        // Column index BRAM address
        case (state)
            READ_COL_IDX: begin
                col_idx_addr = block_idx_counter;
            end
            default: col_idx_addr = 0;
        endcase

        // Block data BRAM address
        case (state)
            LOAD_BLOCK: begin
                block_addr = block_idx_counter * BLOCK_BYTES; // 64 bytes per block
            end
            default: block_addr = 0;
        endcase
    end

    // Systolic Array Interface
    always_comb begin
        systolic_start = (state == LOAD_BLOCK);
        systolic_col = current_col_idx;  // Which column this block affects

        // Send block data to systolic array
        for (int i = 0; i < 64; i++) begin
            systolic_block[i] = block_buffer[i];
        end
    end

    // Register BRAM read data
    always_ff @(posedge clk) begin
        case (state)
            READ_ROW_PTR: begin
                // In real hardware, this would be pipelined
                // For now, assume immediate read
                block_start_reg <= row_ptr_data;  // This would be row_ptr[i]
                // You'd need another read for row_ptr[i+1]
                block_end_reg <= row_ptr_data + 32'h10; // Placeholder
            end

            READ_COL_IDX: begin
                current_col_idx <= col_idx_data;
            end
        endcase
    end

endmodule
