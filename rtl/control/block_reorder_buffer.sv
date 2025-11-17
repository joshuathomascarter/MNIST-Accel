`timescale 1ns / 1ps

// block_reorder_buffer.sv
// Collects per-row block metadata and emits them sorted by column index.
// Simple insertion/bubble-sort approach suitable for small per-row counts.

module block_reorder_buffer #(
    parameter MAX_BLOCKS_PER_ROW = 128
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
    output logic        out_row_done
);

    logic [15:0] mem_col [0:MAX_BLOCKS_PER_ROW-1];
    logic [31:0] mem_idx [0:MAX_BLOCKS_PER_ROW-1];
    logic [7:0] count;
    logic [7:0] emit_idx;

    typedef enum logic [1:0] {COLLECT=2'd0, SORT=2'd1, EMIT=2'd2} state_t;
    state_t state, next_state;

    // Collection and control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= COLLECT;
            count <= 8'd0;
            emit_idx <= 8'd0;
            out_valid <= 1'b0;
            out_row_done <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                COLLECT: begin
                    if (in_valid) begin
                        mem_col[count] <= in_block_col;
                        mem_idx[count] <= in_block_idx;
                        count <= count + 1;
                    end
                    if (in_row_done) begin
                        // move to sort
                    end
                end

                SORT: begin
                    // TODO: Implement sorting logic
                end

                EMIT: begin
                    out_valid <= 1'b1;
                    out_block_col <= mem_col[emit_idx];
                    out_block_idx <= mem_idx[emit_idx];
                    emit_idx <= emit_idx + 1;
                    if (emit_idx == count - 1) begin
                        out_row_done <= 1'b1;
                        out_valid <= 1'b0;
                        // reset counters for next row
                        count <= 8'd0;
                        emit_idx <= 8'd0;
                    end
                end

                default: begin end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            COLLECT: begin
                if (in_row_done) next_state = SORT;
            end
            SORT: begin
                // after one pass, emit
                next_state = EMIT;
            end
            EMIT: begin
                if (emit_idx >= count - 1) next_state = COLLECT;
            end
        endcase
    end

endmodule
