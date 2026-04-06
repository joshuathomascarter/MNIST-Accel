// max_pool_2x2.sv — Bypassable 2×2 Max Pooling Unit
// =============================================================================
//
// Applies 2×2 non-overlapping max pooling to a stream of INT8 values.
// Used between output_accumulator and output_bram_buffer during CNN layers
// that require spatial downsampling (conv1→conv2 in MNIST).
//
// OPERATION
// =========
// Input:  H×W feature map (INT8, streamed row-major in 64-bit words)
// Output: (H/2)×(W/2) feature map (INT8, same packing)
//
// For each 2×2 window:
//   out[y][x] = max(in[2y][2x], in[2y][2x+1], in[2y+1][2x], in[2y+1][2x+1])
//
// IMPLEMENTATION
// ==============
// Uses a single line buffer to store one row. When the second row arrives,
// computes max of the 2×2 window and emits the result.
//
// BYPASS
// ======
// When bypass=1, input is forwarded directly to output (FC layers, no pooling).
//
// RESOURCE USAGE
// ==============
// ~80 LUTs, 1 BRAM18 (line buffer for max row width of 128), 0 DSPs
//
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module max_pool_2x2 #(
    parameter DATA_W    = 8,         // Element width (INT8)
    parameter MAX_W     = 128,       // Maximum feature map width
    parameter PACK_W    = 64         // Packed word width (8 × INT8)
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // =========================================================================
    // Configuration
    // =========================================================================
    input  wire                 bypass,      // 1 = pass-through (no pooling)
    input  wire [15:0]          feat_h,      // Input feature map height
    input  wire [15:0]          feat_w,      // Input feature map width

    // =========================================================================
    // Input Interface (from quantized accumulator output)
    // =========================================================================
    input  wire                 in_valid,
    input  wire [PACK_W-1:0]   in_data,     // 8 × INT8 packed
    output wire                 in_ready,

    // =========================================================================
    // Output Interface (to output_bram_buffer)
    // =========================================================================
    output reg                  out_valid,
    output reg  [PACK_W-1:0]   out_data,
    input  wire                 out_ready,

    // =========================================================================
    // Status
    // =========================================================================
    output wire                 pool_active
);

    // =========================================================================
    // Bypass Mode
    // =========================================================================
    // In bypass mode, just forward the data through
    wire bypass_valid = bypass & in_valid;
    wire bypass_ready = bypass & out_ready;

    assign in_ready = bypass ? out_ready : pool_ready_int;
    assign pool_active = !bypass && (row_cnt != 0 || col_cnt != 0);

    // =========================================================================
    // Line Buffer (stores one row of packed words for even-row compare)
    // =========================================================================
    // Maximum words per row = MAX_W / 8 = 16 (for 128-wide feature map)
    localparam LINE_BUF_DEPTH = MAX_W / (PACK_W / DATA_W);  // 128/8 = 16
    localparam LB_ADDR_W = $clog2(LINE_BUF_DEPTH);

    reg [PACK_W-1:0] line_buf [0:LINE_BUF_DEPTH-1];
    reg [LB_ADDR_W-1:0] lb_wr_addr;
    reg [LB_ADDR_W-1:0] lb_rd_addr;

    // =========================================================================
    // Position Counters
    // =========================================================================
    reg [15:0] row_cnt;     // Current input row (0 to feat_h-1)
    reg [15:0] col_cnt;     // Current input column in packed-word units
    wire [15:0] words_per_row = {3'b000, feat_w[15:3]};  // feat_w / 8

    wire pool_ready_int;
    reg  pool_out_valid_r;

    // Even/odd row tracking
    wire is_even_row = ~row_cnt[0];
    wire is_odd_row  =  row_cnt[0];

    assign pool_ready_int = !pool_out_valid_r || out_ready;

    // =========================================================================
    // Max Function for signed INT8
    // =========================================================================
    function automatic [DATA_W-1:0] smax;
        input [DATA_W-1:0] a;
        input [DATA_W-1:0] b;
        begin
            smax = ($signed(a) > $signed(b)) ? a : b;
        end
    endfunction

    // =========================================================================
    // Per-Element 2×2 Max Pool
    // =========================================================================
    // Computes max of 4 INT8 values from current packed word + line buffer word
    // Step 1: horizontal max (pairs within each row)
    // Step 2: vertical max (across two rows)

    reg [PACK_W-1:0] line_buf_data;   // Data read from line buffer
    reg [PACK_W-1:0] pool_result;

    // Compute pooled output: max across 2×2 windows for all 8 elements
    // Input packing: [e7|e6|e5|e4|e3|e2|e1|e0]
    // Pool pairs: (e0,e1), (e2,e3), (e4,e5), (e6,e7)
    // Then vertical max with line_buf pair results
    integer i;
    reg [DATA_W-1:0] cur_pair_max [0:3];   // Horizontal max of current row pairs
    reg [DATA_W-1:0] buf_pair_max [0:3];   // Horizontal max of buffered row pairs
    reg [DATA_W-1:0] final_max [0:3];      // Vertical max across rows

    always @(*) begin
        for (i = 0; i < 4; i = i + 1) begin
            // Horizontal max: max(elem[2i], elem[2i+1]) from current word
            cur_pair_max[i] = smax(
                in_data[(2*i)*DATA_W +: DATA_W],
                in_data[(2*i+1)*DATA_W +: DATA_W]
            );
            // Horizontal max from line buffer (even row stored previously)
            buf_pair_max[i] = smax(
                line_buf_data[(2*i)*DATA_W +: DATA_W],
                line_buf_data[(2*i+1)*DATA_W +: DATA_W]
            );
            // Vertical max across even and odd rows
            final_max[i] = smax(cur_pair_max[i], buf_pair_max[i]);
        end
        // Pack 4 pooled results into lower 32 bits of output word
        pool_result = {{(PACK_W-4*DATA_W){1'b0}},
                       final_max[3], final_max[2], final_max[1], final_max[0]};
    end

    // =========================================================================
    // Main Control Logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_cnt         <= 16'd0;
            col_cnt         <= 16'd0;
            lb_wr_addr      <= {LB_ADDR_W{1'b0}};
            lb_rd_addr      <= {LB_ADDR_W{1'b0}};
            line_buf_data   <= {PACK_W{1'b0}};
            out_valid       <= 1'b0;
            out_data        <= {PACK_W{1'b0}};
            pool_out_valid_r <= 1'b0;
        end else if (bypass) begin
            // Bypass mode: direct passthrough
            out_valid <= in_valid;
            out_data  <= in_data;
            pool_out_valid_r <= 1'b0;
            row_cnt <= 16'd0;
            col_cnt <= 16'd0;
        end else begin
            // Handle output handshake
            if (out_ready) begin
                out_valid <= 1'b0;
                pool_out_valid_r <= 1'b0;
            end

            if (in_valid && pool_ready_int) begin
                if (is_even_row) begin
                    // Even row: store to line buffer for later comparison
                    line_buf[lb_wr_addr] <= in_data;
                    lb_wr_addr <= lb_wr_addr + {{(LB_ADDR_W-1){1'b0}}, 1'b1};
                end else begin
                    // Odd row: read line buffer, compute 2×2 max, emit result
                    line_buf_data <= line_buf[lb_rd_addr];
                    lb_rd_addr    <= lb_rd_addr + {{(LB_ADDR_W-1){1'b0}}, 1'b1};

                    out_valid        <= 1'b1;
                    out_data         <= pool_result;
                    pool_out_valid_r <= 1'b1;
                end

                // Advance position
                if (col_cnt == words_per_row - 16'd1) begin
                    col_cnt    <= 16'd0;
                    lb_wr_addr <= {LB_ADDR_W{1'b0}};
                    lb_rd_addr <= {LB_ADDR_W{1'b0}};
                    if (row_cnt == feat_h - 16'd1) begin
                        row_cnt <= 16'd0;
                    end else begin
                        row_cnt <= row_cnt + 16'd1;
                    end
                end else begin
                    col_cnt <= col_cnt + 16'd1;
                end
            end
        end
    end

endmodule

`default_nettype wire
