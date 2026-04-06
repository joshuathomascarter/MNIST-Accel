// output_bram_ctrl.sv — Multi-layer output BRAM controller
// =============================================================================
//
// FSM that controls the data path from output_accumulator → output_bram_buffer
// and decides whether to:
//   (a) FEEDBACK: drain BRAM → act_buffer (intermediate layers)
//   (b) DRAIN_TO_DDR: trigger out_dma → DDR (last layer)
//
// Layer Sequencing (SW-triggered):
//   1. Software sets layer_total, layer_current=0, starts compute
//   2. Accumulator finishes a tile → data written to BRAM via this ctrl
//   3. When all tiles for a layer complete (sched_done):
//      - If layer_current < layer_total-1: FEEDBACK activations, signal layer_done
//      - If layer_current == layer_total-1: DRAIN_TO_DDR, signal last_layer_done
//   4. Software increments layer_current, loads new weights, re-starts
//
// Pooling:
//   When pool_en=1, a max_pool_2x2 stage is applied during the write path
//   (data from accumulator is pooled before being written to BRAM).
//   The pooling is done externally — this module just controls the flow.
//
// Resource Usage: ~120 LUTs, 0 DSPs, 0 BRAM (control logic only)
//
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module output_bram_ctrl #(
    parameter DATA_W     = 64,       // Word width (8 × INT8)
    parameter ADDR_W     = 10,       // BRAM address bits
    parameter ACT_ADDR_W = 10,       // Activation buffer address bits
    parameter ACT_DATA_W = 128,      // Activation buffer data width (16 × INT8)
    parameter NUM_ACCS   = 256       // Number of accumulators (16×16)
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // =========================================================================
    // Layer Configuration (from CSR)
    // =========================================================================
    input  wire [7:0]              layer_total,    // Total layers in network
    input  wire [7:0]              layer_current,  // Current layer index (0-based)
    input  wire                    pool_en,        // Enable 2×2 max pooling on output
    input  wire [15:0]             output_h,       // Output feature map height
    input  wire [15:0]             output_w,       // Output feature map width

    // =========================================================================
    // Accumulator → BRAM Write Path (capture quantized output)
    // =========================================================================
    // The accumulator's DMA read interface is reused. This ctrl replaces
    // out_dma as the reader when in intermediate layers.
    output reg                     accum_rd_en,
    output reg  [ADDR_W-1:0]      accum_rd_addr,
    input  wire [DATA_W-1:0]      accum_rd_data,
    input  wire                    accum_ready,    // Inactive bank ready

    // =========================================================================
    // BRAM Write Port (to output_bram_buffer)
    // =========================================================================
    output reg                     bram_wr_en,
    output reg  [ADDR_W-1:0]      bram_wr_addr,
    output reg  [DATA_W-1:0]      bram_wr_data,
    output wire                    bram_bank_sel,  // Which bank is active for writing

    // =========================================================================
    // BRAM Read Port (from output_bram_buffer — for feedback)
    // =========================================================================
    output reg                     bram_rd_en,
    output reg  [ADDR_W-1:0]      bram_rd_addr,
    input  wire [DATA_W-1:0]      bram_rd_data,
    input  wire                    bram_rd_valid,

    // =========================================================================
    // Feedback → Activation Buffer Write Port
    // =========================================================================
    output reg                     fb_act_we,
    output reg  [ACT_ADDR_W-1:0]  fb_act_waddr,
    output reg  [ACT_DATA_W-1:0]  fb_act_wdata,

    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                    sched_done,     // All tiles for this layer done
    input  wire                    start,          // Begin capture (from start_pulse)

    // =========================================================================
    // Out DMA Control (for last layer only)
    // =========================================================================
    output wire                    out_dma_trigger,// Trigger out_dma to drain to DDR
    input  wire                    out_dma_done,   // Out DMA transfer complete

    // =========================================================================
    // Status
    // =========================================================================
    output reg                     layer_done,      // Pulse: current layer complete
    output reg                     last_layer_done, // Pulse: final layer → DDR done
    output reg                     feedback_busy,   // Feedback transfer in progress
    output reg                     busy              // Any operation in progress
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam NUM_WORDS = (NUM_ACCS + 7) / 8;  // 32 words for 256 accumulators

    // =========================================================================
    // FSM States
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,              // Waiting for sched_done
        S_WAIT_ACCUM,        // Wait for accumulator bank ready
        S_CAPTURE,           // Read accumulators → write to BRAM
        S_CAPTURE_DRAIN,     // Drain pipeline (2-cycle latency)
        S_DECIDE,            // Check if last layer
        S_FEEDBACK_START,    // Begin draining BRAM → act_buffer
        S_FEEDBACK,          // Draining BRAM → act_buffer
        S_FEEDBACK_DRAIN,    // Drain feedback pipeline
        S_DMA_WAIT,          // Wait for out_dma to finish DDR write
        S_DONE               // Signal completion
    } state_t;

    state_t state;

    // =========================================================================
    // Internal Registers
    // =========================================================================
    reg [ADDR_W-1:0]  cap_cnt;      // Capture word counter
    reg [ADDR_W-1:0]  fb_rd_cnt;    // Feedback read counter
    reg [ADDR_W-1:0]  fb_wr_cnt;    // Feedback write counter
    reg                bank_sel_r;   // Current write bank
    reg                is_last_layer;
    reg                trigger_dma_r;

    // Pipeline tracking for accumulator read (2-cycle latency)
    reg pipe_d1, pipe_d2;

    // Feedback pipeline tracking (1-cycle BRAM read latency)
    reg fb_pipe_d1;

    // Number of words to capture (may be reduced by pooling)
    reg [ADDR_W-1:0] num_words_to_capture;

    assign bram_bank_sel  = bank_sel_r;
    assign out_dma_trigger = trigger_dma_r;

    // =========================================================================
    // Main FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            bank_sel_r         <= 1'b0;
            busy               <= 1'b0;
            feedback_busy      <= 1'b0;
            layer_done         <= 1'b0;
            last_layer_done    <= 1'b0;
            accum_rd_en        <= 1'b0;
            accum_rd_addr      <= {ADDR_W{1'b0}};
            bram_wr_en         <= 1'b0;
            bram_wr_addr       <= {ADDR_W{1'b0}};
            bram_wr_data       <= {DATA_W{1'b0}};
            bram_rd_en         <= 1'b0;
            bram_rd_addr       <= {ADDR_W{1'b0}};
            fb_act_we          <= 1'b0;
            fb_act_waddr       <= {ACT_ADDR_W{1'b0}};
            fb_act_wdata       <= {ACT_DATA_W{1'b0}};
            cap_cnt            <= {ADDR_W{1'b0}};
            fb_rd_cnt          <= {ADDR_W{1'b0}};
            fb_wr_cnt          <= {ADDR_W{1'b0}};
            pipe_d1            <= 1'b0;
            pipe_d2            <= 1'b0;
            fb_pipe_d1         <= 1'b0;
            is_last_layer      <= 1'b0;
            trigger_dma_r      <= 1'b0;
            num_words_to_capture <= NUM_WORDS[ADDR_W-1:0];
        end else begin
            // Default pulse signals
            layer_done      <= 1'b0;
            last_layer_done <= 1'b0;
            trigger_dma_r   <= 1'b0;

            case (state)
                // ─────────────────────────────────────────────────────────
                // IDLE: Wait for scheduler to complete all tiles
                // ─────────────────────────────────────────────────────────
                S_IDLE: begin
                    if (sched_done) begin
                        busy <= 1'b1;
                        is_last_layer <= (layer_current >= layer_total - 8'd1);
                        num_words_to_capture <= NUM_WORDS[ADDR_W-1:0];
                        state <= S_WAIT_ACCUM;
                        // synthesis translate_off
                        $display("[OBRAM_CTRL] S_IDLE->S_WAIT_ACCUM  layer_total=%0d layer_current=%0d is_last=%0b  @ %0t",
                                 layer_total, layer_current, (layer_current >= layer_total - 8'd1), $time);
                        // synthesis translate_on
                    end
                end

                // ─────────────────────────────────────────────────────────
                // WAIT_ACCUM: Wait for accumulator inactive bank to be ready
                // ─────────────────────────────────────────────────────────
                S_WAIT_ACCUM: begin
                    if (accum_ready) begin
                        cap_cnt <= {ADDR_W{1'b0}};
                        pipe_d1 <= 1'b0;
                        pipe_d2 <= 1'b0;
                        state   <= S_CAPTURE;
                        // synthesis translate_off
                        $display("[OBRAM_CTRL] S_WAIT_ACCUM->S_CAPTURE  @ %0t", $time);
                        // synthesis translate_on
                    end
                end

                // ─────────────────────────────────────────────────────────
                // CAPTURE: Read quantized INT8 data from accumulator
                //          and write into BRAM buffer
                // ─────────────────────────────────────────────────────────
                S_CAPTURE: begin
                    // Issue reads to accumulator (same interface as out_dma)
                    if (cap_cnt < num_words_to_capture) begin
                        accum_rd_en   <= 1'b1;
                        accum_rd_addr <= cap_cnt[ADDR_W-4:0] << 3;  // word × 8
                        cap_cnt       <= cap_cnt + {{(ADDR_W-1){1'b0}}, 1'b1};
                    end else begin
                        accum_rd_en <= 1'b0;
                    end

                    // Track 2-stage pipeline from accumulator
                    pipe_d1 <= accum_rd_en;
                    pipe_d2 <= pipe_d1;

                    // Write data arriving from accumulator pipeline into BRAM
                    if (pipe_d2) begin
                        bram_wr_en   <= 1'b1;
                        bram_wr_addr <= bram_wr_addr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        bram_wr_data <= accum_rd_data;
                    end else begin
                        bram_wr_en <= 1'b0;
                    end

                    // All reads issued and pipeline drained
                    if (cap_cnt >= num_words_to_capture && !pipe_d1 && !pipe_d2 && !accum_rd_en) begin
                        bram_wr_en  <= 1'b0;
                        bram_wr_addr <= {ADDR_W{1'b0}};
                        state        <= S_CAPTURE_DRAIN;
                    end
                end

                // ─────────────────────────────────────────────────────────
                // CAPTURE_DRAIN: Ensure last write completes, then flip bank
                // ─────────────────────────────────────────────────────────
                S_CAPTURE_DRAIN: begin
                    bram_wr_en <= 1'b0;
                    // Flip bank NOW so subsequent reads (feedback or DMA) access
                    // the just-written bank.  bank_sel semantics:
                    //   bank_sel=0 → write bank0 / read bank1
                    //   bank_sel=1 → write bank1 / read bank0
                    // After flipping, the old write-bank becomes the read-bank.
                    bank_sel_r <= ~bank_sel_r;
                    state <= S_DECIDE;
                    // synthesis translate_off
                    $display("[OBRAM_CTRL] S_CAPTURE_DRAIN->S_DECIDE  bank_sel=%0b->%0b  @ %0t",
                             bank_sel_r, ~bank_sel_r, $time);
                    // synthesis translate_on
                end

                // ─────────────────────────────────────────────────────────
                // DECIDE: Route based on layer position
                // ─────────────────────────────────────────────────────────
                S_DECIDE: begin
                    if (is_last_layer) begin
                        // Last layer: drain BRAM → DDR via out_dma
                        trigger_dma_r <= 1'b1;
                        state <= S_DMA_WAIT;
                        // synthesis translate_off
                        $display("[OBRAM_CTRL] S_DECIDE->S_DMA_WAIT (last layer)  @ %0t", $time);
                        // synthesis translate_on
                    end else begin
                        // Intermediate layer: feedback BRAM → act_buffer
                        feedback_busy <= 1'b1;
                        fb_rd_cnt     <= {ADDR_W{1'b0}};
                        fb_wr_cnt     <= {ADDR_W{1'b0}};
                        fb_pipe_d1    <= 1'b0;
                        state         <= S_FEEDBACK_START;
                    end
                end

                // ─────────────────────────────────────────────────────────
                // FEEDBACK_START: Begin reading from BRAM (completed bank)
                // ─────────────────────────────────────────────────────────
                S_FEEDBACK_START: begin
                    fb_rd_cnt <= {ADDR_W{1'b0}};
                    state <= S_FEEDBACK;
                end

                // ─────────────────────────────────────────────────────────
                // FEEDBACK: Drain BRAM into activation buffer
                //
                // BRAM stores 64-bit words (8 × INT8). Act_buffer expects
                // 128-bit words (16 × INT8). This FSM reads sequential
                // 64-bit words from BRAM and writes them out on the
                // fb_act_* port, zero-extended to ACT_DATA_W (128 bits).
                //
                // The actual 64→128-bit repacking is handled externally
                // by u_fb_packer (dma_pack_112) in accel_top.sv, which
                // accumulates two 64-bit beats (128 bits), then emits
                // the full 128 bits as one act_buffer row (16 × INT8).
                // ─────────────────────────────────────────────────────────
                S_FEEDBACK: begin
                    // Issue BRAM reads
                    if (fb_rd_cnt < num_words_to_capture) begin
                        bram_rd_en   <= 1'b1;
                        bram_rd_addr <= fb_rd_cnt;
                        fb_rd_cnt    <= fb_rd_cnt + {{(ADDR_W-1){1'b0}}, 1'b1};
                    end else begin
                        bram_rd_en <= 1'b0;
                    end

                    // Track BRAM read latency (1 cycle)
                    fb_pipe_d1 <= bram_rd_en;

                    // Write to activation buffer feedback port
                    if (bram_rd_valid) begin
                        fb_act_we    <= 1'b1;
                        fb_act_waddr <= fb_wr_cnt[ACT_ADDR_W-1:0];
                        // Zero-extend 64-bit BRAM data to 128-bit act width
                        fb_act_wdata <= {{(ACT_DATA_W-DATA_W){1'b0}}, bram_rd_data};
                        fb_wr_cnt    <= fb_wr_cnt + {{(ADDR_W-1){1'b0}}, 1'b1};
                    end else begin
                        fb_act_we <= 1'b0;
                    end

                    // All reads issued and pipeline drained
                    if (fb_rd_cnt >= num_words_to_capture && !bram_rd_en && !fb_pipe_d1) begin
                        state <= S_FEEDBACK_DRAIN;
                    end
                end

                // ─────────────────────────────────────────────────────────
                // FEEDBACK_DRAIN: Last write to act_buffer
                // ─────────────────────────────────────────────────────────
                S_FEEDBACK_DRAIN: begin
                    fb_act_we     <= 1'b0;
                    feedback_busy <= 1'b0;
                    // Bank already flipped in S_CAPTURE_DRAIN
                    layer_done <= 1'b1;
                    state <= S_DONE;
                end

                // ─────────────────────────────────────────────────────────
                // DMA_WAIT: Wait for out_dma to finish writing to DDR
                // ─────────────────────────────────────────────────────────
                S_DMA_WAIT: begin
                    if (out_dma_done) begin
                        last_layer_done <= 1'b1;
                        state           <= S_DONE;
                        // synthesis translate_off
                        $display("[OBRAM_CTRL] S_DMA_WAIT->S_DONE  last_layer_done=1  @ %0t", $time);
                        // synthesis translate_on
                    end
                end

                // ─────────────────────────────────────────────────────────
                // DONE: Return to idle
                // ─────────────────────────────────────────────────────────
                S_DONE: begin
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
