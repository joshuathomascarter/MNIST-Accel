// =============================================================================
// out_dma.sv — Output Write DMA (Output Accumulator → DDR via AXI4 Write)
// =============================================================================
//
// OVERVIEW
// ========
// Reads quantized INT8 results from the output accumulator's DMA port
// and writes them to DDR via AXI4 write bursts.  Auto-triggered when
// sched_done fires (if dst_addr ≠ 0).
//
// DATA VOLUME
// ===========
// 256 INT8 accumulators (16×16) packed 8 per 64-bit word = 32 words = 256 bytes.
// Transferred in a single AXI4 INCR burst of 32 beats.
//
// PIPELINE
// ========
//   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//   │ Output Accum │ ──> │  Read Buffer │ ──> │  AXI4 Write  │ ──> DDR
//   │ (2-cyc pipe) │     │ (25 × 64-bit)│     │ (25-beat     │
//   │              │     │              │     │   burst)     │
//   └──────────────┘     └──────────────┘     └──────────────┘
//
// TIMING
// ======
//   - Read phase: 32 reads + 2 pipeline cycles = 34 cycles
//   - AW phase:   1-2 cycles (address handshake)
//   - W phase:    32 cycles (1 beat/cycle at best)
//   - B phase:    1 cycle (response)
//   - Total:      ~70 cycles ≈ 700 ns @ 100 MHz
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module out_dma #(
    parameter AXI_ADDR_W  = 32,
    parameter AXI_DATA_W  = 64,
    parameter AXI_ID_W    = 4,
    parameter BRAM_ADDR_W = 10,
    parameter NUM_ACCS    = 256,    // 16×16 output accumulators
    parameter STREAM_ID   = 2      // AXI transaction ID for writes
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                     start,      // Trigger (from sched_done)
    input  wire [AXI_ADDR_W-1:0]   dst_addr,   // DDR destination (from CSR)
    output reg                      done,       // Transfer complete (pulse)
    output reg                      busy,       // Transfer in progress

    // =========================================================================
    // Output Accumulator Read Interface
    // =========================================================================
    output reg                      accum_rd_en,
    output reg  [BRAM_ADDR_W-1:0]  accum_rd_addr,
    input  wire [63:0]             accum_rd_data,
    input  wire                     accum_ready,  // Inactive bank ready

    // =========================================================================
    // AXI4 Write Address Channel
    // =========================================================================
    output wire [AXI_ID_W-1:0]     m_axi_awid,
    output reg  [AXI_ADDR_W-1:0]   m_axi_awaddr,
    output reg  [7:0]              m_axi_awlen,
    output wire [2:0]              m_axi_awsize,
    output wire [1:0]              m_axi_awburst,
    output reg                      m_axi_awvalid,
    input  wire                     m_axi_awready,

    // =========================================================================
    // AXI4 Write Data Channel
    // =========================================================================
    output reg  [AXI_DATA_W-1:0]   m_axi_wdata,
    output wire [AXI_DATA_W/8-1:0] m_axi_wstrb,
    output reg                      m_axi_wlast,
    output reg                      m_axi_wvalid,
    input  wire                     m_axi_wready,

    // =========================================================================
    // AXI4 Write Response Channel
    // =========================================================================
    input  wire [AXI_ID_W-1:0]     m_axi_bid,
    input  wire [1:0]              m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output reg                      m_axi_bready
);

    // =========================================================================
    // Constants
    // =========================================================================
    // 256 accumulators / 8 per word = 32 words (perfectly aligned)
    localparam NUM_WORDS = (NUM_ACCS + 7) / 8;  // 32

    // =========================================================================
    // FSM States
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_WAIT_READY,   // Wait for accumulator bank swap to complete
        S_READ,         // Read NUM_WORDS from output accumulator
        S_AW,           // Issue AXI4 write address
        S_W,            // Stream write data beats
        S_B             // Wait for write response
    } state_t;

    state_t state;

    // =========================================================================
    // Read Data Buffer (32 × 64 bits = 2048 bits)
    // =========================================================================
    reg [63:0] rd_buf [0:NUM_WORDS-1];

    // =========================================================================
    // Counters
    // =========================================================================
    reg [5:0] rd_cnt;       // Read issue counter (0..31)
    reg [5:0] cap_cnt;      // Capture counter (follows rd_cnt by 2 cycles)
    reg [5:0] wr_cnt;       // AXI write beat counter (0..31)

    // Pipeline tracking (output_bram_buffer has 1-cycle read latency)
    reg pipe_d1, pipe_d2;

    // =========================================================================
    // Fixed AXI Write Parameters
    // =========================================================================
    assign m_axi_awid    = STREAM_ID[AXI_ID_W-1:0];
    assign m_axi_awsize  = 3'b011;  // 8 bytes per beat (2^3)
    assign m_axi_awburst = 2'b01;   // INCR burst
    assign m_axi_wstrb   = {(AXI_DATA_W/8){1'b1}};  // All bytes valid

    // AXI response — single-stream write, not checked
    wire _unused_bresp = &{1'b0, m_axi_bid, m_axi_bresp};

    // =========================================================================
    // Main FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            busy          <= 1'b0;
            accum_rd_en   <= 1'b0;
            accum_rd_addr <= {BRAM_ADDR_W{1'b0}};
            m_axi_awaddr  <= {AXI_ADDR_W{1'b0}};
            m_axi_awlen   <= 8'd0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= {AXI_DATA_W{1'b0}};
            m_axi_wlast   <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
            rd_cnt        <= 6'd0;
            cap_cnt       <= 6'd0;
            wr_cnt        <= 6'd0;
            pipe_d1       <= 1'b0;
            pipe_d2       <= 1'b0;
        end else begin
            done <= 1'b0;  // Pulse — cleared every cycle

            case (state)
                // ─────────────────────────────────────────────────────────
                // IDLE: Wait for start trigger
                // ─────────────────────────────────────────────────────────
                S_IDLE: begin
                    if (start) begin
                        if (dst_addr == {AXI_ADDR_W{1'b0}}) begin
                            done <= 1'b1;
                            // synthesis translate_off
                            $display("[OUT_DMA] start with dst_addr=0 -> immediate done  @ %0t", $time);
                            // synthesis translate_on
                        end else begin
                            busy  <= 1'b1;
                            state <= S_WAIT_READY;
                            // synthesis translate_off
                            $display("[OUT_DMA] S_IDLE->S_WAIT_READY  dst_addr=0x%08h  @ %0t", dst_addr, $time);
                            // synthesis translate_on
                        end
                    end
                end

                // ─────────────────────────────────────────────────────────
                // WAIT_READY: Wait for output accumulator bank swap
                // ─────────────────────────────────────────────────────────
                S_WAIT_READY: begin
                    if (accum_ready) begin
                        rd_cnt  <= 6'd0;
                        cap_cnt <= 6'd0;
                        pipe_d1 <= 1'b0;
                        pipe_d2 <= 1'b0;
                        state   <= S_READ;
                        // synthesis translate_off
                        $display("[OUT_DMA] S_WAIT_READY->S_READ  @ %0t", $time);
                        // synthesis translate_on
                    end
                end

                // ─────────────────────────────────────────────────────────
                // READ: Issue 25 reads from output BRAM buffer
                //
                // BRAM stores data at sequential addresses 0, 1, 2, ..., 24
                // (written by pool_bram_wr_addr in accel_top).
                // ─────────────────────────────────────────────────────────
                S_READ: begin
                    // Issue sequential reads
                    if (rd_cnt < NUM_WORDS[5:0]) begin
                        accum_rd_en   <= 1'b1;
                        accum_rd_addr <= {{(BRAM_ADDR_W-6){1'b0}}, rd_cnt};
                        rd_cnt        <= rd_cnt + 6'd1;
                    end else begin
                        accum_rd_en <= 1'b0;
                    end

                    // Track 2-stage pipeline
                    pipe_d1 <= accum_rd_en;
                    pipe_d2 <= pipe_d1;

                    // Capture data arriving from BRAM (1-cycle latency)
                    if (pipe_d1) begin
                        rd_buf[cap_cnt] <= accum_rd_data;
                        if (cap_cnt == NUM_WORDS[5:0] - 6'd1) begin
                            // Last word captured — proceed to AXI write
                            state <= S_AW;
                            // synthesis translate_off
                            $display("[OUT_DMA] S_READ->S_AW  cap_cnt=%0d  @ %0t", cap_cnt, $time);
                            // synthesis translate_on
                        end
                        cap_cnt <= cap_cnt + 6'd1;
                    end
                end

                // ─────────────────────────────────────────────────────────
                // AW: Issue AXI4 write address (single 25-beat burst)
                // ─────────────────────────────────────────────────────────
                S_AW: begin
                    m_axi_awaddr  <= dst_addr;
                    m_axi_awlen   <= NUM_WORDS[7:0] - 8'd1;  // 24 = 25 beats - 1
                    m_axi_awvalid <= 1'b1;

                    if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        wr_cnt        <= 6'd0;
                        state         <= S_W;
                    end
                end

                // ─────────────────────────────────────────────────────────
                // W: Stream write data beats from buffer
                // ─────────────────────────────────────────────────────────
                S_W: begin
                    m_axi_wvalid <= 1'b1;
                    m_axi_wdata  <= rd_buf[wr_cnt];
                    m_axi_wlast  <= (wr_cnt == NUM_WORDS[5:0] - 6'd1);

                    if (m_axi_wvalid && m_axi_wready) begin
                        if (wr_cnt == NUM_WORDS[5:0] - 6'd1) begin
                            // Last beat accepted
                            m_axi_wvalid <= 1'b0;
                            m_axi_wlast  <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state        <= S_B;
                        end else begin
                            wr_cnt <= wr_cnt + 6'd1;
                        end
                    end
                end

                // ─────────────────────────────────────────────────────────
                // B: Wait for write response
                // ─────────────────────────────────────────────────────────
                S_B: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        done         <= 1'b1;
                        busy         <= 1'b0;
                        state        <= S_IDLE;
                        // synthesis translate_off
                        $display("[OUT_DMA] S_B->S_IDLE  done=1  @ %0t", $time);
                        // synthesis translate_on
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
