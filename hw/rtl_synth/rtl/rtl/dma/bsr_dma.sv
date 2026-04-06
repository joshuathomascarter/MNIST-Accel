// bsr_dma.sv — AXI4 read-only DMA for BSR sparse data (DDR → 3 BRAMs)
// 3-phase transfer: row_ptr (32-bit) → col_idx (16-bit) → weight blocks (64-bit)
// Unpacking: row_ptr = 2 per beat, col_idx = 4 per beat, weights = direct.
// STREAM_ID=0 distinguishes from act_dma in axi_dma_bridge.

`timescale 1ns/1ps
`default_nettype none

module bsr_dma #(
    parameter AXI_ADDR_W  = 32,
    parameter AXI_DATA_W  = 64,
    parameter AXI_ID_W    = 4,
    parameter STREAM_ID   = 0,    // bsr_dma=0, act_dma=1
    parameter BRAM_ADDR_W = 10,
    parameter BURST_LEN   = 8'd15 // 16 beats = 128 bytes
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Control (from CSR)
    input  wire                  start,
    input  wire [AXI_ADDR_W-1:0] src_addr,
    input  wire [31:0]           csr_num_rows,
    input  wire [31:0]           csr_total_blocks,
    output reg                   done,
    output reg                   busy,
    output reg                   error,

    // AXI4 read address channel
    output wire [AXI_ID_W-1:0]   m_axi_arid,
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    output reg [7:0]             m_axi_arlen,
    output wire [2:0]            m_axi_arsize,
    output wire [1:0]            m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    // AXI4 read data channel
    input  wire [AXI_ID_W-1:0]   m_axi_rid,    // unused: single-ID stream
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output reg                   m_axi_rready,

    // BRAM write: row_ptr (32-bit entries)
    output reg                   row_ptr_we,
    output reg [BRAM_ADDR_W-1:0] row_ptr_addr,
    output reg [31:0]            row_ptr_wdata,

    // BRAM write: col_idx (16-bit entries)
    output reg                   col_idx_we,
    output reg [BRAM_ADDR_W-1:0] col_idx_addr,
    output reg [15:0]            col_idx_wdata,

    // BRAM write: weight blocks (64-bit, 8 INT8 per beat)
    output reg                    wgt_we,
    output reg [BRAM_ADDR_W+6:0]  wgt_addr,     // byte address
    output reg [63:0]             wgt_wdata
);

    // FSM: 12 states for 3-phase transfer with sub-word unpacking
    typedef enum logic [3:0] {
        IDLE,
        SETUP_ROW_PTR,
        READ_ROW_PTR,
        WRITE_ROW_PTR_HIGH,
        SETUP_COL_IDX,
        READ_COL_IDX,
        WRITE_COL_IDX_1,
        WRITE_COL_IDX_2,
        WRITE_COL_IDX_3,
        SETUP_WEIGHTS,
        READ_WEIGHTS,
        DONE_STATE
    } state_t;

    (* fsm_encoding = "one_hot" *) state_t state;

    reg [31:0] num_rows, total_blocks;
    reg [AXI_ADDR_W-1:0] current_axi_addr;
    reg [31:0]           words_remaining;
    reg [63:0] rdata_reg;       // buffered beat for multi-cycle unpack
    reg        rlast_reg;       // buffered rlast during unpack

    localparam [2:0] AXI_SIZE_64    = 3'b011;  // 8 bytes/beat
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    assign m_axi_arid    = STREAM_ID[AXI_ID_W-1:0];
    assign m_axi_arsize  = AXI_SIZE_64;
    assign m_axi_arburst = AXI_BURST_INCR;

    // m_axi_rid: AXI protocol requires this port, but since this DMA uses
    // a single fixed STREAM_ID, we never need to match response IDs.
    wire _unused_bsr_rid = &{1'b0, m_axi_rid};

    // rdata_reg[15:0]: In READ_COL_IDX, the first 16-bit col_idx is extracted
    // directly from m_axi_rdata[15:0] (live data). Only the upper three words
    // (bits 31:16, 47:32, 63:48) are extracted from rdata_reg in subsequent
    // unpack cycles. So bits [15:0] of the latch are written but never read.
    wire _unused_rdata_lo = &{1'b0, rdata_reg[15:0]};

    // 4KB AXI boundary guard: beats remaining before page crossing.
    // Full 11-bit range: 0..512 beats (0..4096 bytes).
    // When address is page-aligned, page_max = 512 (full page available).
    wire [10:0] page_max_beats = 11'd512 - {2'b0, current_axi_addr[11:3]};

    function automatic [7:0] safe_arlen(input [9:0] desired);
        if ({1'b0, desired} + 11'd1 > page_max_beats)
            safe_arlen = page_max_beats[7:0] - 8'd1;
        else
            safe_arlen = desired[7:0];
    endfunction

    // Pre-computed burst lengths (10-bit) for each DMA phase.
    // words_remaining is bounded by num_rows+1 (row_ptr), total_blocks (col_idx/wgt).
    // After shift, the beat count fits in 10 bits (max 512 beats = 4KB page).
    wire [9:0] rowptr_arlen_short = 10'(((words_remaining + 32'd1) >> 1) - 32'd1);
    wire [9:0] colidx_arlen_short = 10'(((words_remaining + 32'd3) >> 2) - 32'd1);
    wire [9:0] wgt_arlen_short    = words_remaining[9:0] - 10'd1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            error            <= 1'b0;
            m_axi_arvalid    <= 1'b0;
            m_axi_rready     <= 1'b0;
            row_ptr_we       <= 1'b0;
            col_idx_we       <= 1'b0;
            wgt_we           <= 1'b0;
            row_ptr_addr     <= 0;
            col_idx_addr     <= 0;
            wgt_addr         <= 0;
            current_axi_addr <= 0;
            rdata_reg        <= 0;
            rlast_reg        <= 0;
        end else begin
            row_ptr_we    <= 1'b0;
            col_idx_we    <= 1'b0;
            wgt_we        <= 1'b0;
            m_axi_arvalid <= 1'b0;

            case (state)

                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy             <= 1'b1;
                        error            <= 1'b0;
                        current_axi_addr <= src_addr;
                        num_rows         <= csr_num_rows;
                        total_blocks     <= csr_total_blocks;
                        state            <= SETUP_ROW_PTR;
                    end
                end

                // --- Phase 1: row_ptr ((num_rows+1) × 32-bit) ---

                SETUP_ROW_PTR: begin
                    words_remaining <= num_rows + 1;
                    row_ptr_addr    <= {BRAM_ADDR_W{1'b1}};  // -1: first +1 wraps to 0
                    state           <= READ_ROW_PTR;
                end

                READ_ROW_PTR: begin
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        if (words_remaining > 32)
                            m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
                        else
                            m_axi_arlen <= safe_arlen(rowptr_arlen_short);
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1; busy <= 1'b0; done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write lower 32 bits
                            if (words_remaining > 0) begin
                                row_ptr_we    <= 1'b1;
                                row_ptr_wdata <= m_axi_rdata[31:0];
                                row_ptr_addr  <= row_ptr_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end

                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0; // throttle for upper-word unpack
                                state <= WRITE_ROW_PTR_HIGH;
                            end else if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                                if (words_remaining <= 1) state <= SETUP_COL_IDX;
                            end
                        end
                    end
                end

                WRITE_ROW_PTR_HIGH: begin
                    m_axi_rready  <= 1'b0;
                    row_ptr_we    <= 1'b1;
                    row_ptr_wdata <= rdata_reg[63:32];
                    row_ptr_addr  <= row_ptr_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                        if (words_remaining <= 1) state <= SETUP_COL_IDX;
                        else state <= READ_ROW_PTR;
                    end else begin
                        state <= READ_ROW_PTR;
                    end
                end

                // --- Phase 2: col_idx (total_blocks × 16-bit) ---
                // 4 col_idx values per 64-bit beat → unpack in 4 cycles

                SETUP_COL_IDX: begin
                    words_remaining <= total_blocks;
                    col_idx_addr    <= {BRAM_ADDR_W{1'b1}};  // -1: first +1 wraps to 0
                    state           <= READ_COL_IDX;
                end

                READ_COL_IDX: begin
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        if (words_remaining > 64)
                            m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
                        else
                            m_axi_arlen <= safe_arlen(colidx_arlen_short);
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1; busy <= 1'b0; done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write index 0: bits [15:0]
                            if (words_remaining > 0) begin
                                col_idx_we    <= 1'b1;
                                col_idx_wdata <= m_axi_rdata[15:0];
                                col_idx_addr  <= col_idx_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end

                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0;
                                state <= WRITE_COL_IDX_1;
                            end else if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                                if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                            end
                        end
                    end
                end

                WRITE_COL_IDX_1: begin                         // bits [31:16]
                    m_axi_rready <= 1'b0;
                    col_idx_we    <= 1'b1;
                    col_idx_wdata <= rdata_reg[31:16];
                    col_idx_addr  <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_2;
                    else if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                        state <= SETUP_WEIGHTS;
                    end else state <= READ_COL_IDX;
                end

                WRITE_COL_IDX_2: begin                         // bits [47:32]
                    m_axi_rready <= 1'b0;
                    col_idx_we    <= 1'b1;
                    col_idx_wdata <= rdata_reg[47:32];
                    col_idx_addr  <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_3;
                    else if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                        state <= SETUP_WEIGHTS;
                    end else state <= READ_COL_IDX;
                end

                WRITE_COL_IDX_3: begin                         // bits [63:48]
                    m_axi_rready <= 1'b0;
                    col_idx_we    <= 1'b1;
                    col_idx_wdata <= rdata_reg[63:48];
                    col_idx_addr  <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                        if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                        else state <= READ_COL_IDX;
                    end else state <= READ_COL_IDX;
                end

                // --- Phase 3: weight blocks (total_blocks × 25 beats) ---
                // 14×14 = 196 bytes/block, ceil(196/8) = 25 beats.
                // Last beat per block: 4 valid + 4 padding bytes.

                SETUP_WEIGHTS: begin
                    // Force to LUT fabric — not performance-critical, saves 1-2 DSPs
                    // total_blocks * 25 = total_blocks * (16 + 8 + 1)
                    words_remaining <= (total_blocks << 4) + (total_blocks << 3) + total_blocks;
                    wgt_addr        <= 0;
                    state           <= READ_WEIGHTS;
                end

                READ_WEIGHTS: begin
                    m_axi_rready <= 1'b1;

                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        if (words_remaining > {24'd0, BURST_LEN})
                            m_axi_arlen <= safe_arlen({2'd0, BURST_LEN});
                        else
                            m_axi_arlen <= safe_arlen(wgt_arlen_short);
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1; busy <= 1'b0; done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            wgt_we    <= 1'b1;
                            wgt_wdata <= m_axi_rdata;
                            wgt_addr  <= wgt_addr + 8; // byte address
                            words_remaining <= words_remaining - 1;

                            if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + (({24'd0, m_axi_arlen} + 32'd1) * 32'd8);
                                if (words_remaining <= 1) state <= DONE_STATE;
                            end
                        end
                    end
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (!start) state <= IDLE; // handshake: wait for start deassert
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
`default_nettype wire
