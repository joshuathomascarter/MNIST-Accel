// =============================================================================
// bsr_dma.v — AXI4 Master BSR DMA Engine for Sparse Accelerator
// =============================================================================
// Purpose:
//   Transfers BSR-format sparse matrix data from DDR memory into on-chip BRAMs:
//   - row_ptr[]: cumulative block counts (sparse metadata)
//   - col_idx[]: column position of each block
//   - blocks[]:  14×14 INT8 weight blocks (196 bytes each)
//
// Features:
//   - AXI4 Master Interface (Read Only)
//   - Burst-based data transfer for high bandwidth
//   - Automatic unpacking of metadata and weight blocks
//   - Configurable via CSR interface
//
// Architecture:
//   [DDR Memory] <== AXI4 Read ==> [Burst Buffer] ==> [Unpacker] ==> [BRAMs]
//
// Memory Layout (in DDR):
//   Header (3 words): [Num_Rows, Num_Cols, Total_Blocks]
//   Row_Ptr Array:    [row_ptr[0], row_ptr[1], ... row_ptr[M]]
//   Col_Idx Array:    [col_idx[0], col_idx[1], ... col_idx[Total_Blocks]]
//   Weight Blocks:    [Block0 (196B), Block1 (196B), ... ]  // 14×14 INT8 blocks
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bsr_dma #(
    parameter AXI_ADDR_W = 32,
    parameter AXI_DATA_W = 64,
    parameter AXI_ID_W   = 4,       // NEW: ID Width
    parameter STREAM_ID  = 0,       // NEW: ID for this DMA (0 for BSR)
    parameter BRAM_ADDR_W = 10,
    parameter BURST_LEN   = 8'd15   // 16 beats per burst
)(
    // Clock & Reset
    input  wire                  clk,
    input  wire                  rst_n,

    // Control Interface (from CSR)
    input  wire                  start,
    input  wire [AXI_ADDR_W-1:0] src_addr,
    output reg                   done,
    output reg                   busy,
    output reg                   error,

    // AXI4 Master Read Interface
    output wire [AXI_ID_W-1:0]   m_axi_arid,    // NEW: Output ID
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    output reg [7:0]             m_axi_arlen,
    output reg [2:0]             m_axi_arsize,
    output reg [1:0]             m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    input  wire [AXI_ID_W-1:0]   m_axi_rid,     // NEW: Input ID
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output reg                   m_axi_rready,

    // BRAM Write Interfaces
    // Row Pointers (32-bit data)
    output reg                   row_ptr_we,
    output reg [BRAM_ADDR_W-1:0] row_ptr_addr,
    output reg [31:0]            row_ptr_wdata,

    // Column Indices (16-bit data)
    output reg                   col_idx_we,
    output reg [BRAM_ADDR_W-1:0] col_idx_addr,
    output reg [15:0]            col_idx_wdata,

    // Weight Blocks (64-bit data - 8 weights/row)
    output reg                   wgt_we,
    output reg [BRAM_ADDR_W+6:0] wgt_addr, // Byte address
    output reg [63:0]            wgt_wdata
);

    // ========================================================================
    // Internal State & Registers
    // ========================================================================
    typedef enum logic [3:0] {
        IDLE,
        READ_HEADER,
        WAIT_HEADER,
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

    (* fsm_encoding = "one_hot" *) state_t state, next_state;

    // Header Registers
    reg [31:0] num_rows;
    reg [31:0] num_cols;
    reg [31:0] total_blocks;

    // Address Pointers
    reg [AXI_ADDR_W-1:0] current_axi_addr;
    reg [31:0]           words_remaining;
    
    // Data Unpacking
    reg [1:0]  header_word_idx;
    reg [63:0] rdata_reg; // Buffer for unpacking
    reg        rlast_reg; // Store last signal during unpacking

    // AXI Constants
    localparam [2:0] AXI_SIZE_64 = 3'b011; // 8 bytes
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    // NEW: Drive the ID constantly
    assign m_axi_arid = STREAM_ID[AXI_ID_W-1:0];

    // ========================================================================
    // Main FSM
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            wgt_we <= 1'b0;
            
            row_ptr_addr <= 0;
            col_idx_addr <= 0;
            wgt_addr <= 0;
            
            current_axi_addr <= 0;
            header_word_idx <= 0;
            rdata_reg <= 0;
            rlast_reg <= 0;
        end else begin
            // Default Control Signals
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            wgt_we <= 1'b0;
            m_axi_arvalid <= 1'b0; // Auto-clear unless set in state
            
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        error <= 1'b0;
                        current_axi_addr <= src_addr;
                        state <= READ_HEADER;
                    end
                end

                // ------------------------------------------------------------
                // Header Phase: Read 3 x 32-bit words (aligned to 64-bit)
                // ------------------------------------------------------------
                READ_HEADER: begin
                    m_axi_araddr <= current_axi_addr;
                    m_axi_arlen  <= 8'd1; // Read 2 beats (16 bytes) to get 3 words safely
                    m_axi_arsize <= AXI_SIZE_64;
                    m_axi_arburst <= AXI_BURST_INCR;
                    m_axi_arvalid <= 1'b1;
                    m_axi_rready <= 1'b1;
                    
                    header_word_idx <= 0;
                    
                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0; // Handshake done
                        state <= WAIT_HEADER;
                    end
                end

                WAIT_HEADER: begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Unpack 64-bit data into 32-bit header registers
                            case (header_word_idx)
                                0: begin
                                    num_rows <= m_axi_rdata[31:0];
                                    num_cols <= m_axi_rdata[63:32];
                                    header_word_idx <= 2;
                                end
                                2: begin
                                    total_blocks <= m_axi_rdata[31:0];
                                    header_word_idx <= 3;
                                end
                            endcase
                            
                            if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + 16; // Advance past header
                                state <= SETUP_ROW_PTR;
                            end
                        end
                    end
                end

                // ------------------------------------------------------------
                // Row Pointer Phase: Read (Num_Rows + 1) 32-bit words
                // ------------------------------------------------------------
                SETUP_ROW_PTR: begin
                    words_remaining <= num_rows + 1;
                    row_ptr_addr <= 0;
                    state <= READ_ROW_PTR;
                end

                READ_ROW_PTR: begin
                    // Issue Burst
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        // Calculate burst length (beats = (words + 1) / 2)
                        // Max burst 16 beats = 32 words
                        if (words_remaining > 32) 
                            m_axi_arlen <= BURST_LEN;
                        else 
                            m_axi_arlen <= ((words_remaining + 1) >> 1) - 1;
                            
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    // Ready to accept data
                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Capture data for unpacking
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write Lower 32-bits immediately
                            if (words_remaining > 0) begin
                                row_ptr_we <= 1'b1;
                                row_ptr_wdata <= m_axi_rdata[31:0];
                                row_ptr_addr <= row_ptr_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end
                            
                            // If more words remain in this beat (upper 32), go to unpack state
                            // Note: If words_remaining was 1, we are done with this beat.
                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0; // Throttle AXI
                                state <= WRITE_ROW_PTR_HIGH;
                            end else begin
                                // Beat fully consumed
                                if (m_axi_rlast) begin
                                    current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                    if (words_remaining <= 1) state <= SETUP_COL_IDX;
                                end
                            end
                        end
                    end
                end

                WRITE_ROW_PTR_HIGH: begin
                    m_axi_rready <= 1'b0; // Hold off next beat
                    
                    // Write Upper 32-bits
                    row_ptr_we <= 1'b1;
                    row_ptr_wdata <= rdata_reg[63:32];
                    row_ptr_addr <= row_ptr_addr + 1;
                    words_remaining <= words_remaining - 1;
                    
                    // Check if we are done with the burst
                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                        // If we just wrote the last needed word (or more), move on
                        if (words_remaining <= 1) state <= SETUP_COL_IDX;
                        else state <= READ_ROW_PTR; // Should not happen if logic is correct
                    end else begin
                        state <= READ_ROW_PTR; // Get next beat
                    end
                end

                // ------------------------------------------------------------
                // Column Index Phase: Read Total_Blocks 16-bit words
                // ------------------------------------------------------------
                SETUP_COL_IDX: begin
                    words_remaining <= total_blocks;
                    col_idx_addr <= 0;
                    state <= READ_COL_IDX;
                end

                READ_COL_IDX: begin
                    // Issue Burst
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        // Calculate burst length (beats = (words + 3) / 4)
                        // Max burst 16 beats = 64 words
                        if (words_remaining > 64) 
                            m_axi_arlen <= BURST_LEN;
                        else 
                            m_axi_arlen <= ((words_remaining + 3) >> 2) - 1;
                            
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write Index 0 (Bits 15:0)
                            if (words_remaining > 0) begin
                                col_idx_we <= 1'b1;
                                col_idx_wdata <= m_axi_rdata[15:0];
                                col_idx_addr <= col_idx_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end

                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0; // Throttle
                                state <= WRITE_COL_IDX_1;
                            end else begin
                                // Beat fully consumed
                                if (m_axi_rlast) begin
                                    current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                    if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                                end
                            end
                        end
                    end
                end

                WRITE_COL_IDX_1: begin
                    m_axi_rready <= 1'b0;
                    // Write Index 1 (Bits 31:16)
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[31:16];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_2;
                    else begin
                        if (rlast_reg) begin
                            current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                            state <= SETUP_WEIGHTS;
                        end else state <= READ_COL_IDX;
                    end
                end

                WRITE_COL_IDX_2: begin
                    m_axi_rready <= 1'b0;
                    // Write Index 2 (Bits 47:32)
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[47:32];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_3;
                    else begin
                        if (rlast_reg) begin
                            current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                            state <= SETUP_WEIGHTS;
                        end else state <= READ_COL_IDX;
                    end
                end

                WRITE_COL_IDX_3: begin
                    m_axi_rready <= 1'b0;
                    // Write Index 3 (Bits 63:48)
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[63:48];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                        if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                        else state <= READ_COL_IDX; // Should not happen
                    end else begin
                        state <= READ_COL_IDX;
                    end
                end

                // ------------------------------------------------------------
                // Weight Block Phase: Read Total_Blocks * 64 bytes
                // ------------------------------------------------------------
                SETUP_WEIGHTS: begin
                    // Total bytes = total_blocks * 64
                    // Total 64-bit words = total_blocks * 8
                    words_remaining <= total_blocks * 8; 
                    wgt_addr <= 0;
                    state <= READ_WEIGHTS;
                end

                READ_WEIGHTS: begin
                    // Assert rready when waiting for data
                    m_axi_rready <= 1'b1;
                    
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        if (words_remaining > BURST_LEN) m_axi_arlen <= BURST_LEN;
                        else m_axi_arlen <= words_remaining - 1;
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    // Only process data on valid AXI handshake (rvalid && rready)
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            wgt_we <= 1'b1;
                            wgt_wdata <= m_axi_rdata; // Direct 64-bit write!
                            wgt_addr <= wgt_addr + 8; // Byte address increment
                            words_remaining <= words_remaining - 1;

                            if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                if (words_remaining <= 1) state <= DONE_STATE;
                            end
                        end
                    end
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (!start) state <= IDLE; // Handshake reset
                end
            endcase
        end
    end

endmodule
`default_nettype wire
