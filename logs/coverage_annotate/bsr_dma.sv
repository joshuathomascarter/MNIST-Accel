//      // verilator_coverage annotation
        // =============================================================================
        // bsr_dma.v — AXI4 Master BSR DMA Engine for Sparse Accelerator
        // =============================================================================
        // Purpose:
        //   Transfers BSR-format sparse matrix data from DDR memory into on-chip BRAMs:
        //   - row_ptr[]: cumulative block counts (sparse metadata)
        //   - col_idx[]: column position of each block
        //   - blocks[]:  8×8 INT8 weight blocks
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
        //   Weight Blocks:    [Block0 (64B), Block1 (64B), ... ]
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
 012713     input  wire                  clk,
%000007     input  wire                  rst_n,
        
            // Control Interface (from CSR)
%000003     input  wire                  start,
%000002     input  wire [AXI_ADDR_W-1:0] src_addr,
%000001     output reg                   done,
%000001     output reg                   busy,
%000000     output reg                   error,
        
            // AXI4 Master Read Interface
%000000     output wire [AXI_ID_W-1:0]   m_axi_arid,    // NEW: Output ID
%000001     output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
%000002     output reg [7:0]             m_axi_arlen,
%000001     output reg [2:0]             m_axi_arsize,
%000001     output reg [1:0]             m_axi_arburst,
 000013     output reg                   m_axi_arvalid,
%000005     input  wire                  m_axi_arready,
        
%000001     input  wire [AXI_ID_W-1:0]   m_axi_rid,     // NEW: Input ID
%000007     input  wire [AXI_DATA_W-1:0] m_axi_rdata,
%000000     input  wire [1:0]            m_axi_rresp,
%000006     input  wire                  m_axi_rlast,
%000005     input  wire                  m_axi_rvalid,
%000003     output reg                   m_axi_rready,
        
            // BRAM Write Interfaces
            // Row Pointers (32-bit data)
%000001     output reg                   row_ptr_we,
%000005     output reg [BRAM_ADDR_W-1:0] row_ptr_addr,
%000004     output reg [31:0]            row_ptr_wdata,
        
            // Column Indices (16-bit data)
%000001     output reg                   col_idx_we,
%000002     output reg [BRAM_ADDR_W-1:0] col_idx_addr,
%000000     output reg [15:0]            col_idx_wdata,
        
            // Weight Blocks (64-bit data - 8 weights/row)
%000002     output reg                   wgt_we,
~000016     output reg [BRAM_ADDR_W+6:0] wgt_addr, // Byte address
%000004     output reg [63:0]            wgt_wdata
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
        
~000010     (* fsm_encoding = "one_hot" *) state_t state, next_state;
        
            // Header Registers
%000001     reg [31:0] num_rows;
%000001     reg [31:0] num_cols;
%000001     reg [31:0] total_blocks;
        
            // Address Pointers
%000001     reg [AXI_ADDR_W-1:0] current_axi_addr;
~000023     reg [31:0]           words_remaining;
            
            // Data Unpacking
%000001     reg [1:0]  header_word_idx;
%000002     reg [63:0] rdata_reg; // Buffer for unpacking
%000001     reg        rlast_reg; // Store last signal during unpacking
        
            // AXI Constants
            localparam [2:0] AXI_SIZE_64 = 3'b011; // 8 bytes
            localparam [1:0] AXI_BURST_INCR = 2'b01;
        
            // NEW: Drive the ID constantly
            assign m_axi_arid = STREAM_ID[AXI_ID_W-1:0];
        
            // ========================================================================
            // Main FSM
            // ========================================================================
 012713     always @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             state <= IDLE;
 000069             busy <= 1'b0;
 000069             done <= 1'b0;
 000069             error <= 1'b0;
                    
 000069             m_axi_arvalid <= 1'b0;
 000069             m_axi_rready <= 1'b0;
                    
 000069             row_ptr_we <= 1'b0;
 000069             col_idx_we <= 1'b0;
 000069             wgt_we <= 1'b0;
                    
 000069             row_ptr_addr <= 0;
 000069             col_idx_addr <= 0;
 000069             wgt_addr <= 0;
                    
 000069             current_axi_addr <= 0;
 000069             header_word_idx <= 0;
 000069             rdata_reg <= 0;
 000069             rlast_reg <= 0;
 012644         end else begin
                    // Default Control Signals
 012644             row_ptr_we <= 1'b0;
 012644             col_idx_we <= 1'b0;
 012644             wgt_we <= 1'b0;
 012644             m_axi_arvalid <= 1'b0; // Auto-clear unless set in state
                    
 012644             case (state)
 012565                 IDLE: begin
 012565                     done <= 1'b0;
~012564                     if (start) begin
%000001                         busy <= 1'b1;
%000001                         error <= 1'b0;
%000001                         current_axi_addr <= src_addr;
%000001                         state <= READ_HEADER;
                            end
                        end
        
                        // ------------------------------------------------------------
                        // Header Phase: Read 3 x 32-bit words (aligned to 64-bit)
                        // ------------------------------------------------------------
%000003                 READ_HEADER: begin
%000003                     m_axi_araddr <= current_axi_addr;
%000003                     m_axi_arlen  <= 8'd1; // Read 2 beats (16 bytes) to get 3 words safely
%000003                     m_axi_arsize <= AXI_SIZE_64;
%000003                     m_axi_arburst <= AXI_BURST_INCR;
%000003                     m_axi_arvalid <= 1'b1;
%000003                     m_axi_rready <= 1'b1;
                            
%000003                     header_word_idx <= 0;
                            
~012635                     if (m_axi_arready && m_axi_arvalid) begin
%000001                         m_axi_arvalid <= 1'b0; // Handshake done
%000001                         state <= WAIT_HEADER;
                            end
                        end
        
%000003                 WAIT_HEADER: begin
%000003                     m_axi_rready <= 1'b1;
%000002                     if (m_axi_rvalid) begin
%000002                         if (m_axi_rresp != 2'b00) begin
%000000                             error <= 1'b1;
%000000                             busy <= 1'b0;
%000000                             done <= 1'b1;
%000000                             state <= IDLE;
%000002                         end else begin
                                    // Unpack 64-bit data into 32-bit header registers
%000002                             case (header_word_idx)
%000001                                 0: begin
%000001                                     num_rows <= m_axi_rdata[31:0];
%000001                                     num_cols <= m_axi_rdata[63:32];
%000001                                     header_word_idx <= 2;
                                        end
%000001                                 2: begin
%000001                                     total_blocks <= m_axi_rdata[31:0];
%000001                                     header_word_idx <= 3;
                                        end
                                    endcase
                                    
%000001                             if (m_axi_rlast) begin
%000001                                 current_axi_addr <= current_axi_addr + 16; // Advance past header
%000001                                 state <= SETUP_ROW_PTR;
                                    end
                                end
                            end
                        end
        
                        // ------------------------------------------------------------
                        // Row Pointer Phase: Read (Num_Rows + 1) 32-bit words
                        // ------------------------------------------------------------
%000001                 SETUP_ROW_PTR: begin
%000001                     words_remaining <= num_rows + 1;
%000001                     row_ptr_addr <= 0;
%000001                     state <= READ_ROW_PTR;
                        end
        
 000014                 READ_ROW_PTR: begin
                            // Issue Burst
~012579                     if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
%000003                         m_axi_araddr <= current_axi_addr;
                                // Calculate burst length (beats = (words + 1) / 2)
                                // Max burst 16 beats = 32 words
%000003                         if (words_remaining > 32) 
%000000                             m_axi_arlen <= BURST_LEN;
                                else 
%000003                             m_axi_arlen <= ((words_remaining + 1) >> 1) - 1;
                                    
%000003                         m_axi_arvalid <= 1'b1;
                            end
        
~012635                     if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;
        
                            // Ready to accept data
 000014                     m_axi_rready <= 1'b1;
        
%000009                     if (m_axi_rvalid) begin
%000009                         if (m_axi_rresp != 2'b00) begin
%000000                             error <= 1'b1;
%000000                             busy <= 1'b0;
%000000                             done <= 1'b1;
%000000                             state <= IDLE;
%000009                         end else begin
                                    // Capture data for unpacking
%000009                             rdata_reg <= m_axi_rdata;
%000009                             rlast_reg <= m_axi_rlast;
        
                                    // Write Lower 32-bits immediately
%000005                             if (words_remaining > 0) begin
%000005                                 row_ptr_we <= 1'b1;
%000005                                 row_ptr_wdata <= m_axi_rdata[31:0];
%000005                                 row_ptr_addr <= row_ptr_addr + 1;
%000005                                 words_remaining <= words_remaining - 1;
                                    end
                                    
                                    // If more words remain in this beat (upper 32), go to unpack state
                                    // Note: If words_remaining was 1, we are done with this beat.
%000005                             if (words_remaining > 1) begin
%000004                                 m_axi_rready <= 1'b0; // Throttle AXI
%000004                                 state <= WRITE_ROW_PTR_HIGH;
%000005                             end else begin
                                        // Beat fully consumed
%000004                                 if (m_axi_rlast) begin
%000001                                     current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000001                                     if (words_remaining <= 1) state <= SETUP_COL_IDX;
                                        end
                                    end
                                end
                            end
                        end
        
%000004                 WRITE_ROW_PTR_HIGH: begin
%000004                     m_axi_rready <= 1'b0; // Hold off next beat
                            
                            // Write Upper 32-bits
%000004                     row_ptr_we <= 1'b1;
%000004                     row_ptr_wdata <= rdata_reg[63:32];
%000004                     row_ptr_addr <= row_ptr_addr + 1;
%000004                     words_remaining <= words_remaining - 1;
                            
                            // Check if we are done with the burst
%000004                     if (rlast_reg) begin
%000000                         current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                // If we just wrote the last needed word (or more), move on
%000000                         if (words_remaining <= 1) state <= SETUP_COL_IDX;
%000000                         else state <= READ_ROW_PTR; // Should not happen if logic is correct
%000004                     end else begin
%000004                         state <= READ_ROW_PTR; // Get next beat
                            end
                        end
        
                        // ------------------------------------------------------------
                        // Column Index Phase: Read Total_Blocks 16-bit words
                        // ------------------------------------------------------------
%000001                 SETUP_COL_IDX: begin
%000001                     words_remaining <= total_blocks;
%000001                     col_idx_addr <= 0;
%000001                     state <= READ_COL_IDX;
                        end
        
%000006                 READ_COL_IDX: begin
                            // Issue Burst
~012579                     if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
%000003                         m_axi_araddr <= current_axi_addr;
                                // Calculate burst length (beats = (words + 3) / 4)
                                // Max burst 16 beats = 64 words
%000003                         if (words_remaining > 64) 
%000000                             m_axi_arlen <= BURST_LEN;
                                else 
%000003                             m_axi_arlen <= ((words_remaining + 3) >> 2) - 1;
                                    
%000003                         m_axi_arvalid <= 1'b1;
                            end
        
~012635                     if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;
        
%000006                     m_axi_rready <= 1'b1;
        
%000005                     if (m_axi_rvalid) begin
%000001                         if (m_axi_rresp != 2'b00) begin
%000000                             error <= 1'b1;
%000000                             busy <= 1'b0;
%000000                             done <= 1'b1;
%000000                             state <= IDLE;
%000001                         end else begin
%000001                             rdata_reg <= m_axi_rdata;
%000001                             rlast_reg <= m_axi_rlast;
        
                                    // Write Index 0 (Bits 15:0)
%000001                             if (words_remaining > 0) begin
%000001                                 col_idx_we <= 1'b1;
%000001                                 col_idx_wdata <= m_axi_rdata[15:0];
%000001                                 col_idx_addr <= col_idx_addr + 1;
%000001                                 words_remaining <= words_remaining - 1;
                                    end
        
%000001                             if (words_remaining > 1) begin
%000001                                 m_axi_rready <= 1'b0; // Throttle
%000001                                 state <= WRITE_COL_IDX_1;
%000000                             end else begin
                                        // Beat fully consumed
%000000                                 if (m_axi_rlast) begin
%000000                                     current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000000                                     if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                                        end
                                    end
                                end
                            end
                        end
        
%000001                 WRITE_COL_IDX_1: begin
%000001                     m_axi_rready <= 1'b0;
                            // Write Index 1 (Bits 31:16)
%000001                     col_idx_we <= 1'b1;
%000001                     col_idx_wdata <= rdata_reg[31:16];
%000001                     col_idx_addr <= col_idx_addr + 1;
%000001                     words_remaining <= words_remaining - 1;
        
%000001                     if (words_remaining > 1) state <= WRITE_COL_IDX_2;
%000000                     else begin
%000000                         if (rlast_reg) begin
%000000                             current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000000                             state <= SETUP_WEIGHTS;
%000000                         end else state <= READ_COL_IDX;
                            end
                        end
        
%000001                 WRITE_COL_IDX_2: begin
%000001                     m_axi_rready <= 1'b0;
                            // Write Index 2 (Bits 47:32)
%000001                     col_idx_we <= 1'b1;
%000001                     col_idx_wdata <= rdata_reg[47:32];
%000001                     col_idx_addr <= col_idx_addr + 1;
%000001                     words_remaining <= words_remaining - 1;
        
%000001                     if (words_remaining > 1) state <= WRITE_COL_IDX_3;
%000000                     else begin
%000000                         if (rlast_reg) begin
%000000                             current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000000                             state <= SETUP_WEIGHTS;
%000000                         end else state <= READ_COL_IDX;
                            end
                        end
        
%000001                 WRITE_COL_IDX_3: begin
%000001                     m_axi_rready <= 1'b0;
                            // Write Index 3 (Bits 63:48)
%000001                     col_idx_we <= 1'b1;
%000001                     col_idx_wdata <= rdata_reg[63:48];
%000001                     col_idx_addr <= col_idx_addr + 1;
%000001                     words_remaining <= words_remaining - 1;
        
%000001                     if (rlast_reg) begin
%000001                         current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000001                         if (words_remaining <= 1) state <= SETUP_WEIGHTS;
%000000                         else state <= READ_COL_IDX; // Should not happen
%000000                     end else begin
%000000                         state <= READ_COL_IDX;
                            end
                        end
        
                        // ------------------------------------------------------------
                        // Weight Block Phase: Read Total_Blocks * 64 bytes
                        // ------------------------------------------------------------
%000001                 SETUP_WEIGHTS: begin
                            // Total bytes = total_blocks * 64
                            // Total 64-bit words = total_blocks * 8
%000001                     words_remaining <= total_blocks * 8; 
%000001                     wgt_addr <= 0;
%000001                     state <= READ_WEIGHTS;
                        end
        
 000042                 READ_WEIGHTS: begin
                            // Assert rready when waiting for data
 000042                     m_axi_rready <= 1'b1;
                            
~012579                     if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
%000006                         m_axi_araddr <= current_axi_addr;
%000006                         if (words_remaining > BURST_LEN) m_axi_arlen <= BURST_LEN;
%000000                         else m_axi_arlen <= words_remaining - 1;
%000006                         m_axi_arvalid <= 1'b1;
                            end
        
~012635                     if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;
        
                            // Only process data on valid AXI handshake (rvalid && rready)
 012596                     if (m_axi_rvalid && m_axi_rready) begin
~000032                         if (m_axi_rresp != 2'b00) begin
%000000                             error <= 1'b1;
%000000                             busy <= 1'b0;
%000000                             done <= 1'b1;
%000000                             state <= IDLE;
 000032                         end else begin
 000032                             wgt_we <= 1'b1;
 000032                             wgt_wdata <= m_axi_rdata; // Direct 64-bit write!
 000032                             wgt_addr <= wgt_addr + 8; // Byte address increment
 000032                             words_remaining <= words_remaining - 1;
        
~000030                             if (m_axi_rlast) begin
%000002                                 current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
%000001                                 if (words_remaining <= 1) state <= DONE_STATE;
                                    end
                                end
                            end
                        end
        
%000001                 DONE_STATE: begin
%000001                     busy <= 1'b0;
%000001                     done <= 1'b1;
~012642                     if (!start) state <= IDLE; // Handshake reset
                        end
                    endcase
                end
            end
        
        endmodule
        `default_nettype wire
        
