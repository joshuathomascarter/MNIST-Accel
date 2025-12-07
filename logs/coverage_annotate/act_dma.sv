//      // verilator_coverage annotation
        `timescale 1ns/1ps
        `default_nettype none
        
        module act_dma #(
            parameter AXI_ADDR_W = 32,
            parameter AXI_DATA_W = 64,
            parameter AXI_ID_W   = 4,       // NEW: ID Width
            parameter STREAM_ID  = 1,       // NEW: ID for this DMA (1 for Act, 0 for BSR)
            parameter BURST_LEN  = 8'd15    // 15 means 16 beats (0-based)
        )(
            // 1. System
 012713     input  wire                  clk,
%000007     input  wire                  rst_n,
        
            // 2. Control (CSR)
%000002     input  wire                  start,
%000001     input  wire [AXI_ADDR_W-1:0] src_addr,
%000001     input  wire [31:0]           transfer_length,
%000001     output reg                   done,
%000001     output reg                   busy,
%000000     output reg                   error,
        
            // 3. AXI4 Master Interface (To Bridge)
            // -- Read Address Channel --
%000001     output wire [AXI_ID_W-1:0]   m_axi_arid,    // NEW: Output our ID
%000001     output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
%000001     output reg [7:0]             m_axi_arlen,
%000001     output reg [2:0]             m_axi_arsize,
%000001     output reg [1:0]             m_axi_arburst,
%000001     output reg                   m_axi_arvalid,
%000001     input  wire                  m_axi_arready,
        
            // -- Read Data Channel --
%000001     input  wire [AXI_ID_W-1:0]   m_axi_rid,     // NEW: Input ID from Bridge
%000007     input  wire [AXI_DATA_W-1:0] m_axi_rdata,
%000000     input  wire [1:0]            m_axi_rresp,
%000006     input  wire                  m_axi_rlast,
%000001     input  wire                  m_axi_rvalid,
%000001     output reg                   m_axi_rready,
        
            // 4. Buffer Interface (To act_buffer)
%000001     output reg                   act_we,
%000004     output reg [AXI_ADDR_W-1:0]  act_addr,
%000004     output reg [AXI_DATA_W-1:0]  act_wdata
        );
        
            // ========================================================================
            // Internal State & Registers
            // ========================================================================
            typedef enum logic [1:0] {
                IDLE,
                SEND_ADDR,
                READ_DATA,
                DONE_STATE
            } state_t;
        
%000002     state_t state;
%000001     reg [AXI_ADDR_W-1:0] current_axi_addr;
%000004     reg [31:0]           bytes_remaining;
        
            // AXI Constants
            localparam [2:0] AXI_SIZE_64 = 3'b011;
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
 000069             act_we <= 1'b0;
 000069             act_addr <= 0;
 000069             current_axi_addr <= 0;
 000069             bytes_remaining <= 0;
 012644         end else begin
                    // Default Control Signals
 012644             act_we <= 1'b0;
 012644             done <= 1'b0;
        
 012644             case (state)
 012631                 IDLE: begin
~012630                     if (start) begin
%000001                         busy <= 1'b1;
%000001                         error <= 1'b0;
%000001                         current_axi_addr <= src_addr;
%000001                         bytes_remaining <= transfer_length;
%000001                         act_addr <= 0;
%000001                         state <= SEND_ADDR;
                            end
                        end
        
%000003                 SEND_ADDR: begin
%000003                     m_axi_araddr  <= current_axi_addr;
%000003                     m_axi_arsize  <= AXI_SIZE_64;
%000003                     m_axi_arburst <= AXI_BURST_INCR;
%000003                     m_axi_arvalid <= 1'b1;
                            
%000003                     if (bytes_remaining > (BURST_LEN + 1) * 8) 
%000000                         m_axi_arlen <= BURST_LEN;
                            else 
%000003                         m_axi_arlen <= ((bytes_remaining + 7) >> 3) - 1;
        
~012643                     if (m_axi_arready && m_axi_arvalid) begin
%000001                         m_axi_arvalid <= 1'b0;
%000001                         m_axi_rready  <= 1'b1;
%000001                         state <= READ_DATA;
                            end
                        end
        
%000009                 READ_DATA: begin
                            // Note: We trust the Bridge to only assert m_axi_rvalid 
                            // when the data is actually for us (based on ID routing).
%000008                     if (m_axi_rvalid) begin
%000008                         if (m_axi_rresp != 2'b00) begin
%000000                             error <= 1'b1;
%000000                             busy <= 1'b0;
%000000                             done <= 1'b1; // Pulse done to wake up controller
%000000                             state <= IDLE;
%000008                         end else begin
%000008                             act_we    <= 1'b1;
%000008                             act_wdata <= m_axi_rdata;
%000008                             act_addr  <= act_addr + 1;
        
%000008                             if (bytes_remaining >= 8)
%000008                                 bytes_remaining <= bytes_remaining - 8;
                                    else
%000000                                 bytes_remaining <= 0;
        
%000007                             if (m_axi_rlast) begin
%000001                                 m_axi_rready <= 1'b0;
%000001                                 current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
        
%000001                                 if (bytes_remaining <= 8) begin
%000001                                     state <= DONE_STATE;
%000000                                 end else begin
%000000                                     state <= SEND_ADDR;
                                        end
                                    end
                                end
                            end
                        end
        
%000001                 DONE_STATE: begin
%000001                     busy <= 1'b0;
%000001                     done <= 1'b1;
%000001                     state <= IDLE;
                        end
                    endcase
                end
            end
        
            // ========================================================================
            // Simulation-Only Assertions (Safety Checks)
            // ========================================================================
            `ifndef VERILATOR
            // synthesis translate_off
            
            // 1. Check for Zero Length (The Crash Preventer)
            always @(posedge clk) begin
                if (start && transfer_length == 0) begin
                    $error("ACT_DMA ERROR: Attempted to start DMA with length = 0!");
                end
            end
        
            // 2. Check for 64-bit Alignment (The Data Corruption Preventer)
            always @(posedge clk) begin
                if (start && (src_addr[2:0] != 3'b000)) begin
                    $error("ACT_DMA ERROR: Source Address %h is not 64-bit aligned!", src_addr);
                end
            end
        
            // 3. Check for 4KB Boundary Crossing (The AXI Protocol Enforcer)
            always @(posedge clk) begin
                if (state == SEND_ADDR && m_axi_arvalid) begin
                    if ((m_axi_araddr[11:0] + ({24'd0, m_axi_arlen} + 32'd1) * 8) > 32'hFFF) begin
                        $warning("ACT_DMA WARNING: Burst at %h might cross 4KB boundary!", m_axi_araddr);
                    end
                end
            end
            
            // synthesis translate_on
            `endif
        
        endmodule
        
