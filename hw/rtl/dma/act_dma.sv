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
    input  wire                  clk,
    input  wire                  rst_n,

    // 2. Control (CSR)
    input  wire                  start,
    input  wire [AXI_ADDR_W-1:0] src_addr,
    input  wire [31:0]           transfer_length,
    output reg                   done,
    output reg                   busy,
    output reg                   error,

    // 3. AXI4 Master Interface (To Bridge)
    // -- Read Address Channel --
    output wire [AXI_ID_W-1:0]   m_axi_arid,    // NEW: Output our ID
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    output reg [7:0]             m_axi_arlen,
    output reg [2:0]             m_axi_arsize,
    output reg [1:0]             m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    // -- Read Data Channel --
    input  wire [AXI_ID_W-1:0]   m_axi_rid,     // NEW: Input ID from Bridge
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output reg                   m_axi_rready,

    // 4. Buffer Interface (To act_buffer)
    output reg                   act_we,
    output reg [AXI_ADDR_W-1:0]  act_addr,
    output reg [AXI_DATA_W-1:0]  act_wdata
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

    state_t state;
    reg [AXI_ADDR_W-1:0] current_axi_addr;
    reg [31:0]           bytes_remaining;

    // AXI Constants
    localparam [2:0] AXI_SIZE_64 = 3'b011;
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
            act_we <= 1'b0;
            act_addr <= 0;
            current_axi_addr <= 0;
            bytes_remaining <= 0;
        end else begin
            // Default Control Signals
            act_we <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        error <= 1'b0;
                        current_axi_addr <= src_addr;
                        bytes_remaining <= transfer_length;
                        act_addr <= 0;
                        state <= SEND_ADDR;
                    end
                end

                SEND_ADDR: begin
                    m_axi_araddr  <= current_axi_addr;
                    m_axi_arsize  <= AXI_SIZE_64;
                    m_axi_arburst <= AXI_BURST_INCR;
                    m_axi_arvalid <= 1'b1;
                    
                    if (bytes_remaining > (BURST_LEN + 1) * 8) 
                        m_axi_arlen <= BURST_LEN;
                    else 
                        m_axi_arlen <= ((bytes_remaining + 7) >> 3) - 1;

                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        state <= READ_DATA;
                    end
                end

                READ_DATA: begin
                    // Note: We trust the Bridge to only assert m_axi_rvalid 
                    // when the data is actually for us (based on ID routing).
                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1; // Pulse done to wake up controller
                            state <= IDLE;
                        end else begin
                            act_we    <= 1'b1;
                            act_wdata <= m_axi_rdata;
                            act_addr  <= act_addr + 1;

                            if (bytes_remaining >= 8)
                                bytes_remaining <= bytes_remaining - 8;
                            else
                                bytes_remaining <= 0;

                            if (m_axi_rlast) begin
                                m_axi_rready <= 1'b0;
                                current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);

                                if (bytes_remaining <= 8) begin
                                    state <= DONE_STATE;
                                end else begin
                                    state <= SEND_ADDR;
                                end
                            end
                        end
                    end
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= IDLE;
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
