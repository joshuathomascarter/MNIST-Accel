// act_dma.sv — AXI4 read-only DMA for activation transfer (DDR → act_buffer BRAM)
// FSM: IDLE → SEND_ADDR → READ_DATA (loop) → DONE → IDLE
// 16-beat bursts (128 bytes). STREAM_ID=1 distinguishes from bsr_dma.

`timescale 1ns/1ps
`default_nettype none

module act_dma #(
    parameter AXI_ADDR_W = 32,
    parameter AXI_DATA_W = 64,
    parameter AXI_ID_W   = 4,
    parameter STREAM_ID  = 1,   // bsr_dma=0, act_dma=1
    parameter BURST_LEN  = 8'd15 // 16 beats = 128 bytes
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Control (from CSR)
    input  wire                  start,
    input  wire [AXI_ADDR_W-1:0] src_addr,        // DDR byte address (64-bit aligned)
    input  wire [31:0]           transfer_length,  // bytes to transfer
    output reg                   done,
    output reg                   busy,
    output reg                   error,

    // AXI4 read address channel
    output wire [AXI_ID_W-1:0]   m_axi_arid,
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    output reg [7:0]             m_axi_arlen,
    output reg [2:0]             m_axi_arsize,
    output reg [1:0]             m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    // AXI4 read data channel
    input  wire [AXI_ID_W-1:0]   m_axi_rid,    // unused: single-ID stream
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output reg                   m_axi_rready,

    // Buffer write interface (to act_buffer BRAM)
    output reg                   act_we,
    output reg [AXI_ADDR_W-1:0]  act_addr,
    output reg [AXI_DATA_W-1:0]  act_wdata
);

    typedef enum logic [1:0] {
        IDLE, SEND_ADDR, READ_DATA, DONE_STATE
    } state_t;

    state_t state;

    reg [AXI_ADDR_W-1:0] current_axi_addr;
    reg [31:0]           bytes_remaining;

    localparam [2:0] AXI_SIZE_64    = 3'b011;  // 8 bytes/beat
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    assign m_axi_arid = STREAM_ID[AXI_ID_W-1:0];

    // R13: 4KB boundary guard — cap burst to stay within current page
    wire [9:0] page_max_beats = (13'h1000 - {1'b0, current_axi_addr[11:0]}) >> 3;
    // WIDTHTRUNC fix: explicitly cast burst length calculation to 32-bit
    wire [31:0] max_burst_bytes = 32'd8 * (32'd1 + {24'd0, BURST_LEN});
    wire [7:0] data_arlen     = (bytes_remaining > max_burst_bytes)
                               ? BURST_LEN
                               : ((bytes_remaining + 32'd7) >> 3) - 8'd1;
    wire [7:0] safe_arlen     = ({2'b0, data_arlen} + 10'd1 > page_max_beats)
                               ? page_max_beats[7:0] - 8'd1
                               : data_arlen;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            error            <= 1'b0;
            m_axi_arvalid    <= 1'b0;
            m_axi_rready     <= 1'b0;
            act_we           <= 1'b0;
            act_addr         <= 0;
            current_axi_addr <= 0;
            bytes_remaining  <= 0;
        end else begin
            act_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy             <= 1'b1;
                        error            <= 1'b0;
                        current_axi_addr <= src_addr;
                        bytes_remaining  <= transfer_length;
                        act_addr         <= 0;
                        state            <= SEND_ADDR;
                    end
                end

                SEND_ADDR: begin
                    m_axi_araddr  <= current_axi_addr;
                    m_axi_arsize  <= AXI_SIZE_64;
                    m_axi_arburst <= AXI_BURST_INCR;
                    m_axi_arvalid <= 1'b1;
                    m_axi_arlen   <= safe_arlen; // R13: 4KB-safe burst length

                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        state         <= READ_DATA;
                    end
                end

                READ_DATA: begin
                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy  <= 1'b0;
                            done  <= 1'b1;
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
                                m_axi_rready     <= 1'b0;
                                current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);

                                if (bytes_remaining <= 8)
                                    state <= DONE_STATE;
                                else
                                    state <= SEND_ADDR;
                            end
                        end
                    end
                end

                DONE_STATE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    // Assertions (simulation only)
    `ifndef VERILATOR
    // synthesis translate_off
    always @(posedge clk) begin
        if (start && transfer_length == 0)
            $error("ACT_DMA: zero-length transfer");
        if (start && (src_addr[2:0] != 3'b000))
            $error("ACT_DMA: src_addr %h not 64-bit aligned", src_addr);
    end
    always @(posedge clk) begin
        if (state == SEND_ADDR && m_axi_arvalid)
            if ((m_axi_araddr[11:0] + ({24'd0, m_axi_arlen} + 32'd1) * 8) > 32'hFFF)
                $error("ACT_DMA: 4KB guard failed at %h", m_axi_araddr);
    end
    // synthesis translate_on
    `endif

endmodule
