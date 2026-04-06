// ===========================================================================
// eth_mac_rx.sv — Ethernet MAC Receive (AXI-Stream output)
// ===========================================================================
// Strips preamble/SFD, emits payload bytes on AXI-Stream, checks CRC32.
// Designed for RGMII/GMII receive data (8-bit) at 125 MHz.
//
// AXI-Stream output:
//   tdata[7:0]  — byte
//   tvalid      — valid beat
//   tlast       — last beat of frame
//   tuser       — 1 = CRC error (asserted with tlast)
//
// Resource estimate: ~200 LUTs, 0 DSP, 0 BRAM
// ===========================================================================

module eth_mac_rx #(
    parameter int DATA_W = 8
)(
    input  logic             clk,
    input  logic             rst_n,

    // GMII / MII receive interface
    input  logic [7:0]       gmii_rxd,
    input  logic             gmii_rx_dv,     // data valid
    input  logic             gmii_rx_er,     // error (optional)

    // AXI-Stream output
    output logic [DATA_W-1:0] m_axis_tdata,
    output logic              m_axis_tvalid,
    output logic              m_axis_tlast,
    output logic              m_axis_tuser,   // CRC error flag
    input  logic              m_axis_tready
);

    // -----------------------------------------------------------------------
    // CRC-32 (Ethernet polynomial 0x04C11DB7, bit-reversed)
    // -----------------------------------------------------------------------
    function [31:0] crc32_byte;
        input [31:0] crc_in;
        input [7:0]  data;
        reg [31:0] c;
        integer i;
        begin
            c = crc_in;
            for (i = 0; i < 8; i = i + 1) begin
                if (c[0] ^ data[i])
                    c = {1'b0, c[31:1]} ^ 32'hEDB88320;
                else
                    c = {1'b0, c[31:1]};
            end
            crc32_byte = c;
        end
    endfunction

    // -----------------------------------------------------------------------
    // FSM states
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_PREAMBLE,
        S_PAYLOAD,
        S_CHECK_CRC,
        S_ERROR
    } state_t;

    state_t state, state_next;

    logic [31:0] crc_reg, crc_next;
    logic [15:0] byte_cnt, byte_cnt_next;
    logic [7:0]  rx_data_d;   // 1-cycle delayed data for CRC pipeline
    logic        rx_dv_d;

    // Residue for correct CRC-32 = 0xC704DD7B
    localparam logic [31:0] CRC_RESIDUE = 32'hC704DD7B;

    // -----------------------------------------------------------------------
    // Pipeline delay for GMII signals (helps timing)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_data_d <= '0;
            rx_dv_d   <= 1'b0;
        end else begin
            rx_data_d <= gmii_rxd;
            rx_dv_d   <= gmii_rx_dv;
        end
    end

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            crc_reg  <= 32'hFFFF_FFFF;
            byte_cnt <= '0;
        end else begin
            state    <= state_next;
            crc_reg  <= crc_next;
            byte_cnt <= byte_cnt_next;
        end
    end

    // -----------------------------------------------------------------------
    // Next-state & output logic
    // -----------------------------------------------------------------------
    always_comb begin
        state_next    = state;
        crc_next      = crc_reg;
        byte_cnt_next = byte_cnt;

        m_axis_tdata  = '0;
        m_axis_tvalid = 1'b0;
        m_axis_tlast  = 1'b0;
        m_axis_tuser  = 1'b0;

        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                crc_next      = 32'hFFFF_FFFF;
                byte_cnt_next = '0;
                if (gmii_rx_dv && gmii_rxd == 8'h55)
                    state_next = S_PREAMBLE;
            end

            // -----------------------------------------------------------
            S_PREAMBLE: begin
                if (!gmii_rx_dv)
                    state_next = S_IDLE;
                else if (gmii_rxd == 8'hD5)      // SFD
                    state_next = S_PAYLOAD;
                else if (gmii_rxd != 8'h55)
                    state_next = S_ERROR;
            end

            // -----------------------------------------------------------
            S_PAYLOAD: begin
                if (!rx_dv_d) begin
                    // End of frame — check CRC
                    state_next = S_CHECK_CRC;
                end else begin
                    m_axis_tdata  = rx_data_d;
                    m_axis_tvalid = 1'b1;
                    crc_next      = crc32_byte(crc_reg, rx_data_d);
                    byte_cnt_next = byte_cnt + 1;
                end
            end

            // -----------------------------------------------------------
            S_CHECK_CRC: begin
                m_axis_tvalid = 1'b1;
                m_axis_tlast  = 1'b1;
                m_axis_tdata  = '0;
                // CRC residue check: after running CRC over data+FCS, residue
                // should be CRC_RESIDUE for correct frames.
                m_axis_tuser  = (crc_reg != CRC_RESIDUE) ? 1'b1 : 1'b0;
                state_next    = S_IDLE;
            end

            // -----------------------------------------------------------
            S_ERROR: begin
                if (!gmii_rx_dv)
                    state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

endmodule
