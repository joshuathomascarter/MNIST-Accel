// ===========================================================================
// eth_udp_parser.sv — Ethernet/IP/UDP Header Parser (AXI-Stream → AXI-Stream)
// ===========================================================================
// Consumes AXI-Stream byte frames from eth_mac_rx, strips ETH/IP/UDP headers,
// and emits only the UDP payload with metadata sideband.
//
// Protocol stack:
//   ETH:  14 bytes (DA[6] + SA[6] + EtherType[2])  — expect 0x0800
//   IP:   20 bytes (no options)                       — protocol=0x11 (UDP)
//   UDP:   8 bytes (SrcPort + DstPort + Len + Cksum)
//
// Sideband output (valid with first payload beat):
//   src_ip[31:0], dst_ip[31:0], src_port[15:0], dst_port[15:0], udp_len[15:0]
//
// Resource estimate: ~180 LUTs, 0 DSP, 0 BRAM
// ===========================================================================

module eth_udp_parser (
    input  logic        clk,
    input  logic        rst_n,

    // AXI-Stream input (from eth_mac_rx)
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    input  logic        s_axis_tlast,
    input  logic        s_axis_tuser,       // CRC error
    output logic        s_axis_tready,

    // AXI-Stream output (UDP payload only)
    output logic [7:0]  m_axis_tdata,
    output logic        m_axis_tvalid,
    output logic        m_axis_tlast,
    input  logic        m_axis_tready,

    // Sideband — latched on first payload byte
    output logic [31:0] src_ip,
    output logic [31:0] dst_ip,
    output logic [15:0] src_port,
    output logic [15:0] dst_port,
    output logic [15:0] udp_len,
    output logic        hdr_valid,          // pulses 1 cycle when header latched
    output logic        frame_error         // bad EtherType, IP proto, or CRC
);

    // -----------------------------------------------------------------------
    // Header byte offsets (0-indexed from first byte after SFD)
    // -----------------------------------------------------------------------
    localparam int ETH_LEN  = 14;
    localparam int IP_LEN   = 20;
    localparam int UDP_LEN  = 8;
    localparam int HDR_TOTAL = ETH_LEN + IP_LEN + UDP_LEN;  // 42

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_ETH_HDR,
        S_IP_HDR,
        S_UDP_HDR,
        S_PAYLOAD,
        S_DROP
    } state_t;

    state_t state, state_next;

    logic [15:0] byte_cnt, byte_cnt_next;

    // Header accumulators
    logic [15:0] ethertype_reg, ethertype_next;
    logic [7:0]  ip_proto_reg, ip_proto_next;
    logic [31:0] src_ip_reg, src_ip_next;
    logic [31:0] dst_ip_reg, dst_ip_next;
    logic [15:0] src_port_reg, src_port_next;
    logic [15:0] dst_port_reg, dst_port_next;
    logic [15:0] udp_len_reg, udp_len_next;
    logic [15:0] payload_left, payload_left_next;

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            byte_cnt     <= '0;
            ethertype_reg <= '0;
            ip_proto_reg  <= '0;
            src_ip_reg    <= '0;
            dst_ip_reg    <= '0;
            src_port_reg  <= '0;
            dst_port_reg  <= '0;
            udp_len_reg   <= '0;
            payload_left  <= '0;
        end else begin
            state        <= state_next;
            byte_cnt     <= byte_cnt_next;
            ethertype_reg <= ethertype_next;
            ip_proto_reg  <= ip_proto_next;
            src_ip_reg    <= src_ip_next;
            dst_ip_reg    <= dst_ip_next;
            src_port_reg  <= src_port_next;
            dst_port_reg  <= dst_port_next;
            udp_len_reg   <= udp_len_next;
            payload_left  <= payload_left_next;
        end
    end

    // -----------------------------------------------------------------------
    // Next-state logic
    // -----------------------------------------------------------------------
    always_comb begin
        state_next      = state;
        byte_cnt_next   = byte_cnt;
        ethertype_next  = ethertype_reg;
        ip_proto_next   = ip_proto_reg;
        src_ip_next     = src_ip_reg;
        dst_ip_next     = dst_ip_reg;
        src_port_next   = src_port_reg;
        dst_port_next   = dst_port_reg;
        udp_len_next    = udp_len_reg;
        payload_left_next = payload_left;

        s_axis_tready   = 1'b1;          // always accept
        m_axis_tdata    = '0;
        m_axis_tvalid   = 1'b0;
        m_axis_tlast    = 1'b0;
        hdr_valid       = 1'b0;
        frame_error     = 1'b0;

        // Sideband outputs — hold registered values
        src_ip   = src_ip_reg;
        dst_ip   = dst_ip_reg;
        src_port = src_port_reg;
        dst_port = dst_port_reg;
        udp_len  = udp_len_reg;

        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                byte_cnt_next = '0;
                if (s_axis_tvalid && !s_axis_tlast) begin
                    state_next    = S_ETH_HDR;
                    byte_cnt_next = 16'd1;
                end
            end

            // -----------------------------------------------------------
            S_ETH_HDR: begin
                if (s_axis_tvalid) begin
                    byte_cnt_next = byte_cnt + 1;
                    // EtherType at bytes 12-13
                    if (byte_cnt == 12)
                        ethertype_next[15:8] = s_axis_tdata;
                    if (byte_cnt == 13)
                        ethertype_next[7:0]  = s_axis_tdata;

                    if (byte_cnt == (ETH_LEN - 1)) begin
                        byte_cnt_next = '0;
                        if (ethertype_next != 16'h0800)
                            state_next = S_DROP;
                        else
                            state_next = S_IP_HDR;
                    end

                    if (s_axis_tlast)
                        state_next = S_IDLE;  // runt
                end
            end

            // -----------------------------------------------------------
            S_IP_HDR: begin
                if (s_axis_tvalid) begin
                    byte_cnt_next = byte_cnt + 1;
                    // Protocol at byte 9
                    if (byte_cnt == 9)
                        ip_proto_next = s_axis_tdata;
                    // Src IP at bytes 12-15
                    case (byte_cnt)
                        12: src_ip_next[31:24] = s_axis_tdata;
                        13: src_ip_next[23:16] = s_axis_tdata;
                        14: src_ip_next[15:8]  = s_axis_tdata;
                        15: src_ip_next[7:0]   = s_axis_tdata;
                        default: ;
                    endcase
                    // Dst IP at bytes 16-19
                    case (byte_cnt)
                        16: dst_ip_next[31:24] = s_axis_tdata;
                        17: dst_ip_next[23:16] = s_axis_tdata;
                        18: dst_ip_next[15:8]  = s_axis_tdata;
                        19: dst_ip_next[7:0]   = s_axis_tdata;
                        default: ;
                    endcase

                    if (byte_cnt == (IP_LEN - 1)) begin
                        byte_cnt_next = '0;
                        if (ip_proto_next != 8'h11)
                            state_next = S_DROP;
                        else
                            state_next = S_UDP_HDR;
                    end

                    if (s_axis_tlast)
                        state_next = S_IDLE;
                end
            end

            // -----------------------------------------------------------
            S_UDP_HDR: begin
                if (s_axis_tvalid) begin
                    byte_cnt_next = byte_cnt + 1;
                    case (byte_cnt)
                        0: src_port_next[15:8] = s_axis_tdata;
                        1: src_port_next[7:0]  = s_axis_tdata;
                        2: dst_port_next[15:8] = s_axis_tdata;
                        3: dst_port_next[7:0]  = s_axis_tdata;
                        4: udp_len_next[15:8]  = s_axis_tdata;
                        5: udp_len_next[7:0]   = s_axis_tdata;
                        default: ;
                    endcase

                    if (byte_cnt == (UDP_LEN - 1)) begin
                        byte_cnt_next     = '0;
                        hdr_valid         = 1'b1;
                        // Payload bytes = udp_len - 8 (UDP header)
                        payload_left_next = udp_len_next - 16'd8;
                        state_next        = S_PAYLOAD;
                    end

                    if (s_axis_tlast)
                        state_next = S_IDLE;
                end
            end

            // -----------------------------------------------------------
            S_PAYLOAD: begin
                if (s_axis_tvalid) begin
                    m_axis_tdata  = s_axis_tdata;
                    m_axis_tvalid = 1'b1;
                    payload_left_next = payload_left - 1;

                    if (payload_left == 1 || s_axis_tlast) begin
                        m_axis_tlast = 1'b1;
                        if (s_axis_tuser)
                            frame_error = 1'b1;
                        state_next   = S_IDLE;
                    end
                end
            end

            // -----------------------------------------------------------
            S_DROP: begin
                frame_error = 1'b1;
                if (s_axis_tvalid && s_axis_tlast)
                    state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

endmodule
