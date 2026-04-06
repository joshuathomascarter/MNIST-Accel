// =============================================================================
// noc_network_interface.sv — Network Interface (Packetizer/Depacketizer)
// =============================================================================
// Bridges between the tile's AXI/OBI bus and the NoC local port.
// 
// TX path: AXI requests → packetize into flit stream → inject to NoC
// RX path: NoC flits → depacketize → AXI responses
//
// Supports:
//   - Read (AR→R) and Write (AW/W→B) AXI transactions
//   - Sparsity hint CSR: tile firmware sets a "sparse" flag that the NI
//     embeds into HEAD flit msg_type field for the VC allocator

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module noc_network_interface #(
  parameter int NODE_ID    = 0,
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int NUM_VCS    = noc_pkg::NUM_VCS,
  parameter int GW_NODE_ID = noc_pkg::NUM_NODES - 1  // DMA gateway node
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   clk_en,

  // =====================================================================
  // Tile-side AXI-like interface (simplified for NI)
  // =====================================================================
  // --- Write request ---
  input  logic                   aw_valid,
  output logic                   aw_ready,
  input  logic [ADDR_WIDTH-1:0]  aw_addr,
  input  logic [7:0]             aw_len,
  input  logic [3:0]             aw_id,

  input  logic                   w_valid,
  output logic                   w_ready,
  input  logic [DATA_WIDTH-1:0]  w_data,
  input  logic                   w_last,

  output logic                   b_valid,
  input  logic                   b_ready,
  output logic [3:0]             b_id,
  output logic [1:0]             b_resp,

  // --- Read request ---
  input  logic                   ar_valid,
  output logic                   ar_ready,
  input  logic [ADDR_WIDTH-1:0]  ar_addr,
  input  logic [3:0]             ar_id,
  input  logic [7:0]             ar_len,

  output logic                   r_valid,
  input  logic                   r_ready,
  output logic [DATA_WIDTH-1:0]  r_data,
  output logic [3:0]             r_id,
  output logic [1:0]             r_resp,
  output logic                   r_last,

  // =====================================================================
  // Sparsity hint CSR
  // =====================================================================
  input  logic                   sparse_hint,  // Set by tile controller

  // =====================================================================
  // NoC local port
  // =====================================================================
  output flit_t                  noc_flit_out,
  output logic                   noc_valid_out,
  input  logic [NUM_VCS-1:0]     noc_credit_in,

  input  flit_t                  noc_flit_in,
  input  logic                   noc_valid_in,
  output logic [NUM_VCS-1:0]     noc_credit_out
);

  localparam int PAYLOAD_BITS = noc_pkg::PAYLOAD_HI - noc_pkg::PAYLOAD_LO + 1;

  // =========================================================================
  // Address → Node ID mapping
  // =========================================================================
  // Addresses in DRAM range (0x4000_0000+) go to the DMA gateway node.
  // Other addresses use addr[31:28] as direct tile ID (for inter-tile comms).
  function automatic [NODE_BITS-1:0] addr_to_node;
    input [ADDR_WIDTH-1:0] addr;
    begin
      if (addr[31:28] >= 4'h4)
        addr_to_node = NODE_BITS'(GW_NODE_ID);
      else
        addr_to_node = NODE_BITS'(addr[31:28]);
    end
  endfunction

  // =========================================================================
  // TX Packetizer
  // =========================================================================
  typedef enum logic [2:0] {
    TX_IDLE,
    TX_WR_HEAD,
    TX_WR_DATA,
    TX_WR_TAIL,
    TX_RD_HEAD,
    TX_RD_TAIL
  } tx_state_e;

  tx_state_e tx_state;

  logic [VC_BITS-1:0]     tx_vc;
  logic [NODE_BITS-1:0]   tx_dst;
  msg_type_e              tx_msg;
  logic [ADDR_WIDTH-1:0]  tx_addr_reg;
  logic [7:0]             tx_len_reg;
  logic [3:0]             tx_id_reg;

  // Credit tracking for TX
  logic [$clog2(BUF_DEPTH+1)-1:0] tx_credits [NUM_VCS];
  logic tx_has_credit;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int v = 0; v < NUM_VCS; v++)
        tx_credits[v] <= $clog2(BUF_DEPTH+1)'(BUF_DEPTH);
    end else if (clk_en) begin
      for (int v = 0; v < NUM_VCS; v++) begin
        logic inc, dec;
        inc = noc_credit_in[v];
        dec = noc_valid_out && (tx_vc == VC_BITS'(v));
        case ({inc, dec})
          2'b10:   tx_credits[v] <= tx_credits[v] + 1;
          2'b01:   tx_credits[v] <= tx_credits[v] - 1;
          default: tx_credits[v] <= tx_credits[v];
        endcase
      end
    end
  end

  assign tx_has_credit = (tx_credits[tx_vc] != '0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx_state <= TX_IDLE;
      tx_vc    <= '0;
      tx_dst   <= '0;
      tx_msg   <= MSG_DATA;
      tx_addr_reg <= '0;
      tx_len_reg  <= '0;
      tx_id_reg   <= '0;
    end else if (clk_en) begin
      case (tx_state)
        TX_IDLE: begin
          if (aw_valid) begin
            tx_dst   <= addr_to_node(aw_addr);
            tx_vc    <= sparse_hint ? VC_BITS'(NUM_VCS - 1) : '0;
            tx_msg   <= sparse_hint ? MSG_SPARSE_HINT : MSG_WRITE_REQ;
            tx_addr_reg <= aw_addr;
            tx_len_reg  <= aw_len;
            tx_id_reg   <= aw_id;
            tx_state <= TX_WR_HEAD;
          end else if (ar_valid) begin
            tx_dst   <= addr_to_node(ar_addr);
            tx_vc    <= sparse_hint ? VC_BITS'(NUM_VCS - 1) : '0;
            tx_msg   <= sparse_hint ? MSG_SPARSE_HINT : MSG_READ_REQ;
            tx_addr_reg <= ar_addr;
            tx_len_reg  <= ar_len;
            tx_id_reg   <= ar_id;
            tx_state <= TX_RD_HEAD;
          end
        end

        TX_WR_HEAD: begin
          if (tx_has_credit)
            tx_state <= TX_WR_DATA;
        end

        TX_WR_DATA: begin
          if (tx_has_credit && w_valid) begin
            if (w_last)
              tx_state <= TX_IDLE;
          end
        end

        TX_RD_HEAD: begin
          if (tx_has_credit)
            tx_state <= TX_IDLE;  // Read request is single HEAD_TAIL flit
        end

        default: tx_state <= TX_IDLE;
      endcase
    end
  end

  // TX flit output
  always_comb begin
    noc_flit_out  = '0;
    noc_valid_out = 1'b0;
    aw_ready      = 1'b0;
    w_ready       = 1'b0;
    ar_ready      = 1'b0;

    case (tx_state)
      TX_IDLE: begin
        aw_ready = 1'b1;  // Accept AW in idle
        ar_ready = !aw_valid;  // AR only if no AW (write priority)
      end

      TX_WR_HEAD: begin
        if (tx_has_credit) begin
          noc_valid_out = 1'b1;
          // Pack: [47:16]=addr, [15:12]=burst_len_minus_1, [11:8]=txn_id, [7:0]=0
          noc_flit_out  = make_head_flit(
            NODE_BITS'(NODE_ID),
            tx_dst,
            tx_vc,
            tx_msg,
            {tx_addr_reg, tx_len_reg[3:0], tx_id_reg, 8'h0}
          );
        end
      end

      TX_WR_DATA: begin
        if (tx_has_credit && w_valid) begin
          noc_valid_out       = 1'b1;
          noc_flit_out.vc_id  = tx_vc;
          w_ready             = 1'b1;

          if (w_last) begin
            noc_flit_out.flit_type = FLIT_TAIL;
          end else begin
            noc_flit_out.flit_type = FLIT_BODY;
          end
          noc_flit_out.payload = PAYLOAD_BITS'(w_data);
        end
      end

      TX_RD_HEAD: begin
        if (tx_has_credit) begin
          noc_valid_out = 1'b1;
          // Pack: [47:16]=addr, [15:12]=burst_len_minus_1, [11:8]=txn_id, [7:0]=0
          noc_flit_out  = make_head_flit(
            NODE_BITS'(NODE_ID),
            tx_dst,
            tx_vc,
            tx_msg,
            {tx_addr_reg, tx_len_reg[3:0], tx_id_reg, 8'h0}
          );
          noc_flit_out.flit_type = FLIT_HEADTAIL;
          ar_ready = 1'b1;
        end
      end

      default: ;
    endcase
  end

  // =========================================================================
  // RX Depacketizer
  // =========================================================================
  typedef enum logic [1:0] {
    RX_IDLE,
    RX_RD_DATA,
    RX_WR_RESP
  } rx_state_e;

  rx_state_e rx_state;
  logic [3:0] rx_id;
  logic [7:0] rx_beat_cnt;
  logic [7:0] rx_beat_total;
  logic                   rx_hold_valid;
  logic [DATA_WIDTH-1:0]  rx_hold_data;
  logic                   rx_hold_last;
  logic                   rx_accept_flit;

  always_comb begin
    rx_accept_flit = 1'b0;

    case (rx_state)
      RX_IDLE: begin
        if (noc_valid_in &&
            (noc_flit_in.flit_type == FLIT_HEAD ||
             noc_flit_in.flit_type == FLIT_HEADTAIL) &&
            ((noc_flit_in.msg_type == MSG_READ_RESP) ||
             (noc_flit_in.msg_type == MSG_WRITE_ACK))) begin
          rx_accept_flit = 1'b1;
        end
      end

      RX_RD_DATA: begin
        if (!rx_hold_valid && noc_valid_in && r_ready)
          rx_accept_flit = 1'b1;
      end

      default: ;
    endcase

    noc_credit_out = '0;
    if (noc_valid_in && rx_accept_flit)
      noc_credit_out[noc_flit_in.vc_id] = 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_state      <= RX_IDLE;
      rx_id         <= '0;
      rx_beat_cnt   <= '0;
      rx_beat_total <= '0;
      rx_hold_valid <= 1'b0;
      rx_hold_data  <= '0;
      rx_hold_last  <= 1'b0;
    end else if (clk_en) begin
      case (rx_state)
        RX_IDLE: begin
          if (noc_valid_in &&
              (noc_flit_in.flit_type == FLIT_HEAD ||
               noc_flit_in.flit_type == FLIT_HEADTAIL)) begin
            case (noc_flit_in.msg_type)
              MSG_READ_RESP: begin
                rx_state      <= RX_RD_DATA;
                rx_id         <= noc_flit_in.payload[47:44];
                rx_beat_cnt   <= '0;
                rx_beat_total <= noc_flit_in.payload[43:36];
                rx_hold_valid <= 1'b1;
                rx_hold_data  <= noc_flit_in.payload[31:0];
                rx_hold_last  <= (noc_flit_in.flit_type == FLIT_HEADTAIL);
              end
              MSG_WRITE_ACK: begin
                rx_state <= RX_WR_RESP;
                rx_id    <= noc_flit_in.payload[47:44];
              end
              default: ; // Ignore unknown messages
            endcase
          end
        end

        RX_RD_DATA: begin
          if (rx_hold_valid) begin
            if (r_ready) begin
              rx_hold_valid <= 1'b0;
              rx_beat_cnt   <= rx_beat_cnt + 1'b1;
              if (rx_hold_last)
                rx_state <= RX_IDLE;
            end
          end else if (noc_valid_in && r_ready) begin
            rx_beat_cnt <= rx_beat_cnt + 1'b1;
            if (noc_flit_in.flit_type == FLIT_TAIL ||
                noc_flit_in.flit_type == FLIT_HEADTAIL)
              rx_state <= RX_IDLE;
          end
        end

        RX_WR_RESP: begin
          if (b_ready)
            rx_state <= RX_IDLE;
        end

        default: begin
          rx_state <= RX_IDLE;
        end
      endcase
    end
  end

  // RX outputs
  always_comb begin
    r_valid = 1'b0;
    r_data  = '0;
    r_id    = rx_id;
    r_resp  = 2'b00; // OKAY
    r_last  = 1'b0;
    b_valid = 1'b0;
    b_id    = rx_id;
    b_resp  = 2'b00;

    case (rx_state)
      RX_RD_DATA: begin
        if (rx_hold_valid) begin
          r_valid = 1'b1;
          r_data  = rx_hold_data;
          r_last  = rx_hold_last;
        end else if (noc_valid_in) begin
          r_valid = 1'b1;
          r_data  = noc_flit_in.payload[31:0];
          r_last  = (noc_flit_in.flit_type == FLIT_TAIL ||
                     noc_flit_in.flit_type == FLIT_HEADTAIL);
        end
      end
      RX_WR_RESP: begin
        b_valid = 1'b1;
      end
      default: ;
    endcase
  end

endmodule
