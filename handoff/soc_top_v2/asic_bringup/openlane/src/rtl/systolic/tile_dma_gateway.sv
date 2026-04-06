// =============================================================================
// tile_dma_gateway.sv — Tile-to-DRAM DMA Gateway
// =============================================================================
// Translates NoC DMA request flits from tiles into AXI4 master transactions.
// Tiles send DMA_RD / DMA_WR flits via the NoC to a designated gateway node.
// This module sits at the gateway node's local port and drives the AXI master.
//
// Protocol:
//   Tile → NoC → Gateway local port → this module → AXI master → DRAM
//
// DMA_RD flit payload: [47:16] = DRAM addr[31:0], [15:12] = burst_len, [11:8] = src_tile
// DMA_WR flit:         HEAD carries addr+len, subsequent BODY flits carry data
// DMA_RD_RESP:         Gateway sends response flits back to requesting tile

/* verilator lint_off UNUSEDSIGNAL */
module tile_dma_gateway
  import noc_pkg::*;
#(
  parameter int AXI_ADDR_W = 32,
  parameter int AXI_DATA_W = 32,
  parameter int AXI_ID_W   = 4,
  parameter int MAX_BURST  = 16    // Max AXI burst length
) (
  input  logic              clk,
  input  logic              rst_n,

  // =====================================================================
  // NoC local port (connected to one mesh node's local port)
  // =====================================================================
  input  flit_t             noc_flit_in,
  input  logic              noc_valid_in,
  output logic              noc_credit_out,

  output flit_t             noc_flit_out,
  output logic              noc_valid_out,
  input  logic              noc_credit_in,

  // =====================================================================
  // AXI4 Master port (to crossbar → DRAM)
  // =====================================================================
  // AR
  output logic [AXI_ID_W-1:0]    m_axi_arid,
  output logic [AXI_ADDR_W-1:0]  m_axi_araddr,
  output logic [7:0]             m_axi_arlen,
  output logic [2:0]             m_axi_arsize,
  output logic [1:0]             m_axi_arburst,
  output logic                    m_axi_arvalid,
  input  logic                    m_axi_arready,
  // R
  input  logic [AXI_ID_W-1:0]    m_axi_rid,
  input  logic [AXI_DATA_W-1:0]  m_axi_rdata,
  input  logic [1:0]             m_axi_rresp,
  input  logic                    m_axi_rlast,
  input  logic                    m_axi_rvalid,
  output logic                    m_axi_rready,
  // AW
  output logic [AXI_ID_W-1:0]    m_axi_awid,
  output logic [AXI_ADDR_W-1:0]  m_axi_awaddr,
  output logic [7:0]             m_axi_awlen,
  output logic [2:0]             m_axi_awsize,
  output logic [1:0]             m_axi_awburst,
  output logic                    m_axi_awvalid,
  input  logic                    m_axi_awready,
  // W
  output logic [AXI_DATA_W-1:0]  m_axi_wdata,
  output logic [AXI_DATA_W/8-1:0] m_axi_wstrb,
  output logic                    m_axi_wlast,
  output logic                    m_axi_wvalid,
  input  logic                    m_axi_wready,
  // B
  input  logic [AXI_ID_W-1:0]    m_axi_bid,
  input  logic [1:0]             m_axi_bresp,
  input  logic                    m_axi_bvalid,
  output logic                    m_axi_bready
);

  // =========================================================================
  // FSM
  // =========================================================================
  typedef enum logic [3:0] {
    G_IDLE,
    G_DECODE,         // Decode incoming flit
    G_RD_ISSUE,       // Issue AXI AR for DMA read
    G_RD_WAIT,        // Receive AXI R data beats
    G_RD_RESP,        // Send response flits to tile via NoC
    G_WR_COLLECT,     // Collect BODY flits carrying write data
    G_WR_ISSUE,       // Issue AXI AW
    G_WR_DATA,        // Send AXI W data beats
    G_WR_RESP,        // Wait for AXI B response
    G_WR_ACK          // Return write completion to the tile
  } gw_state_e;

  gw_state_e state, state_next;

  localparam int BURST_LEN_W   = (MAX_BURST <= 1) ? 1 : $clog2(MAX_BURST);
  localparam int BURST_COUNT_W = $clog2(MAX_BURST + 1);

  // =========================================================================
  // Request registers
  // =========================================================================
  logic [AXI_ADDR_W-1:0] req_addr;
  logic [BURST_LEN_W-1:0] req_burst_len;  // 0-based (0 = 1 beat)
  logic [3:0]            req_src_tile;
  logic                  req_is_write;
  logic [3:0]            req_dst_node;   // Gateway's own node ID (from flit dst)

  // Read response buffer
  logic [AXI_DATA_W-1:0] rd_buf [MAX_BURST];
  logic [BURST_COUNT_W-1:0] rd_buf_count;
  logic [BURST_LEN_W-1:0]   rd_resp_idx;

  // Write data buffer
  logic [AXI_DATA_W-1:0] wr_buf [MAX_BURST];
  logic [BURST_COUNT_W-1:0] wr_buf_count;
  logic [BURST_LEN_W-1:0]   wr_data_idx;

  // =========================================================================
  // Receive credit for NoC input
  // =========================================================================
  assign noc_credit_out = noc_valid_in &&
                          ((state == G_IDLE) || (state == G_WR_COLLECT));

  // =========================================================================
  // Flit decode
  // =========================================================================
  logic [1:0]  flit_type;
  logic [3:0]  flit_msg_type;
  logic [47:0] flit_payload;
  logic [3:0]  flit_src;
  logic [3:0]  flit_dst;

  assign flit_type     = noc_flit_in[63:62];
  assign flit_src      = noc_flit_in[61:58];
  assign flit_dst      = noc_flit_in[57:54];
  assign flit_msg_type = noc_flit_in[51:48];
  assign flit_payload  = noc_flit_in[47:0];

  // =========================================================================
  // State register
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= G_IDLE;
    else
      state <= state_next;
  end

  // =========================================================================
  // Next state
  // =========================================================================
  always_comb begin
    state_next = state;
    case (state)
      G_IDLE: begin
        if (noc_valid_in)
          state_next = G_DECODE;
      end

      G_DECODE: begin
        if (req_is_write)
          state_next = G_WR_COLLECT;
        else
          state_next = G_RD_ISSUE;
      end

      G_RD_ISSUE: begin
        if (m_axi_arready)
          state_next = G_RD_WAIT;
      end

      G_RD_WAIT: begin
        if (m_axi_rvalid && m_axi_rready && m_axi_rlast)
          state_next = G_RD_RESP;
      end

      G_RD_RESP: begin
        if (noc_credit_in && ({1'b0, rd_resp_idx} == (rd_buf_count - 1'b1)))
          state_next = G_IDLE;
      end

      G_WR_COLLECT: begin
        if (noc_valid_in && (wr_buf_count == {1'b0, req_burst_len}))
          state_next = G_WR_ISSUE;
      end

      G_WR_ISSUE: begin
        if (m_axi_awready)
          state_next = G_WR_DATA;
      end

      G_WR_DATA: begin
        if (m_axi_wvalid && m_axi_wready && m_axi_wlast)
          state_next = G_WR_RESP;
      end

      G_WR_RESP: begin
        if (m_axi_bvalid && m_axi_bready)
          state_next = G_WR_ACK;
      end

      G_WR_ACK: begin
        if (noc_credit_in)
          state_next = G_IDLE;
      end

      default: state_next = G_IDLE;
    endcase
  end

  // =========================================================================
  // Datapath
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      req_addr      <= '0;
      req_burst_len <= '0;
      req_src_tile  <= '0;
      req_is_write  <= 1'b0;
      req_dst_node  <= '0;
      rd_buf_count  <= '0;
      rd_resp_idx   <= '0;
      wr_buf_count  <= '0;
      wr_data_idx   <= '0;
    end else begin
      case (state)
        G_IDLE: begin
          if (noc_valid_in && noc_credit_out) begin
            // Extract fields from HEAD flit
            req_addr      <= {flit_payload[47:16]};
            req_burst_len <= flit_payload[15:12];
            req_src_tile  <= flit_src;
            req_dst_node  <= flit_dst;
            req_is_write  <= (flit_msg_type == MSG_WRITE_REQ);
            rd_buf_count  <= '0;
            rd_resp_idx   <= '0;
            wr_buf_count  <= '0;
            wr_data_idx   <= '0;
          end
        end

        G_RD_WAIT: begin
          if (m_axi_rvalid && m_axi_rready) begin
            rd_buf[rd_buf_count[BURST_LEN_W-1:0]] <= m_axi_rdata;
            rd_buf_count         <= rd_buf_count + 1;
          end
        end

        G_RD_RESP: begin
          if (noc_credit_in)
            rd_resp_idx <= rd_resp_idx + 1;
        end

        G_WR_COLLECT: begin
          if (noc_valid_in && noc_credit_out) begin
            wr_buf[wr_buf_count[BURST_LEN_W-1:0]] <= flit_payload[AXI_DATA_W-1:0];
            wr_buf_count         <= wr_buf_count + 1;
          end
        end

        G_WR_DATA: begin
          if (m_axi_wvalid && m_axi_wready)
            wr_data_idx <= wr_data_idx + 1;
        end

        default: ;
      endcase
    end
  end

  // =========================================================================
  // AXI AR (read)
  // =========================================================================
  assign m_axi_arvalid = (state == G_RD_ISSUE);
  assign m_axi_araddr  = req_addr;
  assign m_axi_arid    = {req_src_tile};
  assign m_axi_arlen   = {4'b0, req_burst_len};
  assign m_axi_arsize  = 3'b010;   // 4 bytes
  assign m_axi_arburst = 2'b01;    // INCR

  // AXI R
  assign m_axi_rready  = (state == G_RD_WAIT);

  // =========================================================================
  // AXI AW (write)
  // =========================================================================
  assign m_axi_awvalid = (state == G_WR_ISSUE);
  assign m_axi_awaddr  = req_addr;
  assign m_axi_awid    = {req_src_tile};
  assign m_axi_awlen   = {4'b0, req_burst_len};
  assign m_axi_awsize  = 3'b010;
  assign m_axi_awburst = 2'b01;

  // AXI W
  assign m_axi_wvalid  = (state == G_WR_DATA);
  assign m_axi_wdata   = wr_buf[wr_data_idx];
  assign m_axi_wstrb   = '1;
  assign m_axi_wlast   = (wr_data_idx == req_burst_len);

  // AXI B
  assign m_axi_bready  = (state == G_WR_RESP);

  // =========================================================================
  // NoC response flits (read data back to requesting tile)
  // =========================================================================
  always_comb begin
    noc_valid_out = 1'b0;
    noc_flit_out  = '0;

    if (state == G_RD_RESP) begin
      noc_valid_out = 1'b1;
      noc_flit_out.src_id   = req_dst_node;
      noc_flit_out.dst_id   = req_src_tile;
      noc_flit_out.vc_id    = '0;
      noc_flit_out.msg_type = MSG_READ_RESP;

      // Build response flit
      if (rd_buf_count == 1) begin
        noc_flit_out.flit_type = FLIT_HEADTAIL;
        noc_flit_out.payload   = {req_src_tile, 8'(rd_buf_count), 4'h0, rd_buf[0]};
      end else if (rd_resp_idx == '0) begin
        noc_flit_out.flit_type = FLIT_HEAD;
        noc_flit_out.payload   = {req_src_tile, 8'(rd_buf_count), 4'h0, rd_buf[0]};
      end else if ({1'b0, rd_resp_idx} == (rd_buf_count - 1'b1)) begin
        noc_flit_out.flit_type = FLIT_TAIL;
        noc_flit_out.payload   = {16'h0, rd_buf[rd_resp_idx]};
      end else begin
        noc_flit_out.flit_type = FLIT_BODY;
        noc_flit_out.payload   = {16'h0, rd_buf[rd_resp_idx]};
      end
    end else if (state == G_WR_ACK) begin
      noc_valid_out           = 1'b1;
      noc_flit_out.flit_type  = FLIT_HEADTAIL;
      noc_flit_out.src_id     = req_dst_node;
      noc_flit_out.dst_id     = req_src_tile;
      noc_flit_out.vc_id      = '0;
      noc_flit_out.msg_type   = MSG_WRITE_ACK;
      noc_flit_out.payload    = {req_src_tile, 44'h0};
    end
  end

endmodule
