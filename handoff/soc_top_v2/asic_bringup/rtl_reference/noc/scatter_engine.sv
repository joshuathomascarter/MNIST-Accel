// =============================================================================
// scatter_engine.sv — Multi-Tile Scatter (Broadcast/Multicast)
// =============================================================================
// Takes a local data block and sends it to one or more destination tiles
// via the NoC. Used for broadcasting activations to multiple tiles that
// need the same input feature map partition.
//
// Flow:
//   1. Host/controller writes scatter descriptor (src_addr, dst_mask, length)
//   2. Engine reads from local scratchpad
//   3. Packetizes into NoC flits with MSG_SCATTER type
//   4. Sends to each destination in the mask sequentially

/* verilator lint_off IMPORTSTAR */
import noc_pkg::*;

module scatter_engine #(
  parameter int NODE_ID    = 0,
  parameter int NUM_TILES  = noc_pkg::MESH_ROWS * noc_pkg::MESH_COLS,
  parameter int SP_ADDR_W  = 12,
  parameter int SP_DATA_W  = 64
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Descriptor input ---
  input  logic                   desc_valid,
  output logic                   desc_ready,
  input  logic [SP_ADDR_W-1:0]  desc_src_addr,
  input  logic [NUM_TILES-1:0]   desc_dst_mask,    // Bitmask of destinations
  input  logic [15:0]            desc_length,       // Words to scatter

  // --- Scratchpad read port ---
  output logic                   sp_rd_en,
  output logic [SP_ADDR_W-1:0]  sp_rd_addr,
  input  logic [SP_DATA_W-1:0]  sp_rd_data,

  // --- NoC inject ---
  output flit_t                  noc_flit,
  output logic                   noc_valid,
  input  logic [noc_pkg::NUM_VCS-1:0] noc_credit,

  // --- Status ---
  output logic                   busy
);

  typedef enum logic [2:0] {
    SC_IDLE,
    SC_NEXT_DST,
    SC_HEAD,
    SC_BODY,
    SC_TAIL
  } sc_state_e;

  sc_state_e state;
  logic [NUM_TILES-1:0]   remaining_mask;
  logic [$clog2(NUM_TILES)-1:0] cur_dst;
  logic [SP_ADDR_W-1:0]  rd_ptr;
  logic [15:0]            word_cnt;
  logic [15:0]            length_reg;
  logic [SP_ADDR_W-1:0]  base_addr;

  // Find first set bit in remaining mask
  logic [$clog2(NUM_TILES)-1:0] first_dst;
  logic has_dst;

  always_comb begin
    first_dst = '0;
    has_dst   = 1'b0;
    for (int i = 0; i < NUM_TILES; i++) begin
      if (!has_dst && remaining_mask[i]) begin
        first_dst = $clog2(NUM_TILES)'(i);
        has_dst   = 1'b1;
      end
    end
  end

  // Credit check (use VC 0)
  logic has_credit;
  assign has_credit = noc_credit[0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= SC_IDLE;
      remaining_mask <= '0;
      cur_dst        <= '0;
      rd_ptr         <= '0;
      word_cnt       <= '0;
      length_reg     <= '0;
      base_addr      <= '0;
    end else begin
      case (state)
        SC_IDLE: begin
          if (desc_valid) begin
            remaining_mask <= desc_dst_mask;
            length_reg     <= desc_length;
            base_addr      <= desc_src_addr;
            state          <= SC_NEXT_DST;
          end
        end

        SC_NEXT_DST: begin
          if (has_dst) begin
            cur_dst  <= first_dst;
            rd_ptr   <= base_addr;
            word_cnt <= '0;
            remaining_mask[first_dst] <= 1'b0;
            state    <= SC_HEAD;
          end else begin
            state <= SC_IDLE;
          end
        end

        SC_HEAD: begin
          if (has_credit) begin
            rd_ptr   <= rd_ptr + 1;
            word_cnt <= word_cnt + 1;
            if (length_reg == 16'd1)
              state <= SC_NEXT_DST;  // Single-flit packet
            else
              state <= SC_BODY;
          end
        end

        SC_BODY: begin
          if (has_credit) begin
            rd_ptr   <= rd_ptr + 1;
            word_cnt <= word_cnt + 1;
            if (word_cnt == length_reg - 2)
              state <= SC_TAIL;
          end
        end

        SC_TAIL: begin
          if (has_credit) begin
            state <= SC_NEXT_DST;
          end
        end
      endcase
    end
  end

  // Scratchpad read
  assign sp_rd_en   = (state == SC_HEAD || state == SC_BODY || state == SC_TAIL);
  assign sp_rd_addr = rd_ptr;

  // NoC output
  always_comb begin
    noc_flit  = '0;
    noc_valid = 1'b0;

    case (state)
      SC_HEAD: begin
        if (has_credit) begin
          noc_valid = 1'b1;
          noc_flit  = make_head_flit(
            (length_reg == 16'd1) ? FLIT_HEAD_TAIL : FLIT_HEAD,
            NODE_BITS'(NODE_ID),
            NODE_BITS'(cur_dst),
            '0,  // VC 0
            MSG_SCATTER,
            PAYLOAD_BITS'(sp_rd_data)
          );
        end
      end

      SC_BODY: begin
        if (has_credit) begin
          noc_valid          = 1'b1;
          noc_flit.flit_type = FLIT_BODY;
          noc_flit.vc_id     = '0;
          noc_flit.payload   = PAYLOAD_BITS'(sp_rd_data);
        end
      end

      SC_TAIL: begin
        if (has_credit) begin
          noc_valid          = 1'b1;
          noc_flit.flit_type = FLIT_TAIL;
          noc_flit.vc_id     = '0;
          noc_flit.payload   = PAYLOAD_BITS'(sp_rd_data);
        end
      end

      default: ;
    endcase
  end

  assign desc_ready = (state == SC_IDLE);
  assign busy       = (state != SC_IDLE);

endmodule
