// =============================================================================
// directory_controller.sv — MESI Directory-Based Coherence Controller
// =============================================================================
// Standalone demo module (NOT integrated into main SoC).
// Implements a full-map directory for cache coherence.
//
// Protocol:
//   GetS (read):
//     I→S: Supply data from memory, add to sharer vector
//     M→S: Forward to owner, owner writes back + sends data, becomes S
//     E→S: Send data, owner becomes S
//     S→S: Send data, add to sharers
//
//   GetM (write):
//     I→M: Supply data, set owner
//     S→M: Invalidate all sharers, wait for acks, supply data, set owner
//     M→M: Forward to owner, old owner writes back, new owner gets M
//     E→M: Forward to owner (no WB needed, was clean)
//
//   PutM (eviction of Modified):
//     Write back data to memory, clear directory entry
//
// One outstanding request at a time per cache line (simplification).

/* verilator lint_off IMPORTSTAR */
import coherence_pkg::*;

module directory_controller #(
  parameter int NUM_ENTRIES = 256,   // Directory entries (cache lines tracked)
  parameter int ADDR_WIDTH  = COH_ADDR_W,
  parameter int LINE_BYTES  = 32     // Bytes per cache line
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Request input (from L1 caches via interconnect) ---
  input  logic                   req_valid,
  output logic                   req_ready,
  input  coh_req_t               req,

  // --- Response output (to requesting cache) ---
  output logic                   resp_valid,
  input  logic                   resp_ready,
  output coh_req_t               resp,

  // --- Snoop/forward output (to other caches) ---
  output logic                   snoop_valid,
  input  logic                   snoop_ready,
  output coh_req_t               snoop,

  // --- Snoop response input (from other caches) ---
  input  logic                   snoop_resp_valid,
  output logic                   snoop_resp_ready,
  input  coh_req_t               snoop_resp,

  // --- Memory interface (simplified) ---
  output logic                   mem_rd_valid,
  input  logic                   mem_rd_ready,
  output logic [ADDR_WIDTH-1:0]  mem_rd_addr,
  input  logic                   mem_rd_data_valid,
  input  logic [COH_DATA_W-1:0]  mem_rd_data,

  output logic                   mem_wr_valid,
  input  logic                   mem_wr_ready,
  output logic [ADDR_WIDTH-1:0]  mem_wr_addr,
  output logic [COH_DATA_W-1:0]  mem_wr_data
);

  localparam int IDX_BITS = $clog2(NUM_ENTRIES);
  localparam int LINE_OFFSET = $clog2(LINE_BYTES);

  // =========================================================================
  // Directory storage
  // =========================================================================
  dir_entry_t dir_mem [NUM_ENTRIES];
  logic [ADDR_WIDTH-1:0] dir_tag [NUM_ENTRIES]; // Tag for associative lookup

  // =========================================================================
  // State machine
  // =========================================================================
  typedef enum logic [3:0] {
    DIR_IDLE,
    DIR_LOOKUP,
    DIR_GET_S_FETCH,      // Fetching from memory for Shared response
    DIR_GET_S_FWD,        // Forwarding to owner (M→S transition)
    DIR_GET_M_INV,        // Sending invalidations
    DIR_GET_M_WAIT_ACK,   // Waiting for invalidation acks
    DIR_GET_M_FETCH,      // Fetching from memory
    DIR_GET_M_FWD,        // Forwarding to owner
    DIR_PUT_WB,           // Writing back to memory
    DIR_RESPOND
  } dir_state_e;

  dir_state_e state;

  // Saved request
  coh_req_t saved_req;
  dir_entry_t cur_entry;
  logic [IDX_BITS-1:0] cur_idx;

  // Invalidation tracking
  logic [MAX_SHARERS-1:0] pending_inv;
  logic [MAX_SHARERS-1:0] inv_sent;
  logic [$clog2(MAX_SHARERS):0] inv_ack_cnt;
  logic [$clog2(MAX_SHARERS):0] inv_total;

  // Index from address
  function automatic [IDX_BITS-1:0] addr_to_idx;
    input [ADDR_WIDTH-1:0] addr;
    begin
      addr_to_idx = IDX_BITS'(addr[LINE_OFFSET +: IDX_BITS]);
    end
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= DIR_IDLE;
      saved_req    <= '0;
      cur_entry    <= '0;
      cur_idx      <= '0;
      pending_inv  <= '0;
      inv_sent     <= '0;
      inv_ack_cnt  <= '0;
      inv_total    <= '0;

      for (int i = 0; i < NUM_ENTRIES; i++) begin
        dir_mem[i] <= '0;  // All Invalid
        dir_tag[i] <= '0;
      end
    end else begin
      case (state)
        DIR_IDLE: begin
          if (req_valid) begin
            saved_req <= req;
            cur_idx   <= addr_to_idx(req.addr);
            state     <= DIR_LOOKUP;
          end
        end

        DIR_LOOKUP: begin
          cur_entry <= dir_mem[cur_idx];

          case (saved_req.msg_type)
            COH_GET_S: begin
              case (dir_mem[cur_idx].state)
                MESI_I, MESI_E, MESI_S: state <= DIR_GET_S_FETCH;
                MESI_M: state <= DIR_GET_S_FWD;
              endcase
            end

            COH_GET_M: begin
              case (dir_mem[cur_idx].state)
                MESI_I: state <= DIR_GET_M_FETCH;
                MESI_S: begin
                  // Need to invalidate all sharers
                  pending_inv <= dir_mem[cur_idx].sharer_vec &
                                 ~(MAX_SHARERS'(1) << saved_req.src);
                  inv_sent    <= '0;
                  inv_ack_cnt <= '0;
                  // Count sharers to invalidate
                  inv_total   <= '0;
                  for (int i = 0; i < MAX_SHARERS; i++)
                    if (dir_mem[cur_idx].sharer_vec[i] && (i != int'(saved_req.src)))
                      inv_total <= inv_total + 1;

                  if (dir_mem[cur_idx].sharer_vec == (MAX_SHARERS'(1) << saved_req.src))
                    state <= DIR_GET_M_FETCH;  // Only sharer is requester
                  else
                    state <= DIR_GET_M_INV;
                end
                MESI_M, MESI_E: state <= DIR_GET_M_FWD;
              endcase
            end

            COH_PUT_M: begin
              state <= DIR_PUT_WB;
            end

            default: state <= DIR_IDLE;
          endcase
        end

        // --- GetS handling ---
        DIR_GET_S_FETCH: begin
          // Read data from memory
          if (mem_rd_data_valid) begin
            // Update directory: add requester to sharers
            dir_mem[cur_idx].state <= MESI_S;
            dir_mem[cur_idx].sharer_vec[saved_req.src] <= 1'b1;
            state <= DIR_RESPOND;
          end
        end

        DIR_GET_S_FWD: begin
          // Forward to owner: ask them to send data to requester
          if (snoop_ready) begin
            // Owner transitions M→S
            dir_mem[cur_idx].state <= MESI_S;
            dir_mem[cur_idx].sharer_vec[saved_req.src] <= 1'b1;
            dir_mem[cur_idx].sharer_vec[cur_entry.owner] <= 1'b1;
            state <= DIR_RESPOND;
          end
        end

        // --- GetM handling ---
        DIR_GET_M_INV: begin
          if (snoop_ready) begin
            // Send invalidation to first pending sharer
            for (int i = 0; i < MAX_SHARERS; i++) begin
              if (pending_inv[i] && !inv_sent[i]) begin
                inv_sent[i] <= 1'b1;
                pending_inv[i] <= 1'b0;
              end
            end

            if ((pending_inv & ~inv_sent) == '0)
              state <= DIR_GET_M_WAIT_ACK;
          end
        end

        DIR_GET_M_WAIT_ACK: begin
          if (snoop_resp_valid && snoop_resp.msg_type == COH_INV_ACK) begin
            inv_ack_cnt <= inv_ack_cnt + 1;
            if (inv_ack_cnt + 1 >= inv_total)
              state <= DIR_GET_M_FETCH;
          end
        end

        DIR_GET_M_FETCH: begin
          if (mem_rd_data_valid) begin
            dir_mem[cur_idx].state <= MESI_M;
            dir_mem[cur_idx].owner <= saved_req.src;
            dir_mem[cur_idx].sharer_vec <= '0;
            state <= DIR_RESPOND;
          end
        end

        DIR_GET_M_FWD: begin
          if (snoop_ready) begin
            dir_mem[cur_idx].state <= MESI_M;
            dir_mem[cur_idx].owner <= saved_req.src;
            dir_mem[cur_idx].sharer_vec <= '0;
            state <= DIR_RESPOND;
          end
        end

        // --- PutM handling ---
        DIR_PUT_WB: begin
          if (mem_wr_ready) begin
            dir_mem[cur_idx].state <= MESI_I;
            dir_mem[cur_idx].owner <= '0;
            dir_mem[cur_idx].sharer_vec <= '0;
            state <= DIR_RESPOND;
          end
        end

        DIR_RESPOND: begin
          if (resp_ready)
            state <= DIR_IDLE;
        end
      endcase
    end
  end

  // =========================================================================
  // Output assignments
  // =========================================================================
  assign req_ready = (state == DIR_IDLE);

  // Memory read
  assign mem_rd_valid = (state == DIR_GET_S_FETCH || state == DIR_GET_M_FETCH);
  assign mem_rd_addr  = saved_req.addr;

  // Memory write
  assign mem_wr_valid = (state == DIR_PUT_WB);
  assign mem_wr_addr  = saved_req.addr;
  assign mem_wr_data  = saved_req.data;

  // Snoop messages
  always_comb begin
    snoop       = '0;
    snoop_valid = 1'b0;

    case (state)
      DIR_GET_S_FWD: begin
        snoop_valid     = 1'b1;
        snoop.msg_type  = COH_FWD_GET_S;
        snoop.src       = saved_req.src;
        snoop.dst       = cur_entry.owner;
        snoop.addr      = saved_req.addr;
      end

      DIR_GET_M_INV: begin
        snoop_valid     = 1'b1;
        snoop.msg_type  = COH_INV;
        snoop.src       = saved_req.src;
        snoop.addr      = saved_req.addr;
        // dst = first pending sharer
        snoop.dst       = '0;
        for (int i = 0; i < MAX_SHARERS; i++)
          if (pending_inv[i] && !inv_sent[i]) begin
            snoop.dst = COH_NODE_W'(i);
          end
      end

      DIR_GET_M_FWD: begin
        snoop_valid     = 1'b1;
        snoop.msg_type  = COH_FWD_GET_M;
        snoop.src       = saved_req.src;
        snoop.dst       = cur_entry.owner;
        snoop.addr      = saved_req.addr;
      end

      default: ;
    endcase
  end

  assign snoop_resp_ready = (state == DIR_GET_M_WAIT_ACK);

  // Response to requester
  always_comb begin
    resp       = '0;
    resp_valid = 1'b0;

    if (state == DIR_RESPOND) begin
      resp_valid     = 1'b1;
      resp.msg_type  = (saved_req.msg_type == COH_PUT_M) ? COH_WB_ACK : COH_DATA;
      resp.src       = '0;  // From directory
      resp.dst       = saved_req.src;
      resp.addr      = saved_req.addr;
      resp.has_data  = (saved_req.msg_type != COH_PUT_M);
      resp.data      = mem_rd_data;
    end
  end

endmodule
