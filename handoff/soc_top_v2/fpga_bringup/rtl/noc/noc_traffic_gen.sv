// =============================================================================
// noc_traffic_gen.sv — Synthetic Traffic Generator for NoC Comparison
// =============================================================================
// Generates configurable traffic patterns for baseline vs. sparsity-aware
// VC allocator performance comparison. Measures latency and throughput.
//
// Patterns:
//   0 = Uniform random     (any src → any dst, equal probability)
//   1 = Nearest neighbor    (adjacent tile only)
//   2 = Hotspot            (80% to node 0, 20% random — models reduce)
//   3 = Scatter/Gather     (node 0 broadcasts, all reply — models inference)
//   4 = Transpose          (src (r,c) → dst (c,r))

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off GENUNNAMED */
import noc_pkg::*;

module noc_traffic_gen #(
  parameter int  NODE_ID     = 0,
  parameter int  MESH_ROWS   = noc_pkg::MESH_ROWS,
  parameter int  MESH_COLS   = noc_pkg::MESH_COLS,
  parameter int  NUM_VCS     = noc_pkg::NUM_VCS,
  parameter int  PKT_LEN     = 4,           // Flits per packet (including head)
  parameter int  INJECT_RATE = 50           // Percentage injection rate (0-100)
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // --- Control ---
  input  logic                  enable,
  input  logic [2:0]            pattern,     // Traffic pattern selector
  input  logic [31:0]           seed,        // LFSR seed

  // --- NoC local port interface ---
  output flit_t                 flit_out,
  output logic                  flit_valid,
  input  logic [NUM_VCS-1:0]    credit_in,   // Credits from router

  input  flit_t                 flit_in,
  input  logic                  flit_valid_in,
  output logic [NUM_VCS-1:0]    credit_out,  // Credits to router

  // --- Statistics ---
  output logic [31:0]           stat_sent,       // Packets injected
  output logic [31:0]           stat_received,   // Packets ejected
  output logic [63:0]           stat_latency_sum // Total latency (sum of per-pkt)
);

  localparam int NUM_NODES = MESH_ROWS * MESH_COLS;

  // =========================================================================
  // LFSR for pseudo-random number generation
  // =========================================================================
  logic [31:0] lfsr;
  logic [31:0] lfsr_next;

  // Galois LFSR: x^32 + x^22 + x^2 + x + 1
  assign lfsr_next = {lfsr[0], lfsr[31:1]} ^ (lfsr[0] ? 32'h8040_0003 : 32'h0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      lfsr <= (seed != '0) ? seed : {16'hACE1, NODE_ID[15:0]};
    else
      lfsr <= lfsr_next;
  end

  // =========================================================================
  // Destination computation based on pattern
  // =========================================================================
  logic [NODE_BITS-1:0] dst_node;
  logic [ROW_BITS-1:0]  my_row;
  logic [COL_BITS-1:0]  my_col;

  assign my_row = node_row(NODE_BITS'(NODE_ID));
  assign my_col = node_col(NODE_BITS'(NODE_ID));

  always_comb begin
    case (pattern)
      3'd0: begin // Uniform random (avoid self)
        dst_node = NODE_BITS'(lfsr[NODE_BITS-1:0] % NUM_NODES);
        if (dst_node == NODE_BITS'(NODE_ID))
          dst_node = NODE_BITS'((NODE_ID + 1) % NUM_NODES);
      end

      3'd1: begin // Nearest neighbor (East, wrap)
        if (my_col < COL_BITS'(MESH_COLS - 1))
          dst_node = make_node_id(my_row, my_col + 1);
        else
          dst_node = make_node_id(my_row, '0);
      end

      3'd2: begin // Hotspot: 80% → node 0, 20% random
        if (lfsr[7:0] < 8'd204) // ~80%
          dst_node = '0;
        else begin
          dst_node = NODE_BITS'(lfsr[NODE_BITS+7:8] % NUM_NODES);
          if (dst_node == NODE_BITS'(NODE_ID))
            dst_node = NODE_BITS'((NODE_ID + 1) % NUM_NODES);
        end
      end

      3'd3: begin // Scatter/Gather: node 0 sends to all, others reply to 0
        if (NODE_ID == 0)
          dst_node = NODE_BITS'(lfsr[NODE_BITS-1:0] % (NUM_NODES - 1)) + 1;
        else
          dst_node = '0;
      end

      3'd4: begin // Transpose: (r,c) → (c,r)
        dst_node = make_node_id(COL_BITS'(my_col), ROW_BITS'(my_row));
        if (dst_node == NODE_BITS'(NODE_ID))
          dst_node = NODE_BITS'((NODE_ID + 1) % NUM_NODES);
      end

      default: dst_node = NODE_BITS'((NODE_ID + 1) % NUM_NODES);
    endcase
  end

  // =========================================================================
  // Injection state machine
  // =========================================================================
  typedef enum logic [1:0] {
    INJ_IDLE,
    INJ_HEAD,
    INJ_BODY,
    INJ_TAIL
  } inj_state_e;

  inj_state_e inj_state;
  logic [$clog2(PKT_LEN)-1:0] flit_cnt;
  logic [VC_BITS-1:0] inject_vc;
  logic inject_ok;

  // Can inject? Rate limiting via LFSR
  assign inject_ok = enable && (lfsr[6:0] < 7'(INJECT_RATE * 127 / 100));

  // Credit tracking (simplified: just use VC 0)
  logic [NUM_VCS-1:0] has_credit;
  logic [$clog2(BUF_DEPTH+1)-1:0] credit_cnt [NUM_VCS];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int v = 0; v < NUM_VCS; v++)
        credit_cnt[v] <= $clog2(BUF_DEPTH+1)'(BUF_DEPTH);
    end else begin
      for (int v = 0; v < NUM_VCS; v++) begin
        logic inc, dec;
        inc = credit_in[v];
        dec = flit_valid && (inject_vc == VC_BITS'(v));
        case ({inc, dec})
          2'b10:   credit_cnt[v] <= credit_cnt[v] + 1;
          2'b01:   credit_cnt[v] <= credit_cnt[v] - 1;
          default: credit_cnt[v] <= credit_cnt[v];
        endcase
      end
    end
  end

  generate
    for (genvar v = 0; v < NUM_VCS; v++)
      assign has_credit[v] = (credit_cnt[v] != '0);
  endgenerate

  // Pick a VC with credit (round-robin)
  logic [VC_BITS-1:0] chosen_vc;
  logic any_credit;
  always_comb begin
    chosen_vc  = '0;
    any_credit = 1'b0;
    for (int v = 0; v < NUM_VCS; v++) begin
      if (!any_credit && has_credit[v]) begin
        chosen_vc  = VC_BITS'(v);
        any_credit = 1'b1;
      end
    end
  end

  // Timestamp for latency measurement (stored in payload)
  logic [31:0] cycle_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cycle_count <= '0;
    else        cycle_count <= cycle_count + 1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      inj_state <= INJ_IDLE;
      flit_cnt  <= '0;
      inject_vc <= '0;
    end else begin
      case (inj_state)
        INJ_IDLE: begin
          if (inject_ok && any_credit) begin
            inject_vc <= chosen_vc;
            if (PKT_LEN == 1)
              inj_state <= INJ_HEAD; // Will send HEAD_TAIL
            else
              inj_state <= INJ_HEAD;
            flit_cnt <= '0;
          end
        end

        INJ_HEAD: begin
          if (has_credit[inject_vc]) begin
            flit_cnt <= flit_cnt + 1;
            if (PKT_LEN == 1)
              inj_state <= INJ_IDLE;
            else if (PKT_LEN == 2)
              inj_state <= INJ_TAIL;
            else
              inj_state <= INJ_BODY;
          end
        end

        INJ_BODY: begin
          if (has_credit[inject_vc]) begin
            flit_cnt <= flit_cnt + 1;
            if (flit_cnt == $clog2(PKT_LEN)'(PKT_LEN - 2))
              inj_state <= INJ_TAIL;
          end
        end

        INJ_TAIL: begin
          if (has_credit[inject_vc]) begin
            inj_state <= INJ_IDLE;
            flit_cnt  <= '0;
          end
        end
      endcase
    end
  end

  // =========================================================================
  // Flit output mux
  // =========================================================================
  always_comb begin
    flit_out   = '0;
    flit_valid = 1'b0;

    case (inj_state)
      INJ_HEAD: begin
        if (has_credit[inject_vc]) begin
          flit_valid = 1'b1;
          if (PKT_LEN == 1)
            flit_out = make_head_flit(FLIT_HEAD_TAIL, NODE_BITS'(NODE_ID),
                                      dst_node, inject_vc, MSG_DATA,
                                      {16'(cycle_count[15:0]), 32'h0});
          else
            flit_out = make_head_flit(FLIT_HEAD, NODE_BITS'(NODE_ID),
                                      dst_node, inject_vc, MSG_DATA,
                                      {16'(cycle_count[15:0]), 32'h0});
        end
      end

      INJ_BODY: begin
        if (has_credit[inject_vc]) begin
          flit_valid           = 1'b1;
          flit_out.flit_type   = FLIT_BODY;
          flit_out.vc_id       = inject_vc;
          flit_out.payload     = lfsr[PAYLOAD_BITS-1:0];
        end
      end

      INJ_TAIL: begin
        if (has_credit[inject_vc]) begin
          flit_valid           = 1'b1;
          flit_out.flit_type   = FLIT_TAIL;
          flit_out.vc_id       = inject_vc;
          flit_out.payload     = lfsr[PAYLOAD_BITS-1:0];
        end
      end

      default: ;
    endcase
  end

  // =========================================================================
  // Ejection + statistics
  // =========================================================================
  // Always accept incoming flits immediately (infinite sink)
  always_comb begin
    credit_out = '0;
    if (flit_valid_in)
      credit_out[flit_in.vc_id] = 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stat_sent        <= '0;
      stat_received    <= '0;
      stat_latency_sum <= '0;
    end else begin
      // Count injected packets (on HEAD or HEAD_TAIL)
      if (flit_valid && (flit_out.flit_type == FLIT_HEAD ||
                         flit_out.flit_type == FLIT_HEAD_TAIL))
        stat_sent <= stat_sent + 1;

      // Count received packets + accumulate latency
      if (flit_valid_in && (flit_in.flit_type == FLIT_HEAD ||
                            flit_in.flit_type == FLIT_HEAD_TAIL)) begin
        stat_received <= stat_received + 1;
        // Extract injection timestamp from payload bits [47:32]
        stat_latency_sum <= stat_latency_sum +
                           (cycle_count - {16'h0, flit_in.payload[47:32]});
      end
    end
  end

endmodule
