// =============================================================================
// noc_qos_shaper.sv — NoC Traffic Shaper with QoS Support
// =============================================================================
// Sits at each router's output port and shapes traffic according to
// per-VC bandwidth allocations.
//
// Features:
//   - Per-VC token bucket rate limiter
//   - Priority queuing across VCs (VC0 = highest priority)
//   - Back-pressure propagation via credit protocol
//   - Configurable burst allowance

module noc_qos_shaper
  import noc_pkg::*;
#(
  parameter int NUM_VCS       = 4,
  parameter int TOKEN_DEPTH   = 8,
  parameter int REFILL_PERIOD = 32
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- Input flits (from switch output) ---
  input  logic              flit_valid_i,
  output logic              flit_ready_o,
  input  flit_t             flit_i,
  input  logic [$clog2(NUM_VCS)-1:0] flit_vc_i,

  // --- Output flits (to link) ---
  output logic              flit_valid_o,
  input  logic              flit_ready_i,
  output flit_t             flit_o,
  output logic [$clog2(NUM_VCS)-1:0] flit_vc_o,

  // --- Per-VC bandwidth configuration ---
  input  logic [7:0]        vc_bw_alloc [NUM_VCS],  // Tokens per refill period
  input  logic              shaper_enable
);

  localparam int VC_BITS = $clog2(NUM_VCS);

  // =========================================================================
  // Per-VC FIFO buffers
  // =========================================================================
  localparam int BUF_DEPTH = 4;

  flit_t             vc_buf     [NUM_VCS][BUF_DEPTH];
  logic [$clog2(BUF_DEPTH):0] vc_wr_ptr [NUM_VCS];
  logic [$clog2(BUF_DEPTH):0] vc_rd_ptr [NUM_VCS];
  logic [NUM_VCS-1:0] vc_empty;
  logic [NUM_VCS-1:0] vc_full;

  generate
    for (genvar v = 0; v < NUM_VCS; v++) begin : g_vc_buf
      assign vc_empty[v] = (vc_wr_ptr[v] == vc_rd_ptr[v]);
      assign vc_full[v]  = (vc_wr_ptr[v][$clog2(BUF_DEPTH)] != vc_rd_ptr[v][$clog2(BUF_DEPTH)]) &&
                            (vc_wr_ptr[v][$clog2(BUF_DEPTH)-1:0] == vc_rd_ptr[v][$clog2(BUF_DEPTH)-1:0]);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          vc_wr_ptr[v] <= '0;
          vc_rd_ptr[v] <= '0;
        end else begin
          // Write (enqueue)
          if (flit_valid_i && flit_ready_o && (flit_vc_i == v[VC_BITS-1:0])) begin
            vc_buf[v][vc_wr_ptr[v][$clog2(BUF_DEPTH)-1:0]] <= flit_i;
            vc_wr_ptr[v] <= vc_wr_ptr[v] + 1;
          end
          // Read (dequeue)
          if (flit_valid_o && flit_ready_i && (selected_vc == v[VC_BITS-1:0])) begin
            vc_rd_ptr[v] <= vc_rd_ptr[v] + 1;
          end
        end
      end
    end
  endgenerate

  // Accept input if target VC buffer is not full
  assign flit_ready_o = !vc_full[flit_vc_i];

  // =========================================================================
  // Token buckets per VC
  // =========================================================================
  logic [$clog2(TOKEN_DEPTH):0] vc_tokens [NUM_VCS];
  logic [$clog2(REFILL_PERIOD)-1:0] refill_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      refill_cnt <= '0;
    else if (refill_cnt == REFILL_PERIOD - 1)
      refill_cnt <= '0;
    else
      refill_cnt <= refill_cnt + 1;
  end

  logic refill_tick;
  assign refill_tick = (refill_cnt == '0);

  generate
    for (genvar v = 0; v < NUM_VCS; v++) begin : g_vc_tokens
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          vc_tokens[v] <= TOKEN_DEPTH;
        end else begin
          // Refill
          if (refill_tick) begin
            if (vc_tokens[v] + vc_bw_alloc[v] > TOKEN_DEPTH)
              vc_tokens[v] <= TOKEN_DEPTH;
            else
              vc_tokens[v] <= vc_tokens[v] + vc_bw_alloc[v];
          end

          // Consume on send
          if (flit_valid_o && flit_ready_i && (selected_vc == v[VC_BITS-1:0])) begin
            if (vc_tokens[v] > 0)
              vc_tokens[v] <= vc_tokens[v] - 1;
          end
        end
      end
    end
  endgenerate

  // =========================================================================
  // VC selection — strict priority (VC0 highest), with token gating
  // =========================================================================
  logic [VC_BITS-1:0] selected_vc;
  logic               any_eligible;

  always_comb begin
    selected_vc  = '0;
    any_eligible = 1'b0;

    for (int v = 0; v < NUM_VCS; v++) begin
      if (!vc_empty[v] && (vc_tokens[v] > 0 || !shaper_enable)) begin
        if (!any_eligible) begin
          selected_vc  = v[VC_BITS-1:0];
          any_eligible = 1'b1;
        end
      end
    end
  end

  // =========================================================================
  // Output
  // =========================================================================
  assign flit_valid_o = any_eligible;
  assign flit_o       = vc_buf[selected_vc][vc_rd_ptr[selected_vc][$clog2(BUF_DEPTH)-1:0]];
  assign flit_vc_o    = selected_vc;

endmodule
