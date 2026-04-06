// =============================================================================
// coherence_demo_top.sv — Standalone Coherence Demo Top-Level
// =============================================================================
// Self-contained demo integrating:
//   - 4 L1 caches (from Month 2)
//   - 1 directory controller
//   - 1 snoop filter
//   - Simple bus interconnect (shared bus, not NoC)
//   - Small shared memory
//
// Used for verification and demonstration only. NOT part of the main SoC.
// The coherence subsystem can be integrated later as a separate chip project.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off PINCONNECTEMPTY */
import coherence_pkg::*;

module coherence_demo_top #(
  parameter int NUM_NODES     = 4,
  parameter int ADDR_WIDTH    = 32,
  parameter int DATA_WIDTH    = 32,
  parameter int LINE_BYTES    = 32,
  parameter int MEM_DEPTH     = 1024   // Shared memory words
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Simple test interface (drives cache requests) ---
  input  logic [NUM_NODES-1:0]   test_req_valid,
  output logic [NUM_NODES-1:0]   test_req_ready,
  input  logic [ADDR_WIDTH-1:0]  test_req_addr  [NUM_NODES],
  input  logic                   test_req_write [NUM_NODES],
  input  logic [DATA_WIDTH-1:0]  test_req_wdata [NUM_NODES],
  output logic [DATA_WIDTH-1:0]  test_req_rdata [NUM_NODES],
  output logic [NUM_NODES-1:0]   test_resp_valid,

  // --- Debug / status ---
  output logic [NUM_NODES-1:0]   cache_hit,
  output logic [NUM_NODES-1:0]   cache_miss
);

  // =========================================================================
  // Coherence bus arbitration (simple round-robin)
  // =========================================================================
  // One request at a time on the shared bus
  logic [NUM_NODES-1:0] bus_req;
  logic [NUM_NODES-1:0] bus_grant;
  logic [$clog2(NUM_NODES)-1:0] bus_rr_ptr;

  // Bus request from each cache's miss handler
  coh_req_t cache_coh_req  [NUM_NODES];
  logic     cache_coh_valid[NUM_NODES];

  // Active transaction
  logic bus_busy;
  logic [$clog2(NUM_NODES)-1:0] active_node;

  // Arbiter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bus_rr_ptr  <= '0;
      bus_busy    <= 1'b0;
      active_node <= '0;
    end else begin
      if (!bus_busy) begin
        // Find next requester
        for (int r = 0; r < NUM_NODES; r++) begin
          int idx;
          idx = (int'(bus_rr_ptr) + r) % NUM_NODES;
          if (cache_coh_valid[idx] && !bus_busy) begin
            bus_busy    <= 1'b1;
            active_node <= $clog2(NUM_NODES)'(idx);
            bus_rr_ptr  <= $clog2(NUM_NODES)'((idx + 1) % NUM_NODES);
          end
        end
      end

      // Release bus when directory responds
      if (bus_busy && dir_resp_valid && dir_resp_ready)
        bus_busy <= 1'b0;
    end
  end

  // =========================================================================
  // Directory Controller
  // =========================================================================
  logic      dir_req_valid, dir_req_ready;
  coh_req_t  dir_req;
  logic      dir_resp_valid, dir_resp_ready;
  coh_req_t  dir_resp;
  logic      dir_snoop_valid, dir_snoop_ready;
  coh_req_t  dir_snoop;
  logic      dir_snoop_resp_valid, dir_snoop_resp_ready;
  coh_req_t  dir_snoop_resp;

  // Memory interface
  logic                    mem_rd_valid, mem_rd_ready, mem_rd_data_valid;
  logic [ADDR_WIDTH-1:0]   mem_rd_addr;
  logic [COH_DATA_W-1:0]   mem_rd_data;
  logic                    mem_wr_valid, mem_wr_ready;
  logic [ADDR_WIDTH-1:0]   mem_wr_addr;
  logic [COH_DATA_W-1:0]   mem_wr_data;

  assign dir_req_valid = bus_busy;
  assign dir_req       = cache_coh_req[active_node];
  assign dir_resp_ready = 1'b1;

  directory_controller #(
    .NUM_ENTRIES (256),
    .ADDR_WIDTH  (ADDR_WIDTH),
    .LINE_BYTES  (LINE_BYTES)
  ) u_dir (
    .clk               (clk),
    .rst_n             (rst_n),
    .req_valid         (dir_req_valid),
    .req_ready         (dir_req_ready),
    .req               (dir_req),
    .resp_valid        (dir_resp_valid),
    .resp_ready        (dir_resp_ready),
    .resp              (dir_resp),
    .snoop_valid       (dir_snoop_valid),
    .snoop_ready       (dir_snoop_ready),
    .snoop             (dir_snoop),
    .snoop_resp_valid  (dir_snoop_resp_valid),
    .snoop_resp_ready  (dir_snoop_resp_ready),
    .snoop_resp        (dir_snoop_resp),
    .mem_rd_valid      (mem_rd_valid),
    .mem_rd_ready      (mem_rd_ready),
    .mem_rd_addr       (mem_rd_addr),
    .mem_rd_data_valid (mem_rd_data_valid),
    .mem_rd_data       (mem_rd_data),
    .mem_wr_valid      (mem_wr_valid),
    .mem_wr_ready      (mem_wr_ready),
    .mem_wr_addr       (mem_wr_addr),
    .mem_wr_data       (mem_wr_data)
  );

  // =========================================================================
  // Snoop filter
  // =========================================================================
  snoop_filter #(
    .NUM_ENTRIES (128),
    .NUM_NODES   (NUM_NODES),
    .ADDR_WIDTH  (ADDR_WIDTH)
  ) u_sf (
    .clk              (clk),
    .rst_n            (rst_n),
    .lookup_valid     (dir_snoop_valid),
    .lookup_ready     (),
    .lookup_addr      (dir_snoop.addr),
    .lookup_src       (dir_snoop.src),
    .lookup_hit       (),
    .sharer_mask      (),
    .line_state       (),
    .update_valid     (dir_resp_valid),
    .update_addr      (dir_resp.addr),
    .update_state     (MESI_S),
    .update_node      (dir_resp.dst),
    .update_add       (1'b1),
    .update_clear_all (1'b0)
  );

  // =========================================================================
  // Shared Memory (simple model)
  // =========================================================================
  logic [COH_DATA_W-1:0] shared_mem [MEM_DEPTH];
  logic mem_rd_pending;
  logic [COH_DATA_W-1:0] mem_rd_data_reg;

  assign mem_rd_ready = !mem_rd_pending;
  assign mem_wr_ready = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mem_rd_pending   <= 1'b0;
      mem_rd_data_valid <= 1'b0;
      mem_rd_data_reg  <= '0;
    end else begin
      mem_rd_data_valid <= 1'b0;

      if (mem_wr_valid)
        shared_mem[mem_wr_addr[$clog2(MEM_DEPTH)+4:5]] <= mem_wr_data;

      if (mem_rd_valid && !mem_rd_pending) begin
        mem_rd_pending <= 1'b1;
      end

      if (mem_rd_pending) begin
        mem_rd_data_valid <= 1'b1;
        mem_rd_pending    <= 1'b0;
      end
    end
  end

  assign mem_rd_data = shared_mem[mem_rd_addr[$clog2(MEM_DEPTH)+4:5]];

  // =========================================================================
  // Per-node cache request handling (simplified stubs)
  // =========================================================================
  // In full implementation, each node would have an L1 cache with MESI FSM.
  // For the demo, we use simple request → coherence message translation.

  generate
    for (genvar n = 0; n < NUM_NODES; n++) begin : gen_node
      // Translate test requests to coherence messages
      always_comb begin
        cache_coh_req[n]          = '0;
        cache_coh_valid[n]        = test_req_valid[n];
        cache_coh_req[n].src      = COH_NODE_W'(n);
        cache_coh_req[n].addr     = test_req_addr[n];
        cache_coh_req[n].msg_type = test_req_write[n] ? COH_GET_M : COH_GET_S;
        cache_coh_req[n].data     = COH_DATA_W'(test_req_wdata[n]);
        cache_coh_req[n].has_data = test_req_write[n];
      end

      assign test_req_ready[n] = (active_node == $clog2(NUM_NODES)'(n)) &&
                                  dir_req_ready && bus_busy;
      assign test_resp_valid[n] = (active_node == $clog2(NUM_NODES)'(n)) &&
                                   dir_resp_valid;
      assign test_req_rdata[n]  = dir_resp.data[DATA_WIDTH-1:0];

      // Placeholder hit/miss signals
      assign cache_hit[n]  = test_req_valid[n] && 1'b0; // Always miss in demo
      assign cache_miss[n] = test_req_valid[n];
    end
  endgenerate

  // Snoop bus connection (simplified — broadcast to all nodes)
  assign dir_snoop_ready = 1'b1;
  assign dir_snoop_resp_valid = 1'b0;
  assign dir_snoop_resp = '0;

endmodule
