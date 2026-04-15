// =============================================================================
// noc_router.sv — 5-Port Wormhole Router (Top-Level)
// =============================================================================
// Instantiates: 5× input_port, 1× vc_allocator, 1× switch_allocator,
//               1× crossbar_5x5, 5× credit_counter
//
// Pipeline: Route Compute → VC Allocate → Switch Allocate → Crossbar Traverse
// (route compute is inside input_port, so 3 external pipeline stages)
//
// Parameter SPARSE_VC_ALLOC swaps in the sparsity-aware VC allocator.
//
// Parameter INNET_REDUCE enables the in-network reduction engine.
// When set, MSG_REDUCE single-flit packets are intercepted and accumulated
// at intermediate routers — only the final reduced flit is forwarded.
// Traffic reduction for N-tile all-reduce: (N-1)/N flits eliminated.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module noc_router #(
  parameter int  ROUTER_ID       = 0,
  parameter int  NUM_PORTS       = 5,
  parameter int  NUM_VCS         = noc_pkg::NUM_VCS,
  parameter int  BUF_DEPTH       = noc_pkg::BUF_DEPTH,
  parameter int  MESH_ROWS       = noc_pkg::MESH_ROWS,
  parameter int  MESH_COLS       = noc_pkg::MESH_COLS,
  parameter bit  SPARSE_VC_ALLOC = 1'b0,  // 0 = baseline RR, 1 = sparsity-aware
  parameter bit  INNET_REDUCE    = 1'b0   // 0 = pass-through, 1 = in-network reduction
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // Optional per-group subtree metadata programming for this router.
  input  logic                  inr_meta_cfg_valid,
  input  logic [7:0]            inr_meta_cfg_reduce_id,
  input  logic [3:0]            inr_meta_cfg_target,
  input  logic                  inr_meta_cfg_enable,

  // --- External link interfaces (5 ports: N, S, E, W, Local) ---
  input  flit_t                 link_flit_in   [NUM_PORTS],
  input  logic                  link_valid_in  [NUM_PORTS],
  output logic [NUM_VCS-1:0]    link_credit_out[NUM_PORTS],

  output flit_t                 link_flit_out  [NUM_PORTS],
  output logic                  link_valid_out [NUM_PORTS],
  input  logic [NUM_VCS-1:0]    link_credit_in [NUM_PORTS]
);

  // Router coordinates derived from ID
  localparam logic [ROW_BITS-1:0] CUR_ROW = node_row(NODE_BITS'(ROUTER_ID));
  localparam logic [COL_BITS-1:0] CUR_COL = node_col(NODE_BITS'(ROUTER_ID));

  // =========================================================================
  // Internal wires
  // =========================================================================

  // Input port → VC allocator + switch allocator
  logic [NUM_VCS-1:0]     ip_vc_has_flit  [NUM_PORTS];
  logic [NUM_VCS-1:0]     ip_vc_has_head  [NUM_PORTS];
  logic [PORT_BITS-1:0]   ip_vc_route     [NUM_PORTS][NUM_VCS];
  flit_t                  ip_vc_head_flit [NUM_PORTS][NUM_VCS];

  // Switch allocator → input port (read grant)
  logic [NUM_VCS-1:0]     ip_vc_read      [NUM_PORTS];

  // Input port → crossbar
  flit_t                  ip_read_flit    [NUM_PORTS];
  logic [VC_BITS-1:0]     ip_read_vc      [NUM_PORTS];

  // VC allocator
  logic [NUM_VCS-1:0]     vca_req         [NUM_PORTS];
  logic [PORT_BITS-1:0]   vca_req_port    [NUM_PORTS][NUM_VCS];
  msg_type_e              vca_req_msg     [NUM_PORTS][NUM_VCS];
  logic [NUM_VCS-1:0]     vca_grant       [NUM_PORTS];
  logic [VC_BITS-1:0]     vca_grant_vc    [NUM_PORTS][NUM_VCS];
  logic [NUM_VCS-1:0]     vca_vc_busy     [NUM_PORTS];
  logic [NUM_VCS-1:0]     vca_release     [NUM_PORTS];
  logic [VC_BITS-1:0]     vca_release_id  [NUM_PORTS][NUM_VCS];

  // Output VC state tracking
  logic [NUM_PORTS-1:0][NUM_VCS-1:0] out_vc_busy;
  logic [VC_BITS-1:0]     out_vc_owner_vc [NUM_PORTS][NUM_VCS]; // which input VC owns this output VC

  // Switch allocator
  logic [NUM_VCS-1:0]     sa_req          [NUM_PORTS];
  logic [PORT_BITS-1:0]   sa_target       [NUM_PORTS][NUM_VCS];
  logic [NUM_VCS-1:0]     sa_grant        [NUM_PORTS];
  logic [PORT_BITS-1:0]   xbar_sel        [NUM_PORTS];
  logic                   xbar_valid      [NUM_PORTS];

  // Credit counters
  logic [NUM_VCS-1:0]     out_has_credit  [NUM_PORTS];

  // Crossbar
  flit_t                  xbar_out_flit   [NUM_PORTS];
  logic                   xbar_out_valid  [NUM_PORTS];

  // In-network reduce intercept wires
  logic  inr_intercept   [NUM_PORTS];    // 1 → absorb this flit, don't forward
  flit_t inr_inject_flit;               // new accumulated flit to inject
  logic  inr_inject_valid;              // injection request
  logic  inr_inject_ready;             // injection credit available
  logic [PORT_BITS-1:0] inr_inject_port;

  // Final output flit (after inject mux)
  flit_t                  out_flit_final  [NUM_PORTS];
  logic                   out_valid_final [NUM_PORTS];

  // Per-(ip,iv) allocated output VC  (registered after VC alloc grant)
  logic [NUM_PORTS-1:0][NUM_VCS-1:0][VC_BITS-1:0] alloc_out_vc;
  logic [NUM_PORTS-1:0][NUM_VCS-1:0] alloc_valid; // VC allocation is active

  // =========================================================================
  // 1. Input Ports
  // =========================================================================
  generate
    for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_ip
      noc_input_port #(
        .NUM_VCS   (NUM_VCS),
        .BUF_DEPTH (BUF_DEPTH),
        .MESH_ROWS (MESH_ROWS),
        .MESH_COLS (MESH_COLS)
      ) u_ip (
        .clk           (clk),
        .rst_n         (rst_n),
        .flit_in       (link_flit_in[p]),
        .flit_valid_in (link_valid_in[p]),
        .credit_out    (link_credit_out[p]),
        .cur_row       (CUR_ROW),
        .cur_col       (CUR_COL),
        .vc_has_flit   (ip_vc_has_flit[p]),
        .vc_has_head   (ip_vc_has_head[p]),
        .vc_route      (ip_vc_route[p]),
        .vc_head_flit  (ip_vc_head_flit[p]),
        .vc_read       (ip_vc_read[p]),
        .read_flit     (ip_read_flit[p]),
        .read_vc       (ip_read_vc[p])
      );
    end
  endgenerate

  // =========================================================================
  // 2. VC request generation
  // =========================================================================
  // A (ip, iv) pair requests VC allocation when:
  //   - It has a HEAD flit at the front
  //   - It hasn't been allocated yet
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++) begin
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        vca_req[ip][iv]      = ip_vc_has_head[ip][iv] && !alloc_valid[ip][iv];
        vca_req_port[ip][iv] = ip_vc_route[ip][iv];
        vca_req_msg[ip][iv]  = msg_type_e'(ip_vc_head_flit[ip][iv].msg_type);
      end
    end
    for (int op = 0; op < NUM_PORTS; op++)
      vca_vc_busy[op] = out_vc_busy[op];
  end

  // =========================================================================
  // 3. VC Allocator (baseline or sparsity-aware)
  // =========================================================================
  // Release when TAIL traverses
  always_comb begin
    for (int op = 0; op < NUM_PORTS; op++) begin
      vca_release[op] = '0;
      for (int ov = 0; ov < NUM_VCS; ov++)
        vca_release_id[op][ov] = '0;
    end

    for (int op = 0; op < NUM_PORTS; op++) begin
      if (xbar_valid[op] && xbar_out_valid[op]) begin
        if (xbar_out_flit[op].flit_type == FLIT_TAIL ||
            xbar_out_flit[op].flit_type == FLIT_HEADTAIL) begin
          vca_release[op][xbar_out_flit[op].vc_id] = 1'b1;
          vca_release_id[op][xbar_out_flit[op].vc_id] = xbar_out_flit[op].vc_id;
        end
      end
    end
  end

  generate
    if (SPARSE_VC_ALLOC) begin : gen_sparse_vca
      noc_vc_allocator_sparse #(
        .NUM_PORTS (NUM_PORTS),
        .NUM_VCS   (NUM_VCS)
      ) u_vca (
        .clk        (clk),
        .rst_n      (rst_n),
        .req        (vca_req),
        .req_port   (vca_req_port),
        .req_msg    (vca_req_msg),
        .vc_busy    (vca_vc_busy),
        .grant      (vca_grant),
        .grant_vc   (vca_grant_vc),
        .release_vc (vca_release),
        .release_id (vca_release_id)
      );
    end else begin : gen_baseline_vca
      noc_vc_allocator #(
        .NUM_PORTS (NUM_PORTS),
        .NUM_VCS   (NUM_VCS)
      ) u_vca (
        .clk        (clk),
        .rst_n      (rst_n),
        .req        (vca_req),
        .req_port   (vca_req_port),
        .vc_busy    (vca_vc_busy),
        .grant      (vca_grant),
        .grant_vc   (vca_grant_vc),
        .release_vc (vca_release),
        .release_id (vca_release_id)
      );
    end
  endgenerate

  // =========================================================================
  // 4. Output VC state tracking + alloc_out_vc registers
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int ip = 0; ip < NUM_PORTS; ip++)
        for (int iv = 0; iv < NUM_VCS; iv++) begin
          alloc_valid[ip][iv]  <= 1'b0;
          alloc_out_vc[ip][iv] <= '0;
        end
      for (int op = 0; op < NUM_PORTS; op++)
        out_vc_busy[op] <= '0;
    end else begin
      // Grant → register allocation
      for (int ip = 0; ip < NUM_PORTS; ip++)
        for (int iv = 0; iv < NUM_VCS; iv++)
          if (vca_grant[ip][iv]) begin
            alloc_valid[ip][iv]  <= 1'b1;
            alloc_out_vc[ip][iv] <= vca_grant_vc[ip][iv];
            // Mark output VC as busy (explicit loop avoids nested variable-index write)
            for (int op2 = 0; op2 < NUM_PORTS; op2++)
              for (int ov2 = 0; ov2 < NUM_VCS; ov2++)
                if (op2 == vca_req_port[ip][iv] && ov2 == vca_grant_vc[ip][iv])
                  out_vc_busy[op2][ov2] <= 1'b1;
          end

      // Release → clear allocation on TAIL
      for (int op = 0; op < NUM_PORTS; op++)
        for (int ov = 0; ov < NUM_VCS; ov++)
          if (vca_release[op][ov])
            out_vc_busy[op][ov] <= 1'b0;

      // Clear alloc_valid when TAIL traverses from this (ip, iv)
      for (int ip = 0; ip < NUM_PORTS; ip++)
        for (int iv = 0; iv < NUM_VCS; iv++)
          if (sa_grant[ip][iv] && alloc_valid[ip][iv]) begin
            flit_t f;
            f = ip_vc_head_flit[ip][iv];
            if (f.flit_type == FLIT_TAIL || f.flit_type == FLIT_HEADTAIL)
              alloc_valid[ip][iv] <= 1'b0;
          end
    end
  end

  // =========================================================================
  // 5. Switch Allocator request generation
  // =========================================================================
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      for (int iv = 0; iv < NUM_VCS; iv++) begin
        // Can request switch if: has flit AND has allocated output VC
        sa_req[ip][iv]    = ip_vc_has_flit[ip][iv] && alloc_valid[ip][iv];
        sa_target[ip][iv] = ip_vc_route[ip][iv];
      end
  end

  // =========================================================================
  // 6. Switch Allocator
  // =========================================================================
  noc_switch_allocator #(
    .NUM_PORTS (NUM_PORTS),
    .NUM_VCS   (NUM_VCS)
  ) u_sa (
    .clk            (clk),
    .rst_n          (rst_n),
    .sa_req         (sa_req),
    .sa_target      (sa_target),
    .out_has_credit (out_has_credit),
    .alloc_vc       (alloc_out_vc),
    .sa_grant       (sa_grant),
    .xbar_sel       (xbar_sel),
    .xbar_valid     (xbar_valid)
  );

  // Connect read grants to input ports
  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      ip_vc_read[ip] = sa_grant[ip];
  end

  // =========================================================================
  // 7. Crossbar
  // =========================================================================
  // Build crossbar input: the flit read from each input port
  logic [VC_BITS-1:0] xbar_in_vc [NUM_PORTS];

  always_comb begin
    for (int ip = 0; ip < NUM_PORTS; ip++)
      xbar_in_vc[ip] = alloc_out_vc[ip][ip_read_vc[ip]];
  end

  noc_crossbar_5x5 #(
    .NUM_PORTS (NUM_PORTS)
  ) u_xbar (
    .in_flit    (ip_read_flit),
    .in_vc      (xbar_in_vc),
    .xbar_sel   (xbar_sel),
    .xbar_valid (xbar_valid),
    .out_flit   (xbar_out_flit),
    .out_valid  (xbar_out_valid)
  );

  // =========================================================================
  // 8. Output links + credit counters
  // =========================================================================
  generate
    for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_credit
      noc_credit_counter #(
        .BUF_DEPTH (BUF_DEPTH),
        .NUM_VCS   (NUM_VCS)
      ) u_cc (
        .clk        (clk),
        .rst_n      (rst_n),
        .credit_in  (link_credit_in[p]),
        .flit_sent  (out_valid_final[p]),
        .flit_vc    (out_flit_final[p].vc_id),
        .has_credit (out_has_credit[p])
      );
    end
  endgenerate

  always_comb begin
    logic [ROW_BITS-1:0] dst_row;
    logic [COL_BITS-1:0] dst_col;
    logic signed [ROW_BITS:0] dst_row_delta;
    logic signed [COL_BITS:0] dst_col_delta;

    dst_row = node_row(NODE_BITS'(inr_inject_flit.dst_id));
    dst_col = node_col(NODE_BITS'(inr_inject_flit.dst_id));
    dst_row_delta = $signed({1'b0, dst_row}) - $signed({1'b0, CUR_ROW});
    dst_col_delta = $signed({1'b0, dst_col}) - $signed({1'b0, CUR_COL});

    if (dst_col_delta > 0)
      inr_inject_port = PORT_BITS'(PORT_EAST);
    else if (dst_col_delta < 0)
      inr_inject_port = PORT_BITS'(PORT_WEST);
    else if (dst_row_delta > 0)
      inr_inject_port = PORT_BITS'(PORT_SOUTH);
    else if (dst_row_delta < 0)
      inr_inject_port = PORT_BITS'(PORT_NORTH);
    else
      inr_inject_port = PORT_BITS'(PORT_LOCAL);
  end

  // =========================================================================
  // 9. In-Network Reduction Engine (conditional)
  // =========================================================================
  generate
    if (INNET_REDUCE) begin : gen_innet_reduce
      noc_innet_reduce #(
        .NODE_ID   (ROUTER_ID),
        .NUM_PORTS (NUM_PORTS),
        .MESH_ROWS (MESH_ROWS),
        .MESH_COLS (MESH_COLS),
        .SP_DEPTH  (noc_pkg::INNET_SP_DEPTH)
      ) u_inr (
        .clk          (clk),
        .rst_n        (rst_n),
        .cfg_valid    (inr_meta_cfg_valid),
        .cfg_reduce_id(inr_meta_cfg_reduce_id),
        .cfg_target   (inr_meta_cfg_target),
        .cfg_enable   (inr_meta_cfg_enable),
        .enable       (1'b1),
        .flit_in      (xbar_out_flit),
        .valid_in     (xbar_out_valid),
        .src_port_in  (xbar_sel),
        .intercept    (inr_intercept),
        .inject_flit  (inr_inject_flit),
        .inject_valid (inr_inject_valid),
        .inject_ready (inr_inject_ready)
      );

      // Emit the accumulated result on the routed output port when that
      // output link is free and the selected VC has downstream credit.
      assign inr_inject_ready = (((!xbar_out_valid[inr_inject_port]) ||
                                  inr_intercept[inr_inject_port]) &&
                                 out_has_credit[inr_inject_port][inr_inject_flit.vc_id]);

      always_comb begin
        for (int p = 0; p < NUM_PORTS; p++) begin
          if (inr_intercept[p]) begin
            // Absorb this flit here, do not drive output
            out_flit_final[p]  = '0;
            out_valid_final[p] = 1'b0;
          end else begin
            out_flit_final[p]  = xbar_out_flit[p];
            out_valid_final[p] = xbar_out_valid[p];
          end
        end
        if (inr_inject_valid && inr_inject_ready) begin
          out_flit_final[inr_inject_port]  = inr_inject_flit;
          out_valid_final[inr_inject_port] = 1'b1;
        end
      end

    end else begin : gen_passthru
      // No in-network reduction: bypass all intercept/inject logic
      for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_passthru_intercept
        assign inr_intercept[p]  = 1'b0;
      end
      assign inr_inject_flit  = '0;
      assign inr_inject_valid = 1'b0;
      assign inr_inject_ready = 1'b0;
      for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_passthru_output
        assign out_flit_final[p]  = xbar_out_flit[p];
        assign out_valid_final[p] = xbar_out_valid[p];
      end
    end
  endgenerate

  // Drive output links
  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++) begin
      link_flit_out[p]  = out_flit_final[p];
      link_valid_out[p] = out_valid_final[p];
    end
  end

endmodule
