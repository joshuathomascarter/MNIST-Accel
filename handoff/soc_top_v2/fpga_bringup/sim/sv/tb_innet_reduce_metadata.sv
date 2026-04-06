`timescale 1ns/1ps

module tb_innet_reduce_metadata #(
  parameter bit INNET_REDUCE = 1'b1
);

  import noc_pkg::*;

  localparam int NUM_NODES   = noc_pkg::NUM_NODES;
  localparam int NUM_GROUPS  = 4;
  localparam int ROOT_NODE   = 0;
  localparam int SOURCE_VC   = noc_pkg::NUM_VCS - 1;
  localparam int BUF_DEPTH_L = noc_pkg::BUF_DEPTH;
  localparam int NUM_SRCS    = 3;
  localparam int EXPECTED_CONTRIB = NUM_SRCS;
  localparam int EXPECTED_SUM = 10;
  localparam int MAX_WAIT    = 10000;
  localparam int EXPECTED_ROOT_PACKETS_PER_GROUP = INNET_REDUCE ? 2 : 3;

  logic clk = 1'b0;
  logic rst_n = 1'b0;

  always #5 clk = ~clk;

  flit_t               local_flit_in   [NUM_NODES];
  logic                local_valid_in  [NUM_NODES];
  logic [NUM_VCS-1:0]  local_credit_out[NUM_NODES];
  flit_t               local_flit_out  [NUM_NODES];
  logic                local_valid_out [NUM_NODES];
  logic [NUM_VCS-1:0]  local_credit_in [NUM_NODES];
  logic                inr_meta_cfg_valid [NUM_NODES];
  logic [7:0]          inr_meta_cfg_reduce_id [NUM_NODES];
  logic [3:0]          inr_meta_cfg_target [NUM_NODES];
  logic                inr_meta_cfg_enable [NUM_NODES];

  flit_t               root_flit_out;
  logic                root_valid_out;
  logic [NUM_VCS-1:0]  root_credit_in;
  flit_t               root_flit_in;
  logic                root_valid_in;
  logic [NUM_VCS-1:0]  root_credit_out;

  logic [31:0] dummy_csr_rdata;
  logic root_barrier_req;
  logic root_tile_busy;
  logic root_tile_done;

  integer cycle_count;
  integer start_cycle;
  integer end_cycle;
  integer root_packets_seen;
  integer link_flit_count;
  integer src_credits [NUM_NODES];
  integer root_group_packets [256];
  integer root_group_contrib [256];

  noc_mesh_4x4 #(
    .MESH_ROWS       (4),
    .MESH_COLS       (4),
    .NUM_VCS         (NUM_VCS),
    .BUF_DEPTH       (BUF_DEPTH_L),
    .SPARSE_VC_ALLOC (1'b0),
    .INNET_REDUCE    (INNET_REDUCE)
  ) u_mesh (
    .clk              (clk),
    .rst_n            (rst_n),
    .inr_meta_cfg_valid     (inr_meta_cfg_valid),
    .inr_meta_cfg_reduce_id (inr_meta_cfg_reduce_id),
    .inr_meta_cfg_target    (inr_meta_cfg_target),
    .inr_meta_cfg_enable    (inr_meta_cfg_enable),
    .local_flit_in    (local_flit_in),
    .local_valid_in   (local_valid_in),
    .local_credit_out (local_credit_out),
    .local_flit_out   (local_flit_out),
    .local_valid_out  (local_valid_out),
    .local_credit_in  (local_credit_in)
  );

  accel_tile #(
    .TILE_ID   (ROOT_NODE),
    .N_ROWS    (16),
    .N_COLS    (16),
    .DATA_W    (8),
    .ACC_W     (32),
    .SP_DEPTH  (4096),
    .SP_DATA_W (64),
    .NUM_VCS   (NUM_VCS)
  ) u_root (
    .clk            (clk),
    .rst_n          (rst_n),
    .noc_flit_out   (root_flit_out),
    .noc_valid_out  (root_valid_out),
    .noc_credit_in  (root_credit_in),
    .noc_flit_in    (root_flit_in),
    .noc_valid_in   (root_valid_in),
    .noc_credit_out (root_credit_out),
    .csr_wdata      ('0),
    .csr_addr       ('0),
    .csr_wen        (1'b0),
    .csr_rdata      (dummy_csr_rdata),
    .barrier_req    (root_barrier_req),
    .barrier_done   (1'b0),
    .tile_busy      (root_tile_busy),
    .tile_done      (root_tile_done)
  );

  function automatic bit is_source(input int node);
    begin
      is_source = (node == 1) || (node == 4) || (node == 5);
    end
  endfunction

  genvar n;
  generate
    for (n = 0; n < NUM_NODES; n++) begin : gen_local_bind
      if (n == ROOT_NODE) begin : gen_root_bind
        assign local_flit_in[n]   = root_flit_out;
        assign local_valid_in[n]  = root_valid_out;
        assign root_credit_in     = local_credit_out[n];
        assign root_flit_in       = local_flit_out[n];
        assign root_valid_in      = local_valid_out[n];
        assign local_credit_in[n] = root_credit_out;
      end else begin : gen_dummy_bind
        always_comb begin
          local_credit_in[n] = '0;
          if (local_valid_out[n])
            local_credit_in[n][local_flit_out[n].vc_id] = 1'b1;
        end
      end
    end
  endgenerate

  task automatic clear_metadata_bus;
    begin
      for (int node = 0; node < NUM_NODES; node++) begin
        inr_meta_cfg_valid[node]     = 1'b0;
        inr_meta_cfg_reduce_id[node] = '0;
        inr_meta_cfg_target[node]    = '0;
        inr_meta_cfg_enable[node]    = 1'b0;
      end
    end
  endtask

  task automatic program_target(
    input int node,
    input int group_id,
    input int target,
    input bit enable
  );
    begin
      clear_metadata_bus();
      inr_meta_cfg_valid[node]     = 1'b1;
      inr_meta_cfg_reduce_id[node] = 8'(group_id);
      inr_meta_cfg_target[node]    = 4'(target);
      inr_meta_cfg_enable[node]    = enable;
      @(posedge clk);
      clear_metadata_bus();
    end
  endtask

  task automatic program_sparse_tree(input int group_id);
    begin
      // Custom 3-source tree for sources {1,4,5} rooted at node 0.
      program_target(5, group_id, 1, 1'b1);
      program_target(4, group_id, 2, 1'b1);
      program_target(1, group_id, 1, 1'b1);
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    int cycle_link_flits;

    if (!rst_n) begin
      cycle_count       <= 0;
      root_packets_seen <= 0;
      link_flit_count   <= 0;
      for (int i = 0; i < NUM_NODES; i++)
        src_credits[i] <= BUF_DEPTH_L;
      for (int gid = 0; gid < 256; gid++) begin
        root_group_packets[gid] <= 0;
        root_group_contrib[gid] <= 0;
      end
    end else begin
      cycle_count <= cycle_count + 1;

      for (int i = 0; i < NUM_NODES; i++) begin
        if (is_source(i)) begin
          if (local_valid_in[i])
            src_credits[i] <= src_credits[i] - 1;
          if (local_credit_out[i][SOURCE_VC])
            src_credits[i] <= src_credits[i] + 1;
        end
      end

      if (root_valid_in && (root_credit_out != '0)) begin
        root_packets_seen <= root_packets_seen + 1;
        if (root_flit_in.msg_type == MSG_REDUCE) begin
          root_group_packets[root_flit_in.payload[REDUCE_ID_HI:REDUCE_ID_LO]] <=
            root_group_packets[root_flit_in.payload[REDUCE_ID_HI:REDUCE_ID_LO]] + 1;
          root_group_contrib[root_flit_in.payload[REDUCE_ID_HI:REDUCE_ID_LO]] <=
            root_group_contrib[root_flit_in.payload[REDUCE_ID_HI:REDUCE_ID_LO]] +
            root_flit_in.payload[REDUCE_COUNT_HI:REDUCE_COUNT_LO];
        end
      end

      cycle_link_flits = 0;
      for (int node = 0; node < NUM_NODES; node++) begin
        for (int port = 0; port < 4; port++) begin
          if (u_mesh.r_valid_out[node][port])
            cycle_link_flits++;
        end
      end
      link_flit_count <= link_flit_count + cycle_link_flits;
    end
  end

  task automatic drive_group(input int group_id);
    begin
      wait (rst_n);
      if (INNET_REDUCE)
        program_sparse_tree(group_id);

      wait (src_credits[1] > 0);
      wait (src_credits[4] > 0);
      wait (src_credits[5] > 0);

      local_flit_in[1]  = make_reduce_flit(4'd1, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                           4'(EXPECTED_CONTRIB), 32'd1, 4'd1);
      local_valid_in[1] = 1'b1;
      local_flit_in[4]  = make_reduce_flit(4'd4, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                           4'(EXPECTED_CONTRIB), 32'd4, 4'd1);
      local_valid_in[4] = 1'b1;
      local_flit_in[5]  = make_reduce_flit(4'd5, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                           4'(EXPECTED_CONTRIB), 32'd5, 4'd1);
      local_valid_in[5] = 1'b1;

      @(posedge clk);

      local_flit_in[1]  = '0;
      local_valid_in[1] = 1'b0;
      local_flit_in[4]  = '0;
      local_valid_in[4] = 1'b0;
      local_flit_in[5]  = '0;
      local_valid_in[5] = 1'b0;
    end
  endtask

  initial begin
    for (int i = 0; i < NUM_NODES; i++) begin
      if (i != ROOT_NODE) begin
        local_flit_in[i]  = '0;
        local_valid_in[i] = 1'b0;
      end
    end
    clear_metadata_bus();

    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);
    start_cycle = cycle_count;

    for (int gid = 0; gid < NUM_GROUPS; gid++) begin
      drive_group(gid);

      for (int wait_cycles = 0; wait_cycles < MAX_WAIT; wait_cycles++) begin
        @(posedge clk);
        if (u_root.reduce_result_valid[gid])
          break;
      end

      if (!u_root.reduce_result_valid[gid]) begin
        $display("[FAIL] group %0d did not complete", gid);
        $display("[DEBUG] root_packets=%0d group_packets=%0d group_contrib=%0d groups_completed=%0d last_id=%0d last_value=%0d reduce_packets=%0d",
                 root_packets_seen,
                 root_group_packets[gid],
                 root_group_contrib[gid],
                 u_root.reduce_groups_completed,
                 u_root.reduce_last_id,
                 u_root.reduce_last_value,
                 u_root.reduce_packets_consumed);
        $fatal(1, "tb_innet_reduce_metadata failed: group %0d incomplete", gid);
      end
    end

    end_cycle = cycle_count;

    for (int gid = 0; gid < NUM_GROUPS; gid++) begin
      if (u_root.reduce_results[gid] != EXPECTED_SUM) begin
        $display("[FAIL] group %0d result mismatch: got %0d expected %0d",
                 gid, u_root.reduce_results[gid], EXPECTED_SUM);
        $fatal(1, "tb_innet_reduce_metadata failed: group %0d result mismatch", gid);
      end
      if (root_group_contrib[gid] != EXPECTED_CONTRIB) begin
        $display("[FAIL] group %0d contrib mismatch: got %0d expected %0d",
                 gid, root_group_contrib[gid], EXPECTED_CONTRIB);
        $fatal(1, "tb_innet_reduce_metadata failed: group %0d contrib mismatch", gid);
      end
      if (root_group_packets[gid] != EXPECTED_ROOT_PACKETS_PER_GROUP) begin
        $display("[FAIL] group %0d root packet mismatch: got %0d expected %0d",
                 gid, root_group_packets[gid], EXPECTED_ROOT_PACKETS_PER_GROUP);
        $fatal(1, "tb_innet_reduce_metadata failed: group %0d root packet mismatch", gid);
      end
    end

    $display("[METRIC] INNET_REDUCE=%0d subset={1,4,5} root_packets=%0d link_flits=%0d cycles=%0d groups=%0d expected_sum=%0d",
             INNET_REDUCE, root_packets_seen, link_flit_count, end_cycle - start_cycle,
             u_root.reduce_groups_completed, EXPECTED_SUM);
    $display("[PASS] tb_innet_reduce_metadata completed with explicit subtree metadata");
    $finish;
  end

endmodule
