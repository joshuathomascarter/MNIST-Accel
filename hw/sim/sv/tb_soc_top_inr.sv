// =============================================================================
// tb_soc_top_inr.sv — Novelty-On SoC Regression for In-Network Reduction
// =============================================================================
// Instantiates the full soc_top_v2 with INNET_REDUCE enabled and injects a
// real reduce workload through the live tile-array mesh inside the SoC.
//
// The workload uses a non-corner root to keep the regression broader than the
// original root-0 focused checker. The root tile must observe collapsed partial
// traffic rather than all raw contributors.
// =============================================================================
`timescale 1ns/1ps

module tb_soc_top_inr;

  import noc_pkg::*;

  localparam int    MAX_CYCLES  = 500_000;
  localparam string BOOT_ROM_HEX = "firmware.hex";
  localparam int    CLK_HALF_NS = 10;

  localparam int MESH_ROWS_L = 4;
  localparam int MESH_COLS_L = 4;
  localparam int NUM_NODES_L = MESH_ROWS_L * MESH_COLS_L;
  localparam int NUM_GROUPS  = 4;
  localparam int ROOT_NODE   = 5;
  localparam int ROOT_ROW    = ROOT_NODE / MESH_COLS_L;
  localparam int ROOT_COL    = ROOT_NODE % MESH_COLS_L;
  localparam int SOURCE_VC   = noc_pkg::NUM_VCS - 1;
  localparam int EXPECTED_CONTRIB = NUM_NODES_L - 1;
  localparam int TOTAL_NODE_SUM   = (NUM_NODES_L * (NUM_NODES_L - 1)) / 2;
  localparam int EXPECTED_SUM     = TOTAL_NODE_SUM - ROOT_NODE;
  localparam int EXPECTED_ROOT_PARTIALS_PER_GROUP =
      ((ROOT_COL > 0) ? 1 : 0) +
      ((ROOT_COL < MESH_COLS_L - 1) ? 1 : 0) +
      ((ROOT_ROW > 0) ? 1 : 0) +
      ((ROOT_ROW < MESH_ROWS_L - 1) ? 1 : 0);

  logic clk   = 1'b0;
  logic rst_n = 1'b0;

  always #(CLK_HALF_NS) clk = ~clk;

  logic        uart_rx   = 1'b1;
  logic        uart_tx;
  logic [7:0]  gpio_o;
  logic [7:0]  gpio_i    = 8'h0;
  logic [7:0]  gpio_oe;
  logic        irq_external;
  logic        irq_timer;
  logic        accel_busy;
  logic        accel_done;

  logic [7:0]  dram_phy_act;
  logic [7:0]  dram_phy_read;
  logic [7:0]  dram_phy_write;
  logic [7:0]  dram_phy_pre;
  logic [13:0] dram_phy_row;
  logic [9:0]  dram_phy_col;
  logic        dram_phy_ref;
  logic [31:0] dram_phy_wdata;
  logic [3:0]  dram_phy_wstrb;
  logic [31:0] dram_phy_rdata = 32'h0;
  logic        dram_phy_rdata_valid = 1'b0;
  logic        dram_ctrl_busy;

  int cycle_count = 0;
  int tests_passed = 0;
  int tests_failed = 0;
  int link_flit_count = 0;

  logic [31:0] dram_rdata_pipe [0:7];
  logic [7:0]  dram_rvalid_pipe;

  soc_top_v2 #(
    .BOOT_ROM_FILE    (BOOT_ROM_HEX),
    .CLK_FREQ         (50_000_000),
    .UART_BAUD        (115_200),
    .MESH_ROWS        (MESH_ROWS_L),
    .MESH_COLS        (MESH_COLS_L),
    .SPARSE_VC_ALLOC  (1'b0),
    .INNET_REDUCE     (1'b1)
  ) dut (
    .clk                  (clk),
    .rst_n                (rst_n),
    .uart_rx              (uart_rx),
    .uart_tx              (uart_tx),
    .gpio_o               (gpio_o),
    .gpio_i               (gpio_i),
    .gpio_oe              (gpio_oe),
    .irq_external         (irq_external),
    .irq_timer            (irq_timer),
    .accel_busy           (accel_busy),
    .accel_done           (accel_done),
    .dram_phy_act         (dram_phy_act),
    .dram_phy_read        (dram_phy_read),
    .dram_phy_write       (dram_phy_write),
    .dram_phy_pre         (dram_phy_pre),
    .dram_phy_row         (dram_phy_row),
    .dram_phy_col         (dram_phy_col),
    .dram_phy_ref         (dram_phy_ref),
    .dram_phy_wdata       (dram_phy_wdata),
    .dram_phy_wstrb       (dram_phy_wstrb),
    .dram_phy_rdata       (dram_phy_rdata),
    .dram_phy_rdata_valid (dram_phy_rdata_valid),
    .dram_ctrl_busy       (dram_ctrl_busy)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dram_rvalid_pipe <= '0;
      for (int i = 0; i < 8; i++)
        dram_rdata_pipe[i] <= '0;
    end else begin
      for (int i = 7; i > 0; i--) begin
        dram_rdata_pipe[i]  <= dram_rdata_pipe[i-1];
        dram_rvalid_pipe[i] <= dram_rvalid_pipe[i-1];
      end
      dram_rdata_pipe[0]  <= $urandom_range(0, 255);
      dram_rvalid_pipe[0] <= |dram_phy_read;
    end
  end

  assign dram_phy_rdata       = dram_rdata_pipe[7];
  assign dram_phy_rdata_valid = dram_rvalid_pipe[7];

  task automatic check(
    input string test_name,
    input logic  condition,
    input string fail_msg = ""
  );
    if (condition) begin
      $display("[PASS] %-40s  @cycle %0d", test_name, cycle_count);
      tests_passed++;
    end else begin
      $display("[FAIL] %-40s  @cycle %0d  %s", test_name, cycle_count, fail_msg);
      tests_failed++;
    end
  endtask

  task automatic drive_source(
    input int src,
    input flit_t flit
  );
    begin
      case (src)
        0: begin dut.u_tile_array.mesh_flit_in[0]  = flit; dut.u_tile_array.mesh_valid_in[0]  = 1'b1; end
        1: begin dut.u_tile_array.mesh_flit_in[1]  = flit; dut.u_tile_array.mesh_valid_in[1]  = 1'b1; end
        2: begin dut.u_tile_array.mesh_flit_in[2]  = flit; dut.u_tile_array.mesh_valid_in[2]  = 1'b1; end
        3: begin dut.u_tile_array.mesh_flit_in[3]  = flit; dut.u_tile_array.mesh_valid_in[3]  = 1'b1; end
        4: begin dut.u_tile_array.mesh_flit_in[4]  = flit; dut.u_tile_array.mesh_valid_in[4]  = 1'b1; end
        5: begin dut.u_tile_array.mesh_flit_in[5]  = flit; dut.u_tile_array.mesh_valid_in[5]  = 1'b1; end
        6: begin dut.u_tile_array.mesh_flit_in[6]  = flit; dut.u_tile_array.mesh_valid_in[6]  = 1'b1; end
        7: begin dut.u_tile_array.mesh_flit_in[7]  = flit; dut.u_tile_array.mesh_valid_in[7]  = 1'b1; end
        8: begin dut.u_tile_array.mesh_flit_in[8]  = flit; dut.u_tile_array.mesh_valid_in[8]  = 1'b1; end
        9: begin dut.u_tile_array.mesh_flit_in[9]  = flit; dut.u_tile_array.mesh_valid_in[9]  = 1'b1; end
        10: begin dut.u_tile_array.mesh_flit_in[10] = flit; dut.u_tile_array.mesh_valid_in[10] = 1'b1; end
        11: begin dut.u_tile_array.mesh_flit_in[11] = flit; dut.u_tile_array.mesh_valid_in[11] = 1'b1; end
        12: begin dut.u_tile_array.mesh_flit_in[12] = flit; dut.u_tile_array.mesh_valid_in[12] = 1'b1; end
        13: begin dut.u_tile_array.mesh_flit_in[13] = flit; dut.u_tile_array.mesh_valid_in[13] = 1'b1; end
        14: begin dut.u_tile_array.mesh_flit_in[14] = flit; dut.u_tile_array.mesh_valid_in[14] = 1'b1; end
        15: begin dut.u_tile_array.mesh_flit_in[15] = flit; dut.u_tile_array.mesh_valid_in[15] = 1'b1; end
        default: begin end
      endcase
    end
  endtask

  task automatic clear_source(input int src);
    begin
      case (src)
        0: begin dut.u_tile_array.mesh_flit_in[0]  = '0; dut.u_tile_array.mesh_valid_in[0]  = 1'b0; end
        1: begin dut.u_tile_array.mesh_flit_in[1]  = '0; dut.u_tile_array.mesh_valid_in[1]  = 1'b0; end
        2: begin dut.u_tile_array.mesh_flit_in[2]  = '0; dut.u_tile_array.mesh_valid_in[2]  = 1'b0; end
        3: begin dut.u_tile_array.mesh_flit_in[3]  = '0; dut.u_tile_array.mesh_valid_in[3]  = 1'b0; end
        4: begin dut.u_tile_array.mesh_flit_in[4]  = '0; dut.u_tile_array.mesh_valid_in[4]  = 1'b0; end
        5: begin dut.u_tile_array.mesh_flit_in[5]  = '0; dut.u_tile_array.mesh_valid_in[5]  = 1'b0; end
        6: begin dut.u_tile_array.mesh_flit_in[6]  = '0; dut.u_tile_array.mesh_valid_in[6]  = 1'b0; end
        7: begin dut.u_tile_array.mesh_flit_in[7]  = '0; dut.u_tile_array.mesh_valid_in[7]  = 1'b0; end
        8: begin dut.u_tile_array.mesh_flit_in[8]  = '0; dut.u_tile_array.mesh_valid_in[8]  = 1'b0; end
        9: begin dut.u_tile_array.mesh_flit_in[9]  = '0; dut.u_tile_array.mesh_valid_in[9]  = 1'b0; end
        10: begin dut.u_tile_array.mesh_flit_in[10] = '0; dut.u_tile_array.mesh_valid_in[10] = 1'b0; end
        11: begin dut.u_tile_array.mesh_flit_in[11] = '0; dut.u_tile_array.mesh_valid_in[11] = 1'b0; end
        12: begin dut.u_tile_array.mesh_flit_in[12] = '0; dut.u_tile_array.mesh_valid_in[12] = 1'b0; end
        13: begin dut.u_tile_array.mesh_flit_in[13] = '0; dut.u_tile_array.mesh_valid_in[13] = 1'b0; end
        14: begin dut.u_tile_array.mesh_flit_in[14] = '0; dut.u_tile_array.mesh_valid_in[14] = 1'b0; end
        15: begin dut.u_tile_array.mesh_flit_in[15] = '0; dut.u_tile_array.mesh_valid_in[15] = 1'b0; end
        default: begin end
      endcase
    end
  endtask

  task automatic inject_group(input int group_id);
    begin
      for (int src = 0; src < NUM_NODES_L; src++) begin
        if (src != ROOT_NODE) begin
          drive_source(src, make_reduce_flit(
            4'(src),
            4'(ROOT_NODE),
            2'(SOURCE_VC),
            8'(group_id),
            4'(EXPECTED_CONTRIB),
            32'(src),
            4'd1
          ));
        end
      end

      @(posedge clk);

      for (int src = 0; src < NUM_NODES_L; src++) begin
        if (src != ROOT_NODE) begin
          clear_source(src);
        end
      end
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    int cycle_link_flits;

    if (!rst_n) begin
      cycle_count     <= 0;
      link_flit_count <= 0;
    end else begin
      cycle_count <= cycle_count + 1;
      if (cycle_count >= MAX_CYCLES) begin
        $display("[TB] TIMEOUT: %0d cycles exceeded without test completion.", MAX_CYCLES);
        $finish;
      end

      cycle_link_flits = 0;
      for (int node = 0; node < NUM_NODES_L; node++) begin
        for (int port = 0; port < 4; port++) begin
          if (dut.u_tile_array.u_mesh.r_valid_out[node][port])
            cycle_link_flits++;
        end
      end
      link_flit_count <= link_flit_count + cycle_link_flits;
    end
  end

  initial begin
    $display("========================================================================");
    $display("  tb_soc_top_inr — Full SoC INR Regression");
    $display("  soc_top_v2 with INNET_REDUCE=1, root tile %0d", ROOT_NODE);
    $display("========================================================================");

    repeat (20) @(posedge clk);
    rst_n = 1'b1;
    repeat (20) @(posedge clk);

    check("accel_done low before injection", !accel_done);

    for (int gid = 0; gid < NUM_GROUPS; gid++) begin
      int wait_cycles;

      inject_group(gid);

      wait_cycles = 0;
      while (!dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_result_valid[gid] &&
             (wait_cycles < 20000)) begin
        @(posedge clk);
        wait_cycles++;
      end

      check($sformatf("group %0d completed", gid),
            dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_result_valid[gid],
            $sformatf("groups_completed=%0d reduce_packets=%0d last_id=%0d last_value=%0d",
                      dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_groups_completed,
                      dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_packets_consumed,
                      dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_last_id,
                      dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_last_value));

      check($sformatf("group %0d result correct", gid),
            dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_results[gid] == EXPECTED_SUM,
            $sformatf("got %0d expected %0d",
                      dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_results[gid],
                      EXPECTED_SUM));
    end

    check("all groups completed",
          dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_groups_completed == NUM_GROUPS,
          $sformatf("got %0d expected %0d",
                    dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_groups_completed,
                    NUM_GROUPS));

    check("root packets collapsed",
          dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_packets_consumed ==
            (NUM_GROUPS * EXPECTED_ROOT_PARTIALS_PER_GROUP),
          $sformatf("got %0d expected %0d",
                    dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_packets_consumed,
                    NUM_GROUPS * EXPECTED_ROOT_PARTIALS_PER_GROUP));

    $display("[METRIC] root=%0d groups=%0d root_packets=%0d link_flits=%0d expected_sum=%0d",
             ROOT_NODE,
             dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_groups_completed,
             dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_packets_consumed,
             link_flit_count,
             EXPECTED_SUM);

    $display("========================================================================");
    $display("  tb_soc_top_inr RESULTS: %0d passed, %0d failed", tests_passed, tests_failed);
    $display("  Total simulation cycles: %0d", cycle_count);
    $display("========================================================================");

    if (tests_failed == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  FAILURES: %0d test(s) failed", tests_failed);

    $finish;
  end

endmodule
