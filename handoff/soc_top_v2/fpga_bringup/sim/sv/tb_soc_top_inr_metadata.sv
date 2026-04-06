`timescale 1ns/1ps

module tb_soc_top_inr_metadata;

  import noc_pkg::*;

  localparam int    MAX_CYCLES  = 500_000;
  localparam string BOOT_ROM_HEX = "firmware.hex";
  localparam int    CLK_HALF_NS = 10;

  localparam int MESH_ROWS_L = 4;
  localparam int MESH_COLS_L = 4;
  localparam int NUM_NODES_L = MESH_ROWS_L * MESH_COLS_L;
  localparam int NUM_GROUPS  = 4;
  localparam int ROOT_NODE   = 0;
  localparam int SOURCE_VC   = noc_pkg::NUM_VCS - 1;
  localparam int EXPECTED_CONTRIB = 3;
  localparam int EXPECTED_SUM     = 10;
  localparam int EXPECTED_ROOT_PARTIALS_PER_GROUP = 2;

  localparam logic [31:0] ACCEL_META_ROUTER_ADDR    = 32'h3000_0080;
  localparam logic [31:0] ACCEL_META_REDUCE_ID_ADDR = 32'h3000_0084;
  localparam logic [31:0] ACCEL_META_TARGET_ADDR    = 32'h3000_0088;
  localparam logic [31:0] ACCEL_META_CTRL_ADDR      = 32'h3000_008C;
  localparam logic [31:0] ACCEL_META_STATUS_ADDR    = 32'h3000_0090;

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

  function automatic bit is_source(input int node);
    begin
      is_source = (node == 1) || (node == 4) || (node == 5);
    end
  endfunction

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
        1: begin dut.u_tile_array.mesh_flit_in[1] = flit; dut.u_tile_array.mesh_valid_in[1] = 1'b1; end
        4: begin dut.u_tile_array.mesh_flit_in[4] = flit; dut.u_tile_array.mesh_valid_in[4] = 1'b1; end
        5: begin dut.u_tile_array.mesh_flit_in[5] = flit; dut.u_tile_array.mesh_valid_in[5] = 1'b1; end
        default: begin end
      endcase
    end
  endtask

  task automatic clear_source(input int src);
    begin
      case (src)
        1: begin dut.u_tile_array.mesh_flit_in[1] = '0; dut.u_tile_array.mesh_valid_in[1] = 1'b0; end
        4: begin dut.u_tile_array.mesh_flit_in[4] = '0; dut.u_tile_array.mesh_valid_in[4] = 1'b0; end
        5: begin dut.u_tile_array.mesh_flit_in[5] = '0; dut.u_tile_array.mesh_valid_in[5] = 1'b0; end
        default: begin end
      endcase
    end
  endtask

  task automatic release_accel_array_bus();
    begin
      release dut.u_tile_array.s_axi_awaddr;
      release dut.u_tile_array.s_axi_awvalid;
      release dut.u_tile_array.s_axi_wdata;
      release dut.u_tile_array.s_axi_wstrb;
      release dut.u_tile_array.s_axi_wvalid;
      release dut.u_tile_array.s_axi_bready;
      release dut.u_tile_array.s_axi_araddr;
      release dut.u_tile_array.s_axi_arvalid;
      release dut.u_tile_array.s_axi_rready;
    end
  endtask

  task automatic accel_array_write(
    input logic [31:0] addr,
    input logic [31:0] data
  );
    int wait_cycles;
    bit saw_busy;
    begin
      wait_cycles = 0;
      saw_busy    = 1'b0;

      force dut.u_tile_array.s_axi_awaddr  = addr;
      force dut.u_tile_array.s_axi_awvalid = 1'b1;
      force dut.u_tile_array.s_axi_wdata   = data;
      force dut.u_tile_array.s_axi_wstrb   = 4'hF;
      force dut.u_tile_array.s_axi_wvalid  = 1'b1;
      force dut.u_tile_array.s_axi_bready  = 1'b1;

      repeat (3) begin
        @(posedge clk);
        #1;
      end

      release dut.u_tile_array.s_axi_awaddr;
      release dut.u_tile_array.s_axi_awvalid;
      release dut.u_tile_array.s_axi_wdata;
      release dut.u_tile_array.s_axi_wstrb;
      release dut.u_tile_array.s_axi_wvalid;

      while (wait_cycles < 2000) begin
        @(posedge clk);
        #1;
        if ((dut.u_tile_array.axi_s_state != 0) || dut.u_tile_array.axi_s_write_done)
          saw_busy = 1'b1;
        if (saw_busy && (dut.u_tile_array.axi_s_state == 0) && !dut.u_tile_array.axi_s_write_done)
          break;
        wait_cycles++;
      end
      if (!saw_busy || (wait_cycles >= 2000))
        $fatal(1,
               "metadata CSR write completion timeout addr=0x%08x data=0x%08x state=%0d write_done=%0b csr_addr=0x%02x array=%0b",
               addr,
               data,
               dut.u_tile_array.axi_s_state,
               dut.u_tile_array.axi_s_write_done,
               dut.u_tile_array.csr_addr_r,
               dut.u_tile_array.csr_is_array_reg);

      @(posedge clk);
      release dut.u_tile_array.s_axi_bready;
    end
  endtask

  task automatic accel_array_read(
    input logic [31:0] addr,
    output logic [31:0] data
  );
    int wait_cycles;
    bit saw_busy;
    begin
      wait_cycles = 0;
      saw_busy    = 1'b0;

      force dut.u_tile_array.s_axi_araddr  = addr;
      force dut.u_tile_array.s_axi_arvalid = 1'b1;
      force dut.u_tile_array.s_axi_rready  = 1'b1;

      repeat (3) begin
        @(posedge clk);
        #1;
      end

      release dut.u_tile_array.s_axi_araddr;
      release dut.u_tile_array.s_axi_arvalid;

      while (wait_cycles < 2000) begin
        @(posedge clk);
        #1;
        if (dut.u_tile_array.axi_s_state != 0)
          saw_busy = 1'b1;
        if (saw_busy && (dut.u_tile_array.axi_s_state == 0))
          break;
        wait_cycles++;
      end
      if (!saw_busy || (wait_cycles >= 2000))
        $fatal(1,
               "metadata CSR read completion timeout addr=0x%08x state=%0d csr_addr=0x%02x array=%0b",
               addr,
               dut.u_tile_array.axi_s_state,
               dut.u_tile_array.csr_addr_r,
               dut.u_tile_array.csr_is_array_reg);

      data = dut.u_tile_array.s_axi_rdata;
      @(posedge clk);
      release dut.u_tile_array.s_axi_rready;
    end
  endtask

  task automatic program_target(
    input int router,
    input int group_id,
    input int target,
    input bit enable
  );
    begin
      accel_array_write(ACCEL_META_ROUTER_ADDR,    32'(router));
      accel_array_write(ACCEL_META_REDUCE_ID_ADDR, 32'(group_id));
      accel_array_write(ACCEL_META_TARGET_ADDR,    32'(target));
      accel_array_write(ACCEL_META_CTRL_ADDR,      {30'h0, 1'b1, enable});
    end
  endtask

  task automatic program_sparse_tree(input int group_id);
    begin
      program_target(1, group_id, 1, 1'b1);
      program_target(4, group_id, 2, 1'b1);
      program_target(5, group_id, 1, 1'b1);
    end
  endtask

  task automatic inject_group(input int group_id);
    begin
      program_sparse_tree(group_id);

      drive_source(1, make_reduce_flit(4'd1, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                       4'(EXPECTED_CONTRIB), 32'd1, 4'd1));
      drive_source(4, make_reduce_flit(4'd4, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                       4'(EXPECTED_CONTRIB), 32'd4, 4'd1));
      drive_source(5, make_reduce_flit(4'd5, 4'(ROOT_NODE), 2'(SOURCE_VC), 8'(group_id),
                                       4'(EXPECTED_CONTRIB), 32'd5, 4'd1));

      @(posedge clk);

      clear_source(1);
      clear_source(4);
      clear_source(5);
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
    logic [31:0] meta_status;

    $display("========================================================================");
    $display("  tb_soc_top_inr_metadata — Full SoC Metadata-On INR Regression");
    $display("========================================================================");

    release_accel_array_bus();
    clear_source(1);
    clear_source(4);
    clear_source(5);

    repeat (20) @(posedge clk);
    rst_n = 1'b1;
    repeat (10) @(posedge clk);

    check("accel_done low before injection", !accel_done);

    program_sparse_tree(0);
    accel_array_read(ACCEL_META_STATUS_ADDR, meta_status);
    check("metadata CSR readback valid",
          (meta_status[3:0] == 4'd5) &&
          (meta_status[11:4] == 8'd0) &&
          (meta_status[19:16] == 4'd1) &&
          meta_status[24],
          $sformatf("status=0x%08x", meta_status));

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

    $display("[METRIC] root=%0d groups=%0d root_packets=%0d link_flits=%0d expected_sum=%0d subset={1,4,5}",
             ROOT_NODE,
             dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_groups_completed,
             dut.u_tile_array.gen_tile[ROOT_NODE].u_tile.reduce_packets_consumed,
             link_flit_count,
             EXPECTED_SUM);

    $display("========================================================================");
    $display("  tb_soc_top_inr_metadata RESULTS: %0d passed, %0d failed", tests_passed, tests_failed);
    $display("  Total simulation cycles: %0d", cycle_count);
    $display("========================================================================");

    if (tests_failed == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  FAILURES: %0d test(s) failed", tests_failed);

    $finish;
  end

endmodule
