// =============================================================================
// soc_top_v2.sv — Full SoC Integration v2
// =============================================================================
// Evolution of soc_top.sv with:
//   1. L1 D-Cache between Ibex and crossbar (CPU data path)
//   2. accel_tile_array as the integrated multi-tile compute fabric
//   3. TLB + Page Table Walker in the CPU data path
//   4. L2 Cache between crossbar and DRAM controller
//   5. Performance monitor (perf counters) connected to a CSR slave
//   6. Reserved peripheral window at 0x2004_0000 for future Ethernet
//
// Master 0: CPU (OBI→AXI bridge, through L1 D-cache + TLB)
// Master 1: accel_tile_array DMA gateway (tiles → DRAM)
//
// Slave 0: Boot ROM          (0x0000_0000)
// Slave 1: SRAM              (0x1000_0000)
// Slave 2: Peripherals       (0x2000_0000) — UART, Timer, GPIO, PLIC
// Slave 3: accel_tile_array  (0x3000_0000) — CSR access to tiles
// Slave 4: DRAM (via L2)     (0x4000_0000)
// Slave 5: Perf counters     (0x5000_0000)
// Slaves 6-7: DECERR

`timescale 1ns/1ps

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */
import soc_pkg::*;

module soc_top_v2 #(
  parameter string BOOT_ROM_FILE = "firmware.hex",
  parameter int unsigned CLK_FREQ = 50_000_000,
  parameter int unsigned UART_BAUD = 115_200,
  // Tile array config
  parameter int MESH_ROWS = 4,   // 4x4 mesh (16 tiles)
  parameter int MESH_COLS = 4,
  parameter bit SPARSE_VC_ALLOC = 1'b0,
  parameter bit INNET_REDUCE = 1'b0
) (
  input  logic              clk,
  input  logic              rst_n,

  // UART I/O
  input  logic              uart_rx,
  output logic              uart_tx,

  // GPIO I/O
  output logic [7:0]        gpio_o,
  input  logic [7:0]        gpio_i,
  output logic [7:0]        gpio_oe,

  // Interrupt outputs
  output logic              irq_external,
  output logic              irq_timer,

  // Accelerator status
  output logic              accel_busy,
  output logic              accel_done,

  // DRAM PHY interface
  output logic [7:0]        dram_phy_act,
  output logic [7:0]        dram_phy_read,
  output logic [7:0]        dram_phy_write,
  output logic [7:0]        dram_phy_pre,
  output logic [13:0]       dram_phy_row,
  output logic [9:0]        dram_phy_col,
  output logic              dram_phy_ref,
  output logic [31:0]       dram_phy_wdata,
  output logic [3:0]        dram_phy_wstrb,
  input  logic [31:0]       dram_phy_rdata,
  input  logic              dram_phy_rdata_valid,
  output logic              dram_ctrl_busy
);

  // =========================================================================
  // Parameters
  // =========================================================================
  localparam int NUM_MASTERS = 3;
  localparam int NUM_SLAVES  = 8;
  localparam int ID_WIDTH    = 4;
  localparam int NUM_TILES   = MESH_ROWS * MESH_COLS;

  logic                    inr_meta_cfg_valid [NUM_TILES];
  logic [7:0]              inr_meta_cfg_reduce_id [NUM_TILES];
  logic [3:0]              inr_meta_cfg_target [NUM_TILES];
  logic                    inr_meta_cfg_enable [NUM_TILES];

  // =========================================================================
  // Clock & Reset
  // =========================================================================
  logic [1:0] rst_sync_ff;
  logic clk_core;
  logic rst_core_n;
  assign clk_core   = clk;

  // Keep asynchronous assertion from the pad, but release reset on a clock
  // edge so the first ASIC baseline does not rely on uncontrolled deassertion.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      rst_sync_ff <= 2'b00;
    else
      rst_sync_ff <= {rst_sync_ff[0], 1'b1};
  end

  assign rst_core_n = rst_sync_ff[1];

  // =========================================================================
  // ===== SECTION 1: CPU + L1 D-Cache + TLB/PTW ============================
  // =========================================================================

  // OBI interface from CPU
  logic              obi_req, obi_gnt;
  logic [31:0]       obi_addr;
  logic              obi_we;
  logic [3:0]        obi_be;
  logic [31:0]       obi_wdata;
  logic              obi_rvalid;
  logic [31:0]       obi_rdata;
  logic              obi_err;

  assign obi_err = 1'b0;

  simple_cpu #(
    .ADDR_WIDTH (32),
    .DATA_WIDTH (32),
    .ID_WIDTH   (ID_WIDTH)
  ) u_cpu (
    .clk          (clk_core),
    .rst_n        (rst_core_n),
    .cpu_reset    (~rst_core_n),
    .irq_external (irq_external),
    .irq_timer    (irq_timer_int),
    .req          (obi_req),
    .gnt          (obi_gnt),
    .addr         (obi_addr),
    .we           (obi_we),
    .be           (obi_be),
    .wdata        (obi_wdata),
    .rvalid       (obi_rvalid),
    .rdata        (obi_rdata),
    .err          (obi_err)
  );

  // -----------------------------------------------------------------------
  // TLB — sits between CPU and L1 cache
  // -----------------------------------------------------------------------
  logic              tlb_lookup_valid;
  logic              tlb_lookup_ready;
  logic [31:0]       tlb_va;
  logic              tlb_hit;
  logic [21:0]       tlb_ppn_out;
  logic [33:0]       tlb_pa_out;
  assign             tlb_ppn_out = tlb_pa_out[33:12];
  logic              tlb_miss;

  // TLB fill from PTW
  logic              ptw_fill_valid;
  logic [31:0]       ptw_fill_va;
  logic [8:0]        ptw_fill_asid;
  logic [21:0]       ptw_fill_ppn;
  logic              ptw_fill_superpage;
  logic              ptw_fill_dirty, ptw_fill_accessed;
  logic              ptw_fill_global, ptw_fill_user;
  logic              ptw_fill_exec, ptw_fill_write, ptw_fill_read;

  // SATP CSR (simplified — hardwired for now)
  logic [21:0]       satp_ppn;
  logic              satp_mode;
  assign satp_ppn  = 22'h0;    // Page table root (firmware sets via CSR write)
  assign satp_mode = 1'b0;     // Bare mode for first tapeout (no virtual memory)

  tlb u_tlb (
    .clk             (clk_core),
    .rst_n           (rst_core_n),
    .lookup_valid    (obi_req),
    .lookup_va       (obi_addr),
    .lookup_asid     (9'b0),
    .lookup_hit      (tlb_hit),
    .lookup_pa       (tlb_pa_out),
    .lookup_fault    (),
    .lookup_is_store (1'b0),
    .lookup_is_exec  (1'b0),
    .fill_valid      (ptw_fill_valid),
    .fill_va         (ptw_fill_va),
    .fill_asid       (ptw_fill_asid),
    .fill_ppn        (ptw_fill_ppn),
    .fill_superpage  (ptw_fill_superpage),
    .fill_dirty      (ptw_fill_dirty),
    .fill_accessed   (ptw_fill_accessed),
    .fill_global     (ptw_fill_global),
    .fill_user       (ptw_fill_user),
    .fill_exec       (ptw_fill_exec),
    .fill_write      (ptw_fill_write),
    .fill_read       (ptw_fill_read),
    .sfence_valid    (1'b0),
    .sfence_va       (32'b0),
    .sfence_asid     (9'b0),
    .sfence_all      (1'b0)
  );

  // -----------------------------------------------------------------------
  // Page Table Walker
  // -----------------------------------------------------------------------
  logic              ptw_mem_req_valid;
  logic              ptw_mem_req_ready;
  logic [33:0]       ptw_mem_req_addr;
  logic              ptw_mem_resp_valid;
  logic [31:0]       ptw_mem_resp_data;

  page_table_walker u_ptw (
    .clk              (clk_core),
    .rst_n            (rst_core_n),
    .walk_req_valid   (obi_req && !tlb_hit && satp_mode),
    .walk_req_ready   (),
    .walk_va          (obi_addr),
    .walk_asid        (9'b0),
    .walk_is_store    (obi_we),
    .walk_is_exec     (1'b0),
    .walk_done        (ptw_fill_valid),
    .walk_fault       (),
    .walk_fault_cause (),
    .walk_result_va   (ptw_fill_va),
    .walk_result_asid (ptw_fill_asid),
    .walk_result_ppn  (ptw_fill_ppn),
    .walk_result_superpage (ptw_fill_superpage),
    .walk_result_dirty     (ptw_fill_dirty),
    .walk_result_accessed  (ptw_fill_accessed),
    .walk_result_global    (ptw_fill_global),
    .walk_result_user      (ptw_fill_user),
    .walk_result_exec      (ptw_fill_exec),
    .walk_result_write     (ptw_fill_write),
    .walk_result_read      (ptw_fill_read),
    .mem_req_valid    (ptw_mem_req_valid),
    .mem_req_ready    (ptw_mem_req_ready),
    .mem_req_addr     (ptw_mem_req_addr),
    .mem_resp_valid   (ptw_mem_resp_valid),
    .mem_resp_data    (ptw_mem_resp_data),
    .satp_ppn         (satp_ppn),
    .satp_mode        (satp_mode)
  );

  // PTW memory port → goes through crossbar (not implemented in bare mode)
  assign ptw_mem_req_ready  = 1'b1;
  assign ptw_mem_resp_valid = 1'b0;
  assign ptw_mem_resp_data  = 32'b0;

  // -----------------------------------------------------------------------
  // Translated address (pass-through in bare mode)
  // -----------------------------------------------------------------------
  logic [31:0] cpu_phys_addr;
  assign cpu_phys_addr = satp_mode ? {tlb_ppn_out[19:0], obi_addr[11:0]} : obi_addr;

  // -----------------------------------------------------------------------
  // L1 D-Cache
  // -----------------------------------------------------------------------
  // AXI master signals from L1 cache
  logic              l1_m_axi_awvalid, l1_m_axi_awready;
  logic [31:0]       l1_m_axi_awaddr;
  logic [ID_WIDTH-1:0] l1_m_axi_awid;
  logic [7:0]        l1_m_axi_awlen;
  logic [2:0]        l1_m_axi_awsize;
  logic [1:0]        l1_m_axi_awburst;
  logic              l1_m_axi_wvalid, l1_m_axi_wready;
  logic [31:0]       l1_m_axi_wdata;
  logic [3:0]        l1_m_axi_wstrb;
  logic              l1_m_axi_wlast;
  logic              l1_m_axi_bvalid;
  logic [1:0]        l1_m_axi_bresp;
  logic [ID_WIDTH-1:0] l1_m_axi_bid;
  logic              l1_m_axi_bready;
  logic              l1_m_axi_arvalid, l1_m_axi_arready;
  logic [31:0]       l1_m_axi_araddr;
  logic [ID_WIDTH-1:0] l1_m_axi_arid;
  logic [7:0]        l1_m_axi_arlen;
  logic [2:0]        l1_m_axi_arsize;
  logic [1:0]        l1_m_axi_arburst;
  logic              l1_m_axi_rvalid;
  logic [31:0]       l1_m_axi_rdata;
  logic [1:0]        l1_m_axi_rresp;
  logic [ID_WIDTH-1:0] l1_m_axi_rid;
  logic              l1_m_axi_rlast;
  logic              l1_m_axi_rready;

  // -----------------------------------------------------------------------
  // Cache bypass for IO regions — MMIO peripherals and accelerator CSRs
  // must not be cached. CPU preload/readback also gets uncached aliases:
  //   Cacheable:   0x0xxx (ROM), 0x1xxx (SRAM), 0x4xxx (DRAM)
  //   Uncacheable: 0x2xxx (Periph), 0x3xxx (Accel), 0x5xxx (Perf)
  //   Aliases:     0x6xxx -> 0x4xxx (DRAM uncached), 0x7xxx -> 0x1xxx (SRAM uncached)
  // -----------------------------------------------------------------------
  logic is_io;
  logic [31:0] io_axi_addr;

  function automatic logic [31:0] translate_uncached_alias(input logic [31:0] addr);
    begin
      case (addr[31:28])
        4'h6: translate_uncached_alias = {4'h4, addr[27:0]};
        4'h7: translate_uncached_alias = {4'h1, addr[27:0]};
        default: translate_uncached_alias = addr;
      endcase
    end
  endfunction

  // MMIO regions are uncacheable; ROM/SRAM/DRAM go through L1 D-cache unless
  // the firmware deliberately uses the uncached alias windows above.
  assign is_io = (cpu_phys_addr[31:28] == 4'h2) ||
                 (cpu_phys_addr[31:28] == 4'h3) ||
                 (cpu_phys_addr[31:28] == 4'h5) ||
                 (cpu_phys_addr[31:28] == 4'h6) ||
                 (cpu_phys_addr[31:28] == 4'h7);

  assign io_axi_addr = translate_uncached_alias(io_addr_r);

  // Split OBI signals
  logic dcache_req, dcache_gnt, dcache_rvalid;
  logic [31:0] dcache_rdata;
  logic io_req, io_gnt, io_rvalid;
  logic [31:0] io_rdata;

  always_comb begin
    for (int tile = 0; tile < NUM_TILES; tile++) begin
      inr_meta_cfg_valid[tile]     = 1'b0;
      inr_meta_cfg_reduce_id[tile] = '0;
      inr_meta_cfg_target[tile]    = '0;
      inr_meta_cfg_enable[tile]    = 1'b0;
    end
  end

  assign dcache_req = obi_req && !is_io;
  assign io_req     = obi_req &&  is_io;
  assign obi_gnt    = is_io ? io_gnt    : dcache_gnt;
  assign obi_rvalid = dcache_rvalid | io_rvalid;
  assign obi_rdata  = io_rvalid ? io_rdata : dcache_rdata;

  l1_dcache_top #(
    .ADDR_WIDTH (32),
    .DATA_WIDTH (32),
    .ID_WIDTH   (ID_WIDTH),
    .NUM_SETS   (16),
    .NUM_WAYS   (4),
    .LINE_BYTES (64)
  ) u_l1_dcache (
    .clk            (clk_core),
    .rst_n          (rst_core_n),
    // CPU side (OBI-like) — cacheable requests only
    .cpu_req        (dcache_req),
    .cpu_gnt        (dcache_gnt),
    .cpu_addr       (cpu_phys_addr),
    .cpu_we         (obi_we),
    .cpu_be         (obi_be),
    .cpu_wdata      (obi_wdata),
    .cpu_rvalid     (dcache_rvalid),
    .cpu_rdata      (dcache_rdata),
    // AXI4 master (to crossbar)
    .m_axi_awvalid  (l1_m_axi_awvalid),
    .m_axi_awready  (l1_m_axi_awready),
    .m_axi_awaddr   (l1_m_axi_awaddr),
    .m_axi_awid     (l1_m_axi_awid),
    .m_axi_awlen    (l1_m_axi_awlen),
    .m_axi_awsize   (l1_m_axi_awsize),
    .m_axi_awburst  (l1_m_axi_awburst),
    .m_axi_wvalid   (l1_m_axi_wvalid),
    .m_axi_wready   (l1_m_axi_wready),
    .m_axi_wdata    (l1_m_axi_wdata),
    .m_axi_wstrb    (l1_m_axi_wstrb),
    .m_axi_wlast    (l1_m_axi_wlast),
    .m_axi_bvalid   (l1_m_axi_bvalid),
    .m_axi_bready   (l1_m_axi_bready),
    .m_axi_bresp    (l1_m_axi_bresp),
    .m_axi_bid      (l1_m_axi_bid),
    .m_axi_arvalid  (l1_m_axi_arvalid),
    .m_axi_arready  (l1_m_axi_arready),
    .m_axi_araddr   (l1_m_axi_araddr),
    .m_axi_arid     (l1_m_axi_arid),
    .m_axi_arlen    (l1_m_axi_arlen),
    .m_axi_arsize   (l1_m_axi_arsize),
    .m_axi_arburst  (l1_m_axi_arburst),
    .m_axi_rvalid   (l1_m_axi_rvalid),
    .m_axi_rready   (l1_m_axi_rready),
    .m_axi_rdata    (l1_m_axi_rdata),
    .m_axi_rresp    (l1_m_axi_rresp),
    .m_axi_rid      (l1_m_axi_rid),
    .m_axi_rlast    (l1_m_axi_rlast),
    .cache_busy     ()
  );

  // =========================================================================
  // Master 0: L1 cache AXI → crossbar
  // =========================================================================
  logic m0_awvalid, m0_awready;
  logic [31:0] m0_awaddr;
  logic [ID_WIDTH-1:0] m0_awid;
  logic m0_wvalid, m0_wready;
  logic [31:0] m0_wdata;
  logic [3:0]  m0_wstrb;
  logic m0_wlast;
  logic m0_bvalid;
  logic [1:0] m0_bresp;
  logic [ID_WIDTH-1:0] m0_bid;
  logic m0_bready;
  logic m0_arvalid, m0_arready;
  logic [31:0] m0_araddr;
  logic [ID_WIDTH-1:0] m0_arid;
  logic m0_rvalid;
  logic [31:0] m0_rdata;
  logic [1:0] m0_rresp;
  logic [ID_WIDTH-1:0] m0_rid;
  logic m0_rlast;
  logic m0_rready;

  assign m0_awvalid = l1_m_axi_awvalid;
  assign l1_m_axi_awready = m0_awready;
  assign m0_awaddr  = l1_m_axi_awaddr;
  assign m0_awid    = l1_m_axi_awid;
  assign m0_wvalid  = l1_m_axi_wvalid;
  assign l1_m_axi_wready = m0_wready;
  assign m0_wdata   = l1_m_axi_wdata;
  assign m0_wstrb   = l1_m_axi_wstrb;
  assign m0_wlast   = l1_m_axi_wlast;
  assign l1_m_axi_bvalid = m0_bvalid;
  assign l1_m_axi_bresp  = m0_bresp;
  assign l1_m_axi_bid    = m0_bid;
  assign m0_bready  = l1_m_axi_bready;
  assign m0_arvalid = l1_m_axi_arvalid;
  assign l1_m_axi_arready = m0_arready;
  assign m0_araddr  = l1_m_axi_araddr;
  assign m0_arid    = l1_m_axi_arid;
  assign l1_m_axi_rvalid = m0_rvalid;
  assign l1_m_axi_rdata  = m0_rdata;
  assign l1_m_axi_rresp  = m0_rresp;
  assign l1_m_axi_rid    = m0_rid;
  assign l1_m_axi_rlast  = m0_rlast;
  assign m0_rready  = l1_m_axi_rready;

  // =========================================================================
  // Master 2: Uncacheable I/O bridge (OBI → single-beat AXI)
  // =========================================================================
  logic m2_awvalid, m2_awready;
  logic [31:0] m2_awaddr;
  logic [ID_WIDTH-1:0] m2_awid;
  logic m2_wvalid, m2_wready;
  logic [31:0] m2_wdata;
  logic [3:0]  m2_wstrb;
  logic m2_wlast;
  logic m2_bvalid;
  logic [1:0] m2_bresp;
  logic [ID_WIDTH-1:0] m2_bid;
  logic m2_bready;
  logic m2_arvalid, m2_arready;
  logic [31:0] m2_araddr;
  logic [ID_WIDTH-1:0] m2_arid;
  logic m2_rvalid;
  logic [31:0] m2_rdata;
  logic [1:0] m2_rresp;
  logic [ID_WIDTH-1:0] m2_rid;
  logic m2_rlast;
  logic m2_rready;

  // Simple OBI → AXI bridge FSM for uncacheable I/O
  typedef enum logic [2:0] {
    IO_IDLE,
    IO_WR,          // write: issue AW+W, track acceptance independently
    IO_B_WAIT,      // write: wait for B
    IO_AR,          // read:  issue AR
    IO_R_WAIT       // read:  wait for R
  } io_state_e;

  io_state_e io_state;
  logic [31:0] io_addr_r, io_wdata_r;
  logic [3:0]  io_be_r;
  logic        io_we_r;
  logic        io_aw_done, io_w_done;

  always_ff @(posedge clk_core or negedge rst_core_n) begin
    if (!rst_core_n) begin
      io_state   <= IO_IDLE;
      io_addr_r  <= '0;
      io_wdata_r <= '0;
      io_be_r    <= '0;
      io_we_r    <= 1'b0;
      io_aw_done <= 1'b0;
      io_w_done  <= 1'b0;
    end else begin
      case (io_state)
        IO_IDLE: begin
          io_aw_done <= 1'b0;
          io_w_done  <= 1'b0;
          if (io_req) begin
            io_addr_r  <= cpu_phys_addr;
            io_wdata_r <= obi_wdata;
            io_be_r    <= obi_be;
            io_we_r    <= obi_we;
            io_state   <= obi_we ? IO_WR : IO_AR;
          end
        end
        IO_WR: begin
          if (m2_awready) io_aw_done <= 1'b1;
          if (m2_wready)  io_w_done  <= 1'b1;
          if ((m2_awready || io_aw_done) && (m2_wready || io_w_done))
            io_state <= IO_B_WAIT;
        end
        IO_B_WAIT: if (m2_bvalid)  io_state <= IO_IDLE;
        IO_AR:     if (m2_arready) io_state <= IO_R_WAIT;
        IO_R_WAIT: if (m2_rvalid)  io_state <= IO_IDLE;
        default:                   io_state <= IO_IDLE;
      endcase
    end
  end

  assign io_gnt    = (io_state == IO_IDLE) && io_req;
  assign io_rvalid = (io_state == IO_B_WAIT && m2_bvalid) ||
                     (io_state == IO_R_WAIT && m2_rvalid);
  assign io_rdata  = m2_rdata;

  `ifdef SIMULATION
  always @(posedge clk_core) begin
    if ($test$plusargs("IO_TRACE")) begin
      if (io_req && io_state == IO_IDLE)
        $display("[IO] %s addr=%h data=%h t=%0t", obi_we ? "WR" : "RD", cpu_phys_addr, obi_wdata, $time);
      if (io_state == IO_AR && m2_arready)
        $display("[IO] AR accepted t=%0t", $time);
      if (io_state == IO_R_WAIT && m2_rvalid)
        $display("[IO] RD done rdata=%h t=%0t", m2_rdata, $time);
      if (io_state == IO_B_WAIT && m2_bvalid)
        $display("[IO] WR done t=%0t", $time);
      if (io_state == IO_WR)
        $display("[IO-WR] awv=%b awr=%b aw_done=%b wv=%b wr=%b w_done=%b t=%0t",
                 m2_awvalid, m2_awready, io_aw_done, m2_wvalid, m2_wready, io_w_done, $time);
      if (io_state == IO_B_WAIT && !m2_bvalid && $time > 2100000 && $time < 2300000)
        $display("[IO-BW] bvalid=%b t=%0t", m2_bvalid, $time);
    end
  end
  `endif

  // AXI write
  assign m2_awvalid = (io_state == IO_WR) && !io_aw_done;
  assign m2_awaddr  = io_axi_addr;
  assign m2_awid    = '0;
  assign m2_wvalid  = (io_state == IO_WR) && !io_w_done;
  assign m2_wdata   = io_wdata_r;
  assign m2_wstrb   = io_be_r;
  assign m2_wlast   = 1'b1;
  assign m2_bready  = 1'b1;

  // AXI read
  assign m2_arvalid = (io_state == IO_AR);
  assign m2_araddr  = io_axi_addr;
  assign m2_arid    = '0;
  assign m2_rready  = 1'b1;

  // =========================================================================
  // ===== SECTION 2: Multi-Tile Accelerator (Master 1 + Slave 3) ============
  // =========================================================================
  logic m1_awvalid, m1_awready;
  logic [31:0] m1_awaddr;
  logic [ID_WIDTH-1:0] m1_awid;
  logic m1_wvalid, m1_wready;
  logic [31:0] m1_wdata;
  logic [3:0]  m1_wstrb;
  logic m1_wlast;
  logic m1_bvalid;
  logic [1:0] m1_bresp;
  logic [ID_WIDTH-1:0] m1_bid;
  logic m1_bready;
  logic m1_arvalid, m1_arready;
  logic [31:0] m1_araddr;
  logic [ID_WIDTH-1:0] m1_arid;
  logic [7:0] m1_arlen;
  logic [7:0] m1_awlen;
  logic m1_rvalid;
  logic [31:0] m1_rdata;
  logic [1:0] m1_rresp;
  logic [ID_WIDTH-1:0] m1_rid;
  logic m1_rlast;
  logic m1_rready;

  // Crossbar slave port signals
  logic [NUM_SLAVES-1:0]               s_awvalid, s_awready;
  logic [NUM_SLAVES-1:0][31:0]         s_awaddr;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_awid;
  logic [NUM_SLAVES-1:0]               s_wvalid, s_wready;
  logic [NUM_SLAVES-1:0][31:0]         s_wdata;
  logic [NUM_SLAVES-1:0][3:0]          s_wstrb;
  logic [NUM_SLAVES-1:0]               s_wlast;
  logic [NUM_SLAVES-1:0]               s_bvalid;
  logic [NUM_SLAVES-1:0][1:0]          s_bresp;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_bid;
  logic [NUM_SLAVES-1:0]               s_bready;
  logic [NUM_SLAVES-1:0]               s_arvalid, s_arready;
  logic [NUM_SLAVES-1:0][31:0]         s_araddr;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_arid;
  logic [NUM_SLAVES-1:0][7:0]          s_arlen;
  logic [NUM_SLAVES-1:0][7:0]          s_awlen;
  logic [NUM_SLAVES-1:0]               s_rvalid;
  logic [NUM_SLAVES-1:0][31:0]         s_rdata;
  logic [NUM_SLAVES-1:0][1:0]          s_rresp;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_rid;
  logic [NUM_SLAVES-1:0]               s_rlast;
  logic [NUM_SLAVES-1:0]               s_rready;

  // Tile array busy aggregation
  logic [MESH_ROWS*MESH_COLS-1:0] tile_busy_vec;
  logic [MESH_ROWS*MESH_COLS-1:0] tile_done_vec;

  accel_tile_array #(
    .MESH_ROWS       (MESH_ROWS),
    .MESH_COLS       (MESH_COLS),
    .N_ROWS          (16),
    .N_COLS          (16),
    .DATA_W          (8),
    .ACC_W           (32),
    .SP_DEPTH        (4096),
    .SPARSE_VC_ALLOC (SPARSE_VC_ALLOC),
    .INNET_REDUCE    (INNET_REDUCE),
    .AXI_ADDR_W      (32),
    .AXI_DATA_W      (32),
    .AXI_ID_W        (ID_WIDTH)
  ) u_tile_array (
    .clk   (clk_core),
    .rst_n (rst_core_n),
    .inr_meta_cfg_valid     (inr_meta_cfg_valid),
    .inr_meta_cfg_reduce_id (inr_meta_cfg_reduce_id),
    .inr_meta_cfg_target    (inr_meta_cfg_target),
    .inr_meta_cfg_enable    (inr_meta_cfg_enable),

    // AXI Slave (CSR access from crossbar slave 3)
    .s_axi_awaddr  (s_awaddr[3]),
    .s_axi_awvalid (s_awvalid[3]),
    .s_axi_awready (s_awready[3]),
    .s_axi_wdata   (s_wdata[3]),
    .s_axi_wstrb   (s_wstrb[3]),
    .s_axi_wvalid  (s_wvalid[3]),
    .s_axi_wready  (s_wready[3]),
    .s_axi_bresp   (s_bresp[3]),
    .s_axi_bvalid  (s_bvalid[3]),
    .s_axi_bready  (s_bready[3]),
    .s_axi_araddr  (s_araddr[3]),
    .s_axi_arvalid (s_arvalid[3]),
    .s_axi_arready (s_arready[3]),
    .s_axi_rdata   (s_rdata[3]),
    .s_axi_rresp   (s_rresp[3]),
    .s_axi_rvalid  (s_rvalid[3]),
    .s_axi_rready  (s_rready[3]),

    // AXI Master (DMA gateway → crossbar master 1)
    .m_axi_arid    (m1_arid),
    .m_axi_araddr  (m1_araddr),
    .m_axi_arlen   (m1_arlen),
    .m_axi_arsize  (),
    .m_axi_arburst (),
    .m_axi_arvalid (m1_arvalid),
    .m_axi_arready (m1_arready),
    .m_axi_rid     (m1_rid),
    .m_axi_rdata   (m1_rdata),
    .m_axi_rresp   (m1_rresp),
    .m_axi_rlast   (m1_rlast),
    .m_axi_rvalid  (m1_rvalid),
    .m_axi_rready  (m1_rready),
    .m_axi_awid    (m1_awid),
    .m_axi_awaddr  (m1_awaddr),
    .m_axi_awlen   (m1_awlen),
    .m_axi_awsize  (),
    .m_axi_awburst (),
    .m_axi_awvalid (m1_awvalid),
    .m_axi_awready (m1_awready),
    .m_axi_wdata   (m1_wdata),
    .m_axi_wstrb   (m1_wstrb),
    .m_axi_wlast   (m1_wlast),
    .m_axi_wvalid  (m1_wvalid),
    .m_axi_wready  (m1_wready),
    .m_axi_bid     (m1_bid),
    .m_axi_bresp   (m1_bresp),
    .m_axi_bvalid  (m1_bvalid),
    .m_axi_bready  (m1_bready),
    .tile_busy_o   (tile_busy_vec),
    .tile_done_o   (tile_done_vec)
  );

  // Slave 3: fill in ID/LAST for crossbar compatibility
  assign s_bid[3]   = '0;
  assign s_rid[3]   = '0;
  assign s_rlast[3] = s_rvalid[3];

  assign accel_busy = |tile_busy_vec;

  // Sticky done register per tile: sets on done pulse, clears when tile
  // starts a new command (busy rises).  accel_done = AND of all sticky bits.
  logic [MESH_ROWS*MESH_COLS-1:0] tile_done_sticky;

  always_ff @(posedge clk_core or negedge rst_core_n) begin
    if (!rst_core_n) begin
      tile_done_sticky <= '0;
    end else begin
      for (int i = 0; i < MESH_ROWS*MESH_COLS; i++) begin
        if (tile_done_vec[i])
          tile_done_sticky[i] <= 1'b1;
        else if (tile_busy_vec[i])
          tile_done_sticky[i] <= 1'b0;
      end
    end
  end

  assign accel_done = &tile_done_sticky;

  // =========================================================================
  // ===== SECTION 3: AXI Crossbar ==========================================
  // =========================================================================

  axi_crossbar #(
    .ADDR_WIDTH  (32),
    .DATA_WIDTH  (32),
    .ID_WIDTH    (ID_WIDTH),
    .NUM_MASTERS (NUM_MASTERS),
    .NUM_SLAVES  (NUM_SLAVES)
  ) u_crossbar (
    .clk   (clk_core),
    .rst_n (rst_core_n),

    .m_awvalid ({m2_awvalid, m1_awvalid, m0_awvalid}),
    .m_awready ({m2_awready, m1_awready, m0_awready}),
    .m_awaddr  ({m2_awaddr,  m1_awaddr,  m0_awaddr}),
    .m_awid    ({m2_awid,    m1_awid,    m0_awid}),
    .m_awlen   ({8'd0,       m1_awlen,   l1_m_axi_awlen}),

    .m_wvalid  ({m2_wvalid,  m1_wvalid,  m0_wvalid}),
    .m_wready  ({m2_wready,  m1_wready,  m0_wready}),
    .m_wdata   ({m2_wdata,   m1_wdata,   m0_wdata}),
    .m_wstrb   ({m2_wstrb,   m1_wstrb,   m0_wstrb}),
    .m_wlast   ({m2_wlast,   m1_wlast,   m0_wlast}),

    .m_bvalid  ({m2_bvalid,  m1_bvalid,  m0_bvalid}),
    .m_bready  ({m2_bready,  m1_bready,  m0_bready}),
    .m_bresp   ({m2_bresp,   m1_bresp,   m0_bresp}),
    .m_bid     ({m2_bid,     m1_bid,     m0_bid}),

    .m_arvalid ({m2_arvalid, m1_arvalid, m0_arvalid}),
    .m_arready ({m2_arready, m1_arready, m0_arready}),
    .m_araddr  ({m2_araddr,  m1_araddr,  m0_araddr}),
    .m_arid    ({m2_arid,    m1_arid,    m0_arid}),
    .m_arlen   ({8'd0,       m1_arlen,   l1_m_axi_arlen}),

    .m_rvalid  ({m2_rvalid,  m1_rvalid,  m0_rvalid}),
    .m_rready  ({m2_rready,  m1_rready,  m0_rready}),
    .m_rdata   ({m2_rdata,   m1_rdata,   m0_rdata}),
    .m_rresp   ({m2_rresp,   m1_rresp,   m0_rresp}),
    .m_rid     ({m2_rid,     m1_rid,     m0_rid}),
    .m_rlast   ({m2_rlast,   m1_rlast,   m0_rlast}),

    .s_awvalid (s_awvalid),
    .s_awready (s_awready),
    .s_awaddr  (s_awaddr),
    .s_awid    (s_awid),
    .s_awlen   (s_awlen),

    .s_wvalid  (s_wvalid),
    .s_wready  (s_wready),
    .s_wdata   (s_wdata),
    .s_wstrb   (s_wstrb),
    .s_wlast   (s_wlast),

    .s_bvalid  (s_bvalid),
    .s_bready  (s_bready),
    .s_bresp   (s_bresp),
    .s_bid     (s_bid),

    .s_arvalid (s_arvalid),
    .s_arready (s_arready),
    .s_araddr  (s_araddr),
    .s_arid    (s_arid),
    .s_arlen   (s_arlen),

    .s_rvalid  (s_rvalid),
    .s_rready  (s_rready),
    .s_rdata   (s_rdata),
    .s_rresp   (s_rresp),
    .s_rid     (s_rid),
    .s_rlast   (s_rlast)
  );

  // =========================================================================
  // ===== SECTION 4: Slave 0 — Boot ROM ====================================
  // =========================================================================

  boot_rom #(
    .ADDR_WIDTH (13),
    .DATA_WIDTH (32),
    .INIT_FILE  (BOOT_ROM_FILE)
  ) u_boot_rom (
    .clk     (clk_core),
    .rst_n   (rst_core_n),
    .awvalid (s_awvalid[0]),  .awready (s_awready[0]),
    .awaddr  (s_awaddr[0]),   .awsize  (3'b010),  .awburst (2'b00),  .awid (s_awid[0]),
    .wvalid  (s_wvalid[0]),   .wready  (s_wready[0]),
    .wdata   (s_wdata[0]),    .wstrb   (s_wstrb[0]),  .wlast (s_wlast[0]),
    .bvalid  (s_bvalid[0]),   .bready  (s_bready[0]),
    .bresp   (s_bresp[0]),    .bid     (s_bid[0]),
    .arvalid (s_arvalid[0]),  .arready (s_arready[0]),
    .araddr  (s_araddr[0]),   .arsize  (3'b010),  .arburst (2'b01),  .arid (s_arid[0]),
    .arlen   (s_arlen[0]),
    .rvalid  (s_rvalid[0]),   .rready  (s_rready[0]),
    .rdata   (s_rdata[0]),    .rresp   (s_rresp[0]),
    .rid     (s_rid[0]),      .rlast   (s_rlast[0])
  );

  // =========================================================================
  // ===== SECTION 5: Slave 1 — SRAM ========================================
  // =========================================================================

  sram_ctrl #(
    .ADDR_WIDTH (15),
    .DATA_WIDTH (32)
  ) u_sram (
    .clk     (clk_core),
    .rst_n   (rst_core_n),
    .awvalid (s_awvalid[1]),  .awready (s_awready[1]),
    .awaddr  (s_awaddr[1]),   .awlen   (s_awlen[1]),
    .awsize  (3'b010),        .awburst (2'b01),  .awid (s_awid[1]),
    .wvalid  (s_wvalid[1]),   .wready  (s_wready[1]),
    .wdata   (s_wdata[1]),    .wstrb   (s_wstrb[1]),  .wlast (s_wlast[1]),
    .bvalid  (s_bvalid[1]),   .bready  (s_bready[1]),
    .bresp   (s_bresp[1]),    .bid     (s_bid[1]),
    .arvalid (s_arvalid[1]),  .arready (s_arready[1]),
    .araddr  (s_araddr[1]),   .arsize  (3'b010),  .arburst (2'b01),  .arid (s_arid[1]),
    .arlen   (s_arlen[1]),
    .rvalid  (s_rvalid[1]),   .rready  (s_rready[1]),
    .rdata   (s_rdata[1]),    .rresp   (s_rresp[1]),
    .rid     (s_rid[1]),      .rlast   (s_rlast[1])
  );

  // =========================================================================
  // ===== SECTION 6: Slave 2 — Peripherals =================================
  // =========================================================================
  // Sub-address decode: s_awaddr[2][23:16]
  //   0x00 = UART, 0x01 = Timer, 0x02 = GPIO, 0x03 = PLIC, 0x04 = ETH

  logic [7:0] periph_aw_sel, periph_ar_sel;
  logic [7:0] periph_aw_sel_r, periph_ar_sel_r;
  logic [31:0] periph_awaddr_r, periph_araddr_r;
  logic [ID_WIDTH-1:0] periph_awid_r, periph_arid_r;
  logic periph_aw_active, periph_ar_active;
  logic [7:0] periph_aw_sel_cur, periph_ar_sel_cur;
  logic [31:0] periph_awaddr_cur, periph_araddr_cur;
  logic [ID_WIDTH-1:0] periph_awid_cur, periph_arid_cur;

  assign periph_aw_sel = s_awaddr[2][23:16];
  assign periph_ar_sel = s_araddr[2][23:16];
  assign periph_aw_sel_cur = periph_aw_active ? periph_aw_sel_r : periph_aw_sel;
  assign periph_ar_sel_cur = periph_ar_active ? periph_ar_sel_r : periph_ar_sel;
  assign periph_awaddr_cur = periph_aw_active ? periph_awaddr_r : s_awaddr[2];
  assign periph_araddr_cur = periph_ar_active ? periph_araddr_r : s_araddr[2];
  assign periph_awid_cur = periph_aw_active ? periph_awid_r : s_awid[2];
  assign periph_arid_cur = periph_ar_active ? periph_arid_r : s_arid[2];

  always_ff @(posedge clk_core or negedge rst_core_n) begin
    if (!rst_core_n) begin
      periph_aw_sel_r  <= '0;
      periph_ar_sel_r  <= '0;
      periph_awaddr_r  <= '0;
      periph_araddr_r  <= '0;
      periph_awid_r    <= '0;
      periph_arid_r    <= '0;
      periph_aw_active <= 1'b0;
      periph_ar_active <= 1'b0;
    end else begin
      if (s_awvalid[2] && s_awready[2]) begin
        periph_aw_sel_r  <= s_awaddr[2][23:16];
        periph_awaddr_r  <= s_awaddr[2];
        periph_awid_r    <= s_awid[2];
        periph_aw_active <= 1'b1;
      end else if (s_bvalid[2] && s_bready[2]) begin
        periph_aw_active <= 1'b0;
      end

      if (s_arvalid[2] && s_arready[2]) begin
        periph_ar_sel_r  <= s_araddr[2][23:16];
        periph_araddr_r  <= s_araddr[2];
        periph_arid_r    <= s_arid[2];
        periph_ar_active <= 1'b1;
      end else if (s_rvalid[2] && s_rready[2] && s_rlast[2]) begin
        periph_ar_active <= 1'b0;
      end
    end
  end

  // Interrupt signals
  logic irq_uart, irq_timer_int, irq_plic_ext;
  logic accel_irq;
  assign accel_irq = |tile_done_vec;  // Interrupt when any tile completes

  // Peripheral response locals
  logic uart_awready, uart_wready, uart_bvalid, uart_arready, uart_rvalid;
  logic [1:0]  uart_bresp, uart_rresp;
  logic [ID_WIDTH-1:0] uart_bid, uart_rid;
  logic [31:0] uart_rdata;
  logic uart_rlast;

  logic timer_awready, timer_wready, timer_bvalid, timer_arready, timer_rvalid;
  logic [1:0]  timer_bresp, timer_rresp;
  logic [ID_WIDTH-1:0] timer_bid, timer_rid;
  logic [31:0] timer_rdata;
  logic timer_rlast;

  logic gpio_awready, gpio_wready, gpio_bvalid, gpio_arready, gpio_rvalid;
  logic [1:0]  gpio_bresp, gpio_rresp;
  logic [ID_WIDTH-1:0] gpio_bid, gpio_rid;
  logic [31:0] gpio_rdata;
  logic gpio_rlast;

  logic plic_awready, plic_wready, plic_bvalid, plic_arready, plic_rvalid;
  logic [1:0]  plic_bresp, plic_rresp;
  logic [ID_WIDTH-1:0] plic_bid, plic_rid;
  logic [31:0] plic_rdata;
  logic plic_rlast;

  // UART
  uart_ctrl #(
    .ADDR_WIDTH   (32),
    .DATA_WIDTH   (32),
    .CLK_FREQ     (CLK_FREQ),
    .DEFAULT_BAUD (UART_BAUD)
  ) u_uart (
    .clk(clk_core), .rst_n(rst_core_n),
    .rx(uart_rx), .tx(uart_tx),
    .awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h00)), .awready(uart_awready),
    .awaddr(periph_awaddr_cur), .awsize(3'b010), .awburst(2'b00), .awid(periph_awid_cur),
    .wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h00)), .wready(uart_wready),
    .wdata(s_wdata[2]), .wstrb(s_wstrb[2]), .wlast(s_wlast[2]),
    .bvalid(uart_bvalid), .bready(s_bready[2]), .bresp(uart_bresp), .bid(uart_bid),
    .arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h00)), .arready(uart_arready),
    .araddr(periph_araddr_cur), .arsize(3'b010), .arburst(2'b00), .arid(periph_arid_cur),
    .rvalid(uart_rvalid), .rready(s_rready[2]), .rdata(uart_rdata),
    .rresp(uart_rresp), .rid(uart_rid), .rlast(uart_rlast),
    .irq_o(irq_uart)
  );

  // Timer
  timer_ctrl #(
    .ADDR_WIDTH (32),
    .DATA_WIDTH (32)
  ) u_timer (
    .clk(clk_core), .rst_n(rst_core_n),
    .awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h01)), .awready(timer_awready),
    .awaddr(periph_awaddr_cur), .awsize(3'b010), .awburst(2'b00), .awid(periph_awid_cur),
    .wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h01)), .wready(timer_wready),
    .wdata(s_wdata[2]), .wstrb(s_wstrb[2]), .wlast(s_wlast[2]),
    .bvalid(timer_bvalid), .bready(s_bready[2]), .bresp(timer_bresp), .bid(timer_bid),
    .arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h01)), .arready(timer_arready),
    .araddr(periph_araddr_cur), .arsize(3'b010), .arburst(2'b00), .arid(periph_arid_cur),
    .rvalid(timer_rvalid), .rready(s_rready[2]), .rdata(timer_rdata),
    .rresp(timer_rresp), .rid(timer_rid), .rlast(timer_rlast),
    .irq_timer_o(irq_timer_int)
  );

  // GPIO
  gpio_ctrl #(
    .ADDR_WIDTH (32),
    .DATA_WIDTH (32),
    .GPIO_WIDTH (8)
  ) u_gpio (
    .clk(clk_core), .rst_n(rst_core_n),
    .gpio_o(gpio_o), .gpio_i(gpio_i), .gpio_oe(gpio_oe),
    .awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h02)), .awready(gpio_awready),
    .awaddr(periph_awaddr_cur), .awsize(3'b010), .awburst(2'b00), .awid(periph_awid_cur),
    .wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h02)), .wready(gpio_wready),
    .wdata(s_wdata[2]), .wstrb(s_wstrb[2]), .wlast(s_wlast[2]),
    .bvalid(gpio_bvalid), .bready(s_bready[2]), .bresp(gpio_bresp), .bid(gpio_bid),
    .arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h02)), .arready(gpio_arready),
    .araddr(periph_araddr_cur), .arsize(3'b010), .arburst(2'b00), .arid(periph_arid_cur),
    .rvalid(gpio_rvalid), .rready(s_rready[2]), .rdata(gpio_rdata),
    .rresp(gpio_rresp), .rid(gpio_rid), .rlast(gpio_rlast)
  );

  // PLIC
  plic #(
    .ADDR_WIDTH  (32),
    .DATA_WIDTH  (32),
    .NUM_SOURCES (32),
    .NUM_TARGETS (1)
  ) u_plic (
    .clk(clk_core), .rst_n(rst_core_n),
    .irq_i({25'b0, accel_irq, 1'b0, irq_timer_int, irq_uart, 3'b0}),
    .irq_o(irq_plic_ext),
    .awvalid(s_awvalid[2] && (periph_aw_sel_cur == 8'h03)), .awready(plic_awready),
    .awaddr(periph_awaddr_cur), .awsize(3'b010), .awburst(2'b00), .awid(periph_awid_cur),
    .wvalid(s_wvalid[2] && (periph_aw_sel_cur == 8'h03)), .wready(plic_wready),
    .wdata(s_wdata[2]), .wstrb(s_wstrb[2]), .wlast(s_wlast[2]),
    .bvalid(plic_bvalid), .bready(s_bready[2]), .bresp(plic_bresp), .bid(plic_bid),
    .arvalid(s_arvalid[2] && (periph_ar_sel_cur == 8'h03)), .arready(plic_arready),
    .araddr(periph_araddr_cur), .arsize(3'b010), .arburst(2'b00), .arid(periph_arid_cur),
    .rvalid(plic_rvalid), .rready(s_rready[2]), .rdata(plic_rdata),
    .rresp(plic_rresp), .rid(plic_rid), .rlast(plic_rlast)
  );

  // Peripheral response mux for slave 2
  always_comb begin
    if (timer_rvalid) begin
      s_rvalid[2] = 1'b1;  s_rdata[2] = timer_rdata;
      s_rresp[2] = timer_rresp; s_rid[2] = timer_rid; s_rlast[2] = timer_rlast;
    end else if (gpio_rvalid) begin
      s_rvalid[2] = 1'b1;  s_rdata[2] = gpio_rdata;
      s_rresp[2] = gpio_rresp; s_rid[2] = gpio_rid; s_rlast[2] = gpio_rlast;
    end else if (plic_rvalid) begin
      s_rvalid[2] = 1'b1;  s_rdata[2] = plic_rdata;
      s_rresp[2] = plic_rresp; s_rid[2] = plic_rid; s_rlast[2] = plic_rlast;
    end else if (uart_rvalid) begin
      s_rvalid[2] = 1'b1;  s_rdata[2] = uart_rdata;
      s_rresp[2] = uart_rresp; s_rid[2] = uart_rid; s_rlast[2] = uart_rlast;
    end else begin
      s_rvalid[2] = 1'b0; s_rdata[2] = '0;
      s_rresp[2] = 2'b00; s_rid[2] = '0; s_rlast[2] = 1'b0;
    end
  end

  always_comb begin
    if (timer_bvalid)      begin s_bvalid[2] = 1'b1; s_bresp[2] = timer_bresp; s_bid[2] = timer_bid; end
    else if (gpio_bvalid)  begin s_bvalid[2] = 1'b1; s_bresp[2] = gpio_bresp;  s_bid[2] = gpio_bid;  end
    else if (plic_bvalid)  begin s_bvalid[2] = 1'b1; s_bresp[2] = plic_bresp;  s_bid[2] = plic_bid;  end
    else if (uart_bvalid)  begin s_bvalid[2] = 1'b1; s_bresp[2] = uart_bresp;  s_bid[2] = uart_bid;  end
    else                   begin s_bvalid[2] = 1'b0; s_bresp[2] = 2'b00;       s_bid[2] = '0;        end
  end

  assign s_awready[2] = (periph_aw_sel_cur == 8'h00) ? uart_awready  :
                         (periph_aw_sel_cur == 8'h01) ? timer_awready :
                         (periph_aw_sel_cur == 8'h02) ? gpio_awready  :
                         (periph_aw_sel_cur == 8'h03) ? plic_awready  : 1'b1;
  assign s_wready[2]  = (periph_aw_sel_cur == 8'h00) ? uart_wready   :
                         (periph_aw_sel_cur == 8'h01) ? timer_wready  :
                         (periph_aw_sel_cur == 8'h02) ? gpio_wready   :
                         (periph_aw_sel_cur == 8'h03) ? plic_wready   : 1'b1;
  assign s_arready[2] = (periph_ar_sel_cur == 8'h00) ? uart_arready  :
                         (periph_ar_sel_cur == 8'h01) ? timer_arready :
                         (periph_ar_sel_cur == 8'h02) ? gpio_arready  :
                         (periph_ar_sel_cur == 8'h03) ? plic_arready  : 1'b1;

  // =========================================================================
  // ===== SECTION 7: Slave 4 — DRAM (L2 bypassed) ===========================
  // =========================================================================
  // The L2 cache has miss-readback and burst-fill bugs.  Bypass it entirely
  // by connecting crossbar slave port 4 directly to the DRAM controller.
  // The gateway already issues single-beat AXI reads (arlen=0), so the
  // DRAM controller's single-beat rlast behaviour is correct.

  // DRAM controller AXI signals — direct from crossbar slave[4]
  logic              l2_m_axi_awvalid, l2_m_axi_awready;
  logic [31:0]       l2_m_axi_awaddr;
  logic [ID_WIDTH-1:0] l2_m_axi_awid;
  logic [7:0]        l2_m_axi_awlen;
  logic [2:0]        l2_m_axi_awsize;
  logic [1:0]        l2_m_axi_awburst;
  logic              l2_m_axi_wvalid, l2_m_axi_wready;
  logic [31:0]       l2_m_axi_wdata;
  logic [3:0]        l2_m_axi_wstrb;
  logic              l2_m_axi_wlast;
  logic              l2_m_axi_bvalid;
  logic [1:0]        l2_m_axi_bresp;
  logic [ID_WIDTH-1:0] l2_m_axi_bid;
  logic              l2_m_axi_bready;
  logic              l2_m_axi_arvalid, l2_m_axi_arready;
  logic [31:0]       l2_m_axi_araddr;
  logic [ID_WIDTH-1:0] l2_m_axi_arid;
  logic [7:0]        l2_m_axi_arlen;
  logic [2:0]        l2_m_axi_arsize;
  logic [1:0]        l2_m_axi_arburst;
  logic              l2_m_axi_rvalid;
  logic [31:0]       l2_m_axi_rdata;
  logic [1:0]        l2_m_axi_rresp;
  logic [ID_WIDTH-1:0] l2_m_axi_rid;
  logic              l2_m_axi_rlast;
  logic              l2_m_axi_rready;

  // --- L2 bypass: wire crossbar slave[4] straight through to DRAM ctrl ---
  assign l2_m_axi_arvalid = s_arvalid[4];
  assign s_arready[4]     = l2_m_axi_arready;
  assign l2_m_axi_araddr  = s_araddr[4];
  assign l2_m_axi_arid    = s_arid[4];
  assign l2_m_axi_arlen   = s_arlen[4];
  assign l2_m_axi_arsize  = 3'b010;
  assign l2_m_axi_arburst = 2'b01;

  assign s_rvalid[4]      = l2_m_axi_rvalid;
  assign l2_m_axi_rready  = s_rready[4];
  assign s_rdata[4]       = l2_m_axi_rdata;
  assign s_rresp[4]       = l2_m_axi_rresp;
  assign s_rid[4]         = l2_m_axi_rid;
  assign s_rlast[4]       = l2_m_axi_rlast;

  assign l2_m_axi_awvalid = s_awvalid[4];
  assign s_awready[4]     = l2_m_axi_awready;
  assign l2_m_axi_awaddr  = s_awaddr[4];
  assign l2_m_axi_awid    = s_awid[4];
  assign l2_m_axi_awlen   = s_awlen[4];
  assign l2_m_axi_awsize  = 3'b010;
  assign l2_m_axi_awburst = 2'b01;

  assign l2_m_axi_wvalid  = s_wvalid[4];
  assign s_wready[4]      = l2_m_axi_wready;
  assign l2_m_axi_wdata   = s_wdata[4];
  assign l2_m_axi_wstrb   = s_wstrb[4];
  assign l2_m_axi_wlast   = s_wlast[4];

  assign s_bvalid[4]      = l2_m_axi_bvalid;
  assign l2_m_axi_bready  = s_bready[4];
  assign s_bresp[4]       = l2_m_axi_bresp;
  assign s_bid[4]         = l2_m_axi_bid;

  // DRAM controller
  dram_ctrl_top #(
    .AXI_ADDR_W  (32),
    .AXI_DATA_W  (32),
    .AXI_ID_W    (ID_WIDTH),
    .NUM_BANKS   (8),
    .ROW_BITS    (14),
    .COL_BITS    (10),
    .BANK_BITS   (3),
    .QUEUE_DEPTH (16),
    .ADDR_MODE   (0)
  ) u_dram_ctrl (
    .clk   (clk_core),
    .rst_n (rst_core_n),

    // AXI4 Slave ← L2 cache master port
    .s_axi_awvalid (l2_m_axi_awvalid), .s_axi_awready (l2_m_axi_awready),
    .s_axi_awaddr  (l2_m_axi_awaddr),  .s_axi_awid    (l2_m_axi_awid),
    .s_axi_awlen   (l2_m_axi_awlen),   .s_axi_awsize  (l2_m_axi_awsize),
    .s_axi_wvalid  (l2_m_axi_wvalid),  .s_axi_wready  (l2_m_axi_wready),
    .s_axi_wdata   (l2_m_axi_wdata),   .s_axi_wstrb   (l2_m_axi_wstrb),
    .s_axi_wlast   (l2_m_axi_wlast),
    .s_axi_bvalid  (l2_m_axi_bvalid),  .s_axi_bready  (l2_m_axi_bready),
    .s_axi_bresp   (l2_m_axi_bresp),   .s_axi_bid     (l2_m_axi_bid),
    .s_axi_arvalid (l2_m_axi_arvalid), .s_axi_arready (l2_m_axi_arready),
    .s_axi_araddr  (l2_m_axi_araddr),  .s_axi_arid    (l2_m_axi_arid),
    .s_axi_arlen   (l2_m_axi_arlen),   .s_axi_arsize  (l2_m_axi_arsize),
    .s_axi_rvalid  (l2_m_axi_rvalid),  .s_axi_rready  (l2_m_axi_rready),
    .s_axi_rdata   (l2_m_axi_rdata),   .s_axi_rresp   (l2_m_axi_rresp),
    .s_axi_rid     (l2_m_axi_rid),     .s_axi_rlast   (l2_m_axi_rlast),

    // PHY
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
    .ctrl_busy            (dram_ctrl_busy)
  );

  // =========================================================================
  // ===== SECTION 8: Slave 5 — Performance Counters ========================
  // =========================================================================
  // Simple AXI-Lite readable perf counter block

  perf_axi #(
    .NUM_COUNTERS (6)
  ) u_perf (
    .clk   (clk_core),
    .rst_n (rst_core_n),

    // Events
    .event_valid ({
      l2_m_axi_arvalid && l2_m_axi_arready,  // DRAM reads
      l2_m_axi_awvalid && l2_m_axi_awready,  // DRAM writes
      s_arvalid[4] && s_arready[4],           // L2 read accesses
      s_awvalid[4] && s_awready[4],           // L2 write accesses
      obi_req && obi_gnt,                     // CPU requests
      1'b1                                     // Cycle counter
    }),

    // AXI-Lite slave (mapped to slave 5)
    .s_axi_awvalid (s_awvalid[5]),
    .s_axi_awready (s_awready[5]),
    .s_axi_awaddr  (s_awaddr[5][7:0]),
    .s_axi_wvalid  (s_wvalid[5]),
    .s_axi_wready  (s_wready[5]),
    .s_axi_wdata   (s_wdata[5]),
    .s_axi_wstrb   (s_wstrb[5]),
    .s_axi_bvalid  (s_bvalid[5]),
    .s_axi_bready  (s_bready[5]),
    .s_axi_bresp   (s_bresp[5]),
    .s_axi_arvalid (s_arvalid[5]),
    .s_axi_arready (s_arready[5]),
    .s_axi_araddr  (s_araddr[5][7:0]),
    .s_axi_rvalid  (s_rvalid[5]),
    .s_axi_rready  (s_rready[5]),
    .s_axi_rdata   (s_rdata[5]),
    .s_axi_rresp   (s_rresp[5])
  );

  // Fill in ID/LAST for slave 5 crossbar compatibility
  assign s_bid[5]   = '0;
  assign s_rid[5]   = '0;
  assign s_rlast[5] = s_rvalid[5];

  // =========================================================================
  // ===== SECTION 9: Slaves 6-7 — DECERR Responders ========================
  // =========================================================================
  generate
    for (genvar si = 6; si < NUM_SLAVES; si++) begin : gen_dummy_slaves
      assign s_awready[si] = 1'b1;
      assign s_wready[si]  = 1'b1;
      assign s_arready[si] = 1'b1;
    end
  endgenerate

  always_comb begin
    for (int i = 6; i < NUM_SLAVES; i++) begin
      s_bvalid[i] = 1'b0;
      s_bresp[i]  = 2'b11;   // DECERR
      s_bid[i]    = '0;
      s_rvalid[i] = 1'b0;
      s_rdata[i]  = 32'hDEAD_DEAD;
      s_rresp[i]  = 2'b11;   // DECERR
      s_rid[i]    = '0;
      s_rlast[i]  = 1'b0;
    end
  end

  // =========================================================================
  // ===== Interrupt Outputs =================================================
  // =========================================================================
  assign irq_external = irq_plic_ext;
  assign irq_timer    = irq_timer_int;

endmodule : soc_top_v2
