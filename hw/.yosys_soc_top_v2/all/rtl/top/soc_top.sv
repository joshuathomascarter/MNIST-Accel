// SOC Top Level - Full System Integration
// Ibex CPU + AXI Crossbar + Peripherals + Accelerator

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off PINCONNECTEMPTY */
import soc_pkg::*;

module soc_top #(
  parameter BOOT_ROM_FILE = "firmware.hex",
  parameter int unsigned CLK_FREQ = 50_000_000,
  parameter int unsigned UART_BAUD = 115_200
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
  
  // Interrupt outputs (for external monitoring)
  output logic              irq_external,
  output logic              irq_timer,

  // Accelerator status
  output logic              accel_busy,
  output logic              accel_done,
  output logic              accel_error,

  // DRAM PHY interface (directly out to physical DRAM)
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

  // ===== LOCAL PARAMETERS =====
  
  localparam int NUM_MASTERS = 2;  // CPU I-fetch, CPU D-load
  localparam int NUM_SLAVES = 8;
  localparam int ID_WIDTH = 4;

  // ===== CLOCK & RESET =====
  
  // For now, single clock domain. Real implementation would have CDC.
  logic clk_core;
  logic rst_core_n;
  
  assign clk_core = clk;
  assign rst_core_n = rst_n;

  // ===== INTERNAL SIGNALS =====
  
  // OBI interface from CPU
  logic              obi_gnt;
  logic              obi_req;
  logic [31:0]       obi_addr;
  logic              obi_we;
  logic [3:0]        obi_be;
  logic [31:0]       obi_wdata;
  logic [1:0]        obi_burst;  // Unused in our simple CPU
  logic              obi_rvalid;
  logic [31:0]       obi_rdata;
  logic              obi_err;

  // AXI signals from OBI→AXI bridge (Master 0)
  logic              m0_awvalid, m0_awready;
  logic [31:0]       m0_awaddr;
  logic [ID_WIDTH-1:0] m0_awid;
  logic [7:0]        m0_awlen;
  
  logic              m0_wvalid, m0_wready;
  logic [31:0]       m0_wdata;
  logic [3:0]        m0_wstrb;
  logic              m0_wlast;
  
  logic              m0_bvalid;
  logic [1:0]        m0_bresp;
  logic [ID_WIDTH-1:0] m0_bid;
  logic              m0_bready;
  
  logic              m0_arvalid, m0_arready;
  logic [31:0]       m0_araddr;
  logic [ID_WIDTH-1:0] m0_arid;
  logic [7:0]        m0_arlen;
  
  logic              m0_rvalid;
  logic [31:0]       m0_rdata;
  logic [1:0]        m0_rresp;
  logic [ID_WIDTH-1:0] m0_rid;
  logic              m0_rlast;
  logic              m0_rready;

  // Master 1 (unused for now - would be D-cache or separate data port)
  logic              m1_awvalid, m1_awready;
  logic [31:0]       m1_awaddr;
  logic [ID_WIDTH-1:0] m1_awid;
  logic [7:0]        m1_awlen;
  logic              m1_wvalid, m1_wready;
  logic [31:0]       m1_wdata;
  logic [3:0]        m1_wstrb;
  logic              m1_wlast;
  logic              m1_bvalid;
  logic [1:0]        m1_bresp;
  logic [ID_WIDTH-1:0] m1_bid;
  logic              m1_bready;
  logic              m1_arvalid, m1_arready;
  logic [31:0]       m1_araddr;
  logic [ID_WIDTH-1:0] m1_arid;
  logic [7:0]        m1_arlen;
  logic              m1_rvalid;
  logic [31:0]       m1_rdata;
  logic [1:0]        m1_rresp;
  logic [ID_WIDTH-1:0] m1_rid;
  logic              m1_rlast;
  logic              m1_rready;

  // AXI signals to slaves via crossbar
  logic [NUM_SLAVES-1:0] s_awvalid, s_awready;
  logic [NUM_SLAVES-1:0][31:0] s_awaddr;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_awid;
  logic [NUM_SLAVES-1:0][7:0] s_awlen;
  
  logic [NUM_SLAVES-1:0] s_wvalid, s_wready;
  logic [NUM_SLAVES-1:0][31:0] s_wdata;
  logic [NUM_SLAVES-1:0][3:0] s_wstrb;
  logic [NUM_SLAVES-1:0] s_wlast;
  
  logic [NUM_SLAVES-1:0] s_bvalid;
  logic [NUM_SLAVES-1:0][1:0] s_bresp;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_bid;
  logic [NUM_SLAVES-1:0] s_bready;
  
  logic [NUM_SLAVES-1:0] s_arvalid, s_arready;
  logic [NUM_SLAVES-1:0][31:0] s_araddr;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_arid;
  logic [NUM_SLAVES-1:0][7:0] s_arlen;
  
  logic [NUM_SLAVES-1:0] s_rvalid;
  logic [NUM_SLAVES-1:0][31:0] s_rdata;
  logic [NUM_SLAVES-1:0][1:0] s_rresp;
  logic [NUM_SLAVES-1:0][ID_WIDTH-1:0] s_rid;
  logic [NUM_SLAVES-1:0] s_rlast;
  logic [NUM_SLAVES-1:0] s_rready;

  // Interrupt signals
  logic irq_uart;
  logic irq_timer_int;
  logic irq_plic_ext;
  logic [31:0] irq_sources;

  // PLIC local signals
  logic plic_awready, plic_wready, plic_bvalid, plic_arready, plic_rvalid;
  logic [1:0]  plic_bresp, plic_rresp;
  logic [ID_WIDTH-1:0] plic_bid, plic_rid;
  logic [31:0] plic_rdata;
  logic plic_rlast;

  // Timer local signals
  logic timer_awready, timer_wready, timer_bvalid, timer_arready, timer_rvalid;
  logic [1:0]  timer_bresp, timer_rresp;
  logic [ID_WIDTH-1:0] timer_bid, timer_rid;
  logic [31:0] timer_rdata;
  logic timer_rlast;

  // GPIO local signals
  logic gpio_awready, gpio_wready, gpio_bvalid, gpio_arready, gpio_rvalid;
  logic [1:0]  gpio_bresp, gpio_rresp;
  logic [ID_WIDTH-1:0] gpio_bid, gpio_rid;
  logic [31:0] gpio_rdata;
  logic gpio_rlast;

  // UART local signals (intermediate wires to avoid multi-driver on s_*[2])
  logic uart_awready, uart_wready, uart_bvalid, uart_arready, uart_rvalid;
  logic [1:0]  uart_bresp, uart_rresp;
  logic [ID_WIDTH-1:0] uart_bid, uart_rid;
  logic [31:0] uart_rdata;
  logic uart_rlast;

  // Accelerator signals
  logic accel_irq;
  logic accel_busy_w, accel_done_w, accel_error_w;

  // ===== CPU INSTANTIATION =====
  
  simple_cpu #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .ID_WIDTH(ID_WIDTH)
  ) u_cpu (
    .clk(clk_core),
    .rst_n(rst_core_n),
    .cpu_reset(~rst_core_n),
    .irq_external(irq_external),
    .irq_timer(irq_timer_int),
    .req(obi_req),
    .gnt(obi_gnt),
    .addr(obi_addr),
    .we(obi_we),
    .be(obi_be),
    .wdata(obi_wdata),
    .rvalid(obi_rvalid),
    .rdata(obi_rdata),
    .err(obi_err)
  );

  // ===== OBI TO AXI BRIDGE =====
  
  obi_to_axi #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .ID_WIDTH(ID_WIDTH)
  ) u_obi2axi (
    .clk(clk_core),
    .rst_n(rst_core_n),
    .obi_gnt(obi_gnt),
    .obi_req(obi_req),
    .obi_addr(obi_addr),
    .obi_we(obi_we),
    .obi_be(obi_be),
    .obi_wdata(obi_wdata),
    .obi_burst(obi_burst),
    .obi_rvalid(obi_rvalid),
    .obi_rdata(obi_rdata),
    .obi_err(obi_err),
    .axi_awvalid(m0_awvalid),
    .axi_awready(m0_awready),
    .axi_awaddr(m0_awaddr),
    .axi_awid(m0_awid),
    .axi_wvalid(m0_wvalid),
    .axi_wready(m0_wready),
    .axi_wdata(m0_wdata),
    .axi_wstrb(m0_wstrb),
    .axi_wlast(m0_wlast),
    .axi_bvalid(m0_bvalid),
    .axi_bready(m0_bready),
    .axi_bresp(m0_bresp),
    .axi_bid(m0_bid),
    .axi_arvalid(m0_arvalid),
    .axi_arready(m0_arready),
    .axi_araddr(m0_araddr),
    .axi_arid(m0_arid),
    .axi_rvalid(m0_rvalid),
    .axi_rready(m0_rready),
    .axi_rdata(m0_rdata),
    .axi_rresp(m0_rresp),
    .axi_rid(m0_rid)
  );

  assign m0_awlen = 8'd0;
  assign m0_arlen = 8'd0;

  // ===== ACCELERATOR (Master 1 DMA + Slave 3 CSR) =====

  accel_top #(
    .N_ROWS      (16),
    .N_COLS      (16),
    .DATA_W      (8),
    .ACC_W       (32),
    .AXI_ADDR_W  (32),
    .AXI_DATA_W  (32),       // Match crossbar data width (32-bit SoC bus)
    .AXI_ID_W    (ID_WIDTH),
    .CSR_ADDR_W  (8),
    .BRAM_ADDR_W (10)
  ) u_accel (
    .clk   (clk_core),
    .rst_n (rst_core_n),

    // --- AXI4 Master: Read DMA → crossbar master 1 ---
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

    // --- AXI4 Master: Write DMA → crossbar master 1 ---
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

    // --- AXI-Lite Slave: CSR ← crossbar slave 3 ---
    .s_axi_awaddr  (s_awaddr[3]),
    .s_axi_awprot  (3'b000),
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
    .s_axi_arprot  (3'b000),
    .s_axi_arvalid (s_arvalid[3]),
    .s_axi_arready (s_arready[3]),

    .s_axi_rdata   (s_rdata[3]),
    .s_axi_rresp   (s_rresp[3]),
    .s_axi_rvalid  (s_rvalid[3]),
    .s_axi_rready  (s_rready[3]),

    // Status
    .busy  (accel_busy_w),
    .done  (accel_done_w),
    .error (accel_error_w),
    .irq   (accel_irq)
  );

  // Slave 3 doesn't have AXI4 ID/LAST — fill in for crossbar
  assign s_bid[3]   = '0;
  assign s_rid[3]   = '0;
  assign s_rlast[3] = s_rvalid[3];  // single-beat: rlast = rvalid

  // Export accel status to top-level ports
  assign accel_busy  = accel_busy_w;
  assign accel_done  = accel_done_w;
  assign accel_error = accel_error_w;

  // ===== AXI CROSSBAR =====
  
  axi_crossbar #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .ID_WIDTH(ID_WIDTH),
    .NUM_MASTERS(NUM_MASTERS),
    .NUM_SLAVES(NUM_SLAVES)
  ) u_crossbar (
    .clk(clk_core),
    .rst_n(rst_core_n),
    
    .m_awvalid({m1_awvalid, m0_awvalid}),
    .m_awready({m1_awready, m0_awready}),
    .m_awaddr({m1_awaddr, m0_awaddr}),
    .m_awid({m1_awid, m0_awid}),
    .m_awlen({m1_awlen, m0_awlen}),
    
    .m_wvalid({m1_wvalid, m0_wvalid}),
    .m_wready({m1_wready, m0_wready}),
    .m_wdata({m1_wdata, m0_wdata}),
    .m_wstrb({m1_wstrb, m0_wstrb}),
    .m_wlast({m1_wlast, m0_wlast}),
    
    .m_bvalid({m1_bvalid, m0_bvalid}),
    .m_bready({m1_bready, m0_bready}),
    .m_bresp({m1_bresp, m0_bresp}),
    .m_bid({m1_bid, m0_bid}),
    
    .m_arvalid({m1_arvalid, m0_arvalid}),
    .m_arready({m1_arready, m0_arready}),
    .m_araddr({m1_araddr, m0_araddr}),
    .m_arid({m1_arid, m0_arid}),
    .m_arlen({m1_arlen, m0_arlen}),
    
    .m_rvalid({m1_rvalid, m0_rvalid}),
    .m_rready({m1_rready, m0_rready}),
    .m_rdata({m1_rdata, m0_rdata}),
    .m_rresp({m1_rresp, m0_rresp}),
    .m_rid({m1_rid, m0_rid}),
    .m_rlast({m1_rlast, m0_rlast}),
    
    .s_awvalid(s_awvalid),
    .s_awready(s_awready),
    .s_awaddr(s_awaddr),
    .s_awid(s_awid),
    .s_awlen(s_awlen),
    
    .s_wvalid(s_wvalid),
    .s_wready(s_wready),
    .s_wdata(s_wdata),
    .s_wstrb(s_wstrb),
    .s_wlast(s_wlast),
    
    .s_bvalid(s_bvalid),
    .s_bready(s_bready),
    .s_bresp(s_bresp),
    .s_bid(s_bid),
    
    .s_arvalid(s_arvalid),
    .s_arready(s_arready),
    .s_araddr(s_araddr),
    .s_arid(s_arid),
    .s_arlen(s_arlen),
    
    .s_rvalid(s_rvalid),
    .s_rready(s_rready),
    .s_rdata(s_rdata),
    .s_rresp(s_rresp),
    .s_rid(s_rid),
    .s_rlast(s_rlast)
  );

  // ===== SLAVE 0: BOOT ROM =====
  
  boot_rom #(
    .ADDR_WIDTH(12),
    .DATA_WIDTH(32),
    .INIT_FILE(BOOT_ROM_FILE)
  ) u_boot_rom (
    .clk(clk_core),
    .rst_n(rst_core_n),
    
    .awvalid(s_awvalid[0]),
    .awready(s_awready[0]),
    .awaddr(s_awaddr[0]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[0]),
    
    .wvalid(s_wvalid[0]),
    .wready(s_wready[0]),
    .wdata(s_wdata[0]),
    .wstrb(s_wstrb[0]),
    .wlast(s_wlast[0]),
    
    .bvalid(s_bvalid[0]),
    .bready(s_bready[0]),
    .bresp(s_bresp[0]),
    .bid(s_bid[0]),
    
    .arvalid(s_arvalid[0]),
    .arready(s_arready[0]),
    .araddr(s_araddr[0]),
    .arlen(s_arlen[0]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[0]),
    
    .rvalid(s_rvalid[0]),
    .rready(s_rready[0]),
    .rdata(s_rdata[0]),
    .rresp(s_rresp[0]),
    .rid(s_rid[0]),
    .rlast(s_rlast[0])
  );

  // ===== SLAVE 1: SRAM =====
  
  sram_ctrl #(
    .ADDR_WIDTH(15),
    .DATA_WIDTH(32)
  ) u_sram (
    .clk(clk_core),
    .rst_n(rst_core_n),
    
    .awvalid(s_awvalid[1]),
    .awready(s_awready[1]),
    .awaddr(s_awaddr[1]),
    .awlen(s_awlen[1]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[1]),
    
    .wvalid(s_wvalid[1]),
    .wready(s_wready[1]),
    .wdata(s_wdata[1]),
    .wstrb(s_wstrb[1]),
    .wlast(s_wlast[1]),
    
    .bvalid(s_bvalid[1]),
    .bready(s_bready[1]),
    .bresp(s_bresp[1]),
    .bid(s_bid[1]),
    
    .arvalid(s_arvalid[1]),
    .arready(s_arready[1]),
    .araddr(s_araddr[1]),
    .arlen(s_arlen[1]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[1]),
    
    .rvalid(s_rvalid[1]),
    .rready(s_rready[1]),
    .rdata(s_rdata[1]),
    .rresp(s_rresp[1]),
    .rid(s_rid[1]),
    .rlast(s_rlast[1])
  );

  // ===== SLAVE 2: PERIPHERALS (UART, Timer, GPIO, PLIC) =====
  
  // Sub-address decode: s_*addr[2][15:8]
  //   0x00 = UART     (0x2000_0000)
  //   0x01 = TIMER    (0x2001_0000)
  //   0x02 = GPIO     (0x2002_0000)
  //   0x03 = PLIC     (0x2003_0000)
  
  logic [7:0] periph_aw_sel, periph_ar_sel;
  assign periph_aw_sel = s_awaddr[2][15:8];
  assign periph_ar_sel = s_araddr[2][15:8];

  // UART
  uart_ctrl #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .CLK_FREQ(CLK_FREQ),
    .DEFAULT_BAUD(UART_BAUD)
  ) u_uart (
    .clk(clk_core),
    .rst_n(rst_core_n),
    .rx(uart_rx),
    .tx(uart_tx),
    
    .awvalid(s_awvalid[2] && (periph_aw_sel == 8'h00)),
    .awready(uart_awready),
    .awaddr(s_awaddr[2]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[2]),
    
    .wvalid(s_wvalid[2] && (periph_aw_sel == 8'h00)),
    .wready(uart_wready),
    .wdata(s_wdata[2]),
    .wstrb(s_wstrb[2]),
    .wlast(s_wlast[2]),
    
    .bvalid(uart_bvalid),
    .bready(s_bready[2]),
    .bresp(uart_bresp),
    .bid(uart_bid),
    
    .arvalid(s_arvalid[2] && (periph_ar_sel == 8'h00)),
    .arready(uart_arready),
    .araddr(s_araddr[2]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[2]),
    
    .rvalid(uart_rvalid),
    .rready(s_rready[2]),
    .rdata(uart_rdata),
    .rresp(uart_rresp),
    .rid(uart_rid),
    .rlast(uart_rlast),
    
    .irq_o(irq_uart)
  );

  // Timer
  timer_ctrl #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32)
  ) u_timer (
    .clk(clk_core),
    .rst_n(rst_core_n),
    
    .awvalid(s_awvalid[2] && (periph_aw_sel == 8'h01)),
    .awready(timer_awready),
    .awaddr(s_awaddr[2]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[2]),
    
    .wvalid(s_wvalid[2] && (periph_aw_sel == 8'h01)),
    .wready(timer_wready),
    .wdata(s_wdata[2]),
    .wstrb(s_wstrb[2]),
    .wlast(s_wlast[2]),
    
    .bvalid(timer_bvalid),
    .bready(s_bready[2]),
    .bresp(timer_bresp),
    .bid(timer_bid),
    
    .arvalid(s_arvalid[2] && (periph_ar_sel == 8'h01)),
    .arready(timer_arready),
    .araddr(s_araddr[2]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[2]),
    
    .rvalid(timer_rvalid),
    .rready(s_rready[2]),
    .rdata(timer_rdata),
    .rresp(timer_rresp),
    .rid(timer_rid),
    .rlast(timer_rlast),
    
    .irq_timer_o(irq_timer_int)
  );

  // GPIO
  gpio_ctrl #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .GPIO_WIDTH(8)
  ) u_gpio (
    .clk(clk_core),
    .rst_n(rst_core_n),
    .gpio_o(gpio_o),
    .gpio_i(gpio_i),
    .gpio_oe(gpio_oe),
    
    .awvalid(s_awvalid[2] && (periph_aw_sel == 8'h02)),
    .awready(gpio_awready),
    .awaddr(s_awaddr[2]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[2]),
    
    .wvalid(s_wvalid[2] && (periph_aw_sel == 8'h02)),
    .wready(gpio_wready),
    .wdata(s_wdata[2]),
    .wstrb(s_wstrb[2]),
    .wlast(s_wlast[2]),
    
    .bvalid(gpio_bvalid),
    .bready(s_bready[2]),
    .bresp(gpio_bresp),
    .bid(gpio_bid),
    
    .arvalid(s_arvalid[2] && (periph_ar_sel == 8'h02)),
    .arready(gpio_arready),
    .araddr(s_araddr[2]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[2]),
    
    .rvalid(gpio_rvalid),
    .rready(s_rready[2]),
    .rdata(gpio_rdata),
    .rresp(gpio_rresp),
    .rid(gpio_rid),
    .rlast(gpio_rlast)
  );

  // PLIC
  plic #(
    .ADDR_WIDTH(32),
    .DATA_WIDTH(32),
    .NUM_SOURCES(32),
    .NUM_TARGETS(1)
  ) u_plic (
    .clk(clk_core),
    .rst_n(rst_core_n),
    
    .irq_i({25'b0, accel_irq, 1'b0, irq_timer_int, irq_uart, 3'b0}),
    .irq_o(irq_plic_ext),
    
    .awvalid(s_awvalid[2] && (periph_aw_sel == 8'h03)),
    .awready(plic_awready),
    .awaddr(s_awaddr[2]),
    .awsize(3'b010),
    .awburst(2'b00),
    .awid(s_awid[2]),
    
    .wvalid(s_wvalid[2] && (periph_aw_sel == 8'h03)),
    .wready(plic_wready),
    .wdata(s_wdata[2]),
    .wstrb(s_wstrb[2]),
    .wlast(s_wlast[2]),
    
    .bvalid(plic_bvalid),
    .bready(s_bready[2]),
    .bresp(plic_bresp),
    .bid(plic_bid),
    
    .arvalid(s_arvalid[2] && (periph_ar_sel == 8'h03)),
    .arready(plic_arready),
    .araddr(s_araddr[2]),
    .arsize(3'b010),
    .arburst(2'b00),
    .arid(s_arid[2]),
    
    .rvalid(plic_rvalid),
    .rready(s_rready[2]),
    .rdata(plic_rdata),
    .rresp(plic_rresp),
    .rid(plic_rid),
    .rlast(plic_rlast)
  );

  // Peripheral response mux for slave 2
  // Priority encode: whoever has a valid response wins
  always_comb begin
    if (timer_rvalid) begin
      s_rvalid[2] = timer_rvalid;
      s_rdata[2]  = timer_rdata;
      s_rresp[2]  = timer_rresp;
      s_rid[2]    = timer_rid;
      s_rlast[2]  = timer_rlast;
    end else if (gpio_rvalid) begin
      s_rvalid[2] = gpio_rvalid;
      s_rdata[2]  = gpio_rdata;
      s_rresp[2]  = gpio_rresp;
      s_rid[2]    = gpio_rid;
      s_rlast[2]  = gpio_rlast;
    end else if (plic_rvalid) begin
      s_rvalid[2] = plic_rvalid;
      s_rdata[2]  = plic_rdata;
      s_rresp[2]  = plic_rresp;
      s_rid[2]    = plic_rid;
      s_rlast[2]  = plic_rlast;
    end else if (uart_rvalid) begin
      s_rvalid[2] = uart_rvalid;
      s_rdata[2]  = uart_rdata;
      s_rresp[2]  = uart_rresp;
      s_rid[2]    = uart_rid;
      s_rlast[2]  = uart_rlast;
    end else begin
      // No peripheral has a response
      s_rvalid[2] = 1'b0;
      s_rdata[2]  = '0;
      s_rresp[2]  = 2'b00;
      s_rid[2]    = '0;
      s_rlast[2]  = 1'b0;
    end
  end

  always_comb begin
    if (timer_bvalid) begin
      s_bvalid[2] = timer_bvalid;
      s_bresp[2]  = timer_bresp;
      s_bid[2]    = timer_bid;
    end else if (gpio_bvalid) begin
      s_bvalid[2] = gpio_bvalid;
      s_bresp[2]  = gpio_bresp;
      s_bid[2]    = gpio_bid;
    end else if (plic_bvalid) begin
      s_bvalid[2] = plic_bvalid;
      s_bresp[2]  = plic_bresp;
      s_bid[2]    = plic_bid;
    end else if (uart_bvalid) begin
      s_bvalid[2] = uart_bvalid;
      s_bresp[2]  = uart_bresp;
      s_bid[2]    = uart_bid;
    end else begin
      s_bvalid[2] = 1'b0;
      s_bresp[2]  = 2'b00;
      s_bid[2]    = '0;
    end
  end

  // awready/arready/wready: OR of all selected slaves
  assign s_awready[2] = (periph_aw_sel == 8'h00) ? uart_awready  :
                         (periph_aw_sel == 8'h01) ? timer_awready :
                         (periph_aw_sel == 8'h02) ? gpio_awready  :
                         (periph_aw_sel == 8'h03) ? plic_awready  :
                         1'b1;
  assign s_wready[2]  = (periph_aw_sel == 8'h00) ? uart_wready   :
                         (periph_aw_sel == 8'h01) ? timer_wready  :
                         (periph_aw_sel == 8'h02) ? gpio_wready   :
                         (periph_aw_sel == 8'h03) ? plic_wready   :
                         1'b1;
  assign s_arready[2] = (periph_ar_sel == 8'h00) ? uart_arready  :
                         (periph_ar_sel == 8'h01) ? timer_arready :
                         (periph_ar_sel == 8'h02) ? gpio_arready  :
                         (periph_ar_sel == 8'h03) ? plic_arready  :
                         1'b1;

  // ===== SLAVE 4: DRAM CONTROLLER =====

  dram_ctrl_top #(
    .AXI_ADDR_W  (32),
    .AXI_DATA_W  (32),
    .AXI_ID_W    (ID_WIDTH),
    .NUM_BANKS   (8),
    .ROW_BITS    (14),
    .COL_BITS    (10),
    .BANK_BITS   (3),
    .QUEUE_DEPTH (16),
    .ADDR_MODE   (0)       // Bank-interleaved (streaming-friendly)
  ) u_dram_ctrl (
    .clk   (clk_core),
    .rst_n (rst_core_n),

    // AXI4 Slave ← crossbar slave port 4
    .s_axi_awvalid (s_awvalid[4]),
    .s_axi_awready (s_awready[4]),
    .s_axi_awaddr  (s_awaddr[4]),
    .s_axi_awid    (s_awid[4]),
    .s_axi_awlen   (8'd0),          // single-beat (crossbar doesn't carry burst len)
    .s_axi_awsize  (3'b010),        // 4 bytes per beat

    .s_axi_wvalid  (s_wvalid[4]),
    .s_axi_wready  (s_wready[4]),
    .s_axi_wdata   (s_wdata[4]),
    .s_axi_wstrb   (s_wstrb[4]),
    .s_axi_wlast   (s_wlast[4]),

    .s_axi_bvalid  (s_bvalid[4]),
    .s_axi_bready  (s_bready[4]),
    .s_axi_bresp   (s_bresp[4]),
    .s_axi_bid     (s_bid[4]),

    .s_axi_arvalid (s_arvalid[4]),
    .s_axi_arready (s_arready[4]),
    .s_axi_araddr  (s_araddr[4]),
    .s_axi_arid    (s_arid[4]),
    .s_axi_arlen   (8'd0),
    .s_axi_arsize  (3'b010),

    .s_axi_rvalid  (s_rvalid[4]),
    .s_axi_rready  (s_rready[4]),
    .s_axi_rdata   (s_rdata[4]),
    .s_axi_rresp   (s_rresp[4]),
    .s_axi_rid     (s_rid[4]),
    .s_axi_rlast   (s_rlast[4]),

    // DRAM PHY → top-level pins
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

  // Slaves 5-7: Unused — DECERR responder (ready signals are wire, so assign is ok)
  genvar si;
  generate
    for (si = 5; si < NUM_SLAVES; si++) begin : gen_dummy_slaves_ready
      assign s_awready[si] = 1'b1;
      assign s_wready[si]  = 1'b1;
      assign s_arready[si] = 1'b1;
    end
  endgenerate

  // Dummy slave response channels driven from always_comb to match s_bvalid/s_rvalid
  // inference (those signals are inferred as reg due to always_comb drivers on index 2).
  always_comb begin
    for (int i = 5; i < NUM_SLAVES; i++) begin
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

  // ===== INTERRUPT OUTPUTS =====
  
  assign irq_external = irq_plic_ext;
  assign irq_timer = irq_timer_int;

endmodule : soc_top
