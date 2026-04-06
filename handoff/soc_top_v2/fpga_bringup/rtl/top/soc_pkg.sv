`timescale 1ns/1ps
/* verilator lint_off UNUSEDPARAM */

// SoC Package - Address Map and AXI Type Definitions
package soc_pkg;

  // ===== Global Address Map =====
  localparam logic [31:0] BOOT_ROM_BASE   = 32'h0000_0000;  // 8KB
  localparam logic [31:0] SRAM_BASE       = 32'h1000_0000;  // 32KB
  localparam logic [31:0] UART_BASE       = 32'h2000_0000;  // 256B
  localparam logic [31:0] TIMER_BASE      = 32'h2001_0000;  // 256B
  localparam logic [31:0] GPIO_BASE       = 32'h2002_0000;  // 256B
  localparam logic [31:0] PLIC_BASE       = 32'h2003_0000;  // 64KB
  localparam logic [31:0] ETH_BASE        = 32'h2004_0000;  // 256B reserved window
  localparam logic [31:0] ACCEL_BASE      = 32'h3000_0000;  // 64KB (reuse ACCEL-v1 CSR space)
  localparam logic [31:0] DRAM_BASE       = 32'h4000_0000;  // 256MB DRAM
  localparam logic [31:0] DRAM_BASE_UC    = 32'h6000_0000;  // uncached alias for CPU preload/readback
  localparam logic [31:0] SRAM_BASE_UC    = 32'h7000_0000;  // uncached alias for CPU preload/readback

  // ===== Memory Sizes =====
  localparam int unsigned BOOT_ROM_SIZE = 4*1024;           // 4 KB
  localparam int unsigned SRAM_SIZE      = 32*1024;          // 32 KB

  // ===== AXI Lite Parameters =====
  localparam int unsigned AXI_ADDR_WIDTH = 32;
  localparam int unsigned AXI_DATA_WIDTH = 32;
  localparam int unsigned AXI_ID_WIDTH   = 4;

  // ===== AXI Lite Request Struct =====
  typedef struct packed {
    // Write Address Channel
    logic                           awvalid;
    logic [AXI_ADDR_WIDTH-1:0]      awaddr;
    logic [2:0]                     awsize;
    logic [1:0]                     awburst;
    logic [AXI_ID_WIDTH-1:0]        awid;
    
    // Write Data Channel
    logic                           wvalid;
    logic [AXI_DATA_WIDTH-1:0]      wdata;
    logic [AXI_DATA_WIDTH/8-1:0]    wstrb;
    logic                           wlast;
    
    // Read Address Channel
    logic                           arvalid;
    logic [AXI_ADDR_WIDTH-1:0]      araddr;
    logic [2:0]                     arsize;
    logic [1:0]                     arburst;
    logic [AXI_ID_WIDTH-1:0]        arid;
    
    // Read/Write Response Ready signals
    logic                           bready;
    logic                           rready;
  } axi_req_t;

  // ===== AXI Lite Response Struct =====
  typedef struct packed {
    // Write Response Channel
    logic                           bvalid;
    logic [1:0]                     bresp;
    logic [AXI_ID_WIDTH-1:0]        bid;
    
    // Read Data Channel
    logic                           rvalid;
    logic [AXI_DATA_WIDTH-1:0]      rdata;
    logic [1:0]                     rresp;
    logic [AXI_ID_WIDTH-1:0]        rid;
    logic                           rlast;
    
    // Ready signals to slave
    logic                           awready;
    logic                           wready;
    logic                           arready;
  } axi_resp_t;

  // ===== AXI Response Codes =====
  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_EXOKAY = 2'b01;
  localparam logic [1:0] RESP_SLVERR = 2'b10;
  localparam logic [1:0] RESP_DECERR = 2'b11;

endpackage : soc_pkg
