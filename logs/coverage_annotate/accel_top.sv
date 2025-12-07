//      // verilator_coverage annotation
        // =============================================================================
        // accel_top_v2.sv — AXI4-based Sparse Accelerator Top-Level (UART-Free)
        // =============================================================================
        // Architecture:
        //   Host CPU → AXI-Lite → CSR (Configuration)
        //   Host CPU → AXI4 → axi_dma_bridge → act_dma + bsr_dma → DDR
        //   bsr_dma → meta_decode → bsr_scheduler → systolic_array_sparse → Output
        //
        // Design Choice: SPARSE-ONLY (Dense systolic array removed to save area)
        // =============================================================================
        
        `timescale 1ns/1ps
        `default_nettype none
        
        module accel_top #(
            parameter N_ROWS     = 16,          // Systolic array rows (16x16)
            parameter N_COLS     = 16,          // Systolic array columns
            parameter DATA_W     = 8,           // INT8 data width
            parameter ACC_W      = 32,          // Accumulator width
            parameter AXI_ADDR_W = 32,          // AXI address width
            parameter AXI_DATA_W = 64,          // AXI data width (64-bit for DDR)
            parameter AXI_ID_W   = 4,           // AXI transaction ID width
            parameter CSR_ADDR_W = 8,           // CSR address width (256 registers)
            parameter BRAM_ADDR_W = 10          // BRAM address width
        )(
            // =========================================================================
            // Clock & Reset
            // =========================================================================
 012713     input  wire clk,
%000007     input  wire rst_n,
        
            // =========================================================================
            // AXI4 Master Interface (To DDR via Zynq HP Port)
            // =========================================================================
            // Read Address Channel
%000001     output wire [AXI_ID_W-1:0]   m_axi_arid,
%000006     output wire [AXI_ADDR_W-1:0] m_axi_araddr,
%000004     output wire [7:0]            m_axi_arlen,
%000006     output wire [2:0]            m_axi_arsize,
%000006     output wire [1:0]            m_axi_arburst,
%000006     output wire                  m_axi_arvalid,
%000007     input  wire                  m_axi_arready,
        
            // Read Data Channel
%000001     input  wire [AXI_ID_W-1:0]   m_axi_rid,
%000007     input  wire [AXI_DATA_W-1:0] m_axi_rdata,
%000000     input  wire [1:0]            m_axi_rresp,
%000006     input  wire                  m_axi_rlast,
%000006     input  wire                  m_axi_rvalid,
%000007     output wire                  m_axi_rready,
        
            // =========================================================================
            // AXI4-Lite Slave Interface (CSR from Zynq GP Port)
            // =========================================================================
~000028     input  wire [AXI_ADDR_W-1:0] s_axi_awaddr,
%000000     input  wire [2:0]            s_axi_awprot,
 000048     input  wire                  s_axi_awvalid,
 000096     output wire                  s_axi_awready,
~000014     input  wire [31:0]           s_axi_wdata,
%000007     input  wire [3:0]            s_axi_wstrb,
 000048     input  wire                  s_axi_wvalid,
 000096     output wire                  s_axi_wready,
%000000     output wire [1:0]            s_axi_bresp,
 000096     output wire                  s_axi_bvalid,
 000052     input  wire                  s_axi_bready,
~000021     input  wire [AXI_ADDR_W-1:0] s_axi_araddr,
%000000     input  wire [2:0]            s_axi_arprot,
 000152     input  wire                  s_axi_arvalid,
 000153     output wire                  s_axi_arready,
~000011     output wire [31:0]           s_axi_rdata,
%000000     output wire [1:0]            s_axi_rresp,
 000152     output wire                  s_axi_rvalid,
 000152     input  wire                  s_axi_rready,
        
            // =========================================================================
            // Status Outputs (directly from CSR control signals)
            // =========================================================================
%000006     output wire busy,
%000004     output wire done,
%000000     output wire error
        );
        
            // =========================================================================
            // Internal Signal Declarations
            // =========================================================================
        
            // -------------------------------------------------------------------------
            // CSR Interface Signals
            // -------------------------------------------------------------------------
 000152     wire                    csr_wen, csr_ren;
~000094     wire [CSR_ADDR_W-1:0]   csr_addr;
~000075     wire [31:0]             csr_wdata, csr_rdata;
%000005     wire                    start_pulse, abort_pulse;
        
            // CSR Configuration Outputs
%000007     wire [31:0] cfg_M, cfg_N, cfg_K;
%000001     wire [31:0] cfg_act_src_addr;
%000002     wire [31:0] cfg_bsr_src_addr;
%000001     wire [31:0] cfg_act_xfer_len;
%000002     wire        cfg_act_dma_start;
        
            // -------------------------------------------------------------------------
            // DMA Control Signals
            // -------------------------------------------------------------------------
            // Activation DMA
%000002     wire                    act_dma_start;
%000001     wire                    act_dma_done;
%000001     wire                    act_dma_busy;
%000000     wire                    act_dma_error;
%000001     wire [AXI_ADDR_W-1:0]   act_dma_src_addr;
%000001     wire [31:0]             act_dma_xfer_len;
        
            // BSR DMA
%000003     wire                    bsr_dma_start;
%000001     wire                    bsr_dma_done;
%000001     wire                    bsr_dma_busy;
%000000     wire                    bsr_dma_error;
%000002     wire [AXI_ADDR_W-1:0]   bsr_dma_src_addr;
        
            // -------------------------------------------------------------------------
            // AXI DMA Bridge Internal Signals (Arbiter between two DMAs)
            // -------------------------------------------------------------------------
            // act_dma → Bridge (Slave 1)
%000001     wire [AXI_ID_W-1:0]     act_arid;
%000001     wire [AXI_ADDR_W-1:0]   act_araddr;
%000001     wire [7:0]              act_arlen;
%000001     wire [2:0]              act_arsize;
%000001     wire [1:0]              act_arburst;
%000001     wire                    act_arvalid;
%000001     wire                    act_arready;
%000001     wire [AXI_ID_W-1:0]     act_rid;
%000007     wire [AXI_DATA_W-1:0]   act_rdata;
%000000     wire [1:0]              act_rresp;
%000006     wire                    act_rlast;
%000001     wire                    act_rvalid;
%000001     wire                    act_rready;
        
            // bsr_dma → Bridge (Slave 0)
%000000     wire [AXI_ID_W-1:0]     bsr_arid;
%000001     wire [AXI_ADDR_W-1:0]   bsr_araddr;
%000002     wire [7:0]              bsr_arlen;
%000001     wire [2:0]              bsr_arsize;
%000001     wire [1:0]              bsr_arburst;
 000013     wire                    bsr_arvalid;
%000005     wire                    bsr_arready;
%000001     wire [AXI_ID_W-1:0]     bsr_rid;
%000007     wire [AXI_DATA_W-1:0]   bsr_rdata;
%000000     wire [1:0]              bsr_rresp;
%000006     wire                    bsr_rlast;
%000005     wire                    bsr_rvalid;
%000003     wire                    bsr_rready;
        
            // -------------------------------------------------------------------------
            // Buffer Write Interfaces (from DMAs)
            // -------------------------------------------------------------------------
            // Activation Buffer Write (from act_dma)
%000001     wire                    act_buf_we;
%000004     wire [AXI_ADDR_W-1:0]   act_buf_waddr;
%000004     wire [AXI_DATA_W-1:0]   act_buf_wdata;
        
            // BSR Metadata BRAMs Write (from bsr_dma)
%000001     wire                    row_ptr_we;
%000005     wire [BRAM_ADDR_W-1:0]  row_ptr_waddr;
%000004     wire [31:0]             row_ptr_wdata;
        
%000001     wire                    col_idx_we;
%000002     wire [BRAM_ADDR_W-1:0]  col_idx_waddr;
%000000     wire [15:0]             col_idx_wdata;
        
%000002     wire                    wgt_we;
~000016     wire [BRAM_ADDR_W+6:0]  wgt_waddr;
%000004     wire [63:0]             wgt_wdata;
        
            // -------------------------------------------------------------------------
            // Metadata Decoder Interface (BSR BRAM → Scheduler)
            // -------------------------------------------------------------------------
%000005     wire                    meta_req_valid;
%000004     wire [31:0]             meta_req_addr;
 000015     wire                    meta_req_ready;
 000014     wire                    meta_valid;
%000000     wire [31:0]             meta_rdata;
%000001     wire                    meta_ready;
        
            // Memory interface for meta_decode
%000009     wire                    meta_mem_en;
%000004     wire [31:0]             meta_mem_addr;
%000000     wire [31:0]             meta_mem_rdata;
        
            // -------------------------------------------------------------------------
            // BSR Scheduler Interface
            // -------------------------------------------------------------------------
%000005     wire                    sched_start;
%000004     wire                    sched_busy;
%000004     wire                    sched_done;
%000007     wire [9:0]              sched_MT;
%000004     wire [11:0]             sched_KT;
        
            // Scheduler → Systolic Array
%000000     wire                    load_weight;
%000000     wire                    pe_en;
%000000     wire                    accum_en;
        
            // Scheduler → Buffer Read
%000000     wire                    wgt_rd_en;
%000000     wire [AXI_ADDR_W-1:0]   wgt_rd_addr;
%000000     wire                    act_rd_en;
%000000     wire [AXI_ADDR_W-1:0]   act_rd_addr;
        
            // -------------------------------------------------------------------------
            // Activation Buffer Read Interface
            // -------------------------------------------------------------------------
%000000     wire [N_ROWS*DATA_W-1:0] act_rd_data;
        
            // -------------------------------------------------------------------------
            // Weight Buffer (Block Data) Read Interface
            // -------------------------------------------------------------------------
%000000     wire [N_COLS*DATA_W-1:0] wgt_rd_data;
        
            // -------------------------------------------------------------------------
            // Sparse Systolic Array Interface
            // -------------------------------------------------------------------------
            wire [N_ROWS*N_COLS*ACC_W-1:0] systolic_out_flat;
        
            // -------------------------------------------------------------------------
            // Performance Counters
            // -------------------------------------------------------------------------
%000003     wire [31:0] perf_total_cycles;
%000004     wire [31:0] perf_active_cycles;
%000004     wire [31:0] perf_idle_cycles;
%000004     wire        perf_done;
        
            // -------------------------------------------------------------------------
            // Status Aggregation
            // -------------------------------------------------------------------------
            assign busy  = act_dma_busy | bsr_dma_busy | sched_busy;
            assign done  = sched_done;
            assign error = bsr_dma_error | act_dma_error;
        
            // =========================================================================
            // Module Instantiations
            // =========================================================================
        
            // -------------------------------------------------------------------------
            // 1. AXI4-Lite Slave (CSR Access from Host CPU)
            // -------------------------------------------------------------------------
            axi_lite_slave #(
                .CSR_ADDR_WIDTH(CSR_ADDR_W),
                .CSR_DATA_WIDTH(32)
            ) u_axi_lite_slave (
                .clk            (clk),
                .rst_n          (rst_n),
                // AXI4-Lite Write
                .s_axi_awaddr   (s_axi_awaddr[CSR_ADDR_W-1:0]),
                .s_axi_awprot   (s_axi_awprot),
                .s_axi_awvalid  (s_axi_awvalid),
                .s_axi_awready  (s_axi_awready),
                .s_axi_wdata    (s_axi_wdata),
                .s_axi_wstrb    (s_axi_wstrb),
                .s_axi_wvalid   (s_axi_wvalid),
                .s_axi_wready   (s_axi_wready),
                .s_axi_bresp    (s_axi_bresp),
                .s_axi_bvalid   (s_axi_bvalid),
                .s_axi_bready   (s_axi_bready),
                // AXI4-Lite Read
                .s_axi_araddr   (s_axi_araddr[CSR_ADDR_W-1:0]),
                .s_axi_arprot   (s_axi_arprot),
                .s_axi_arvalid  (s_axi_arvalid),
                .s_axi_arready  (s_axi_arready),
                .s_axi_rdata    (s_axi_rdata),
                .s_axi_rresp    (s_axi_rresp),
                .s_axi_rvalid   (s_axi_rvalid),
                .s_axi_rready   (s_axi_rready),
                // CSR Interface
                .csr_addr       (csr_addr),
                .csr_wen        (csr_wen),
                .csr_ren        (csr_ren),
                .csr_wdata      (csr_wdata),
                .csr_rdata      (csr_rdata),
                .axi_error      ()
            );
        
            // -------------------------------------------------------------------------
            // 2. CSR Module (Configuration & Control Registers)
            // -------------------------------------------------------------------------
            csr #(
                .ADDR_W(CSR_ADDR_W)
            ) u_csr (
                .clk                    (clk),
                .rst_n                  (rst_n),
                // CSR Bus
                .csr_wen                (csr_wen),
                .csr_ren                (csr_ren),
                .csr_addr               (csr_addr),
                .csr_wdata              (csr_wdata),
                .csr_rdata              (csr_rdata),
                // Status Inputs
                .core_busy              (busy),
                .core_done_tile_pulse   (sched_done),
                .core_bank_sel_rd_A     (1'b0),
                .core_bank_sel_rd_B     (1'b0),
                .rx_illegal_cmd         (1'b0),  // No UART, no illegal commands
                // Control Outputs
                .start_pulse            (start_pulse),
                .abort_pulse            (abort_pulse),
                .irq_en                 (),
                // Matrix Dimensions
                .M                      (cfg_M),
                .N                      (cfg_N),
                .K                      (cfg_K),
                // Tile Sizes (not used in sparse mode, but kept for compatibility)
                .Tm                     (),
                .Tn                     (),
                .Tk                     (),
                .m_idx                  (),
                .n_idx                  (),
                .k_idx                  (),
                // Bank Selection (legacy, unused in sparse)
                .bank_sel_wr_A          (),
                .bank_sel_wr_B          (),
                .bank_sel_rd_A          (),
                .bank_sel_rd_B          (),
                // Scaling Factors (for future quantization)
                .Sa_bits                (),
                .Sw_bits                (),
                // Performance Counters
                .perf_total_cycles      (perf_total_cycles),
                .perf_active_cycles     (perf_active_cycles),
                .perf_idle_cycles       (perf_idle_cycles),
                .perf_cache_hits        (32'd0),  // TODO: Connect from meta_decode
                .perf_cache_misses      (32'd0),
                .perf_decode_count      (32'd0),
                // Result Data (first 4 accumulators for quick read)
                .result_data            (systolic_out_flat[127:0]),
                // DMA Control/Status
                .dma_busy_in            (act_dma_busy | bsr_dma_busy),
                .dma_done_in            (act_dma_done & bsr_dma_done),
                .dma_bytes_xferred_in   (32'd0),  // TODO: Add counters
                .dma_src_addr           (cfg_bsr_src_addr),
                .dma_dst_addr           (),       // Not used (read-only DMAs)
                .dma_xfer_len           (),
                .dma_start_pulse        (bsr_dma_start),
                // Activation DMA
                .act_dma_src_addr       (cfg_act_src_addr),
                .act_dma_len            (cfg_act_xfer_len),
                .act_dma_start_pulse    (cfg_act_dma_start)
            );
        
            // BSR DMA uses a separate CSR address for its source
            assign bsr_dma_src_addr = cfg_bsr_src_addr;
            assign act_dma_start    = cfg_act_dma_start;
            assign act_dma_src_addr = cfg_act_src_addr;
            assign act_dma_xfer_len = cfg_act_xfer_len;
        
            // Scheduler config
            assign sched_start = start_pulse;
            assign sched_MT    = cfg_M[9:0] >> 3;   // M / 8 (block rows)
            assign sched_KT    = cfg_K[11:0] >> 3;  // K / 8 (block columns)
        
            // -------------------------------------------------------------------------
            // 3. AXI DMA Bridge (Arbitrates act_dma and bsr_dma to single AXI Master)
            // -------------------------------------------------------------------------
            axi_dma_bridge #(
                .DATA_WIDTH (AXI_DATA_W),
                .ADDR_WIDTH (AXI_ADDR_W),
                .ID_WIDTH   (AXI_ID_W)
            ) u_axi_dma_bridge (
                .clk            (clk),
                .rst_n          (rst_n),
                // Slave 0: BSR DMA
                .s0_arid        (bsr_arid),
                .s0_araddr      (bsr_araddr),
                .s0_arlen       (bsr_arlen),
                .s0_arsize      (bsr_arsize),
                .s0_arburst     (bsr_arburst),
                .s0_arvalid     (bsr_arvalid),
                .s0_arready     (bsr_arready),
                .s0_rid         (bsr_rid),
                .s0_rdata       (bsr_rdata),
                .s0_rresp       (bsr_rresp),
                .s0_rlast       (bsr_rlast),
                .s0_rvalid      (bsr_rvalid),
                .s0_rready      (bsr_rready),
                // Slave 1: Activation DMA
                .s1_arid        (act_arid),
                .s1_araddr      (act_araddr),
                .s1_arlen       (act_arlen),
                .s1_arsize      (act_arsize),
                .s1_arburst     (act_arburst),
                .s1_arvalid     (act_arvalid),
                .s1_arready     (act_arready),
                .s1_rid         (act_rid),
                .s1_rdata       (act_rdata),
                .s1_rresp       (act_rresp),
                .s1_rlast       (act_rlast),
                .s1_rvalid      (act_rvalid),
                .s1_rready      (act_rready),
                // Master to DDR
                .m_arid         (m_axi_arid),
                .m_araddr       (m_axi_araddr),
                .m_arlen        (m_axi_arlen),
                .m_arsize       (m_axi_arsize),
                .m_arburst      (m_axi_arburst),
                .m_arvalid      (m_axi_arvalid),
                .m_arready      (m_axi_arready),
                .m_rid          (m_axi_rid),
                .m_rdata        (m_axi_rdata),
                .m_rresp        (m_axi_rresp),
                .m_rlast        (m_axi_rlast),
                .m_rvalid       (m_axi_rvalid),
                .m_rready       (m_axi_rready)
            );
        
            // -------------------------------------------------------------------------
            // 4. Activation DMA (Loads Dense Activations from DDR → act_buffer)
            // -------------------------------------------------------------------------
            act_dma #(
                .AXI_ADDR_W (AXI_ADDR_W),
                .AXI_DATA_W (AXI_DATA_W),
                .AXI_ID_W   (AXI_ID_W),
                .STREAM_ID  (1),        // ID = 1 for Act DMA
                .BURST_LEN  (8'd15)     // 16-beat bursts
            ) u_act_dma (
                .clk                (clk),
                .rst_n              (rst_n),
                // Control
                .start              (act_dma_start),
                .src_addr           (act_dma_src_addr),
                .transfer_length    (act_dma_xfer_len),
                .done               (act_dma_done),
                .busy               (act_dma_busy),
                .error              (act_dma_error),
                // AXI Master (to Bridge)
                .m_axi_arid         (act_arid),
                .m_axi_araddr       (act_araddr),
                .m_axi_arlen        (act_arlen),
                .m_axi_arsize       (act_arsize),
                .m_axi_arburst      (act_arburst),
                .m_axi_arvalid      (act_arvalid),
                .m_axi_arready      (act_arready),
                .m_axi_rid          (act_rid),
                .m_axi_rdata        (act_rdata),
                .m_axi_rresp        (act_rresp),
                .m_axi_rlast        (act_rlast),
                .m_axi_rvalid       (act_rvalid),
                .m_axi_rready       (act_rready),
                // Buffer Write Interface
                .act_we             (act_buf_we),
                .act_addr           (act_buf_waddr),
                .act_wdata          (act_buf_wdata)
            );
        
            // -------------------------------------------------------------------------
            // 5. BSR DMA (Loads Sparse Weights from DDR → Row Ptr, Col Idx, Block BRAMs)
            // -------------------------------------------------------------------------
            bsr_dma #(
                .AXI_ADDR_W  (AXI_ADDR_W),
                .AXI_DATA_W  (AXI_DATA_W),
                .AXI_ID_W    (AXI_ID_W),
                .STREAM_ID   (0),       // ID = 0 for BSR DMA
                .BRAM_ADDR_W (BRAM_ADDR_W),
                .BURST_LEN   (8'd15)
            ) u_bsr_dma (
                .clk                (clk),
                .rst_n              (rst_n),
                // Control
                .start              (bsr_dma_start),
                .src_addr           (bsr_dma_src_addr),
                .done               (bsr_dma_done),
                .busy               (bsr_dma_busy),
                .error              (bsr_dma_error),
                // AXI Master (to Bridge)
                .m_axi_arid         (bsr_arid),
                .m_axi_araddr       (bsr_araddr),
                .m_axi_arlen        (bsr_arlen),
                .m_axi_arsize       (bsr_arsize),
                .m_axi_arburst      (bsr_arburst),
                .m_axi_arvalid      (bsr_arvalid),
                .m_axi_arready      (bsr_arready),
                .m_axi_rid          (bsr_rid),
                .m_axi_rdata        (bsr_rdata),
                .m_axi_rresp        (bsr_rresp),
                .m_axi_rlast        (bsr_rlast),
                .m_axi_rvalid       (bsr_rvalid),
                .m_axi_rready       (bsr_rready),
                // BRAM Write Interfaces
                .row_ptr_we         (row_ptr_we),
                .row_ptr_addr       (row_ptr_waddr),
                .row_ptr_wdata      (row_ptr_wdata),
                .col_idx_we         (col_idx_we),
                .col_idx_addr       (col_idx_waddr),
                .col_idx_wdata      (col_idx_wdata),
                .wgt_we             (wgt_we),
                .wgt_addr           (wgt_waddr),
                .wgt_wdata          (wgt_wdata)
            );
        
            // -------------------------------------------------------------------------
            // 6. BSR Metadata BRAMs (Store sparse structure)
            // -------------------------------------------------------------------------
            // Row Pointer BRAM (32-bit entries)
            reg [31:0] row_ptr_bram [0:(1<<BRAM_ADDR_W)-1];
%000000     reg [31:0] row_ptr_rdata_r;
        
 012713     always @(posedge clk) begin
~012704         if (row_ptr_we)
%000009             row_ptr_bram[row_ptr_waddr] <= row_ptr_wdata;
 012713         row_ptr_rdata_r <= row_ptr_bram[meta_mem_addr[BRAM_ADDR_W-1:0]];
            end
        
            // Column Index BRAM (16-bit entries)
            reg [15:0] col_idx_bram [0:(1<<BRAM_ADDR_W)-1];
%000000     reg [15:0] col_idx_rdata_r;
        
 012713     always @(posedge clk) begin
~012709         if (col_idx_we)
%000004             col_idx_bram[col_idx_waddr] <= col_idx_wdata;
            end
        
            // Weight Block BRAM (64-bit entries)
            reg [63:0] wgt_block_bram [0:(1<<BRAM_ADDR_W)-1];
%000000     reg [63:0] wgt_block_rdata_r;
        
 012713     always @(posedge clk) begin
 012681         if (wgt_we)
 000032             wgt_block_bram[wgt_waddr[BRAM_ADDR_W-1:0]] <= wgt_wdata;
 012713         wgt_block_rdata_r <= wgt_block_bram[wgt_rd_addr[BRAM_ADDR_W-1:0]];
            end
        
            // Metadata memory interface (row_ptr for now, mux if needed)
            assign meta_mem_rdata = row_ptr_rdata_r;
        
            // -------------------------------------------------------------------------
            // 7. Metadata Decoder (Cache for BSR Metadata)
            // -------------------------------------------------------------------------
            meta_decode #(
                .DATA_WIDTH  (32),
                .CACHE_DEPTH (64)
            ) u_meta_decode (
                .clk            (clk),
                .rst_n          (rst_n),
                // Scheduler Interface
                .req_valid      (meta_req_valid),
                .req_addr       (meta_req_addr),
                .req_ready      (meta_req_ready),
                // Memory Interface
                .mem_en         (meta_mem_en),
                .mem_addr       (meta_mem_addr),
                .mem_rdata      (meta_mem_rdata),
                // Output to Scheduler
                .meta_valid     (meta_valid),
                .meta_rdata     (meta_rdata),
                .meta_ready     (meta_ready)
            );
        
            // -------------------------------------------------------------------------
            // 8. BSR Scheduler (Traverses Sparse Blocks)
            // -------------------------------------------------------------------------
            bsr_scheduler #(
                .M_W        (10),
                .N_W        (10),
                .K_W        (12),
                .ADDR_W     (AXI_ADDR_W),
                .BLOCK_SIZE (N_ROWS)  // Block size matches systolic array dimension
            ) u_bsr_scheduler (
                .clk            (clk),
                .rst_n          (rst_n),
                // Control
                .start          (sched_start),
                .abort          (abort_pulse),
                .busy           (sched_busy),
                .done           (sched_done),
                // Configuration
                .MT             (sched_MT),
                .KT             (sched_KT),
                // Metadata Interface
                .meta_raddr     (meta_req_addr),
                .meta_ren       (meta_req_valid),
                .meta_rdata     (meta_rdata),
                .meta_rvalid    (meta_valid),
                .meta_ready     (meta_ready),
                // Buffer Interfaces
                .wgt_rd_en      (wgt_rd_en),
                .wgt_addr       (wgt_rd_addr),
                .act_rd_en      (act_rd_en),
                .act_addr       (act_rd_addr),
                // Systolic Control
                .load_weight    (load_weight),
                .pe_en          (pe_en),
                .accum_en       (accum_en)
            );
        
            // -------------------------------------------------------------------------
            // 9. Activation Buffer (Dense Activation Storage)
            // -------------------------------------------------------------------------
            // Simple dual-port RAM: DMA writes, Scheduler reads
            reg [AXI_DATA_W-1:0] act_buffer_ram [0:(1<<BRAM_ADDR_W)-1];
%000000     reg [AXI_DATA_W-1:0] act_rd_data_r;
        
 012713     always @(posedge clk) begin
~012705         if (act_buf_we)
%000008             act_buffer_ram[act_buf_waddr[BRAM_ADDR_W-1:0]] <= act_buf_wdata;
~012713         if (act_rd_en)
%000000             act_rd_data_r <= act_buffer_ram[act_rd_addr[BRAM_ADDR_W-1:0]];
            end
        
            assign act_rd_data = act_rd_data_r[N_ROWS*DATA_W-1:0];
        
            // -------------------------------------------------------------------------
            // 10. Weight Read Path (Block data from wgt_block_bram)
            // -------------------------------------------------------------------------
            assign wgt_rd_data = wgt_block_rdata_r[N_COLS*DATA_W-1:0];
        
            // -------------------------------------------------------------------------
            // 11. Sparse Systolic Array (2D PE Array with Skip-Zero Logic)
            // -------------------------------------------------------------------------
            systolic_array_sparse #(
                .N_ROWS (N_ROWS),
                .N_COLS (N_COLS),
                .DATA_W (DATA_W),
                .ACC_W  (ACC_W)
            ) u_systolic_sparse (
                .clk            (clk),
                .rst_n          (rst_n),
                // Control
                .block_valid    (pe_en),
                .load_weight    (load_weight),
                // Data
                .a_in_flat      (act_rd_data),
                .b_in_flat      (wgt_rd_data),
                // Output
                .c_out_flat     (systolic_out_flat)
            );
        
            // -------------------------------------------------------------------------
            // 12. Performance Monitor
            // -------------------------------------------------------------------------
            perf #(
                .COUNTER_WIDTH(32)
            ) u_perf (
                .clk                    (clk),
                .rst_n                  (rst_n),
                .start_pulse            (start_pulse),
                .done_pulse             (sched_done),
                .busy_signal            (busy),  // OR of all busy signals
                .meta_cache_hits        (32'd0),
                .meta_cache_misses      (32'd0),
                .meta_decode_cycles     (32'd0),
                .total_cycles_count     (perf_total_cycles),
                .active_cycles_count    (perf_active_cycles),
                .idle_cycles_count      (perf_idle_cycles),
                .cache_hit_count        (),
                .cache_miss_count       (),
                .decode_count           (),
                .measurement_done       (perf_done)
            );
        
        endmodule
        
        `default_nettype wire
        
