//      // verilator_coverage annotation
        // -----------------------------------------------------------------------------
        // csr.v — Accel v1 Control/Status (single source of truth)
        // -----------------------------------------------------------------------------
        `timescale 1ns/1ps
        `default_nettype none
        
        module csr #(
          parameter ADDR_W = 8,  // 256B map
          parameter ENABLE_CLOCK_GATING = 1
        )(
 012713   input  wire              clk,
%000007   input  wire              rst_n,
        
          // Host CSR bus
 000141   input  wire              csr_wen,
 000152   input  wire              csr_ren,
~000094   input  wire [ADDR_W-1:0] csr_addr,
~000023   input  wire [31:0]       csr_wdata, // <--- Data comes in here from AXI Slave
~000075   output reg  [31:0]       csr_rdata,
        
          // Events
%000006   input  wire              core_busy,
%000004   input  wire              core_done_tile_pulse,
%000000   input  wire              core_bank_sel_rd_A,
%000000   input  wire              core_bank_sel_rd_B,
%000000   input  wire              rx_illegal_cmd,
          
          // Performance monitor inputs
%000003   input  wire [31:0]       perf_total_cycles,
%000004   input  wire [31:0]       perf_active_cycles,
%000004   input  wire [31:0]       perf_idle_cycles,
%000000   input  wire [31:0]       perf_cache_hits,
%000000   input  wire [31:0]       perf_cache_misses,
%000000   input  wire [31:0]       perf_decode_count,
          
          // Results
%000000   input  wire [127:0]      result_data,
        
          // DMA status inputs (BSR)
%000002   input  wire              dma_busy_in,
%000000   input  wire              dma_done_in,
%000000   input  wire [31:0]       dma_bytes_xferred_in,
        
          // Pulses / config to core
%000005   output wire              start_pulse,
%000001   output wire              abort_pulse,
%000002   output wire              irq_en,
        
%000007   output wire [31:0]       M, N, K,
%000005   output wire [31:0]       Tm, Tn, Tk,
%000001   output wire [31:0]       m_idx, n_idx, k_idx,
%000001   output wire              bank_sel_wr_A, bank_sel_wr_B,
%000000   output wire              bank_sel_rd_A, bank_sel_rd_B,
%000002   output wire [31:0]       Sa_bits, Sw_bits,
        
          // DMA control outputs (BSR)
%000002   output wire [31:0]       dma_src_addr,
%000001   output wire [31:0]       dma_dst_addr,
%000002   output wire [31:0]       dma_xfer_len,
%000003   output wire              dma_start_pulse,
        
          // NEW: DMA control outputs (Activation)
%000001   output wire [31:0]       act_dma_src_addr,
%000001   output wire [31:0]       act_dma_len,
%000002   output wire              act_dma_start_pulse
        );
        
          // ========================================================================
          // 1. Clock Gating Logic (Power Optimization)
          // ========================================================================
          // Engineer's Note:
          // CSRs are accessed infrequently (only at start/end of jobs).
          // Gating the clock here saves dynamic power in the flip-flops.
 000392   wire csr_clk_en, clk_gated;
          assign csr_clk_en = csr_wen | csr_ren | core_done_tile_pulse;
          
          generate
            if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
              // Simplified for simulation/synthesis compatibility
              assign clk_gated = clk & csr_clk_en; 
            end else begin : gen_no_gate
              assign clk_gated = clk;
            end
          endgenerate
        
          // ========================================================================
          // 2. Address Map Definition
          // ========================================================================
          // Engineer's Note:
          // This map MUST match the Python driver (csr_map.py).
          // 0x00 - 0x3F: Control & Configuration
          // 0x40 - 0x7F: Status & Performance
          // 0x80 - 0xBF: DMA Configuration
          localparam CTRL         = 8'h00; // [2]=irq_en (RW), [1]=abort (W1P), [0]=start (W1P)
          localparam DIMS_M       = 8'h04;
          localparam DIMS_N       = 8'h08;
          localparam DIMS_K       = 8'h0C;
          localparam TILES_Tm     = 8'h10;
          localparam TILES_Tn     = 8'h14;
          localparam TILES_Tk     = 8'h18;
          localparam INDEX_m      = 8'h1C;
          localparam INDEX_n      = 8'h20;
          localparam INDEX_k      = 8'h24;
          localparam BUFF         = 8'h28; // [0]=wrA (RW), [1]=wrB (RW), [8]=rdA (RO), [9]=rdB (RO)
          localparam SCALE_Sa     = 8'h2C; // float32 bits
          localparam SCALE_Sw     = 8'h30; // float32 bits
          // UART registers removed
          localparam STATUS       = 8'h3C; // [0]=busy(RO), [1]=done_tile(R/W1C), [9]=err_illegal(R/W1C)
          // Performance monitor registers (Read-Only)
          localparam PERF_TOTAL   = 8'h40; // Total cycles from start to done
          localparam PERF_ACTIVE  = 8'h44; // Cycles where busy was high  
          localparam PERF_IDLE    = 8'h48; // Cycles where busy was low
          localparam PERF_CACHE_HITS   = 8'h4C; // Metadata cache hits
          localparam PERF_CACHE_MISSES = 8'h50; // Metadata cache misses
          localparam PERF_DECODE_COUNT = 8'h54; // Metadata decode operations
          // Result registers (Read-Only, captured on done)
          localparam RESULT_0     = 8'h80; // c_out[0]
          localparam RESULT_1     = 8'h84; // c_out[1]
          localparam RESULT_2     = 8'h88; // c_out[2]
          localparam RESULT_3     = 8'h8C; // c_out[3]
          // DMA control registers (AXI DMA mode only)
          localparam [7:0] DMA_SRC_ADDR      = 8'h90; // Weight Address
          localparam DMA_DST_ADDR = 8'h94; // Destination address (buffer select in MSBs)
          localparam DMA_XFER_LEN = 8'h98; // Transfer length in bytes
          localparam DMA_CTRL     = 8'h9C; // [0]=start (W1P), [1]=busy (RO), [2]=done (R/W1C)
          localparam DMA_BYTES_XFERRED = 8'hB8; // Bytes transferred (RO)
        
          // Activation DMA registers
          localparam [7:0] ACT_DMA_SRC_ADDR  = 8'hA0; // Activation Address
          localparam [7:0] ACT_DMA_LEN       = 8'hA4; // Activation Length
          localparam [7:0] ACT_DMA_CTRL      = 8'hA8; // Activation Start
        
          // Backing regs
%000002   reg        r_irq_en;
%000007   reg [31:0] r_M, r_N, r_K;
%000005   reg [31:0] r_Tm, r_Tn, r_Tk;
%000001   reg [31:0] r_m_idx, r_n_idx, r_k_idx;
%000001   reg        r_bank_sel_wr_A, r_bank_sel_wr_B;
%000002   reg [31:0] r_Sa_bits, r_Sw_bits;
          // UART regs removed
        
          // Sticky status
%000004   reg        st_done_tile;
          // reg        st_err_crc; // Removed
%000005   reg        st_err_illegal;
          
          // Result capture registers
%000000   reg [31:0] r_result_0, r_result_1, r_result_2, r_result_3;
          
          // DMA registers (BSR)
%000002   reg [31:0] r_dma_src_addr;
%000001   reg [31:0] r_dma_dst_addr;
%000002   reg [31:0] r_dma_xfer_len;
%000000   reg        st_dma_done;
        
          // NEW: DMA registers (Activation)
%000001   reg [31:0] r_act_dma_src_addr;
%000001   reg [31:0] r_act_dma_len;
%000000   reg        st_act_dma_done;
        
          // Coverage hooks (for UVM or functional coverage)
          // covergroup cg_csr_write @(posedge clk);
          //   coverpoint csr_addr;
          //   coverpoint csr_wdata;
          // endgroup
          // cg_csr_write cg = new();
        
          // Reset defaults
 000398   always @(posedge clk_gated or negedge rst_n) begin
~000392     if (!rst_n) begin
%000006       r_irq_en         <= 1'b0;
%000006       r_M <= 0; r_N <= 0; r_K <= 0;
%000006       r_Tm <= 0; r_Tn <= 0; r_Tk <= 0;
%000006       r_m_idx <= 0; r_n_idx <= 0; r_k_idx <= 0;
%000006       r_bank_sel_wr_A  <= 1'b0;
%000006       r_bank_sel_wr_B  <= 1'b0;
%000006       r_Sa_bits        <= 32'h3F80_0000; // 1.0f
%000006       r_Sw_bits        <= 32'h3F80_0000; // 1.0f
              // UART defaults removed
%000006       st_done_tile     <= 1'b0;
              // st_err_crc       <= 1'b0; // Removed
%000006       st_err_illegal   <= 1'b0;
%000006       r_result_0       <= 32'd0;
%000006       r_result_1       <= 32'd0;
%000006       r_result_2       <= 32'd0;
%000006       r_result_3       <= 32'd0;
%000006       r_dma_src_addr   <= 32'h0;
%000006       r_dma_dst_addr   <= 32'h0;
%000006       r_dma_xfer_len   <= 32'h0;
%000006       st_dma_done      <= 1'b0;
              
              // NEW: Reset Act DMA regs
%000006       r_act_dma_src_addr <= 32'h0;
%000006       r_act_dma_len      <= 32'h0;
%000006       st_act_dma_done    <= 1'b0;
 000392     end else begin
              // Sticky setters
~000384       if (core_done_tile_pulse) st_done_tile <= 1'b1;
              // if (rx_crc_error)         st_err_crc   <= 1'b1; // Removed
~000392       if (rx_illegal_cmd)       st_err_illegal <= 1'b1;
~000392       if (dma_done_in)          st_dma_done  <= 1'b1;
              // Note: You need an input for act_dma_done_in if you want status tracking
              
              // CSR writes
 000203       if (csr_wen) begin
 000189         case (csr_addr)
 000022           CTRL: begin
 000022             r_irq_en <= csr_wdata[2];
                    // W1P pulses handled below; readback appears as 0
                  end
 000043           DIMS_M:       r_M <= csr_wdata;
%000009           DIMS_N:       r_N <= csr_wdata;
 000016           DIMS_K:       r_K <= csr_wdata;
%000005           TILES_Tm:     r_Tm <= csr_wdata;
 000015           TILES_Tn:     r_Tn <= csr_wdata;
%000005           TILES_Tk:     r_Tk <= csr_wdata;
%000001           INDEX_m:      r_m_idx <= csr_wdata;
%000003           INDEX_n:      r_n_idx <= csr_wdata;
%000001           INDEX_k:      r_k_idx <= csr_wdata;
%000003           BUFF: begin
%000003             r_bank_sel_wr_A <= csr_wdata[0];
%000003             r_bank_sel_wr_B <= csr_wdata[1];
                  end
%000008           SCALE_Sa:     r_Sa_bits <= csr_wdata;
%000004           SCALE_Sw:     r_Sw_bits <= csr_wdata;
                  // UART writes removed
%000004           STATUS: begin
                    // R/W1C clears
%000003             if (csr_wdata[1]) st_done_tile   <= 1'b0;
                    // if (csr_wdata[8]) st_err_crc     <= 1'b0; // Removed
%000003             if (csr_wdata[9]) st_err_illegal <= 1'b0;
                  end
 000011           DMA_SRC_ADDR: r_dma_src_addr <= csr_wdata;
%000001           DMA_DST_ADDR: r_dma_dst_addr <= csr_wdata;
 000019           DMA_XFER_LEN: r_dma_xfer_len <= csr_wdata;
%000004           DMA_CTRL: begin
                    // R/W1C clear done
%000004             if (csr_wdata[2]) st_dma_done <= 1'b0;
                    // Start bit (bit 0) is W1P, handled by w_dma_start logic
                  end
                  // NEW: Correctly write to Act DMA registers
%000004           ACT_DMA_SRC_ADDR: r_act_dma_src_addr <= csr_wdata;
%000004           ACT_DMA_LEN:      r_act_dma_len      <= csr_wdata;
%000003           ACT_DMA_CTRL: begin
%000003             if (csr_wdata[2]) st_act_dma_done <= 1'b0;
                    // Start bit (bit 0) is W1P, handled by w_dma_start logic
                  end
%000004           default: ;
                endcase
              end
            end
          end
        
          // W1P pulse generation (single-cycle)
 000011   wire w_start = (csr_wen && csr_addr==CTRL && csr_wdata[0]);
%000001   wire w_abort = (csr_wen && csr_addr==CTRL && csr_wdata[1]);
%000003   wire w_dma_start = (csr_wen && csr_addr==DMA_CTRL && csr_wdata[0]);
          
          // NEW: Act DMA Start Pulse
%000002   wire w_act_dma_start = (csr_wen && csr_addr==ACT_DMA_CTRL && csr_wdata[0]);
        
          // Illegal start guard (e.g., zero tiles or start while busy)
%000005   wire dims_illegal  = (r_Tm==0 || r_Tn==0 || r_Tk==0);
          assign start_pulse = w_start && !core_busy && !dims_illegal;
          assign abort_pulse = w_abort;
          // set illegal if blocked
 000398   always @(posedge clk_gated or negedge rst_n) begin
~000392     if (!rst_n) begin
              // Reset handled in main always block above
 000392     end else begin
~000381       if (w_start && (core_busy || dims_illegal)) st_err_illegal <= 1'b1;
            end
          end
        
          // Read bank selectors are RO mirrors
          assign bank_sel_rd_A = core_bank_sel_rd_A;
          assign bank_sel_rd_B = core_bank_sel_rd_B;
        
          // Expose config
          assign irq_en         = r_irq_en;
          assign M = r_M;  assign N = r_N;  assign K = r_K;
          assign Tm = r_Tm; assign Tn = r_Tn; assign Tk = r_Tk;
          assign m_idx = r_m_idx; assign n_idx = r_n_idx; assign k_idx = r_k_idx;
          assign bank_sel_wr_A = r_bank_sel_wr_A;
          assign bank_sel_wr_B = r_bank_sel_wr_B;
          assign Sa_bits = r_Sa_bits;  assign Sw_bits = r_Sw_bits;
          // UART assigns removed
          assign dma_src_addr = r_dma_src_addr;
          assign dma_dst_addr = r_dma_dst_addr;
          assign dma_xfer_len = r_dma_xfer_len;
          assign dma_start_pulse = w_dma_start;
        
          // NEW: Expose Act DMA config
          assign act_dma_src_addr = r_act_dma_src_addr;
          assign act_dma_len      = r_act_dma_len;
          assign act_dma_start_pulse = w_act_dma_start;
        
          // Read mux (note CTRL start/abort read as 0)
 025523   always @(*) begin
 025523     unique case (csr_addr)
 001985       CTRL:         csr_rdata = {29'b0, r_irq_en, 2'b00};
 002659       DIMS_M:       csr_rdata = r_M;
 000226       DIMS_N:       csr_rdata = r_N;
 001054       DIMS_K:       csr_rdata = r_K;
 000012       TILES_Tm:     csr_rdata = r_Tm;
 001052       TILES_Tn:     csr_rdata = r_Tn;
 000012       TILES_Tk:     csr_rdata = r_Tk;
%000004       INDEX_m:      csr_rdata = r_m_idx;
 000232       INDEX_n:      csr_rdata = r_n_idx;
%000004       INDEX_k:      csr_rdata = r_k_idx;
 000216       BUFF:         csr_rdata = {22'b0,  // keep tidy for future use
 000216                                  6'b0,
 000216                                  bank_sel_rd_B, bank_sel_rd_A,
 000216                                  r_bank_sel_wr_B, r_bank_sel_wr_A};
 000438       SCALE_Sa:     csr_rdata = r_Sa_bits;
 000230       SCALE_Sw:     csr_rdata = r_Sw_bits;
 000074       STATUS:       csr_rdata = {22'b0,  // reserved
 000074                                  6'b0,
 000074                                  st_err_illegal, 1'b0, st_done_tile, core_busy};
              // Performance monitor registers (Read-Only)
 000223       PERF_TOTAL:   csr_rdata = perf_total_cycles;
%000002       PERF_ACTIVE:  csr_rdata = perf_active_cycles;
%000002       PERF_IDLE:    csr_rdata = perf_idle_cycles;
%000002       PERF_CACHE_HITS:   csr_rdata = perf_cache_hits;
%000002       PERF_CACHE_MISSES: csr_rdata = perf_cache_misses;
%000000       PERF_DECODE_COUNT: csr_rdata = perf_decode_count;
              // Result registers (Read-Only)
%000006       RESULT_0:     csr_rdata = r_result_0;
%000002       RESULT_1:     csr_rdata = r_result_1;
%000002       RESULT_2:     csr_rdata = r_result_2;
%000002       RESULT_3:     csr_rdata = r_result_3;
              // DMA registers
 000440       DMA_SRC_ADDR: csr_rdata = r_dma_src_addr;
%000002       DMA_DST_ADDR: csr_rdata = r_dma_dst_addr;
 007428       DMA_XFER_LEN: csr_rdata = r_dma_xfer_len;
 004396       DMA_CTRL:     csr_rdata = {29'b0, st_dma_done, dma_busy_in, 1'b0};
%000002       DMA_BYTES_XFERRED: csr_rdata = dma_bytes_xferred_in;
              
              // NEW: Act DMA Readback
 000222       ACT_DMA_SRC_ADDR: csr_rdata = r_act_dma_src_addr;
 000254       ACT_DMA_LEN:      csr_rdata = r_act_dma_len;
 004330       ACT_DMA_CTRL:     csr_rdata = {29'b0, st_act_dma_done, 1'b0, 1'b0}; // Busy bit needs input
              
%000008       default:      csr_rdata = 32'hDEAD_BEEF;
            endcase
          end
        
        endmodule
        `default_nettype wire
        
