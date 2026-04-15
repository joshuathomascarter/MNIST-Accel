// SRAM Controller - AXI-Lite Slave with Byte Enables
// 32KB SRAM for CPU stack, globals, and working data
//
// Uses two sram_1rw_wrapper blackbox macros (32-bit × 4096-word = 16KB each)
// for synthesis/P&R.  Bank 0 = lower 16KB (word-addr[12]=0),
// Bank 1 = upper 16KB (word-addr[12]=1).
// Synchronous SRAM adds 1-cycle read latency; rvalid is delayed accordingly.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off UNUSEDSIGNAL */
import soc_pkg::*;

module sram_ctrl #(
  parameter int unsigned ADDR_WIDTH = 15,  // 32KB = 2^15 bytes
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Write address channel
  input  logic                    awvalid,
  output logic                    awready,
  input  logic [31:0]             awaddr,
  input  logic [7:0]              awlen,
  input  logic [2:0]              awsize,
  input  logic [1:0]              awburst,
  input  logic [3:0]              awid,

  // Write data channel
  input  logic                    wvalid,
  output logic                    wready,
  input  logic [DATA_WIDTH-1:0]   wdata,
  input  logic [DATA_WIDTH/8-1:0] wstrb,
  input  logic                    wlast,

  // Write response channel
  output logic                    bvalid,
  input  logic                    bready,
  output logic [1:0]              bresp,
  output logic [3:0]              bid,

  // Read address channel
  input  logic                    arvalid,
  output logic                    arready,
  input  logic [31:0]             araddr,
  input  logic [2:0]              arsize,
  input  logic [1:0]              arburst,
  input  logic [3:0]              arid,
  input  logic [7:0]              arlen,

  // Read data channel
  output logic                    rvalid,
  input  logic                    rready,
  output logic [DATA_WIDTH-1:0]   rdata,
  output logic [1:0]              rresp,
  output logic [3:0]              rid,
  output logic                    rlast
);

  // Word-address width: 13 bits (8192 words × 4 bytes = 32KB)
  // Each bank: 12 bits (4096 words × 4 bytes = 16KB)
  localparam int unsigned WORD_W = ADDR_WIDTH - 2;  // 13

  // ── Write path ──────────────────────────────────────────────────────────
  logic [WORD_W-1:0] aw_addr_cur;
  logic [3:0]        aw_id;
  logic              aw_valid;

  assign awready = !aw_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_valid    <= 1'b0;
      aw_addr_cur <= '0;
      aw_id       <= '0;
    end else begin
      if (awvalid && awready) begin
        aw_valid    <= 1'b1;
        aw_addr_cur <= awaddr[ADDR_WIDTH-1:2];
        aw_id       <= awid;
      end else if (wvalid && wready) begin
        aw_addr_cur <= aw_addr_cur + 1;
        if (wlast) aw_valid <= 1'b0;
      end
    end
  end

  assign wready = aw_valid;

  // SRAM write enables per bank
  logic        sram_we;
  logic        sram_w_bank;   // which bank the write targets
  logic [11:0] sram_waddr;

  assign sram_we     = wvalid && wready;
  assign sram_w_bank = aw_addr_cur[12];
  assign sram_waddr  = aw_addr_cur[11:0];

  // Write response
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bvalid <= 1'b0;
      bid    <= '0;
    end else begin
      if (wvalid && wready && wlast) begin
        bvalid <= 1'b1;
        bid    <= aw_id;
      end else if (bready) begin
        bvalid <= 1'b0;
      end
    end
  end
  assign bresp = RESP_OKAY;

  // ── Read path ────────────────────────────────────────────────────────────
  logic [WORD_W-1:0] ar_addr;
  logic [3:0]        ar_id;
  logic              ar_valid;
  logic [7:0]        ar_len;
  logic [7:0]        ar_beat_cnt;

  assign arready = !ar_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_valid    <= 1'b0;
      ar_addr     <= '0;
      ar_id       <= '0;
      ar_len      <= '0;
      ar_beat_cnt <= '0;
    end else begin
      if (arvalid && arready) begin
        ar_valid    <= 1'b1;
        ar_addr     <= araddr[ADDR_WIDTH-1:2];
        ar_id       <= arid;
        ar_len      <= arlen;
        ar_beat_cnt <= '0;
      end else if (ar_valid && rready && rvalid) begin
        if (ar_beat_cnt == ar_len) begin
          ar_valid <= 1'b0;
        end else begin
          ar_beat_cnt <= ar_beat_cnt + 1;
          ar_addr     <= ar_addr + 1;
        end
      end
    end
  end

  // SRAM read enable per bank
  logic        sram_r_bank;
  logic [11:0] sram_raddr;

  assign sram_r_bank = ar_addr[12];
  assign sram_raddr  = ar_addr[11:0];

  // ── SRAM macro instances ─────────────────────────────────────────────────
  logic [DATA_WIDTH-1:0] sram0_rdata, sram1_rdata;

  sram_1rw_wrapper u_sram_bank0 (
    .clk   (clk),
    .rst_n (rst_n),
    .en    ((ar_valid && !sram_r_bank) | (sram_we && !sram_w_bank)),
    .we    (sram_we && !sram_w_bank),
    .addr  ((sram_we && !sram_w_bank) ? sram_waddr : sram_raddr),
    .wdata (wdata),
    .rdata (sram0_rdata)
  );

  sram_1rw_wrapper u_sram_bank1 (
    .clk   (clk),
    .rst_n (rst_n),
    .en    ((ar_valid && sram_r_bank) | (sram_we && sram_w_bank)),
    .we    (sram_we && sram_w_bank),
    .addr  ((sram_we && sram_w_bank) ? sram_waddr : sram_raddr),
    .wdata (wdata),
    .rdata (sram1_rdata)
  );

  // ── Read output pipeline (1-cycle SRAM latency) ──────────────────────────
  logic        rvalid_r;
  logic [3:0]  rid_r;
  logic        rlast_r;
  logic        rbank_r;   // registered bank select tracks data output

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rvalid_r <= 1'b0;
      rid_r    <= '0;
      rlast_r  <= 1'b0;
      rbank_r  <= 1'b0;
    end else begin
      rvalid_r <= ar_valid;
      rid_r    <= ar_id;
      rlast_r  <= (ar_beat_cnt == ar_len);
      rbank_r  <= sram_r_bank;
    end
  end

  assign rvalid = rvalid_r;
  assign rdata  = rbank_r ? sram1_rdata : sram0_rdata;
  assign rresp  = RESP_OKAY;
  assign rid    = rid_r;
  assign rlast  = rlast_r;

endmodule : sram_ctrl
