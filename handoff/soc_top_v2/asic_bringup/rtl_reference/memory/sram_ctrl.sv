// SRAM Controller - AXI-Lite Slave with Byte Enables
// 32KB SRAM for CPU stack, globals, and working data

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

  localparam int unsigned DEPTH = 2**(ADDR_WIDTH - 2);  // Word-addressable

  // Dual-port SRAM
  logic [DATA_WIDTH-1:0] sram [0:DEPTH-1];
  
  // Write path
  logic [ADDR_WIDTH-3:0] aw_addr_cur;  // burst-aware write address (increments per beat)
  logic [3:0]            aw_id;
  logic                  aw_valid;
  logic                  w_valid;
  
  // Read path
  logic [ADDR_WIDTH-3:0] ar_addr;
  logic [3:0]           ar_id;
  logic                  ar_valid;
  logic [7:0]            ar_len;     // burst length (0-based)
  logic [7:0]            ar_beat_cnt; // current beat

  // Initialize SRAM with zeros
  initial begin
    for (int i = 0; i < DEPTH; i++) begin
      sram[i] = '0;
    end
  end

  // ===== WRITE PATH =====

  // Write Address Channel
  assign awready = !aw_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_valid   <= 1'b0;
      aw_addr_cur <= '0;
      aw_id      <= '0;
    end else begin
      if (awvalid && awready) begin
        aw_valid    <= 1'b1;
        aw_addr_cur <= awaddr[ADDR_WIDTH-1:2];  // latch word-aligned base
        aw_id       <= awid;
      end else if (wvalid && wready) begin
        // Advance write address each accepted beat (INCR burst support)
        aw_addr_cur <= aw_addr_cur + 1;
        if (wlast) aw_valid <= 1'b0;            // release after final beat
      end
    end
  end

  // Write Data Channel
  // wready = aw_valid: slave is ready as soon as AW is registered,
  // decoupled from wvalid to avoid a combinatorial loop through the
  // crossbar's wready back-propagation path.
  assign wready = aw_valid;

  always_ff @(posedge clk) begin
    if (wvalid && wready) begin               // AXI W handshake
      for (int i = 0; i < DATA_WIDTH/8; i++) begin
        if (wstrb[i]) begin
          sram[aw_addr_cur][i*8+:8] <= wdata[i*8+:8];
        end
      end
    end
  end

  // Write Response Channel
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

  // ===== READ PATH =====

  // Read Address Channel
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
      end else if (ar_valid && rready) begin
        if (ar_beat_cnt == ar_len) begin
          ar_valid <= 1'b0;
        end else begin
          ar_beat_cnt <= ar_beat_cnt + 1;
          ar_addr     <= ar_addr + 1;  // INCR burst: next word
        end
      end
    end
  end

  // Read Data Channel - burst capable
  assign rvalid = ar_valid;
  assign rdata  = sram[ar_addr];
  assign rresp  = RESP_OKAY;
  assign rid    = ar_id;
  assign rlast  = (ar_beat_cnt == ar_len);

endmodule : sram_ctrl
