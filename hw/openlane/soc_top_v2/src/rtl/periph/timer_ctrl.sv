// Timer Controller - RISC-V Machine Timer Compatible
// Implements mtime and mtimecmp per RISC-V Privileged Spec

/* verilator lint_off UNUSEDSIGNAL */
module timer_ctrl #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,
  
  // AXI-Lite Slave Interface
  input  logic                    awvalid,
  output logic                    awready,
  input  logic [ADDR_WIDTH-1:0]   awaddr,
  input  logic [2:0]              awsize,
  input  logic [1:0]              awburst,
  input  logic [3:0]              awid,
  
  input  logic                    wvalid,
  output logic                    wready,
  input  logic [DATA_WIDTH-1:0]   wdata,
  input  logic [DATA_WIDTH/8-1:0] wstrb,
  input  logic                    wlast,
  
  output logic                    bvalid,
  input  logic                    bready,
  output logic [1:0]              bresp,
  output logic [3:0]              bid,
  
  input  logic                    arvalid,
  output logic                    arready,
  input  logic [ADDR_WIDTH-1:0]   araddr,
  input  logic [2:0]              arsize,
  input  logic [1:0]              arburst,
  input  logic [3:0]              arid,
  
  output logic                    rvalid,
  input  logic                    rready,
  output logic [DATA_WIDTH-1:0]   rdata,
  output logic [1:0]              rresp,
  output logic [3:0]              rid,
  output logic                    rlast,
  
  output logic                    irq_timer_o
);

  // Register offsets
  localparam logic [7:0] MTIME_LO    = 8'h00;
  localparam logic [7:0] MTIME_HI    = 8'h04;
  localparam logic [7:0] MTIMECMP_LO = 8'h08;
  localparam logic [7:0] MTIMECMP_HI = 8'h0C;

  // 64-bit mtime counter and mtimecmp register
  logic [63:0] mtime;
  logic [63:0] mtimecmp;
  logic [3:0] aw_id, ar_id;
  logic [7:0] ar_addr_r;
  logic b_pending, ar_valid;

  // ===== AUTO-INCREMENTING mtime + AXI WRITE PATH (merged to avoid multi-driver) =====

  // ===== Timer Interrupt =====
  
  assign irq_timer_o = (mtime >= mtimecmp);

  // ===== AXI WRITE PATH =====
  
  assign awready = !b_pending;
  assign wready = awvalid && !b_pending;

  // Combined: auto-increment mtime and handle AXI register writes.
  // Single always_ff block avoids multi-driver on mtime and mtimecmp.
  // Non-blocking semantics: "mtime <= mtime+1" fires every cycle; any
  // byte-lane write to MTIME_LO/HI on the same cycle overrides those bits
  // because the slice assignment is scheduled last (last-wins for NBAs).
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      b_pending <= 1'b0;
      aw_id    <= '0;
      mtime    <= 64'b0;
      mtimecmp <= 64'hFFFF_FFFF_FFFF_FFFF;
    end else begin
      // B response handshake
      if (b_pending && bready) begin
        b_pending <= 1'b0;
      end

      // Auto-increment every cycle
      mtime <= mtime + 64'b1;

      // AXI write (AW+W arrive together)
      if (awvalid && wvalid && !b_pending) begin
        b_pending <= 1'b1;
        aw_id <= awid;
        case (awaddr[7:0])
          MTIME_LO: begin
            if (wstrb[0]) mtime[7:0]   <= wdata[7:0];
            if (wstrb[1]) mtime[15:8]  <= wdata[15:8];
            if (wstrb[2]) mtime[23:16] <= wdata[23:16];
            if (wstrb[3]) mtime[31:24] <= wdata[31:24];
          end
          MTIME_HI: begin
            if (wstrb[0]) mtime[39:32] <= wdata[7:0];
            if (wstrb[1]) mtime[47:40] <= wdata[15:8];
            if (wstrb[2]) mtime[55:48] <= wdata[23:16];
            if (wstrb[3]) mtime[63:56] <= wdata[31:24];
          end
          MTIMECMP_LO: begin
            if (wstrb[0]) mtimecmp[7:0]   <= wdata[7:0];
            if (wstrb[1]) mtimecmp[15:8]  <= wdata[15:8];
            if (wstrb[2]) mtimecmp[23:16] <= wdata[23:16];
            if (wstrb[3]) mtimecmp[31:24] <= wdata[31:24];
          end
          MTIMECMP_HI: begin
            if (wstrb[0]) mtimecmp[39:32] <= wdata[7:0];
            if (wstrb[1]) mtimecmp[47:40] <= wdata[15:8];
            if (wstrb[2]) mtimecmp[55:48] <= wdata[23:16];
            if (wstrb[3]) mtimecmp[63:56] <= wdata[31:24];
          end
          default: ;
        endcase
      end
    end
  end

  assign bvalid = b_pending;
  assign bresp = 2'b00;
  assign bid = aw_id;

  // ===== AXI READ PATH =====
  
  assign arready = 1'b1;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_valid <= 1'b0;
      ar_id <= '0;
      ar_addr_r <= '0;
    end else begin
      if (arvalid && arready) begin
        ar_valid <= 1'b1;
        ar_id <= arid;
        ar_addr_r <= araddr[7:0];
      end else if (rvalid && rready) begin
        ar_valid <= 1'b0;
      end
    end
  end

  // Read response
  assign rvalid = ar_valid;
  assign rid = ar_id;
  assign rresp = 2'b00;
  assign rlast = 1'b1;

  always_comb begin
    case (ar_addr_r)
      MTIME_LO: rdata = mtime[31:0];
      MTIME_HI: rdata = mtime[63:32];
      MTIMECMP_LO: rdata = mtimecmp[31:0];
      MTIMECMP_HI: rdata = mtimecmp[63:32];
      default: rdata = '0;
    endcase
  end

endmodule : timer_ctrl
