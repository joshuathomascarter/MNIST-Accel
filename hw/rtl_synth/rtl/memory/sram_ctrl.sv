// SRAM Controller - AXI-Lite Slave with Byte Enables
// 32KB dual-port SRAM for code/data

module sram_ctrl #(
  parameter int unsigned ADDR_WIDTH = 15,  // 32KB = 2^15 bytes
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,
  
  // AXI-Lite Slave Interface
  input  logic                    awvalid,
  output logic                    awready,
  input  logic [31:0]             awaddr,
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
  input  logic [31:0]             araddr,
  input  logic [2:0]              arsize,
  input  logic [1:0]              arburst,
  input  logic [3:0]              arid,
  
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
  logic [ADDR_WIDTH-3:0] aw_addr;
  logic [3:0]           aw_id;
  logic                  aw_valid;
  logic                  w_valid;
  
  // Read path
  logic [ADDR_WIDTH-3:0] ar_addr;
  logic [3:0]           ar_id;
  logic                  ar_valid;

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
      aw_valid <= 1'b0;
      aw_addr  <= '0;
      aw_id    <= '0;
    end else begin
      if (awvalid && awready) begin
        aw_valid <= 1'b1;
        aw_addr  <= awaddr[ADDR_WIDTH-1:2];
        aw_id    <= awid;
      end else if (wvalid && wready && wlast) begin
        aw_valid <= 1'b0;
      end
    end
  end

  // Write Data Channel
  assign wready = aw_valid & wvalid;
  
  always_ff @(posedge clk) begin
    if (wready) begin
      // Apply byte enables
      for (int i = 0; i < DATA_WIDTH/8; i++) begin
        if (wstrb[i]) begin
          sram[aw_addr][i*8+:8] <= wdata[i*8+:8];
        end
      end
    end
  end

  // Write Response Channel
  assign bvalid = wvalid & wready & wlast & aw_valid;
  assign bresp  = 2'b00;  // OKAY
  assign bid    = aw_id;

  // ===== READ PATH =====
  
  // Read Address Channel
  assign arready = !ar_valid;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_valid <= 1'b0;
      ar_addr  <= '0;
      ar_id    <= '0;
    end else begin
      if (arvalid && arready) begin
        ar_valid <= 1'b1;
        ar_addr  <= araddr[ADDR_WIDTH-1:2];
        ar_id    <= arid;
      end else if (rvalid && rready) begin
        ar_valid <= 1'b0;
      end
    end
  end

  // Read Data Channel - 1 cycle latency
  assign rvalid = ar_valid;
  assign rdata  = sram[ar_addr];
  assign rresp  = 2'b00;  // OKAY
  assign rid    = ar_id;
  assign rlast  = 1'b1;   // Single beat response

endmodule : sram_ctrl
