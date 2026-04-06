// GPIO Controller - General Purpose I/O with AXI-Lite Interface
// 8 GPIO pins with configurable direction

module gpio_ctrl #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned GPIO_WIDTH = 8
) (
  input  logic                    clk,
  input  logic                    rst_n,
  
  // GPIO I/O
  output logic [GPIO_WIDTH-1:0]   gpio_o,
  input  logic [GPIO_WIDTH-1:0]   gpio_i,
  output logic [GPIO_WIDTH-1:0]   gpio_oe,
  
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
  output logic                    rlast
);

  // Register offsets
  localparam logic [7:0] DIR  = 8'h00;  // Direction: 1=output, 0=input
  localparam logic [7:0] OUT  = 8'h04;  // Output value
  localparam logic [7:0] IN   = 8'h08;  // Input value (read-only)

  // GPIO registers
  logic [GPIO_WIDTH-1:0] dir_reg;
  logic [GPIO_WIDTH-1:0] out_reg;
  logic [GPIO_WIDTH-1:0] in_sync;
  logic [GPIO_WIDTH-1:0] in_sync_r;
  logic [3:0] aw_id, ar_id;
  logic aw_valid, ar_valid;

  // Synchronize inputs with 2-FF filter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_sync_r <= '0;
      in_sync <= '0;
    end else begin
      in_sync_r <= gpio_i;
      in_sync <= in_sync_r;
    end
  end

  // GPIO output assignment
  assign gpio_o = out_reg;
  assign gpio_oe = dir_reg;

  // ===== AXI WRITE PATH =====
  
  assign awready = 1'b1;
  assign wready = awvalid;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_valid <= 1'b0;
      aw_id <= '0;
    end else begin
      if (awvalid && awready) begin
        aw_valid <= 1'b1;
        aw_id <= awid;
      end else if (bvalid && bready) begin
        aw_valid <= 1'b0;
      end
    end
  end

  // Write transaction handler
  always_ff @(posedge clk) begin
    if (wvalid && wready && aw_valid) begin
      case (awaddr[7:0])
        DIR: begin
          if (wstrb[0]) dir_reg <= wdata[GPIO_WIDTH-1:0];
        end
        OUT: begin
          if (wstrb[0]) out_reg <= wdata[GPIO_WIDTH-1:0];
        end
        default: begin
          // IN is read-only
        end
      endcase
    end
  end

  assign bvalid = wvalid & wready & aw_valid;
  assign bresp = 2'b00;
  assign bid = aw_id;

  // ===== AXI READ PATH =====
  
  assign arready = 1'b1;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_valid <= 1'b0;
      ar_id <= '0;
    end else begin
      if (arvalid && arready) begin
        ar_valid <= 1'b1;
        ar_id <= arid;
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
    case (araddr[7:0])
      DIR: rdata = {24'b0, dir_reg};
      OUT: rdata = {24'b0, out_reg};
      IN:  rdata = {24'b0, in_sync};
      default: rdata = '0;
    endcase
  end

endmodule : gpio_ctrl
