// Boot ROM - AXI-Lite Read-Only Slave
// Loads firmware.hex at initialization

module boot_rom #(
  parameter int unsigned ADDR_WIDTH = 13,  // 8KB = 2^13 bytes
  parameter int unsigned DATA_WIDTH = 32,
  parameter INIT_FILE = "firmware.hex"
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

  localparam int unsigned DEPTH = 2**ADDR_WIDTH / (DATA_WIDTH/8);

  // ROM storage
  logic [DATA_WIDTH-1:0] rom_array [0:DEPTH-1];
  
  // Read port registers
  logic [ADDR_WIDTH-1:0] ar_addr;
  logic [3:0]           ar_id;
  logic                  ar_valid;
  
  // Write response registers (always return error)
  logic [3:0] aw_id;
  logic       aw_valid;

  // Initialize from file
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, rom_array);
    end
  end

  // Write Address Channel - Accept but mark for error response
  assign awready = 1'b1;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_valid <= 1'b0;
      aw_id    <= '0;
    end else begin
      if (awvalid && awready) begin
        aw_valid <= 1'b1;
        aw_id    <= awid;
      end else if (bvalid && bready) begin
        aw_valid <= 1'b0;
      end
    end
  end

  // Write Data Channel - Accept but ignore
  assign wready = wvalid & aw_valid;

  // Write Response Channel - Always return SLVERR (ROM is read-only)
  assign bvalid = aw_valid & wvalid & wlast;
  assign bresp  = 2'b10;  // SLVERR
  assign bid    = aw_id;

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
        ar_addr  <= araddr[ADDR_WIDTH-1:0];
        ar_id    <= arid;
      end else if (rvalid && rready) begin
        ar_valid <= 1'b0;
      end
    end
  end

  // Read Data Channel - 1 cycle latency
  assign rvalid = ar_valid;
  assign rdata  = rom_array[ar_addr];
  assign rresp  = 2'b00;  // OKAY
  assign rid    = ar_id;
  assign rlast  = 1'b1;   // Single beat response

endmodule : boot_rom
