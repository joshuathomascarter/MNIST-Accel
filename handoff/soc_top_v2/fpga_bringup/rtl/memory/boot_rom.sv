// Boot ROM - AXI-Lite Read-Only Slave
// Loads firmware.hex at initialization

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off UNUSEDSIGNAL */
import soc_pkg::*;

module boot_rom #(
  parameter int unsigned ADDR_WIDTH = 12,  // 4KB = 2^12 bytes
  parameter int unsigned DATA_WIDTH = 32,
  parameter string INIT_FILE = "firmware.hex"
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Write address channel
  input  logic                    awvalid,
  output logic                    awready,
  input  logic [31:0]             awaddr,
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
  input  logic [7:0]              arlen,
  input  logic [2:0]              arsize,
  input  logic [1:0]              arburst,
  input  logic [3:0]              arid,

  // Read data channel
  output logic                    rvalid,
  input  logic                    rready,
  output logic [DATA_WIDTH-1:0]   rdata,
  output logic [1:0]              rresp,
  output logic [3:0]              rid,
  output logic                    rlast
);

  localparam int unsigned BYTE_DEPTH = 2**ADDR_WIDTH;

  // ROM storage — byte-addressed to match objcopy -O verilog output
  logic [7:0] rom_array [0:BYTE_DEPTH-1];

  // Write response registers (always return error)
  logic [3:0] aw_id;
  logic       aw_valid;

  // Initialize from file
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, rom_array);
    end
  end

  // Write Address Channel - accept but mark for error response
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

  // Write Data Channel - accept but ignore data
  assign wready = wvalid & aw_valid;

  // Write Response Channel - always SLVERR (ROM is read-only)
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
  assign bresp = RESP_SLVERR;

  // Read Address Channel — supports AXI4 bursts
  logic [ADDR_WIDTH-1:0] ar_addr;
  logic [3:0]           ar_id;
  logic                  ar_active;
  logic [7:0]           ar_len;      // remaining beats (0 = last)
  logic [7:0]           ar_beat_cnt; // current beat

  assign arready = !ar_active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_active   <= 1'b0;
      ar_addr     <= '0;
      ar_id       <= '0;
      ar_len      <= '0;
      ar_beat_cnt <= '0;
    end else begin
      if (!ar_active && arvalid) begin
        ar_active   <= 1'b1;
        ar_addr     <= araddr[ADDR_WIDTH-1:0];
        ar_id       <= arid;
        ar_len      <= arlen;
        ar_beat_cnt <= '0;
      end else if (ar_active && rvalid && rready) begin
        if (ar_beat_cnt == ar_len) begin
          ar_active <= 1'b0;  // burst done
        end else begin
          ar_beat_cnt <= ar_beat_cnt + 1;
          ar_addr     <= ar_addr + (ADDR_WIDTH'(DATA_WIDTH / 8));  // INCR burst: advance by word size
        end
      end
    end
  end

  // Read Data Channel — one beat per cycle while active
  assign rvalid = ar_active;
  assign rdata  = {rom_array[{ar_addr[ADDR_WIDTH-1:2], 2'd3}],
                    rom_array[{ar_addr[ADDR_WIDTH-1:2], 2'd2}],
                    rom_array[{ar_addr[ADDR_WIDTH-1:2], 2'd1}],
                    rom_array[{ar_addr[ADDR_WIDTH-1:2], 2'd0}]};
  assign rresp  = RESP_OKAY;
  assign rid    = ar_id;
  assign rlast  = (ar_beat_cnt == ar_len);

endmodule : boot_rom
