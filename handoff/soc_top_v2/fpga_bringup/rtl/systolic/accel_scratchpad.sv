// =============================================================================
// accel_scratchpad.sv — Per-Tile Scratchpad SRAM
// =============================================================================
// Simple dual-port SRAM for tile-local activation/weight/output storage.
// Each tile has its own scratchpad to avoid coherence overhead.
//
// Three logical banks:
//   Bank 0: Activations  (ACT_BASE)
//   Bank 1: Weights/BSR  (WGT_BASE)
//   Bank 2: Output/Accum (OUT_BASE)
//
// Accessed by tile controller (DMA fill) and systolic array (compute).

module accel_scratchpad #(
  parameter int DEPTH      = 4096,    // Total words
  parameter int DATA_WIDTH = 32,      // 32-bit words (matches NoC DMA payload)
  parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   clk_en,

  // --- Port A: Tile controller / DMA (read/write) ---
  input  logic                   a_en,
  input  logic                   a_we,
  input  logic [ADDR_WIDTH-1:0]  a_addr,
  input  logic [DATA_WIDTH-1:0]  a_wdata,
  output logic [DATA_WIDTH-1:0]  a_rdata,

  // --- Port B: Systolic array (read-only) ---
  input  logic                   b_en,
  input  logic [ADDR_WIDTH-1:0]  b_addr,
  output logic [DATA_WIDTH-1:0]  b_rdata
);

  // Simple dual-port RAM inference
`ifdef XILINX_FPGA
  (* ram_style = "block" *) logic [DATA_WIDTH-1:0] mem [DEPTH];
`else
  logic [DATA_WIDTH-1:0] mem [DEPTH];
`endif

  // Port A: read/write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_rdata <= '0;
    end else if (clk_en && a_en) begin
      if (a_we)
        mem[a_addr] <= a_wdata;
      a_rdata <= mem[a_addr];
    end
  end

  // Port B: read-only
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      b_rdata <= '0;
    end else if (clk_en && b_en)
      b_rdata <= mem[b_addr];
  end

endmodule
