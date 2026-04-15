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

  // -------------------------------------------------------------------------
  // Dual-port scratchpad using two sram_1rw_wrapper blackbox instances.
  // Port A (RW) drives the write side; both ports share reads via separate
  // macros (write-through coherence handled in the controller).
  // -------------------------------------------------------------------------

  // Blackbox declaration — body excluded during synthesis
`ifndef SYNTHESIS
  // simulation stub: infer simple arrays when not synthesizing
  logic [DATA_WIDTH-1:0] _sim_mem [DEPTH];
  always_ff @(posedge clk) begin
    if (clk_en && a_en) begin
      if (a_we) _sim_mem[a_addr] <= a_wdata;
      a_rdata <= _sim_mem[a_addr];
    end
    if (clk_en && b_en) b_rdata <= _sim_mem[b_addr];
  end
`else
  // No module-level parameters — sram_1rw_wrapper is non-parameterized to
  // match the flat Liberty cell that Yosys reads from EXTRA_LIBS.
  // Widths are fixed: DATA_W=32, ADDR_W=12 (DEPTH=4096 always in this design).
  sram_1rw_wrapper u_sram_a (
    .clk   (clk),
    .rst_n (rst_n),
    .en    (clk_en & a_en),
    .we    (a_we),
    .addr  (a_addr),
    .wdata (a_wdata),
    .rdata (a_rdata)
  );

  sram_1rw_wrapper u_sram_b (
    .clk   (clk),
    .rst_n (rst_n),
    .en    (clk_en & b_en),
    .we    (1'b0),
    .addr  (b_addr),
    .wdata (32'b0),
    .rdata (b_rdata)
  );
`endif

endmodule
