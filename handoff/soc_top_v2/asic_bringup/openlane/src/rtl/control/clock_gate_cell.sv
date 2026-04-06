// =============================================================================
// clock_gate_cell.sv — ICG-friendly clock-gate wrapper
// =============================================================================
// FPGA builds bypass the gate and keep downstream clock-enable logic active.
// ASIC builds can hook a library cell by defining ASIC_TECH_ICG plus
// CLOCK_GATE_CELL_MODULE and, if needed, CLOCK_GATE_CELL_PORTS.
// Without those overrides the wrapper falls back to a latch-based gate model.

`default_nettype none

module clock_gate_cell (
  input  logic clk_i,
  input  logic en_i,
  input  logic test_en_i,
  output logic clk_o
);

`ifdef ASIC_TECH_ICG
  `ifdef CLOCK_GATE_CELL_MODULE
    `ifndef CLOCK_GATE_CELL_PORTS
      `define CLOCK_GATE_CELL_PORTS \
        .CLK(clk_i), \
        .GATE(en_i), \
        .SCE(test_en_i), \
        .GCLK(clk_o)
    `endif

    `CLOCK_GATE_CELL_MODULE u_icg (
      `CLOCK_GATE_CELL_PORTS
    );
  `else
    logic gate_en_latched;

    always_latch begin
      if (!clk_i)
        gate_en_latched = en_i | test_en_i;
    end

    assign clk_o = clk_i & gate_en_latched;
  `endif
`elsif XILINX_FPGA
  assign clk_o = clk_i;
`else
  logic gate_en_latched;

  always_latch begin
    if (!clk_i)
      gate_en_latched = en_i | test_en_i;
  end

  assign clk_o = clk_i & gate_en_latched;
`endif

endmodule

`default_nettype wire
