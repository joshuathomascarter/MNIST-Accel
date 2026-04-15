// =============================================================================
// sram_1rw_wrapper_bb.v — Yosys blackbox declaration
// =============================================================================
// Parameterized port-only stub so Yosys accepts instantiations with any
// DATA_W / ADDR_W without inferring logic.  The (* blackbox *) attribute
// tells Yosys: "I know this module; do NOT elaborate internals."
//
// Physical timing arcs come from macros/sram_1rw_wrapper.lib (OpenSTA).
// Geometry comes from macros/sram_1rw_wrapper.lef (OpenROAD P&R).
// SYNTH_READ_BLACKBOX_LIB is disabled so the Liberty cell does not
// shadow this parameterized definition during Yosys elaboration.
// =============================================================================

// Non-parameterized blackbox stub — fixed widths match Liberty cell exactly.
// DATA_W=32 (32-bit words), ADDR_W=12 (DEPTH=4096) — only dimensions used in
// this design.  Parameters omitted so Yosys does not conflict with Liberty.
(* blackbox *)
module sram_1rw_wrapper (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        we,
    input  wire [11:0] addr,
    input  wire [31:0] wdata,
    output wire [31:0] rdata
);
endmodule
