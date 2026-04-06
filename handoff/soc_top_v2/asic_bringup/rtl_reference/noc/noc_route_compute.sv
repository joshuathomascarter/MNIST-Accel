// =============================================================================
// noc_route_compute.sv — XY Dimension-Order Routing
// =============================================================================
// Given current router (row, col) and destination (dst_row, dst_col),
// compute the output port. XY routing: go X (col) first, then Y (row).
// Guarantees deadlock freedom for minimal routing in a mesh.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDPARAM */
import noc_pkg::*;

module noc_route_compute #(
  parameter int MESH_ROWS = noc_pkg::MESH_ROWS,
  parameter int MESH_COLS = noc_pkg::MESH_COLS
) (
  input  logic [ROW_BITS-1:0]   cur_row,
  input  logic [COL_BITS-1:0]   cur_col,
  input  logic [ROW_BITS-1:0]   dst_row,
  input  logic [COL_BITS-1:0]   dst_col,
  output logic [PORT_BITS-1:0]  out_port,
  output logic                  route_valid
);

  always_comb begin
    route_valid = 1'b1;

    if (dst_col > cur_col) begin
      out_port = PORT_BITS'(PORT_EAST);         // X first: go East
    end else if (dst_col < cur_col) begin
      out_port = PORT_BITS'(PORT_WEST);         // X first: go West
    end else if (dst_row > cur_row) begin
      out_port = PORT_BITS'(PORT_SOUTH);        // Y second: go South
    end else if (dst_row < cur_row) begin
      out_port = PORT_BITS'(PORT_NORTH);        // Y second: go North
    end else begin
      out_port    = PORT_BITS'(PORT_LOCAL);     // Arrived at destination
      route_valid = 1'b1;
    end
  end

endmodule
