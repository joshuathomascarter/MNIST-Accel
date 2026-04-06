// =============================================================================
// noc_input_port.sv — Router Input Port with Per-VC Buffers
// =============================================================================
// Each of the 5 input ports has NUM_VCS virtual-channel buffers.
// Head flit is decoded to determine output port (via route_compute).
// Buffers are standard FIFOs; credits are sent back when a flit is consumed.
//
// Interface:
//   upstream  → flit_in / flit_valid_in / credit_out (one per VC)
//   internal  → per-VC: head flit, buffer status, route result
//               flit read is triggered by switch allocator grant

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module noc_input_port #(
  parameter int NUM_VCS   = noc_pkg::NUM_VCS,
  parameter int BUF_DEPTH = noc_pkg::BUF_DEPTH,
  parameter int MESH_ROWS = noc_pkg::MESH_ROWS,
  parameter int MESH_COLS = noc_pkg::MESH_COLS
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Upstream interface ---
  input  flit_t                  flit_in,
  input  logic                   flit_valid_in,
  output logic [NUM_VCS-1:0]     credit_out,       // One credit per VC freed

  // --- Current router coordinates (for routing) ---
  input  logic [ROW_BITS-1:0]    cur_row,
  input  logic [COL_BITS-1:0]    cur_col,

  // --- Per-VC status to allocator/switch stages ---
  output logic [NUM_VCS-1:0]     vc_has_flit,      // Buffer non-empty
  output logic [NUM_VCS-1:0]     vc_has_head,      // Head-of-line is HEAD flit
  output logic [PORT_BITS-1:0]   vc_route [NUM_VCS],  // Computed output port for head
  output flit_t                  vc_head_flit [NUM_VCS],

  // --- Read interface (from switch allocator) ---
  input  logic [NUM_VCS-1:0]     vc_read,          // One-hot: consume flit from this VC
  output flit_t                  read_flit,         // The consumed flit
  output logic [VC_BITS-1:0]     read_vc            // Which VC was read
);

  // ---------------------------------------------------------------------------
  // Per-VC FIFO storage
  // ---------------------------------------------------------------------------
  flit_t fifo_mem [NUM_VCS][BUF_DEPTH];
  logic [$clog2(BUF_DEPTH)-1:0] wr_ptr [NUM_VCS];
  logic [$clog2(BUF_DEPTH)-1:0] rd_ptr [NUM_VCS];
  logic [$clog2(BUF_DEPTH+1)-1:0] count [NUM_VCS];

  // Which VC does the incoming flit target?
  logic [VC_BITS-1:0] in_vc;
  assign in_vc = flit_in.vc_id;

  // ---------------------------------------------------------------------------
  // Write logic (incoming flit → correct VC buffer)
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int v = 0; v < NUM_VCS; v++) begin
        wr_ptr[v] <= '0;
        rd_ptr[v] <= '0;
        count[v]  <= '0;
      end
    end else begin
      // Write
      if (flit_valid_in) begin
        fifo_mem[in_vc][wr_ptr[in_vc]] <= flit_in;
        wr_ptr[in_vc] <= (wr_ptr[in_vc] == $clog2(BUF_DEPTH)'(BUF_DEPTH - 1))
                         ? '0 : wr_ptr[in_vc] + 1;
      end

      // Read
      for (int v = 0; v < NUM_VCS; v++) begin
        if (vc_read[v]) begin
          rd_ptr[v] <= (rd_ptr[v] == $clog2(BUF_DEPTH)'(BUF_DEPTH - 1))
                       ? '0 : rd_ptr[v] + 1;
        end
      end

      // Count update
      for (int v = 0; v < NUM_VCS; v++) begin
        logic wr_this, rd_this;
        wr_this = flit_valid_in && (in_vc == VC_BITS'(v));
        rd_this = vc_read[v];
        case ({wr_this, rd_this})
          2'b10:   count[v] <= count[v] + 1;
          2'b01:   count[v] <= count[v] - 1;
          default: count[v] <= count[v];
        endcase
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Credit generation — send credit when a slot is freed
  // ---------------------------------------------------------------------------
  always_comb begin
    credit_out = '0;
    for (int v = 0; v < NUM_VCS; v++)
      credit_out[v] = vc_read[v];
  end

  // ---------------------------------------------------------------------------
  // Head-of-line peek + route compute (per VC)
  // ---------------------------------------------------------------------------
  generate
    for (genvar v = 0; v < NUM_VCS; v++) begin : gen_vc

      assign vc_has_flit[v] = (count[v] != '0);

      // Head flit at read pointer
      assign vc_head_flit[v] = fifo_mem[v][rd_ptr[v]];

      // Is it a HEAD or HEAD_TAIL flit?
      assign vc_has_head[v] = vc_has_flit[v] &&
                              (fifo_mem[v][rd_ptr[v]].flit_type == FLIT_HEAD ||
                               fifo_mem[v][rd_ptr[v]].flit_type == FLIT_HEADTAIL);

      // Route compute for head flit
      logic [ROW_BITS-1:0] dst_row_v;
      logic [COL_BITS-1:0] dst_col_v;
      logic [PORT_BITS-1:0] route_port_v;
      logic route_valid_v;

      assign dst_row_v = node_row(fifo_mem[v][rd_ptr[v]].dst_id);
      assign dst_col_v = node_col(fifo_mem[v][rd_ptr[v]].dst_id);

      noc_route_compute #(
        .MESH_ROWS (MESH_ROWS),
        .MESH_COLS (MESH_COLS)
      ) u_rc (
        .cur_row     (cur_row),
        .cur_col     (cur_col),
        .dst_row     (dst_row_v),
        .dst_col     (dst_col_v),
        .out_port    (route_port_v),
        .route_valid (route_valid_v)
      );

      assign vc_route[v] = route_port_v;

    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Read mux — which VC is being read?
  // ---------------------------------------------------------------------------
  always_comb begin
    read_flit = '0;
    read_vc   = '0;
    for (int v = 0; v < NUM_VCS; v++) begin
      if (vc_read[v]) begin
        read_flit = fifo_mem[v][rd_ptr[v]];
        read_vc   = VC_BITS'(v);
      end
    end
  end

endmodule
