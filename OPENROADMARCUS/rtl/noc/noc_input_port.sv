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
  // Under SYNTHESIS: per-VC sram_1rw_wrapper blackboxes (no behavioral array).
  // Otherwise:       explicit registered FIFO (Verilator / FPGA simulation).
  // ---------------------------------------------------------------------------
  logic [NUM_VCS-1:0][$clog2(BUF_DEPTH)-1:0]   wr_ptr;
  logic [NUM_VCS-1:0][$clog2(BUF_DEPTH)-1:0]   rd_ptr;
  logic [NUM_VCS-1:0][$clog2(BUF_DEPTH+1)-1:0] count;

  // Which VC does the incoming flit target?
  logic [VC_BITS-1:0] in_vc;
  assign in_vc = flit_in.vc_id;

`ifndef SYNTHESIS
  // -------------------------------------------------------------------------
  // Behavioral FIFO — Verilator / FPGA path
  // -------------------------------------------------------------------------
  flit_t fifo_mem [NUM_VCS][BUF_DEPTH];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
      count  <= '0;
    end else begin
      // Write — explicit loop avoids double-variable-indexed write to unpacked array
      if (flit_valid_in) begin
        for (int v2 = 0; v2 < NUM_VCS; v2++)
          for (int b2 = 0; b2 < BUF_DEPTH; b2++)
            if (v2 == in_vc && b2 == wr_ptr[v2])
              fifo_mem[v2][b2] <= flit_in;
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

`else
  // -------------------------------------------------------------------------
  // Synthesis path — sram_1rw_wrapper blackboxes, one per VC
  // Each SRAM holds BUF_DEPTH flit entries split across two 32-bit words.
  // Pointer and count logic is identical; only the storage differs.
  // -------------------------------------------------------------------------
  // Pointer / count updates
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
      count  <= '0;
    end else begin
      if (flit_valid_in) begin
        wr_ptr[in_vc] <= (wr_ptr[in_vc] == $clog2(BUF_DEPTH)'(BUF_DEPTH - 1))
                         ? '0 : wr_ptr[in_vc] + 1;
      end
      for (int v = 0; v < NUM_VCS; v++) begin
        if (vc_read[v])
          rd_ptr[v] <= (rd_ptr[v] == $clog2(BUF_DEPTH)'(BUF_DEPTH - 1))
                       ? '0 : rd_ptr[v] + 1;
      end
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

  // Per-VC SRAM instances (flit_t = 64 bits = 2 × 32-bit words)
  // Word 0 = flit[31:0], word 1 = flit[63:32]; addr[0] selects word within flit.
  // head_flit[v] output below uses registered read (1-cycle latency = vc_head_flit).
  generate
    for (genvar vs = 0; vs < NUM_VCS; vs++) begin : gen_vc_sram
      wire        sram_we_lo  = flit_valid_in && (in_vc == VC_BITS'(vs));
      wire        sram_we_hi  = sram_we_lo;
      wire [11:0] sram_wa     = {10'h0, wr_ptr[vs]};
      wire [11:0] sram_ra     = {10'h0, rd_ptr[vs]};
      // During write use write addr; during read use read addr (non-overlapping by FIFO invariant)
      wire [11:0] sram_addr_lo = sram_we_lo ? sram_wa : sram_ra;
      wire [11:0] sram_addr_hi = sram_we_hi ? sram_wa : sram_ra;
      wire [31:0] sram_rdata_lo, sram_rdata_hi;

      sram_1rw_wrapper u_vc_lo (
        .clk   (clk), .rst_n (rst_n),
        .en    (1'b1),
        .we    (sram_we_lo),
        .addr  (sram_addr_lo),
        .wdata (flit_in[31:0]),
        .rdata (sram_rdata_lo)
      );
      sram_1rw_wrapper u_vc_hi (
        .clk   (clk), .rst_n (rst_n),
        .en    (1'b1),
        .we    (sram_we_hi),
        .addr  (sram_addr_hi),
        .wdata (flit_in[63:32]),
        .rdata (sram_rdata_hi)
      );
      // Registered head-flit output (valid 1 cycle after rd_ptr presented)
      assign vc_head_flit[vs] = {sram_rdata_hi, sram_rdata_lo};
    end
  endgenerate

`endif

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

`ifndef SYNTHESIS
      // Behavioral: head flit driven combinationally from fifo_mem
      assign vc_head_flit[v] = fifo_mem[v][rd_ptr[v]];

      assign vc_has_head[v] = vc_has_flit[v] &&
                              (fifo_mem[v][rd_ptr[v]].flit_type == FLIT_HEAD ||
                               fifo_mem[v][rd_ptr[v]].flit_type == FLIT_HEADTAIL);
`else
      // Synthesis: vc_head_flit driven by SRAM registered output (gen_vc_sram).
      assign vc_has_head[v] = vc_has_flit[v] &&
                              (vc_head_flit[v].flit_type == FLIT_HEAD ||
                               vc_head_flit[v].flit_type == FLIT_HEADTAIL);
`endif

      // Route compute for head flit
      logic [ROW_BITS-1:0] dst_row_v;
      logic [COL_BITS-1:0] dst_col_v;
      logic [PORT_BITS-1:0] route_port_v;
      logic route_valid_v;

      assign dst_row_v = node_row(vc_head_flit[v].dst_id);
      assign dst_col_v = node_col(vc_head_flit[v].dst_id);

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
  // Both paths (behavioral and SRAM) expose the head flit via vc_head_flit[v].
  // ---------------------------------------------------------------------------
  always_comb begin
    read_flit = '0;
    read_vc   = '0;
    for (int v = 0; v < NUM_VCS; v++) begin
      if (vc_read[v]) begin
        read_flit = vc_head_flit[v];
        read_vc   = VC_BITS'(v);
      end
    end
  end

endmodule
