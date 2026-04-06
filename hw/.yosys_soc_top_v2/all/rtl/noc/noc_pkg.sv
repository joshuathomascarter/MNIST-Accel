`timescale 1ns/1ps
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */

// =============================================================================
// noc_pkg.sv — NoC Package: Flit Formats, Parameters, Types
// =============================================================================

package noc_pkg;

  // -----------------------------------------------------------------------
  // Mesh parameters (parameterizable — same RTL targets 2×2, 3×3, 4×4)
  // -----------------------------------------------------------------------
  parameter int MESH_ROWS   = 4;
  parameter int MESH_COLS   = 4;
  parameter int NUM_NODES   = MESH_ROWS * MESH_COLS;

  // -----------------------------------------------------------------------
  // Router parameters
  // -----------------------------------------------------------------------
  parameter int NUM_PORTS   = 5;   // N, E, S, W, Local
  parameter int NUM_VCS     = 4;   // virtual channels per port
  parameter int BUF_DEPTH   = 4;   // flits per VC buffer
  parameter int FLIT_WIDTH  = 64;  // bits per flit
  parameter int CREDIT_WIDTH = $clog2(BUF_DEPTH) + 1;

  // -----------------------------------------------------------------------
  // Port indices
  // -----------------------------------------------------------------------
  parameter int PORT_NORTH  = 0;
  parameter int PORT_EAST   = 1;
  parameter int PORT_SOUTH  = 2;
  parameter int PORT_WEST   = 3;
  parameter int PORT_LOCAL  = 4;

  // -----------------------------------------------------------------------
  // Coordinate encoding
  // -----------------------------------------------------------------------
  parameter int ROW_BITS    = $clog2(MESH_ROWS);
  parameter int COL_BITS    = $clog2(MESH_COLS);
  parameter int NODE_BITS   = $clog2(NUM_NODES);
  parameter int VC_BITS     = $clog2(NUM_VCS);
  parameter int PORT_BITS   = $clog2(NUM_PORTS);

  // -----------------------------------------------------------------------
  // Flit types
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] {
    FLIT_HEAD    = 2'b00,
    FLIT_BODY    = 2'b01,
    FLIT_TAIL    = 2'b10,
    FLIT_HEADTAIL = 2'b11   // single-flit packet
  } flit_type_e;

  // -----------------------------------------------------------------------
  // Flit format (64 bits)
  //
  //  [63:62]  flit_type       (2 bits)
  //  [61:58]  src_id          (4 bits — up to 16 nodes)
  //  [57:54]  dst_id          (4 bits)
  //  [53:52]  vc_id           (2 bits — 4 VCs)
  //  [51:48]  msg_type        (4 bits — command encoding)
  //  [47:0]   payload         (48 bits)
  // -----------------------------------------------------------------------
  parameter int FLIT_TYPE_HI  = 63;
  parameter int FLIT_TYPE_LO  = 62;
  parameter int SRC_ID_HI     = 61;
  parameter int SRC_ID_LO     = 58;
  parameter int DST_ID_HI     = 57;
  parameter int DST_ID_LO     = 54;
  parameter int VC_ID_HI      = 53;
  parameter int VC_ID_LO      = 52;
  parameter int MSG_TYPE_HI   = 51;
  parameter int MSG_TYPE_LO   = 48;
  parameter int PAYLOAD_HI    = 47;
  parameter int PAYLOAD_LO    = 0;

  // -----------------------------------------------------------------------
  // Message types (4-bit encoding)
  // -----------------------------------------------------------------------
  typedef enum logic [3:0] {
    MSG_DATA        = 4'h0,   // Generic data transfer
    MSG_WRITE_REQ   = 4'h1,   // Write request (addr + data)
    MSG_READ_REQ    = 4'h2,   // Read request (addr)
    MSG_READ_RESP   = 4'h3,   // Read response (data)
    MSG_WRITE_ACK   = 4'h4,   // Write acknowledgment
    MSG_SCATTER     = 4'h5,   // Scatter data to tiles
    MSG_REDUCE      = 4'h6,   // Reduction partial sum
    MSG_BARRIER     = 4'h7,   // Barrier synchronization
    MSG_CSR_WRITE   = 4'h8,   // CSR write from Ibex via gateway
    MSG_CSR_READ    = 4'h9,   // CSR read request
    MSG_CSR_RESP    = 4'hA,   // CSR read response
    MSG_SPARSE_HINT = 4'hB,   // Sparsity hint for VC allocator
    MSG_RESERVED    = 4'hF    // Reserved
  } msg_type_e;

  // -----------------------------------------------------------------------
  // Struct view of a flit (for convenience)
  // -----------------------------------------------------------------------
  typedef struct packed {
    flit_type_e flit_type;    // [63:62]
    logic [3:0] src_id;       // [61:58]
    logic [3:0] dst_id;       // [57:54]
    logic [1:0] vc_id;        // [53:52]
    logic [3:0] msg_type;     // [51:48]
    logic [47:0] payload;     // [47:0]
  } flit_t;

  // -----------------------------------------------------------------------
  // In-Network Reduction: payload sub-fields for MSG_REDUCE flits
  //
  // When msg_type == MSG_REDUCE the 48-bit payload is split as:
  //   [47:40]  reduce_id      (8 bits) — identifies the reduction group/wave
  //   [39:36]  reduce_expect  (4 bits) — total expected contributors (1-16)
  //   [35:4]   reduce_val     (32 bits) — INT32 partial-sum / accumulated value
  //   [3:0]    reduce_count   (4 bits) — contributors represented by this flit
  //
  // Non-reduce packets leave [47:36] = 0 — fully backward-compatible.
  // -----------------------------------------------------------------------
  parameter int REDUCE_ID_HI     = 47;
  parameter int REDUCE_ID_LO     = 40;
  parameter int REDUCE_EXPECT_HI = 39;
  parameter int REDUCE_EXPECT_LO = 36;
  parameter int REDUCE_VAL_HI    = 35;
  parameter int REDUCE_VAL_LO    = 4;
  parameter int REDUCE_COUNT_HI  = 3;
  parameter int REDUCE_COUNT_LO  = 0;

  // Maximum contributors in a reduction group (limited by 4-bit expect field)
  parameter int MAX_REDUCE_CONTRIBUTORS = 16;

  // In-network reduce scratchpad size (per router, indexed by reduce_id[2:0])
  parameter int INNET_SP_DEPTH   = 8;

  // -----------------------------------------------------------------------
  // Helper functions
  // -----------------------------------------------------------------------

  // Node ID ↔ (row, col) conversion
  function automatic [ROW_BITS-1:0] node_row;
    input [NODE_BITS-1:0] id;
    begin
      node_row = id[NODE_BITS-1 -: ROW_BITS];
    end
  endfunction

  function automatic [COL_BITS-1:0] node_col;
    input [NODE_BITS-1:0] id;
    begin
      node_col = id[COL_BITS-1:0];
    end
  endfunction

  function automatic [NODE_BITS-1:0] make_node_id;
    input [ROW_BITS-1:0] row;
    input [COL_BITS-1:0] col;
    begin
      make_node_id = {row, col};
    end
  endfunction

  // Build a head flit
  function automatic flit_t make_head_flit;
    input [3:0] src;
    input [3:0] dst;
    input [1:0] vc;
    input [3:0] mtype;
    input [47:0] payload;
    begin
      make_head_flit.flit_type = FLIT_HEAD;
      make_head_flit.src_id    = src;
      make_head_flit.dst_id    = dst;
      make_head_flit.vc_id     = vc;
      make_head_flit.msg_type  = mtype;
      make_head_flit.payload   = payload;
    end
  endfunction

  // Build a single-flit MSG_REDUCE packet with in-network reduction fields
  function automatic flit_t make_reduce_flit;
    input [3:0] src;
    input [3:0] dst;
    input [1:0] vc;
    input [7:0] red_id;
    input [3:0] red_expect;
    input [31:0] red_val;
    input [3:0] red_count;
    begin
      make_reduce_flit.flit_type = FLIT_HEADTAIL;
      make_reduce_flit.src_id    = src;
      make_reduce_flit.dst_id    = dst;
      make_reduce_flit.vc_id     = vc;
      make_reduce_flit.msg_type  = MSG_REDUCE;
      make_reduce_flit.payload   = {red_id, red_expect, red_val, red_count};
    end
  endfunction

endpackage : noc_pkg
