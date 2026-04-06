// =============================================================================
// noc_mesh_4x4.sv — Parameterized 2D Mesh Network
// =============================================================================
// Instantiates MESH_ROWS × MESH_COLS routers and wires inter-router links.
// Boundary ports (North edge has no North neighbor, etc.) are tied off.
// Local ports are exposed for tile/NI connections.
//
// Node numbering: node_id = row * MESH_COLS + col
//   (0,0)=0  (0,1)=1  (0,2)=2  (0,3)=3
//   (1,0)=4  (1,1)=5  (1,2)=6  (1,3)=7
//   (2,0)=8  (2,1)=9  (2,2)=10 (2,3)=11
//   (3,0)=12 (3,1)=13 (3,2)=14 (3,3)=15

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
import noc_pkg::*;

module noc_mesh_4x4 #(
  parameter int  MESH_ROWS       = noc_pkg::MESH_ROWS,
  parameter int  MESH_COLS       = noc_pkg::MESH_COLS,
  parameter int  NUM_VCS         = noc_pkg::NUM_VCS,
  parameter int  BUF_DEPTH       = noc_pkg::BUF_DEPTH,
  parameter bit  SPARSE_VC_ALLOC = 1'b0,
  parameter bit  INNET_REDUCE    = 1'b0
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // Optional per-router subtree metadata configuration keyed by reduce_id.
  input  logic                  inr_meta_cfg_valid [MESH_ROWS * MESH_COLS],
  input  logic [7:0]            inr_meta_cfg_reduce_id [MESH_ROWS * MESH_COLS],
  input  logic [3:0]            inr_meta_cfg_target [MESH_ROWS * MESH_COLS],
  input  logic                  inr_meta_cfg_enable [MESH_ROWS * MESH_COLS],

  // --- Local injection/ejection ports (one per node) ---
  input  flit_t                 local_flit_in  [MESH_ROWS * MESH_COLS],
  input  logic                  local_valid_in [MESH_ROWS * MESH_COLS],
  output logic [NUM_VCS-1:0]    local_credit_out[MESH_ROWS * MESH_COLS],

  output flit_t                 local_flit_out [MESH_ROWS * MESH_COLS],
  output logic                  local_valid_out[MESH_ROWS * MESH_COLS],
  input  logic [NUM_VCS-1:0]    local_credit_in[MESH_ROWS * MESH_COLS]
);

  localparam int NUM_NODES = MESH_ROWS * MESH_COLS;
  localparam int NP = 5; // ports per router

  // =========================================================================
  // Inter-router link wires
  // =========================================================================
  // For each router, 5 directional link bundles
  flit_t                 r_flit_in   [NUM_NODES][NP];
  logic                  r_valid_in  [NUM_NODES][NP];
  logic [NUM_VCS-1:0]    r_credit_out[NUM_NODES][NP];

  flit_t                 r_flit_out  [NUM_NODES][NP];
  logic                  r_valid_out [NUM_NODES][NP];
  logic [NUM_VCS-1:0]    r_credit_in [NUM_NODES][NP];

  // =========================================================================
  // Router instantiation
  // =========================================================================
  generate
    for (genvar r = 0; r < MESH_ROWS; r++) begin : gen_row
      for (genvar c = 0; c < MESH_COLS; c++) begin : gen_col
        localparam int ID = r * MESH_COLS + c;

        noc_router #(
          .ROUTER_ID       (ID),
          .NUM_PORTS       (NP),
          .NUM_VCS         (NUM_VCS),
          .BUF_DEPTH       (BUF_DEPTH),
          .MESH_ROWS       (MESH_ROWS),
          .MESH_COLS       (MESH_COLS),
          .SPARSE_VC_ALLOC (SPARSE_VC_ALLOC),
          .INNET_REDUCE    (INNET_REDUCE)
        ) u_router (
          .clk             (clk),
          .rst_n           (rst_n),
          .inr_meta_cfg_valid     (inr_meta_cfg_valid[ID]),
          .inr_meta_cfg_reduce_id (inr_meta_cfg_reduce_id[ID]),
          .inr_meta_cfg_target    (inr_meta_cfg_target[ID]),
          .inr_meta_cfg_enable    (inr_meta_cfg_enable[ID]),
          .link_flit_in    (r_flit_in[ID]),
          .link_valid_in   (r_valid_in[ID]),
          .link_credit_out (r_credit_out[ID]),
          .link_flit_out   (r_flit_out[ID]),
          .link_valid_out  (r_valid_out[ID]),
          .link_credit_in  (r_credit_in[ID])
        );
      end
    end
  endgenerate

  // =========================================================================
  // Inter-router wiring + boundary tie-off
  // =========================================================================
  generate
    for (genvar r = 0; r < MESH_ROWS; r++) begin : gen_wire_row
      for (genvar c = 0; c < MESH_COLS; c++) begin : gen_wire_col
        localparam int ID = r * MESH_COLS + c;

        // ----- NORTH link -----
        if (r > 0) begin : north_link
          localparam int N_ID = (r - 1) * MESH_COLS + c;
          // My North input ← neighbor's South output
          assign r_flit_in[ID][PORT_NORTH]    = r_flit_out[N_ID][PORT_SOUTH];
          assign r_valid_in[ID][PORT_NORTH]   = r_valid_out[N_ID][PORT_SOUTH];
          assign r_credit_in[ID][PORT_NORTH]  = r_credit_out[N_ID][PORT_SOUTH];
        end else begin : north_tieoff
          assign r_flit_in[ID][PORT_NORTH]    = '0;
          assign r_valid_in[ID][PORT_NORTH]   = 1'b0;
          assign r_credit_in[ID][PORT_NORTH]  = '0;
        end

        // ----- SOUTH link -----
        if (r < MESH_ROWS - 1) begin : south_link
          localparam int S_ID = (r + 1) * MESH_COLS + c;
          assign r_flit_in[ID][PORT_SOUTH]    = r_flit_out[S_ID][PORT_NORTH];
          assign r_valid_in[ID][PORT_SOUTH]   = r_valid_out[S_ID][PORT_NORTH];
          assign r_credit_in[ID][PORT_SOUTH]  = r_credit_out[S_ID][PORT_NORTH];
        end else begin : south_tieoff
          assign r_flit_in[ID][PORT_SOUTH]    = '0;
          assign r_valid_in[ID][PORT_SOUTH]   = 1'b0;
          assign r_credit_in[ID][PORT_SOUTH]  = '0;
        end

        // ----- EAST link -----
        if (c < MESH_COLS - 1) begin : east_link
          localparam int E_ID = r * MESH_COLS + (c + 1);
          assign r_flit_in[ID][PORT_EAST]     = r_flit_out[E_ID][PORT_WEST];
          assign r_valid_in[ID][PORT_EAST]    = r_valid_out[E_ID][PORT_WEST];
          assign r_credit_in[ID][PORT_EAST]   = r_credit_out[E_ID][PORT_WEST];
        end else begin : east_tieoff
          assign r_flit_in[ID][PORT_EAST]     = '0;
          assign r_valid_in[ID][PORT_EAST]    = 1'b0;
          assign r_credit_in[ID][PORT_EAST]   = '0;
        end

        // ----- WEST link -----
        if (c > 0) begin : west_link
          localparam int W_ID = r * MESH_COLS + (c - 1);
          assign r_flit_in[ID][PORT_WEST]     = r_flit_out[W_ID][PORT_EAST];
          assign r_valid_in[ID][PORT_WEST]    = r_valid_out[W_ID][PORT_EAST];
          assign r_credit_in[ID][PORT_WEST]   = r_credit_out[W_ID][PORT_EAST];
        end else begin : west_tieoff
          assign r_flit_in[ID][PORT_WEST]     = '0;
          assign r_valid_in[ID][PORT_WEST]    = 1'b0;
          assign r_credit_in[ID][PORT_WEST]   = '0;
        end

        // ----- LOCAL port ← external injection/ejection -----
        assign r_flit_in[ID][PORT_LOCAL]      = local_flit_in[ID];
        assign r_valid_in[ID][PORT_LOCAL]     = local_valid_in[ID];
        assign r_credit_in[ID][PORT_LOCAL]    = local_credit_in[ID];

        assign local_flit_out[ID]             = r_flit_out[ID][PORT_LOCAL];
        assign local_valid_out[ID]            = r_valid_out[ID][PORT_LOCAL];
        assign local_credit_out[ID]           = r_credit_out[ID][PORT_LOCAL];
      end
    end
  endgenerate

endmodule
