// =============================================================================
// accel_tile_array.sv — Multi-Tile Accelerator on NoC Mesh
// =============================================================================
// Top-level that instantiates NUM_TILES accelerator tiles, the NoC mesh,
// a barrier synchronizer, and an AXI gateway for Ibex host access.
//
// This is the "multi-tile accel" that replaces single accel_top in the SoC.
// The AXI slave port connects to the existing crossbar (slave 3).

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */
import noc_pkg::*;

module accel_tile_array #(
  parameter int  MESH_ROWS       = noc_pkg::MESH_ROWS,
  parameter int  MESH_COLS       = noc_pkg::MESH_COLS,
  parameter int  N_ROWS          = 16,
  parameter int  N_COLS          = 16,
  parameter int  DATA_W          = 8,
  parameter int  ACC_W           = 32,
  parameter int  SP_DEPTH        = 4096,
  parameter int  NUM_VCS         = noc_pkg::NUM_VCS,
  parameter bit  SPARSE_VC_ALLOC = 1'b0,
  parameter bit  INNET_REDUCE    = 1'b0,

  // AXI parameters (match soc_pkg)
  parameter int  AXI_ADDR_W      = 32,
  parameter int  AXI_DATA_W      = 32,
  parameter int  AXI_ID_W        = 4
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Optional per-router subtree metadata programming for INR experiments.
  input  logic                    inr_meta_cfg_valid [MESH_ROWS*MESH_COLS],
  input  logic [7:0]              inr_meta_cfg_reduce_id [MESH_ROWS*MESH_COLS],
  input  logic [3:0]              inr_meta_cfg_target [MESH_ROWS*MESH_COLS],
  input  logic                    inr_meta_cfg_enable [MESH_ROWS*MESH_COLS],

  // =====================================================================
  // AXI4-Lite Slave (host → tile CSR access via crossbar)
  // =====================================================================
  input  logic [AXI_ADDR_W-1:0]  s_axi_awaddr,
  input  logic                    s_axi_awvalid,
  output logic                    s_axi_awready,

  input  logic [AXI_DATA_W-1:0]  s_axi_wdata,
  input  logic [3:0]             s_axi_wstrb,
  input  logic                    s_axi_wvalid,
  output logic                    s_axi_wready,

  output logic [1:0]             s_axi_bresp,
  output logic                    s_axi_bvalid,
  input  logic                    s_axi_bready,

  input  logic [AXI_ADDR_W-1:0]  s_axi_araddr,
  input  logic                    s_axi_arvalid,
  output logic                    s_axi_arready,

  output logic [AXI_DATA_W-1:0]  s_axi_rdata,
  output logic [1:0]             s_axi_rresp,
  output logic                    s_axi_rvalid,
  input  logic                    s_axi_rready,

  // =====================================================================
  // AXI4 Master (tiles → DRAM via crossbar)
  // =====================================================================
  output logic [AXI_ID_W-1:0]    m_axi_arid,
  output logic [AXI_ADDR_W-1:0]  m_axi_araddr,
  output logic [7:0]             m_axi_arlen,
  output logic [2:0]             m_axi_arsize,
  output logic [1:0]             m_axi_arburst,
  output logic                    m_axi_arvalid,
  input  logic                    m_axi_arready,

  input  logic [AXI_ID_W-1:0]    m_axi_rid,
  input  logic [AXI_DATA_W-1:0]  m_axi_rdata,
  input  logic [1:0]             m_axi_rresp,
  input  logic                    m_axi_rlast,
  input  logic                    m_axi_rvalid,
  output logic                    m_axi_rready,

  output logic [AXI_ID_W-1:0]    m_axi_awid,
  output logic [AXI_ADDR_W-1:0]  m_axi_awaddr,
  output logic [7:0]             m_axi_awlen,
  output logic [2:0]             m_axi_awsize,
  output logic [1:0]             m_axi_awburst,
  output logic                    m_axi_awvalid,
  input  logic                    m_axi_awready,

  output logic [AXI_DATA_W-1:0]  m_axi_wdata,
  output logic [AXI_DATA_W/8-1:0] m_axi_wstrb,
  output logic                    m_axi_wlast,
  output logic                    m_axi_wvalid,
  input  logic                    m_axi_wready,

  input  logic [AXI_ID_W-1:0]    m_axi_bid,
  input  logic [1:0]             m_axi_bresp,
  input  logic                    m_axi_bvalid,
  output logic                    m_axi_bready,

  // Status outputs
  output logic [MESH_ROWS*MESH_COLS-1:0] tile_busy_o,
  output logic [MESH_ROWS*MESH_COLS-1:0] tile_done_o
);

  localparam int NUM_TILES = MESH_ROWS * MESH_COLS;

  // =========================================================================
  // NoC mesh local ports
  // =========================================================================
  flit_t                  tile_flit_out  [NUM_TILES];
  logic                   tile_valid_out [NUM_TILES];
  logic [NUM_VCS-1:0]     tile_credit_in [NUM_TILES];

  flit_t                  tile_flit_in   [NUM_TILES];
  logic                   tile_valid_in  [NUM_TILES];
  logic [NUM_VCS-1:0]     tile_credit_out[NUM_TILES];

  // =========================================================================
  // Barrier signals
  // =========================================================================
  logic [NUM_TILES-1:0]   tile_barrier_req;
  logic                   barrier_release;

  // =========================================================================
  // Per-tile CSR routing
  // =========================================================================
  // Address decode: addr[15:12] selects tile, addr[7:0] selects CSR
  logic [3:0]             csr_tile_sel;
  logic                   csr_is_array_reg;
  logic [31:0]            tile_csr_wdata;
  logic [7:0]             tile_csr_addr;
  logic                   tile_csr_wen [NUM_TILES];
  logic [31:0]            tile_csr_rdata [NUM_TILES];
  logic [31:0]            array_csr_rdata;

  // Tile-array level NoC metadata programming window.
  // Offsets are handled inside the AXI slave bridge and do not target an
  // individual tile CSR block.
  logic [3:0]             noc_meta_router_reg;
  logic [7:0]             noc_meta_reduce_id_reg;
  logic [3:0]             noc_meta_target_reg;
  logic                   noc_meta_enable_reg;
  logic                   noc_meta_apply_pulse;
  logic                   mesh_inr_meta_cfg_valid [NUM_TILES];
  logic [7:0]             mesh_inr_meta_cfg_reduce_id [NUM_TILES];
  logic [3:0]             mesh_inr_meta_cfg_target [NUM_TILES];
  logic                   mesh_inr_meta_cfg_enable [NUM_TILES];

  localparam logic [7:0] NOC_META_ROUTER_OFF    = 8'h80;
  localparam logic [7:0] NOC_META_REDUCE_ID_OFF = 8'h84;
  localparam logic [7:0] NOC_META_TARGET_OFF    = 8'h88;
  localparam logic [7:0] NOC_META_CTRL_OFF      = 8'h8C;
  localparam logic [7:0] NOC_META_STATUS_OFF    = 8'h90;

  // Registered CSR address and write data: latched when AW/AR handshake
  // completes in AXI_S_IDLE. The combinatorial s_axi_awaddr/wdata are gone
  // by the time tile_csr_wen fires one cycle later in AXI_S_WRITE.
  logic [7:0]             csr_addr_r;
  logic [31:0]            csr_wdata_r;

  assign csr_tile_sel  = s_axi_awvalid ? s_axi_awaddr[15:12] :
                         s_axi_arvalid ? s_axi_araddr[15:12] : '0;
  assign tile_csr_addr = csr_addr_r;
  assign csr_is_array_reg = csr_addr_r[7];

  // =========================================================================
  // Tile CSR status
  // =========================================================================
  logic [NUM_TILES-1:0] tile_busy;
  logic [NUM_TILES-1:0] tile_done;

  // =========================================================================
  // AXI Slave → CSR bridge (simplified)
  // =========================================================================
  typedef enum logic [1:0] {
    AXI_S_IDLE,
    AXI_S_WRITE,
    AXI_S_READ,
    AXI_S_RESP
  } axi_s_state_e;

  axi_s_state_e axi_s_state;
  logic [31:0] axi_s_rdata_reg;
  logic axi_s_write_done;
  logic [3:0] axi_s_tile;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      axi_s_state      <= AXI_S_IDLE;
      axi_s_rdata_reg  <= '0;
      axi_s_write_done <= 1'b0;
      axi_s_tile       <= '0;
      csr_addr_r       <= '0;
      csr_wdata_r      <= '0;
      noc_meta_router_reg    <= '0;
      noc_meta_reduce_id_reg <= '0;
      noc_meta_target_reg    <= '0;
      noc_meta_enable_reg    <= 1'b0;
      noc_meta_apply_pulse   <= 1'b0;
    end else begin
      noc_meta_apply_pulse <= 1'b0;

      case (axi_s_state)
        AXI_S_IDLE: begin
          if (s_axi_awvalid && s_axi_wvalid) begin
            axi_s_tile  <= csr_tile_sel;
            csr_addr_r  <= s_axi_awaddr[7:0];  // latch before awvalid deasserts
            csr_wdata_r <= s_axi_wdata;         // latch before wvalid deasserts
            axi_s_state <= AXI_S_WRITE;
          end else if (s_axi_arvalid) begin
            axi_s_tile  <= csr_tile_sel;
            csr_addr_r  <= s_axi_araddr[7:0];  // latch before arvalid deasserts
            axi_s_state <= AXI_S_READ;
          end
        end

        AXI_S_WRITE: begin
          if (csr_is_array_reg) begin
            case (csr_addr_r)
              NOC_META_ROUTER_OFF:    noc_meta_router_reg    <= csr_wdata_r[3:0];
              NOC_META_REDUCE_ID_OFF: noc_meta_reduce_id_reg <= csr_wdata_r[7:0];
              NOC_META_TARGET_OFF:    noc_meta_target_reg    <= csr_wdata_r[3:0];
              NOC_META_CTRL_OFF: begin
                noc_meta_enable_reg  <= csr_wdata_r[0];
                noc_meta_apply_pulse <= csr_wdata_r[1];
              end
              default: ;
            endcase
          end
          axi_s_write_done <= 1'b1;
          axi_s_state      <= AXI_S_RESP;
        end

        AXI_S_READ: begin
          axi_s_rdata_reg <= csr_is_array_reg ? array_csr_rdata : tile_csr_rdata[axi_s_tile];
          axi_s_state     <= AXI_S_RESP;
        end

        AXI_S_RESP: begin
          if ((axi_s_write_done && s_axi_bready) ||
              (!axi_s_write_done && s_axi_rready)) begin
            axi_s_state      <= AXI_S_IDLE;
            axi_s_write_done <= 1'b0;
          end
        end
      endcase
    end
  end

  assign s_axi_awready = (axi_s_state == AXI_S_IDLE) && s_axi_awvalid;
  assign s_axi_wready  = (axi_s_state == AXI_S_IDLE) && s_axi_awvalid;
  assign s_axi_arready = (axi_s_state == AXI_S_IDLE) && !s_axi_awvalid && s_axi_arvalid;
  assign s_axi_bresp   = 2'b00;
  assign s_axi_bvalid  = (axi_s_state == AXI_S_RESP) && axi_s_write_done;
  assign s_axi_rdata   = axi_s_rdata_reg;
  assign s_axi_rresp   = 2'b00;
  assign s_axi_rvalid  = (axi_s_state == AXI_S_RESP) && !axi_s_write_done;
  assign tile_csr_wdata = csr_wdata_r;

  // CSR write enable per tile
  always_comb begin
    for (int t = 0; t < NUM_TILES; t++)
      tile_csr_wen[t] = (axi_s_state == AXI_S_WRITE) && !csr_is_array_reg && (axi_s_tile == 4'(t));
  end

  always_comb begin
    case (csr_addr_r)
      NOC_META_ROUTER_OFF:    array_csr_rdata = {28'h0, noc_meta_router_reg};
      NOC_META_REDUCE_ID_OFF: array_csr_rdata = {24'h0, noc_meta_reduce_id_reg};
      NOC_META_TARGET_OFF:    array_csr_rdata = {28'h0, noc_meta_target_reg};
      NOC_META_CTRL_OFF:      array_csr_rdata = {30'h0, 1'b0, noc_meta_enable_reg};
      NOC_META_STATUS_OFF:    array_csr_rdata = {
                                  7'h0,
                                  noc_meta_enable_reg,
                                  4'h0,
                                  noc_meta_target_reg,
                                  4'h0,
                                  noc_meta_reduce_id_reg,
                                  noc_meta_router_reg
                                };
      default:                array_csr_rdata = '0;
    endcase
  end

  always_comb begin
    for (int t = 0; t < NUM_TILES; t++) begin
      mesh_inr_meta_cfg_valid[t]     = inr_meta_cfg_valid[t];
      mesh_inr_meta_cfg_reduce_id[t] = inr_meta_cfg_reduce_id[t];
      mesh_inr_meta_cfg_target[t]    = inr_meta_cfg_target[t];
      mesh_inr_meta_cfg_enable[t]    = inr_meta_cfg_enable[t];

      if (noc_meta_apply_pulse && (noc_meta_router_reg == 4'(t))) begin
        mesh_inr_meta_cfg_valid[t]     = 1'b1;
        mesh_inr_meta_cfg_reduce_id[t] = noc_meta_reduce_id_reg;
        mesh_inr_meta_cfg_target[t]    = noc_meta_target_reg;
        mesh_inr_meta_cfg_enable[t]    = noc_meta_enable_reg;
      end
    end
  end

  // =========================================================================
  // Tile instances
  // =========================================================================
  generate
    for (genvar t = 0; t < NUM_TILES; t++) begin : gen_tile
      accel_tile #(
        .TILE_ID   (t),
        .N_ROWS    (N_ROWS),
        .N_COLS    (N_COLS),
        .DATA_W    (DATA_W),
        .ACC_W     (ACC_W),
        .SP_DEPTH  (SP_DEPTH),
        .SP_DATA_W (32),
        .NUM_VCS   (NUM_VCS)
      ) u_tile (
        .clk            (clk),
        .rst_n          (rst_n),
        .noc_flit_out   (tile_flit_out[t]),
        .noc_valid_out  (tile_valid_out[t]),
        .noc_credit_in  (tile_credit_in[t]),
        .noc_flit_in    (tile_flit_in[t]),
        .noc_valid_in   (tile_valid_in[t]),
        .noc_credit_out (tile_credit_out[t]),
        .csr_wdata      (tile_csr_wdata),
        .csr_addr       (tile_csr_addr),
        .csr_wen        (tile_csr_wen[t]),
        .csr_rdata      (tile_csr_rdata[t]),
        .barrier_req    (tile_barrier_req[t]),
        .barrier_done   (barrier_release),
        .tile_busy      (tile_busy[t]),
        .tile_done      (tile_done[t])
      );
    end
  endgenerate

  // =========================================================================
  // NoC Mesh
  // =========================================================================
  noc_mesh_4x4 #(
    .MESH_ROWS       (MESH_ROWS),
    .MESH_COLS       (MESH_COLS),
    .NUM_VCS         (NUM_VCS),
    .BUF_DEPTH       (noc_pkg::BUF_DEPTH),
    .SPARSE_VC_ALLOC (SPARSE_VC_ALLOC),
    .INNET_REDUCE    (INNET_REDUCE)
  ) u_mesh (
    .clk              (clk),
    .rst_n            (rst_n),
    .inr_meta_cfg_valid     (mesh_inr_meta_cfg_valid),
    .inr_meta_cfg_reduce_id (mesh_inr_meta_cfg_reduce_id),
    .inr_meta_cfg_target    (mesh_inr_meta_cfg_target),
    .inr_meta_cfg_enable    (mesh_inr_meta_cfg_enable),
    .local_flit_in    (mesh_flit_in),
    .local_valid_in   (mesh_valid_in),
    .local_credit_out (tile_credit_in),
    .local_flit_out   (tile_flit_in),
    .local_valid_out  (tile_valid_in),
    .local_credit_in  (mesh_credit_out)
  );

  // =========================================================================
  // Barrier Sync
  // =========================================================================
  barrier_sync #(
    .NUM_TILES (NUM_TILES)
  ) u_barrier (
    .clk              (clk),
    .rst_n            (rst_n),
    .tile_barrier_req (tile_barrier_req),
    .participant_mask ({NUM_TILES{1'b1}}),  // All tiles participate
    .barrier_release  (barrier_release),
    .arrived_mask     (),
    .barrier_active   ()
  );

  // =========================================================================
  // Status outputs
  // =========================================================================
  assign tile_busy_o = tile_busy;
  assign tile_done_o = tile_done;

  // =========================================================================
  // DMA Gateway — translates NoC DMA flits to AXI master transactions
  // =========================================================================
  // The gateway occupies the last mesh node's local port.
  // Tiles send DMA_RD/DMA_WR flits to node (NUM_TILES-1) to access DRAM.
  localparam int GW_NODE = NUM_TILES - 1;

  // Gateway NoC local port signals (steals last tile's NoC connection)
  flit_t gw_flit_in, gw_flit_out;
  logic  gw_valid_in, gw_valid_out;
  logic  gw_credit_in, gw_credit_out;

  // The last tile's NoC connection is already wired to the mesh above.
  // We tap into it by re-assigning: the gateway shares the local port
  // of the last mesh node. In a real design you'd add a dedicated gateway
  // node; here we multiplex tile[GW_NODE] and the gateway on the same port.
  // For simplicity, the gateway listens to all flits addressed to node GW_NODE.
  assign gw_flit_in    = tile_flit_in[GW_NODE];
  assign gw_valid_in   = tile_valid_in[GW_NODE];
  assign gw_credit_in  = tile_credit_in[GW_NODE][0];

  // ---- Override the mesh local-port outputs for GW_NODE ----
  // Mux: if gateway is sending a response flit, use gateway output;
  // otherwise pass through tile[GW_NODE]'s output.
  // Note: gen_tile still drives tile_flit_out[GW_NODE] / tile_valid_out[GW_NODE],
  // but the mesh sees the muxed versions below via the override wires.
  flit_t                  mesh_flit_in   [NUM_TILES];
  logic                   mesh_valid_in  [NUM_TILES];
  logic [NUM_VCS-1:0]     mesh_credit_out[NUM_TILES];

  always_comb begin
    for (int t = 0; t < NUM_TILES; t++) begin
      if (t == GW_NODE) begin
        // Gateway takes priority when it has a response to send
        mesh_flit_in[t]    = gw_valid_out ? gw_flit_out  : tile_flit_out[t];
        mesh_valid_in[t]   = gw_valid_out ? 1'b1         : tile_valid_out[t];
        mesh_credit_out[t] = tile_credit_out[t];  // credits flow back to tiles
      end else begin
        mesh_flit_in[t]    = tile_flit_out[t];
        mesh_valid_in[t]   = tile_valid_out[t];
        mesh_credit_out[t] = tile_credit_out[t];
      end
    end
  end

  tile_dma_gateway #(
    .AXI_ADDR_W (AXI_ADDR_W),
    .AXI_DATA_W (AXI_DATA_W),
    .AXI_ID_W   (AXI_ID_W),
    .MAX_BURST  (16)
  ) u_dma_gw (
    .clk              (clk),
    .rst_n            (rst_n),
    // NoC local port
    .noc_flit_in      (gw_flit_in),
    .noc_valid_in     (gw_valid_in),
    .noc_credit_out   (gw_credit_out),
    .noc_flit_out     (gw_flit_out),
    .noc_valid_out    (gw_valid_out),
    .noc_credit_in    (gw_credit_in),
    // AXI master
    .m_axi_arid       (m_axi_arid),
    .m_axi_araddr     (m_axi_araddr),
    .m_axi_arlen      (m_axi_arlen),
    .m_axi_arsize     (m_axi_arsize),
    .m_axi_arburst    (m_axi_arburst),
    .m_axi_arvalid    (m_axi_arvalid),
    .m_axi_arready    (m_axi_arready),
    .m_axi_rid        (m_axi_rid),
    .m_axi_rdata      (m_axi_rdata),
    .m_axi_rresp      (m_axi_rresp),
    .m_axi_rlast      (m_axi_rlast),
    .m_axi_rvalid     (m_axi_rvalid),
    .m_axi_rready     (m_axi_rready),
    .m_axi_awid       (m_axi_awid),
    .m_axi_awaddr     (m_axi_awaddr),
    .m_axi_awlen      (m_axi_awlen),
    .m_axi_awsize     (m_axi_awsize),
    .m_axi_awburst    (m_axi_awburst),
    .m_axi_awvalid    (m_axi_awvalid),
    .m_axi_awready    (m_axi_awready),
    .m_axi_wdata      (m_axi_wdata),
    .m_axi_wstrb      (m_axi_wstrb),
    .m_axi_wlast      (m_axi_wlast),
    .m_axi_wvalid     (m_axi_wvalid),
    .m_axi_wready     (m_axi_wready),
    .m_axi_bid        (m_axi_bid),
    .m_axi_bresp      (m_axi_bresp),
    .m_axi_bvalid     (m_axi_bvalid),
    .m_axi_bready     (m_axi_bready)
  );

endmodule
