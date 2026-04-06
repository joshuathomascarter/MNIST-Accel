// =============================================================================
// l1_dcache_top.sv — L1 Data Cache Top-Level Wrapper
// =============================================================================
// Wraps tag array, data array, LRU, and controller FSM.
// CPU-side: OBI-like interface (matches Ibex load-store unit)
// Memory-side: AXI4 master for cache fills and writebacks
//
// Parameters sized for first tapeout:
//   4KB = 16 sets × 4 ways × 64B lines

/* verilator lint_off UNUSEDSIGNAL */
module l1_dcache_top #(
  parameter int ADDR_WIDTH  = 32,
  parameter int DATA_WIDTH  = 32,
  parameter int ID_WIDTH    = 4,
  parameter int NUM_SETS    = 16,
  parameter int NUM_WAYS    = 4,
  parameter int LINE_BYTES  = 64
) (
  input  logic              clk,
  input  logic              rst_n,

  // -----------------------------------------------------------------------
  // CPU-side (OBI-like — connects to Ibex data port or OBI bridge)
  // -----------------------------------------------------------------------
  input  logic              cpu_req,
  output logic              cpu_gnt,
  input  logic [ADDR_WIDTH-1:0] cpu_addr,
  input  logic              cpu_we,
  input  logic [DATA_WIDTH/8-1:0] cpu_be,
  input  logic [DATA_WIDTH-1:0] cpu_wdata,
  output logic              cpu_rvalid,
  output logic [DATA_WIDTH-1:0] cpu_rdata,

  // -----------------------------------------------------------------------
  // AXI4 Master — memory-side (fills + writebacks)
  // -----------------------------------------------------------------------
  // AW channel
  output logic              m_axi_awvalid,
  input  logic              m_axi_awready,
  output logic [ADDR_WIDTH-1:0] m_axi_awaddr,
  output logic [ID_WIDTH-1:0]   m_axi_awid,
  output logic [7:0]        m_axi_awlen,
  output logic [2:0]        m_axi_awsize,
  output logic [1:0]        m_axi_awburst,

  // W channel
  output logic              m_axi_wvalid,
  input  logic              m_axi_wready,
  output logic [DATA_WIDTH-1:0] m_axi_wdata,
  output logic [DATA_WIDTH/8-1:0] m_axi_wstrb,
  output logic              m_axi_wlast,

  // B channel
  input  logic              m_axi_bvalid,
  output logic              m_axi_bready,
  input  logic [1:0]        m_axi_bresp,
  input  logic [ID_WIDTH-1:0] m_axi_bid,

  // AR channel
  output logic              m_axi_arvalid,
  input  logic              m_axi_arready,
  output logic [ADDR_WIDTH-1:0] m_axi_araddr,
  output logic [ID_WIDTH-1:0]   m_axi_arid,
  output logic [7:0]        m_axi_arlen,
  output logic [2:0]        m_axi_arsize,
  output logic [1:0]        m_axi_arburst,

  // R channel
  input  logic              m_axi_rvalid,
  output logic              m_axi_rready,
  input  logic [DATA_WIDTH-1:0] m_axi_rdata,
  input  logic [1:0]        m_axi_rresp,
  input  logic [ID_WIDTH-1:0] m_axi_rid,
  input  logic              m_axi_rlast,

  // -----------------------------------------------------------------------
  // Status
  // -----------------------------------------------------------------------
  output logic              cache_busy
);

  // -----------------------------------------------------------------------
  // Internal controller ↔ AXI adapter signals
  // -----------------------------------------------------------------------
  logic              ctrl_mem_req;
  logic              ctrl_mem_gnt;
  logic [ADDR_WIDTH-1:0] ctrl_mem_addr;
  logic              ctrl_mem_we;
  logic [DATA_WIDTH-1:0] ctrl_mem_wdata;
  logic              ctrl_mem_rvalid;
  logic [DATA_WIDTH-1:0] ctrl_mem_rdata;
  logic              ctrl_mem_last;

  // -----------------------------------------------------------------------
  // Cache controller (contains tag, data, LRU internally)
  // -----------------------------------------------------------------------
  l1_cache_ctrl #(
    .ADDR_WIDTH (ADDR_WIDTH),
    .DATA_WIDTH (DATA_WIDTH),
    .NUM_SETS   (NUM_SETS),
    .NUM_WAYS   (NUM_WAYS),
    .LINE_BYTES (LINE_BYTES)
  ) u_ctrl (
    .clk        (clk),
    .rst_n      (rst_n),
    .cpu_req    (cpu_req),
    .cpu_gnt    (cpu_gnt),
    .cpu_addr   (cpu_addr),
    .cpu_we     (cpu_we),
    .cpu_be     (cpu_be),
    .cpu_wdata  (cpu_wdata),
    .cpu_rvalid (cpu_rvalid),
    .cpu_rdata  (cpu_rdata),
    .mem_req    (ctrl_mem_req),
    .mem_gnt    (ctrl_mem_gnt),
    .mem_addr   (ctrl_mem_addr),
    .mem_we     (ctrl_mem_we),
    .mem_wdata  (ctrl_mem_wdata),
    .mem_rvalid (ctrl_mem_rvalid),
    .mem_rdata  (ctrl_mem_rdata),
    .mem_last   (ctrl_mem_last),
    .cache_busy (cache_busy)
  );

  // -----------------------------------------------------------------------
  // AXI4 Master Adapter
  // Converts the simple burst interface to AXI4 protocol
  // -----------------------------------------------------------------------
  localparam int WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH / 8);

  // FSM for AXI transactions
  typedef enum logic [2:0] {
    AXI_IDLE,
    AXI_AR_WAIT,     // issue AR, wait for arready
    AXI_R_BURST,     // receive read data beats
    AXI_AW_WAIT,     // issue AW, wait for awready
    AXI_W_BURST,     // send write data beats
    AXI_B_WAIT       // wait for write response
  } axi_state_e;

  axi_state_e axi_state, axi_state_next;
  logic [7:0] axi_beat_cnt;
  logic [ADDR_WIDTH-1:0] axi_base_addr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      axi_state     <= AXI_IDLE;
      axi_beat_cnt  <= '0;
      axi_base_addr <= '0;
    end else begin
      axi_state <= axi_state_next;

      case (axi_state)
        AXI_IDLE: begin
          if (ctrl_mem_req) begin
            axi_base_addr <= {ctrl_mem_addr[ADDR_WIDTH-1:$clog2(LINE_BYTES)], {$clog2(LINE_BYTES){1'b0}}};
            axi_beat_cnt  <= '0;
          end
        end

        AXI_R_BURST: begin
          if (m_axi_rvalid && m_axi_rready)
            axi_beat_cnt <= axi_beat_cnt + 1;
        end

        AXI_W_BURST: begin
          if (m_axi_wvalid && m_axi_wready)
            axi_beat_cnt <= axi_beat_cnt + 1;
        end

        default: ;
      endcase
    end
  end

  always_comb begin
    axi_state_next = axi_state;
    case (axi_state)
      AXI_IDLE: begin
        if (ctrl_mem_req && !ctrl_mem_we)
          axi_state_next = AXI_AR_WAIT;
        else if (ctrl_mem_req && ctrl_mem_we)
          axi_state_next = AXI_AW_WAIT;
      end

      AXI_AR_WAIT:
        if (m_axi_arready)
          axi_state_next = AXI_R_BURST;

      AXI_R_BURST:
        if (m_axi_rvalid && m_axi_rlast)
          axi_state_next = AXI_IDLE;

      AXI_AW_WAIT:
        if (m_axi_awready)
          axi_state_next = AXI_W_BURST;

      AXI_W_BURST:
        if (m_axi_wvalid && m_axi_wready && m_axi_wlast)
          axi_state_next = AXI_B_WAIT;

      AXI_B_WAIT:
        if (m_axi_bvalid)
          axi_state_next = AXI_IDLE;

      default:
        axi_state_next = AXI_IDLE;
    endcase
  end

  // AR channel
  assign m_axi_arvalid = (axi_state == AXI_AR_WAIT);
  assign m_axi_araddr  = axi_base_addr;
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = 8'(WORDS_PER_LINE - 1);  // burst length
  assign m_axi_arsize  = 3'($clog2(DATA_WIDTH/8));   // 4 bytes
  assign m_axi_arburst = 2'b01;                       // INCR

  // R channel
  assign m_axi_rready    = (axi_state == AXI_R_BURST);
  assign ctrl_mem_rvalid = (axi_state == AXI_R_BURST) && m_axi_rvalid;
  assign ctrl_mem_rdata  = m_axi_rdata;

  // AW channel
  assign m_axi_awvalid = (axi_state == AXI_AW_WAIT);
  assign m_axi_awaddr  = axi_base_addr;
  assign m_axi_awid    = '0;
  assign m_axi_awlen   = 8'(WORDS_PER_LINE - 1);
  assign m_axi_awsize  = 3'($clog2(DATA_WIDTH/8));
  assign m_axi_awburst = 2'b01;

  // W channel
  assign m_axi_wvalid = (axi_state == AXI_W_BURST);
  assign m_axi_wdata  = ctrl_mem_wdata;
  assign m_axi_wstrb  = '1;  // full word
  assign m_axi_wlast  = (axi_beat_cnt == 8'(WORDS_PER_LINE - 1));

  // B channel
  assign m_axi_bready = (axi_state == AXI_B_WAIT);

  // Grant to controller: AXI accepted the request
  assign ctrl_mem_gnt = ((axi_state == AXI_W_BURST) && m_axi_wready) ||
                        ((axi_state == AXI_R_BURST) && m_axi_rvalid);

endmodule
