// ===========================================================================
// dram_ctrl_top.sv — DRAM Controller Top-Level Integration
// ===========================================================================
// Integrates all DRAM controller sub-modules:
//   - AXI4 slave front-end (accepts AXI read/write)
//   - dram_addr_decoder
//   - dram_cmd_queue
//   - dram_write_buffer
//   - dram_scheduler_frfcfs
//   - dram_bank_fsm × NUM_BANKS
//   - dram_refresh_ctrl
//
// AXI4 slave interface on one side, DRAM PHY command interface on the other.
// Suitable for DDR3 on Zynq-7020 (8 banks, 14-bit row, 10-bit col).
//
// Resource estimate: ~1500 LUTs, 0 DSP, 0 BRAM
// ===========================================================================

module dram_ctrl_top #(
    parameter int AXI_ADDR_W  = 32,
    parameter int AXI_DATA_W  = 32,
    parameter int AXI_ID_W    = 4,
    parameter int NUM_BANKS   = 8,
    parameter int ROW_BITS    = 14,
    parameter int COL_BITS    = 10,
    parameter int BANK_BITS   = 3,
    parameter int QUEUE_DEPTH = 16,
    parameter int ADDR_MODE   = 0,      // 0=RBC, 1=BRC
    // Timing
    parameter int T_RCD       = 3,
    parameter int T_RP        = 3,
    parameter int T_RAS       = 7,
    parameter int T_RC        = 10,
    parameter int T_RTP       = 2,
    parameter int T_WR        = 3,
    parameter int T_CAS       = 3,
    parameter int T_REFI      = 1560,
    parameter int T_RFC       = 52
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // ===== AXI4 Slave Interface =====
    // Write address
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [AXI_ADDR_W-1:0]    s_axi_awaddr,
    input  logic [AXI_ID_W-1:0]      s_axi_awid,
    input  logic [7:0]               s_axi_awlen,
    input  logic [2:0]               s_axi_awsize,

    // Write data
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    input  logic [AXI_DATA_W-1:0]    s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0]  s_axi_wstrb,
    input  logic                      s_axi_wlast,

    // Write response
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    output logic [1:0]               s_axi_bresp,
    output logic [AXI_ID_W-1:0]      s_axi_bid,

    // Read address
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    input  logic [AXI_ADDR_W-1:0]    s_axi_araddr,
    input  logic [AXI_ID_W-1:0]      s_axi_arid,
    input  logic [7:0]               s_axi_arlen,
    input  logic [2:0]               s_axi_arsize,

    // Read data
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,
    output logic [AXI_DATA_W-1:0]    s_axi_rdata,
    output logic [1:0]               s_axi_rresp,
    output logic [AXI_ID_W-1:0]      s_axi_rid,
    output logic                      s_axi_rlast,

    // ===== DRAM PHY Interface =====
    output logic [NUM_BANKS-1:0]      dram_phy_act,
    output logic [NUM_BANKS-1:0]      dram_phy_read,
    output logic [NUM_BANKS-1:0]      dram_phy_write,
    output logic [NUM_BANKS-1:0]      dram_phy_pre,
    output logic [ROW_BITS-1:0]       dram_phy_row,
    output logic [COL_BITS-1:0]       dram_phy_col,
    output logic                      dram_phy_ref,      // all-bank refresh
    output logic [AXI_DATA_W-1:0]     dram_phy_wdata,
    output logic [AXI_DATA_W/8-1:0]  dram_phy_wstrb,
    input  logic [AXI_DATA_W-1:0]     dram_phy_rdata,
    input  logic                      dram_phy_rdata_valid,

    // Status
    output logic                      ctrl_busy
);

    // -----------------------------------------------------------------------
    // Internal address field width
    // -----------------------------------------------------------------------
    localparam int DRAM_ADDR_W = BANK_BITS + ROW_BITS + COL_BITS + 1; // +1 byte offset
    localparam int BLEN_W = 4;
    localparam int QIX_W  = $clog2(QUEUE_DEPTH);

    // -----------------------------------------------------------------------
    // Address Decoder
    // -----------------------------------------------------------------------
    logic [BANK_BITS-1:0] dec_bank;
    logic [ROW_BITS-1:0]  dec_row;
    logic [COL_BITS-1:0]  dec_col;

    dram_addr_decoder #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .BANK_BITS(BANK_BITS),
        .ROW_BITS(ROW_BITS),
        .COL_BITS(COL_BITS),
        .BUS_BYTES(AXI_DATA_W/8),
        .MODE(ADDR_MODE)
    ) u_addr_dec (
        .axi_addr(s_axi_arvalid ? s_axi_araddr : s_axi_awaddr),
        .bank(dec_bank),
        .row(dec_row),
        .col(dec_col)
    );

    // -----------------------------------------------------------------------
    // Command Queue
    // -----------------------------------------------------------------------
    logic                       enq_valid, enq_ready;
    logic                       enq_rw;
    logic [DRAM_ADDR_W-1:0]    enq_addr;
    logic [AXI_ID_W-1:0]       enq_id;
    logic [BLEN_W-1:0]         enq_blen;

    logic                       deq_valid;
    logic [QIX_W-1:0]          deq_idx;
    logic                       deq_ready;

    logic [QUEUE_DEPTH-1:0]                  q_entry_valid;
    logic [QUEUE_DEPTH-1:0]                  q_entry_rw;
    logic [QUEUE_DEPTH-1:0][DRAM_ADDR_W-1:0] q_entry_addr;
    logic [QUEUE_DEPTH-1:0][AXI_ID_W-1:0]    q_entry_id;
    logic [QUEUE_DEPTH-1:0][BLEN_W-1:0]      q_entry_blen;
    logic [QUEUE_DEPTH-1:0][7:0]             q_entry_age;
    logic [QIX_W:0]                          q_count;
    logic                                    q_empty, q_full;

    dram_cmd_queue #(
        .DEPTH(QUEUE_DEPTH),
        .ADDR_W(DRAM_ADDR_W),
        .ID_W(AXI_ID_W),
        .BLEN_W(BLEN_W)
    ) u_cmd_queue (
        .clk(clk), .rst_n(rst_n),
        .enq_valid(enq_valid), .enq_ready(enq_ready),
        .enq_rw(enq_rw), .enq_addr(enq_addr),
        .enq_id(enq_id), .enq_blen(enq_blen),
        .deq_valid(deq_valid), .deq_idx(deq_idx), .deq_ready(deq_ready),
        .count(q_count), .empty(q_empty), .full(q_full),
        .entry_valid(q_entry_valid), .entry_rw(q_entry_rw),
        .entry_addr(q_entry_addr), .entry_id(q_entry_id),
        .entry_blen(q_entry_blen), .entry_age(q_entry_age)
    );

    // -----------------------------------------------------------------------
    // Write Buffer
    // -----------------------------------------------------------------------
    logic                    wb_wr_valid, wb_wr_ready;
    logic [AXI_DATA_W-1:0]  wb_wr_data;
    logic [AXI_DATA_W/8-1:0] wb_wr_strb;

    logic                    wb_drain_valid, wb_drain_ready;
    logic [QIX_W-1:0]       wb_drain_idx;
    logic [AXI_DATA_W-1:0]  wb_drain_data;
    logic [AXI_DATA_W/8-1:0] wb_drain_strb;

    dram_write_buffer #(
        .DEPTH(QUEUE_DEPTH),
        .DATA_W(AXI_DATA_W),
        .ID_W(AXI_ID_W)
    ) u_write_buf (
        .clk(clk), .rst_n(rst_n),
        .wr_valid(wb_wr_valid), .wr_ready(wb_wr_ready),
        .wr_data(wb_wr_data), .wr_strb(wb_wr_strb),
        .wr_id(s_axi_awid),
        .drain_valid(wb_drain_valid), .drain_idx(wb_drain_idx),
        .drain_ready(wb_drain_ready),
        .drain_data(wb_drain_data), .drain_strb(wb_drain_strb),
        .count(), .empty(), .full()
    );

    // -----------------------------------------------------------------------
    // Bank FSMs
    // -----------------------------------------------------------------------
    logic [NUM_BANKS-1:0][2:0]             bk_state;
    logic [NUM_BANKS-1:0][ROW_BITS-1:0]    bk_open_row;
    logic [NUM_BANKS-1:0]                  bk_row_open;

    logic [NUM_BANKS-1:0]                  bk_cmd_valid;
    logic [NUM_BANKS-1:0][2:0]             bk_cmd_op;
    logic [NUM_BANKS-1:0][ROW_BITS-1:0]    bk_cmd_row;
    logic [NUM_BANKS-1:0][COL_BITS-1:0]    bk_cmd_col;

    genvar bi;
    generate
        for (bi = 0; bi < NUM_BANKS; bi++) begin : gen_bank
            dram_bank_fsm #(
                .ROW_BITS(ROW_BITS), .COL_BITS(COL_BITS),
                .T_RCD(T_RCD), .T_RP(T_RP), .T_RAS(T_RAS),
                .T_RC(T_RC), .T_RTP(T_RTP), .T_WR(T_WR), .T_CAS(T_CAS)
            ) u_bank (
                .clk(clk), .rst_n(rst_n),
                .cmd_valid(bk_cmd_valid[bi]),
                .cmd_op(bk_cmd_op[bi]),
                .cmd_row(bk_cmd_row[bi]),
                .cmd_col(bk_cmd_col[bi]),
                .cmd_ready(),
                .bank_state(bk_state[bi]),
                .open_row(bk_open_row[bi]),
                .row_open(bk_row_open[bi]),
                .row_hit(),
                .phy_act(dram_phy_act[bi]),
                .phy_read(dram_phy_read[bi]),
                .phy_write(dram_phy_write[bi]),
                .phy_pre(dram_phy_pre[bi]),
                .phy_row(/* muxed below */),
                .phy_col(/* muxed below */)
            );
        end
    endgenerate

    // PHY row/col: OR-reduce from all banks (only one active at a time)
    always_comb begin
        dram_phy_row = '0;
        dram_phy_col = '0;
        for (int b = 0; b < NUM_BANKS; b++) begin
            if (dram_phy_act[b])
                dram_phy_row = bk_cmd_row[b];
            if (dram_phy_read[b] || dram_phy_write[b])
                dram_phy_col = bk_cmd_col[b];
        end
    end

    // -----------------------------------------------------------------------
    // Refresh Controller
    // -----------------------------------------------------------------------
    logic ref_req, ref_ack, ref_cmd, ref_busy;

    dram_refresh_ctrl #(
        .T_REFI(T_REFI), .T_RFC(T_RFC)
    ) u_refresh (
        .clk(clk), .rst_n(rst_n),
        .ref_req(ref_req), .ref_ack(ref_ack),
        .ref_cmd(ref_cmd), .ref_busy(ref_busy)
    );

    assign dram_phy_ref = ref_cmd;

    // -----------------------------------------------------------------------
    // FR-FCFS Scheduler
    // -----------------------------------------------------------------------
    logic sched_data_rd, sched_data_wr;
    logic [AXI_ID_W-1:0] sched_data_id;
    logic sched_busy;

    dram_scheduler_frfcfs #(
        .NUM_BANKS(NUM_BANKS), .QUEUE_DEPTH(QUEUE_DEPTH),
        .ADDR_W(DRAM_ADDR_W), .ROW_BITS(ROW_BITS),
        .COL_BITS(COL_BITS), .BANK_BITS(BANK_BITS),
        .ID_W(AXI_ID_W), .BLEN_W(BLEN_W)
    ) u_scheduler (
        .clk(clk), .rst_n(rst_n),
        .entry_valid(q_entry_valid), .entry_rw(q_entry_rw),
        .entry_addr(q_entry_addr), .entry_id(q_entry_id),
        .entry_blen(q_entry_blen), .entry_age(q_entry_age),
        .deq_valid(deq_valid), .deq_idx(deq_idx),
        .bank_state(bk_state), .bank_open_row(bk_open_row),
        .bank_row_open(bk_row_open),
        .bank_cmd_valid(bk_cmd_valid), .bank_cmd_op(bk_cmd_op),
        .bank_cmd_row(bk_cmd_row), .bank_cmd_col(bk_cmd_col),
        .ref_req(ref_req), .ref_ack(ref_ack), .ref_busy(ref_busy),
        .data_rd_valid(sched_data_rd), .data_wr_valid(sched_data_wr),
        .data_id(sched_data_id), .sched_busy(sched_busy)
    );

    // -----------------------------------------------------------------------
    // AXI Front-End: Accepts AR/AW/W and enqueues into command queue
    // -----------------------------------------------------------------------

    // Simple state machine for AXI acceptance
    typedef enum logic [1:0] {
        AXI_IDLE,
        AXI_WRITE_DATA,
        AXI_WRITE_RESP
    } axi_state_t;

    axi_state_t axi_st, axi_st_next;
    logic [AXI_ID_W-1:0] aw_id_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_st  <= AXI_IDLE;
            aw_id_r <= '0;
        end else begin
            axi_st <= axi_st_next;
            if (s_axi_awvalid && s_axi_awready)
                aw_id_r <= s_axi_awid;
        end
    end

    always_comb begin
        axi_st_next     = axi_st;
        s_axi_awready   = 1'b0;
        s_axi_arready   = 1'b0;
        s_axi_wready    = 1'b0;
        s_axi_bvalid    = 1'b0;
        s_axi_bresp     = 2'b00;
        s_axi_bid       = '0;
        enq_valid        = 1'b0;
        enq_rw           = 1'b0;
        enq_addr         = '0;
        enq_id           = '0;
        enq_blen         = '0;
        wb_wr_valid      = 1'b0;
        wb_wr_data       = '0;
        wb_wr_strb       = '0;

        case (axi_st)
            AXI_IDLE: begin
                // Accept read address
                if (s_axi_arvalid && enq_ready) begin
                    s_axi_arready = 1'b1;
                    enq_valid     = 1'b1;
                    enq_rw        = 1'b0;
                    enq_addr      = s_axi_araddr[DRAM_ADDR_W-1:0];
                    enq_id        = s_axi_arid;
                    enq_blen      = s_axi_arlen[BLEN_W-1:0];
                end
                // Accept write address
                else if (s_axi_awvalid && enq_ready) begin
                    s_axi_awready = 1'b1;
                    enq_valid     = 1'b1;
                    enq_rw        = 1'b1;
                    enq_addr      = s_axi_awaddr[DRAM_ADDR_W-1:0];
                    enq_id        = s_axi_awid;
                    enq_blen      = s_axi_awlen[BLEN_W-1:0];
                    axi_st_next   = AXI_WRITE_DATA;
                end
            end

            AXI_WRITE_DATA: begin
                s_axi_wready = wb_wr_ready;
                wb_wr_valid  = s_axi_wvalid;
                wb_wr_data   = s_axi_wdata;
                wb_wr_strb   = s_axi_wstrb;
                if (s_axi_wvalid && s_axi_wlast && wb_wr_ready)
                    axi_st_next = AXI_WRITE_RESP;
            end

            AXI_WRITE_RESP: begin
                s_axi_bvalid = 1'b1;
                s_axi_bresp  = 2'b00;
                s_axi_bid    = aw_id_r;
                if (s_axi_bready)
                    axi_st_next = AXI_IDLE;
            end

            default: axi_st_next = AXI_IDLE;
        endcase
    end

    // -----------------------------------------------------------------------
    // Read data path: PHY rdata → AXI R channel
    // -----------------------------------------------------------------------
    assign s_axi_rvalid = dram_phy_rdata_valid;
    assign s_axi_rdata  = dram_phy_rdata;
    assign s_axi_rresp  = 2'b00;
    assign s_axi_rid    = sched_data_id;
    assign s_axi_rlast  = dram_phy_rdata_valid;  // single-beat for now

    // Write data to PHY
    assign wb_drain_valid = sched_data_wr;
    assign wb_drain_idx   = deq_idx;
    assign dram_phy_wdata = wb_drain_data;
    assign dram_phy_wstrb = wb_drain_strb;

    // Status
    assign ctrl_busy = sched_busy || !q_empty;

endmodule
