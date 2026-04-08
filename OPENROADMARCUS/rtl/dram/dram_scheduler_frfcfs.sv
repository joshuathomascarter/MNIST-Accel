// ===========================================================================
// dram_scheduler_frfcfs.sv — First-Ready First-Come-First-Serve Scheduler
// ===========================================================================
// Priority order:
//   1. Refresh (highest — must not be starved)
//   2. Row-hit requests (FCFS among hits)
//   3. Row-miss / row-closed requests (FCFS by age)
//
// Scans the command queue entries, classifies each by bank state (hit/miss),
// picks the best candidate, and issues bank commands (ACT/READ/WRITE/PRE).
//
// Interfaces with:
//   - dram_cmd_queue (entry peek + deq)
//   - dram_bank_fsm × NUM_BANKS (status query + cmd issue)
//   - dram_refresh_ctrl (ref_req/ref_ack handshake)
//
// Resource estimate: ~400 LUTs for 8 banks × 16 queue depth
// ===========================================================================

/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */

module dram_scheduler_frfcfs #(
    parameter int NUM_BANKS   = 8,
    parameter int QUEUE_DEPTH = 16,
    parameter int ADDR_W      = 29, // BANK_BITS+ROW_BITS+COL_BITS+2 byte offset
    parameter int ROW_BITS    = 14,
    parameter int COL_BITS    = 10,
    parameter int BANK_BITS   = 3,
    parameter int ID_W        = 4,
    parameter int BLEN_W      = 4
)(
    input  logic  clk,
    input  logic  rst_n,

    // ---- Command queue peek interface ----
    input  logic [QUEUE_DEPTH-1:0]                entry_valid,
    input  logic [QUEUE_DEPTH-1:0]                entry_rw,
    input  logic [QUEUE_DEPTH-1:0][ADDR_W-1:0]    entry_addr,
    input  logic [QUEUE_DEPTH-1:0][ID_W-1:0]      entry_id,
    input  logic [QUEUE_DEPTH-1:0][BLEN_W-1:0]    entry_blen,
    input  logic [QUEUE_DEPTH-1:0][7:0]            entry_age,

    // Dequeue command
    output logic                                   deq_valid,
    output logic [$clog2(QUEUE_DEPTH)-1:0]         deq_idx,

    // ---- Per-bank FSM status (from dram_bank_fsm) ----
    input  logic [NUM_BANKS-1:0][2:0]              bank_state,
    input  logic [NUM_BANKS-1:0][ROW_BITS-1:0]     bank_open_row,
    input  logic [NUM_BANKS-1:0]                   bank_row_open,

    // ---- Per-bank command issue ----
    output logic [NUM_BANKS-1:0]                   bank_cmd_valid,
    output logic [NUM_BANKS-1:0][2:0]              bank_cmd_op,
    output logic [NUM_BANKS-1:0][ROW_BITS-1:0]     bank_cmd_row,
    output logic [NUM_BANKS-1:0][COL_BITS-1:0]     bank_cmd_col,

    // ---- Refresh handshake ----
    input  logic                                   ref_req,
    output logic                                   ref_ack,
    input  logic                                   ref_busy,

    // ---- Read/write data handshake (to PHY / data path) ----
    output logic                                   data_rd_valid,
    output logic                                   data_wr_valid,
    output logic [ID_W-1:0]                        data_id,
    output logic                                   sched_busy
);

    localparam int QIX_W = $clog2(QUEUE_DEPTH);

    // Bank FSM state encodings (match dram_bank_fsm.sv)
    localparam logic [2:0] BS_IDLE   = 3'd0;
    localparam logic [2:0] BS_ACTIVE = 3'd2;

    // Command opcodes
    localparam logic [2:0] OP_ACT   = 3'b001;
    localparam logic [2:0] OP_READ  = 3'b010;
    localparam logic [2:0] OP_WRITE = 3'b011;
    localparam logic [2:0] OP_PRE   = 3'b100;

    // -----------------------------------------------------------------------
    // Address decomposition for each queue entry
    // -----------------------------------------------------------------------
    logic [QUEUE_DEPTH-1:0][BANK_BITS-1:0] q_bank;
    logic [QUEUE_DEPTH-1:0][ROW_BITS-1:0]  q_row;
    logic [QUEUE_DEPTH-1:0][COL_BITS-1:0]  q_col;

    // RBC decode (matching dram_addr_decoder MODE=0)
    localparam int BYTE_OFF = 2;  // 32-bit bus (4 bytes → $clog2(4)=2)
    genvar gi;
    generate
        for (gi = 0; gi < QUEUE_DEPTH; gi++) begin : gen_decode
            assign q_col[gi]  = entry_addr[gi][BYTE_OFF +: COL_BITS];
            assign q_bank[gi] = entry_addr[gi][BYTE_OFF + COL_BITS +: BANK_BITS];
            assign q_row[gi]  = entry_addr[gi][BYTE_OFF + COL_BITS + BANK_BITS +: ROW_BITS];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Classify each entry: row-hit vs row-miss vs bank-busy
    // -----------------------------------------------------------------------
    logic [QUEUE_DEPTH-1:0] is_row_hit;
    logic [QUEUE_DEPTH-1:0] is_bank_ready;   // bank in IDLE or ACTIVE

    always_comb begin
        for (int i = 0; i < QUEUE_DEPTH; i++) begin
            logic [BANK_BITS-1:0] b;
            b = q_bank[i];
            is_bank_ready[i] = entry_valid[i] &&
                                (bank_state[b] == BS_IDLE || bank_state[b] == BS_ACTIVE);
            is_row_hit[i]    = entry_valid[i] && bank_row_open[b] &&
                                (bank_open_row[b] == q_row[i]) &&
                                (bank_state[b] == BS_ACTIVE);
        end
    end

    // -----------------------------------------------------------------------
    // Pipeline-register the hit/ready classification vectors.
    // This removes the bank_state mux + row-address comparator chain
    // (~18 combinational levels) from the FR-FCFS scan critical path.
    // The scheduler FSM takes many cycles per transaction so 1-cycle stale
    // classification is safe — bank FSMs still gate illegal commands.
    // -----------------------------------------------------------------------
    logic [QUEUE_DEPTH-1:0] is_row_hit_r;
    logic [QUEUE_DEPTH-1:0] is_bank_ready_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_row_hit_r    <= '0;
            is_bank_ready_r <= '0;
        end else begin
            is_row_hit_r    <= is_row_hit;
            is_bank_ready_r <= is_bank_ready;
        end
    end

    // -----------------------------------------------------------------------
    // FR-FCFS selection: pick best candidate
    // -----------------------------------------------------------------------
    logic              found_hit, found_ready;
    logic [QIX_W-1:0]  best_idx;
    logic [7:0]         best_age;

    always_comb begin
        found_hit   = 1'b0;
        found_ready = 1'b0;
        best_idx    = '0;
        best_age    = '0;

        // Pass 1: row-hit entries (FCFS = oldest age wins)
        // Uses registered _r vectors: bank_state chain is off this path.
        // Cross-check with live entry_valid to avoid stale picks after dequeue.
        for (int i = 0; i < QUEUE_DEPTH; i++) begin
            if (is_row_hit_r[i] && entry_valid[i] && entry_age[i] >= best_age) begin
                found_hit = 1'b1;
                best_idx  = i[QIX_W-1:0];
                best_age  = entry_age[i];
            end
        end

        // Pass 2: if no hit, take oldest bank-ready entry
        if (!found_hit) begin
            best_age = '0;
            for (int i = 0; i < QUEUE_DEPTH; i++) begin
                if (is_bank_ready_r[i] && entry_valid[i] && entry_age[i] >= best_age) begin
                    found_ready = 1'b1;
                    best_idx    = i[QIX_W-1:0];
                    best_age    = entry_age[i];
                end
            end
        end
    end

    logic has_candidate;
    assign has_candidate = found_hit || found_ready;

    // -----------------------------------------------------------------------
    // Scheduler FSM
    // -----------------------------------------------------------------------
    // One-hot encoding: each state is a single bit, removing binary decode
    // logic from the FSM output path.  Vivado attribute re-enforces this.
    typedef enum logic [4:0] {
        SCH_IDLE      = 5'b00001,
        SCH_ISSUE_PRE = 5'b00010,  // issue precharge for row-miss
        SCH_ISSUE_ACT = 5'b00100,  // issue activate
        SCH_ISSUE_RW  = 5'b01000,  // issue read or write
        SCH_REFRESH   = 5'b10000   // refresh in progress
    } sch_state_t;

    (* fsm_encoding = "one_hot" *) sch_state_t sch_state, sch_state_next;

    logic [QIX_W-1:0]      sel_idx_r;
    logic [BANK_BITS-1:0]  sel_bank_r;
    logic [ROW_BITS-1:0]   sel_row_r;
    logic [COL_BITS-1:0]   sel_col_r;
    logic                   sel_rw_r;
    logic [ID_W-1:0]       sel_id_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sch_state  <= SCH_IDLE;
            sel_idx_r  <= '0;
            sel_bank_r <= '0;
            sel_row_r  <= '0;
            sel_col_r  <= '0;
            sel_rw_r   <= 1'b0;
            sel_id_r   <= '0;
        end else begin
            sch_state <= sch_state_next;
            if (sch_state == SCH_IDLE && has_candidate && !ref_req) begin
                sel_idx_r  <= best_idx;
                sel_bank_r <= q_bank[best_idx];
                sel_row_r  <= q_row[best_idx];
                sel_col_r  <= q_col[best_idx];
                sel_rw_r   <= entry_rw[best_idx];
                sel_id_r   <= entry_id[best_idx];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Output logic
    // -----------------------------------------------------------------------
    always_comb begin
        sch_state_next = sch_state;

        // Default: all outputs inactive
        deq_valid      = 1'b0;
        deq_idx        = '0;
        ref_ack        = 1'b0;
        data_rd_valid  = 1'b0;
        data_wr_valid  = 1'b0;
        data_id        = '0;
        sched_busy     = 1'b1;

        for (int b = 0; b < NUM_BANKS; b++) begin
            bank_cmd_valid[b] = 1'b0;
            bank_cmd_op[b]    = 3'b000;
            bank_cmd_row[b]   = '0;
            bank_cmd_col[b]   = '0;
        end

        case (sch_state)
            SCH_IDLE: begin
                sched_busy = 1'b0;
                // Refresh takes priority
                if (ref_req && !ref_busy) begin
                    ref_ack        = 1'b1;
                    sch_state_next = SCH_REFRESH;
                end else if (has_candidate) begin
                    if (is_row_hit_r[best_idx]) begin
                        // Row hit → go straight to RW
                        sch_state_next = SCH_ISSUE_RW;
                    end else if (bank_row_open[q_bank[best_idx]]) begin
                        // Row miss — need PRE first
                        sch_state_next = SCH_ISSUE_PRE;
                    end else begin
                        // Bank idle — need ACT
                        sch_state_next = SCH_ISSUE_ACT;
                    end
                end
            end

            SCH_ISSUE_PRE: begin
                bank_cmd_valid[sel_bank_r] = 1'b1;
                bank_cmd_op[sel_bank_r]    = OP_PRE;
                sch_state_next             = SCH_ISSUE_ACT;
            end

            SCH_ISSUE_ACT: begin
                // Wait for bank to be IDLE before issuing ACT
                if (bank_state[sel_bank_r] == BS_IDLE) begin
                    bank_cmd_valid[sel_bank_r] = 1'b1;
                    bank_cmd_op[sel_bank_r]    = OP_ACT;
                    bank_cmd_row[sel_bank_r]   = sel_row_r;
                    sch_state_next             = SCH_ISSUE_RW;
                end
            end

            SCH_ISSUE_RW: begin
                // Wait for bank to be ACTIVE
                if (bank_state[sel_bank_r] == BS_ACTIVE) begin
                    bank_cmd_valid[sel_bank_r] = 1'b1;
                    bank_cmd_op[sel_bank_r]    = sel_rw_r ? OP_WRITE : OP_READ;
                    bank_cmd_col[sel_bank_r]   = sel_col_r;

                    data_rd_valid = !sel_rw_r;
                    data_wr_valid = sel_rw_r;
                    data_id       = sel_id_r;

                    // Dequeue the entry
                    deq_valid = 1'b1;
                    deq_idx   = sel_idx_r;

                    sch_state_next = SCH_IDLE;
                end
            end

            SCH_REFRESH: begin
                // Wait for refresh to complete
                if (!ref_busy)
                    sch_state_next = SCH_IDLE;
            end

            default: sch_state_next = SCH_IDLE;
        endcase
    end

endmodule
