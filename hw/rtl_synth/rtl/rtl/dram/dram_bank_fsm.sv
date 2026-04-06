// ===========================================================================
// dram_bank_fsm.sv — Per-Bank State Machine for DDR3/DDR4 DRAM Controller
// ===========================================================================
// Models the state of a single DRAM bank: IDLE → ACTIVATING → ACTIVE →
// READ/WRITE → PRECHARGING, with timing parameter enforcement.
//
// Each bank FSM instance tracks:
//   - Current open row (row hit detection)
//   - Timing counters (tRCD, tRP, tRAS, tCAS)
//   - State for the scheduler to query
//
// Parameters use DDR3-1600 (Zynq-7020 PS DDR) defaults.
// Resource estimate: ~120 LUTs per bank
// ===========================================================================

module dram_bank_fsm #(
    parameter int ROW_BITS   = 14,
    parameter int COL_BITS   = 10,
    // Timing parameters (in controller clock cycles, typically 200 MHz → 5 ns)
    parameter int T_RCD      = 3,     // ACT → READ/WRITE
    parameter int T_RP       = 3,     // PRE → bank idle
    parameter int T_RAS      = 7,     // ACT → PRE minimum
    parameter int T_RC       = 10,    // ACT → ACT same bank
    parameter int T_RTP      = 2,     // READ → PRE
    parameter int T_WR       = 3,     // last WRITE data → PRE
    parameter int T_CAS      = 3      // CAS latency (READ cmd → data)
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Command interface (from scheduler)
    input  logic                  cmd_valid,
    input  logic [2:0]            cmd_op,       // 3'b001=ACT, 010=READ, 011=WRITE, 100=PRE
    input  logic [ROW_BITS-1:0]   cmd_row,
    input  logic [COL_BITS-1:0]   cmd_col,
    output logic                  cmd_ready,    // bank can accept command

    // Status outputs (to scheduler)
    output logic [2:0]            bank_state,   // current FSM state encoding
    output logic [ROW_BITS-1:0]   open_row,     // currently activated row
    output logic                  row_open,     // 1 = a row is active
    output logic                  row_hit,      // cmd_row matches open_row

    // DRAM PHY command output (active for 1 cycle per command)
    output logic                  phy_act,
    output logic                  phy_read,
    output logic                  phy_write,
    output logic                  phy_pre,
    output logic [ROW_BITS-1:0]   phy_row,
    output logic [COL_BITS-1:0]   phy_col
);

    // -----------------------------------------------------------------------
    // Command opcodes
    // -----------------------------------------------------------------------
    localparam logic [2:0] OP_NOP   = 3'b000;
    localparam logic [2:0] OP_ACT   = 3'b001;
    localparam logic [2:0] OP_READ  = 3'b010;
    localparam logic [2:0] OP_WRITE = 3'b011;
    localparam logic [2:0] OP_PRE   = 3'b100;

    // -----------------------------------------------------------------------
    // FSM states
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE       = 3'd0,
        S_ACTIVATING = 3'd1,  // waiting tRCD
        S_ACTIVE     = 3'd2,  // row open, can accept READ/WRITE
        S_READING    = 3'd3,
        S_WRITING    = 3'd4,
        S_PRECHARGING= 3'd5   // waiting tRP
    } state_t;

    state_t state, state_next;

    // -----------------------------------------------------------------------
    // Timing counters
    // -----------------------------------------------------------------------
    logic [3:0] cnt, cnt_next;                 // general timing countdown
    logic [3:0] ras_cnt, ras_cnt_next;         // tRAS timer (ACT → PRE)
    logic [ROW_BITS-1:0] open_row_reg, open_row_next;

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            cnt          <= '0;
            ras_cnt      <= '0;
            open_row_reg <= '0;
        end else begin
            state        <= state_next;
            cnt          <= cnt_next;
            ras_cnt      <= ras_cnt_next;
            open_row_reg <= open_row_next;
        end
    end

    // -----------------------------------------------------------------------
    // Output assignments
    // -----------------------------------------------------------------------
    assign bank_state = state;
    assign open_row   = open_row_reg;
    assign row_open   = (state == S_ACTIVE || state == S_READING || state == S_WRITING);
    assign row_hit    = row_open && (cmd_row == open_row_reg);

    // -----------------------------------------------------------------------
    // Next-state logic
    // -----------------------------------------------------------------------
    always_comb begin
        state_next    = state;
        cnt_next      = (cnt > 0) ? cnt - 1 : '0;
        ras_cnt_next  = (ras_cnt > 0) ? ras_cnt - 1 : '0;
        open_row_next = open_row_reg;

        cmd_ready  = 1'b0;
        phy_act    = 1'b0;
        phy_read   = 1'b0;
        phy_write  = 1'b0;
        phy_pre    = 1'b0;
        phy_row    = '0;
        phy_col    = '0;

        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                cmd_ready = 1'b1;
                if (cmd_valid && cmd_op == OP_ACT) begin
                    phy_act       = 1'b1;
                    phy_row       = cmd_row;
                    open_row_next = cmd_row;
                    cnt_next      = T_RCD[3:0] - 1;
                    ras_cnt_next  = T_RAS[3:0] - 1;
                    state_next    = S_ACTIVATING;
                end
            end

            // -----------------------------------------------------------
            S_ACTIVATING: begin
                if (cnt == 0) begin
                    state_next = S_ACTIVE;
                end
            end

            // -----------------------------------------------------------
            S_ACTIVE: begin
                cmd_ready = 1'b1;
                if (cmd_valid) begin
                    if (cmd_op == OP_READ) begin
                        phy_read   = 1'b1;
                        phy_col    = cmd_col;
                        cnt_next   = T_CAS[3:0] - 1;
                        state_next = S_READING;
                    end else if (cmd_op == OP_WRITE) begin
                        phy_write  = 1'b1;
                        phy_col    = cmd_col;
                        cnt_next   = T_WR[3:0] - 1;
                        state_next = S_WRITING;
                    end else if (cmd_op == OP_PRE && ras_cnt == 0) begin
                        phy_pre    = 1'b1;
                        cnt_next   = T_RP[3:0] - 1;
                        state_next = S_PRECHARGING;
                    end
                end
            end

            // -----------------------------------------------------------
            S_READING: begin
                if (cnt == 0) begin
                    state_next = S_ACTIVE;  // back to open-row state
                end
                // Allow PRE after tRTP
                if (cmd_valid && cmd_op == OP_PRE && cnt == 0 && ras_cnt == 0) begin
                    phy_pre    = 1'b1;
                    cnt_next   = T_RP[3:0] - 1;
                    state_next = S_PRECHARGING;
                end
            end

            // -----------------------------------------------------------
            S_WRITING: begin
                if (cnt == 0) begin
                    state_next = S_ACTIVE;
                end
                if (cmd_valid && cmd_op == OP_PRE && cnt == 0 && ras_cnt == 0) begin
                    phy_pre    = 1'b1;
                    cnt_next   = T_RP[3:0] - 1;
                    state_next = S_PRECHARGING;
                end
            end

            // -----------------------------------------------------------
            S_PRECHARGING: begin
                if (cnt == 0) begin
                    open_row_next = '0;
                    state_next    = S_IDLE;
                end
            end

            default: state_next = S_IDLE;
        endcase
    end

endmodule
