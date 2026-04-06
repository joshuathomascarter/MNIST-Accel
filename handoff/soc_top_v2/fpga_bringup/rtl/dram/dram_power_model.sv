// ===========================================================================
// dram_power_model.sv — DRAM CKE / Power-Down Controller
// ===========================================================================
// Monitors DRAM activity and drives CKE (Clock-Enable) low to enter
// power-down mode when the controller is idle for IDLE_THRESHOLD cycles.
//
// Power-down modes modelled:
//   1. Active power-down (APD): banks stay open, CKE driven low.
//      Wakeup: 1 cycle (tXP ≈ 3 at DDR3-1600).
//   2. Precharge power-down (PPD): all banks precharged, CKE low.
//      Deeper savings, wakeup: tXP cycles.
//
// This module does NOT model self-refresh (requires PS DDRC cooperation
// on Zynq-7020).  It provides a simulation power estimator by counting
// active, idle, and power-down cycles.
//
// Resource estimate: ~40 LUTs, ~30 FFs
// ===========================================================================

module dram_power_model #(
    parameter int IDLE_THRESHOLD = 64,   // cycles idle before power-down
    parameter int T_XP           = 3,    // exit power-down latency
    parameter int NUM_BANKS      = 8
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // From controller
    input  logic                    ctrl_busy,         // any command pending
    input  logic [NUM_BANKS-1:0]    bank_row_open,     // per-bank open flag

    // CKE control (active-high)
    output logic                    cke,

    // Power counters (for simulation analysis)
    output logic [31:0]             cnt_active_cycles,
    output logic [31:0]             cnt_idle_cycles,
    output logic [31:0]             cnt_pd_cycles,

    // State exposure for testbench
    output logic [1:0]              power_state
);

    typedef enum logic [1:0] {
        PWR_ACTIVE   = 2'b00,
        PWR_IDLE     = 2'b01,
        PWR_PD_ENTRY = 2'b10,
        PWR_PD       = 2'b11
    } pwr_state_t;

    pwr_state_t state, state_next;

    logic [$clog2(IDLE_THRESHOLD+1)-1:0] idle_cnt;
    logic [$clog2(T_XP+1)-1:0]          wakeup_cnt;
    logic all_precharged;

    assign all_precharged = (bank_row_open == '0);
    assign power_state = state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= PWR_ACTIVE;
            idle_cnt         <= '0;
            wakeup_cnt       <= '0;
            cnt_active_cycles <= '0;
            cnt_idle_cycles  <= '0;
            cnt_pd_cycles    <= '0;
        end else begin
            state <= state_next;

            // Counters
            case (state)
                PWR_ACTIVE:   cnt_active_cycles <= cnt_active_cycles + 1;
                PWR_IDLE:     cnt_idle_cycles   <= cnt_idle_cycles + 1;
                PWR_PD_ENTRY: cnt_pd_cycles     <= cnt_pd_cycles + 1;
                PWR_PD:       cnt_pd_cycles     <= cnt_pd_cycles + 1;
            endcase

            // Idle counter
            if (ctrl_busy || state == PWR_ACTIVE)
                idle_cnt <= '0;
            else if (state == PWR_IDLE)
                idle_cnt <= idle_cnt + 1;

            // Wakeup counter
            if (state == PWR_PD && ctrl_busy)
                wakeup_cnt <= ($clog2(T_XP+1))'(T_XP);
            else if (wakeup_cnt != 0)
                wakeup_cnt <= wakeup_cnt - 1;
        end
    end

    always_comb begin
        state_next = state;
        cke = 1'b1;

        case (state)
            PWR_ACTIVE: begin
                if (!ctrl_busy)
                    state_next = PWR_IDLE;
            end

            PWR_IDLE: begin
                if (ctrl_busy)
                    state_next = PWR_ACTIVE;
                else if (idle_cnt >= IDLE_THRESHOLD[$clog2(IDLE_THRESHOLD+1)-1:0])
                    state_next = PWR_PD_ENTRY;
            end

            PWR_PD_ENTRY: begin
                cke = 1'b0;
                state_next = PWR_PD;
            end

            PWR_PD: begin
                cke = 1'b0;
                if (ctrl_busy)
                    state_next = PWR_ACTIVE;  // wakeup_cnt handles tXP
            end
        endcase
    end

endmodule
