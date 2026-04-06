// ===========================================================================
// dram_refresh_ctrl.sv — DRAM Refresh Controller
// ===========================================================================
// Issues periodic ALL-BANK REFRESH commands according to tREFI timing.
// When refresh is due, asserts `ref_req`; scheduler must grant via `ref_ack`.
// After grant, issues `ref_cmd` pulse and waits tRFC before releasing.
//
// DDR3-1600 defaults:
//   tREFI = 7.8 µs → 1560 cycles @ 200 MHz
//   tRFC  = 260 ns → 52 cycles @ 200 MHz (for 4 Gbit)
//
// Resource estimate: ~50 LUTs
// ===========================================================================

module dram_refresh_ctrl #(
    parameter int T_REFI  = 1560,   // refresh interval (clk cycles)
    parameter int T_RFC   = 52,     // refresh cycle time
    parameter int CTR_W   = 11      // counter width (must hold T_REFI)
)(
    input  logic  clk,
    input  logic  rst_n,

    // Scheduler handshake
    output logic  ref_req,          // refresh needed
    input  logic  ref_ack,          // scheduler grants refresh slot

    // PHY command
    output logic  ref_cmd,          // pulse: issue REFRESH to DRAM
    output logic  ref_busy          // 1 during tRFC cooldown
);

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_COUNT,       // counting down to next refresh
        S_REQUEST,     // asserting ref_req, waiting for ack
        S_REFRESH      // tRFC cooldown
    } state_t;

    state_t state, state_next;

    logic [CTR_W-1:0] cnt, cnt_next;

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_COUNT;
            cnt   <= T_REFI[CTR_W-1:0] - 1;
        end else begin
            state <= state_next;
            cnt   <= cnt_next;
        end
    end

    // -----------------------------------------------------------------------
    // Next-state
    // -----------------------------------------------------------------------
    always_comb begin
        state_next = state;
        cnt_next   = cnt;
        ref_req    = 1'b0;
        ref_cmd    = 1'b0;
        ref_busy   = 1'b0;

        case (state)
            S_COUNT: begin
                if (cnt == 0) begin
                    state_next = S_REQUEST;
                end else begin
                    cnt_next = cnt - 1;
                end
            end

            S_REQUEST: begin
                ref_req = 1'b1;
                if (ref_ack) begin
                    ref_cmd    = 1'b1;
                    cnt_next   = T_RFC[CTR_W-1:0] - 1;
                    state_next = S_REFRESH;
                end
            end

            S_REFRESH: begin
                ref_busy = 1'b1;
                if (cnt == 0) begin
                    cnt_next   = T_REFI[CTR_W-1:0] - 1;
                    state_next = S_COUNT;
                end else begin
                    cnt_next = cnt - 1;
                end
            end

            default: state_next = S_COUNT;
        endcase
    end

endmodule
