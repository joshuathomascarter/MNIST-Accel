// ===========================================================================
// axi_arbiter.sv — Round-Robin AXI Arbiter
// ===========================================================================
// Per-slave round-robin arbiter. Given N masters requesting access to a
// single slave, grants one master at a time and rotates priority each cycle
// a grant is accepted (handshake completes).
//
// Extracted from axi_crossbar.sv for standalone testability, formal
// verification of starvation-freedom, and reuse in the NoC crossbar.
//
// Properties (verifiable by formal / SVA):
//   1. At most one grant active at a time (one-hot or zero).
//   2. Starvation-free: every persistent requester is granted within
//      NUM_MASTERS cycles of the last grant.
//   3. Grant is stable while valid && !ready (no mid-handshake switch).
//
// Resource estimate: ~30 LUTs per instance (for NUM_MASTERS=2)
// ===========================================================================

module axi_arbiter #(
    parameter int unsigned NUM_MASTERS = 2
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Request vector: one bit per master
    input  logic [NUM_MASTERS-1:0]     req,

    // Handshake feedback: assert when granted transaction completes
    // (valid && ready on the arbitrated channel)
    input  logic                       handshake_done,

    // Grant outputs
    output logic [NUM_MASTERS-1:0]     grant,       // one-hot grant
    output logic [$clog2(NUM_MASTERS)-1:0] grant_idx // binary index of granted master
);

    localparam int unsigned IDX_W = $clog2(NUM_MASTERS);

    // Priority pointer — indicates the highest-priority requester.
    // After a successful handshake, rotates to (granted + 1) % N.
    logic [IDX_W-1:0] priority_r;

    // Masked request: requesters at or above the priority pointer
    logic [NUM_MASTERS-1:0] req_masked;
    // Unmasked winner (from req_masked first, then raw req if masked is zero)
    logic [NUM_MASTERS-1:0] grant_masked;
    logic [NUM_MASTERS-1:0] grant_unmasked;
    logic                   masked_has_req;

    // Mask: zero out requesters below priority_r
    always_comb begin
        for (int i = 0; i < NUM_MASTERS; i++) begin
            req_masked[i] = req[i] && (i >= priority_r);
        end
    end

    assign masked_has_req = |req_masked;

    // Priority encoder — find lowest-index set bit
    function [NUM_MASTERS-1:0] find_first;
        input [NUM_MASTERS-1:0] vec;
        reg [NUM_MASTERS-1:0] result;
        integer i;
        begin
            result = {NUM_MASTERS{1'b0}};
            for (i = 0; i < NUM_MASTERS; i = i + 1) begin
                if (vec[i] && result == {NUM_MASTERS{1'b0}}) begin
                    result[i] = 1'b1;
                end
            end
            find_first = result;
        end
    endfunction

    function [IDX_W-1:0] onehot_to_idx;
        input [NUM_MASTERS-1:0] oh;
        reg [IDX_W-1:0] idx;
        integer i;
        begin
            idx = {IDX_W{1'b0}};
            for (i = 0; i < NUM_MASTERS; i = i + 1) begin
                if (oh[i]) idx = IDX_W'(i);
            end
            onehot_to_idx = idx;
        end
    endfunction

    assign grant_masked   = find_first(req_masked);
    assign grant_unmasked = find_first(req);

    // Raw (unlocked) combinational grant — can change every cycle
    logic [NUM_MASTERS-1:0] grant_raw;
    logic [IDX_W-1:0]       grant_raw_idx;

    assign grant_raw     = masked_has_req ? grant_masked : grant_unmasked;
    assign grant_raw_idx = onehot_to_idx(grant_raw);

    // -----------------------------------------------------------------------
    // Lock register: once a master is granted, freeze it until the AW/AR
    // handshake completes.  Without this, a higher-priority master arriving
    // mid-transaction can steal the grant, causing s_awvalid to glitch low
    // before the slave has seen awready — an AXI protocol violation.
    // -----------------------------------------------------------------------
    logic [NUM_MASTERS-1:0] grant_locked;
    logic [IDX_W-1:0]       grant_locked_idx;
    logic                   lock_active;

    assign grant_locked_idx = onehot_to_idx(grant_locked);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            priority_r   <= '0;
            grant_locked <= '0;
            lock_active  <= 1'b0;
        end else begin
            if (lock_active) begin
                if (handshake_done) begin
                    // Handshake complete: release lock, rotate priority away
                    // from the master that just finished
                    lock_active <= 1'b0;
                    if (grant_locked_idx == IDX_W'(NUM_MASTERS - 1))
                        priority_r <= '0;
                    else
                        priority_r <= grant_locked_idx + 1'b1;
                end
                // else: hold — grant_locked is stable, slave still deciding
            end else begin
                // Free: latch the next winner if one is requesting
                if (|grant_raw) begin
                    if (handshake_done) begin
                        // Same-cycle grant + handshake: slave was immediately
                        // ready, so s_awready/s_arready fired on the very
                        // first cycle the grant appeared.  Skip locking —
                        // just rotate priority so the next requester gets a
                        // fair turn.  Without this, the arbiter enters locked
                        // and waits for a second handshake that never comes
                        // (slave already accepted), causing phantom AW/AR
                        // transactions on every subsequent cycle s_awready
                        // goes high (ghost transactions → SRAM stuck +
                        // IO bridge deadlocked in IO_B_WAIT).
                        if (grant_raw_idx == IDX_W'(NUM_MASTERS - 1))
                            priority_r <= '0;
                        else
                            priority_r <= grant_raw_idx + 1'b1;
                    end else begin
                        grant_locked <= grant_raw;
                        lock_active  <= 1'b1;
                    end
                end
            end
        end
    end

    // Outputs: locked version exposed so crossbar sees a stable grant
    // for the full duration of the handshake
    assign grant     = lock_active ? grant_locked : grant_raw;
    assign grant_idx = lock_active ? grant_locked_idx : grant_raw_idx;

endmodule : axi_arbiter
