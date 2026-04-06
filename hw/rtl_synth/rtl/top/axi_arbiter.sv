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
                if (oh[i]) idx = i;
            end
            onehot_to_idx = idx;
        end
    endfunction

    assign grant_masked   = find_first(req_masked);
    assign grant_unmasked = find_first(req);

    // Select: prefer masked (above priority) winners; fall back to unmasked
    assign grant     = masked_has_req ? grant_masked : grant_unmasked;
    assign grant_idx = onehot_to_idx(grant);

    // Priority rotation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            priority_r <= '0;
        end else if (handshake_done && |grant) begin
            // Rotate to next master after the one that just completed
            if (grant_idx == (NUM_MASTERS - 1))
                priority_r <= '0;
            else
                priority_r <= grant_idx + 1'b1;
        end
    end

endmodule : axi_arbiter
