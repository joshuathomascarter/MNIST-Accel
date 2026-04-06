// ===========================================================================
// dram_deterministic_mode.sv — Fixed-Latency DRAM Access for HFT Path
// ===========================================================================
// Wraps the DRAM controller with a latency-padding stage that guarantees
// every read completes in exactly FIXED_LATENCY cycles from the AXI AR
// handshake.  Used by the HFT critical path to have cycle-deterministic
// memory access regardless of row-hit/miss variability.
//
// Implementation: a shift-register delay line absorbs the variable gap
// between actual data arrival (earliest = T_CAS, worst ≈ T_RP+T_RCD+T_CAS)
// and the fixed deadline.
//
// FIXED_LATENCY must be ≥ worst-case read latency through the controller.
// Recommended: T_RP + T_RCD + T_CAS + 4 (scheduler + pipeline overhead).
//
// Resource estimate: ~80 LUTs, ~40 FFs per outstanding read slot
// ===========================================================================

module dram_deterministic_mode #(
    parameter int DATA_W        = 32,
    parameter int ID_W          = 4,
    parameter int FIXED_LATENCY = 16,  // cycles from AR accept to R valid
    parameter int MAX_OUTSTANDING = 4
)(
    input  logic                 clk,
    input  logic                 rst_n,

    // Enable: when 0, passthrough (variable latency)
    input  logic                 det_enable,

    // From AXI AR handshake (pulse when arvalid & arready)
    input  logic                 ar_accepted,
    input  logic [ID_W-1:0]     ar_id,

    // From DRAM controller R channel
    input  logic                 dram_rvalid,
    input  logic [DATA_W-1:0]   dram_rdata,
    input  logic [ID_W-1:0]     dram_rid,

    // To AXI master R channel
    output logic                 det_rvalid,
    output logic [DATA_W-1:0]   det_rdata,
    output logic [ID_W-1:0]     det_rid,
    output logic                 det_rlast,

    // Error: deadline missed (data didn't arrive before FIXED_LATENCY)
    output logic                 err_deadline_miss
);

    localparam int CTR_W = $clog2(FIXED_LATENCY + 1);
    localparam int SLOT_W = $clog2(MAX_OUTSTANDING);

    // Slot tracking — flat arrays (no struct for Yosys compatibility)
    logic [MAX_OUTSTANDING-1:0]             sl_valid;
    logic [MAX_OUTSTANDING-1:0]             sl_data_ready;
    logic [CTR_W-1:0]                       sl_countdown  [MAX_OUTSTANDING];
    logic [DATA_W-1:0]                      sl_data       [MAX_OUTSTANDING];
    logic [ID_W-1:0]                        sl_id         [MAX_OUTSTANDING];

    // Find free slot
    logic [SLOT_W-1:0] free_idx;
    logic               free_found;

    always_comb begin
        free_found = 1'b0;
        free_idx   = '0;
        for (int i = 0; i < MAX_OUTSTANDING; i++) begin
            if (!sl_valid[i] && !free_found) begin
                free_found = 1'b1;
                free_idx   = i;
            end
        end
    end

    // Match incoming data to slot by ID
    logic [SLOT_W-1:0] match_idx;
    logic               match_found;

    always_comb begin
        match_found = 1'b0;
        match_idx   = '0;
        for (int i = 0; i < MAX_OUTSTANDING; i++) begin
            if (sl_valid[i] && !sl_data_ready[i] &&
                sl_id[i] == dram_rid && !match_found) begin
                match_found = 1'b1;
                match_idx   = i;
            end
        end
    end

    // Find slot that reached zero countdown
    logic [SLOT_W-1:0] fire_idx;
    logic               fire_found;

    always_comb begin
        fire_found = 1'b0;
        fire_idx   = '0;
        for (int i = 0; i < MAX_OUTSTANDING; i++) begin
            if (sl_valid[i] && sl_countdown[i] == '0 && !fire_found) begin
                fire_found = 1'b1;
                fire_idx   = i;
            end
        end
    end

    // Slot state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sl_valid      <= '0;
            sl_data_ready <= '0;
            for (int i = 0; i < MAX_OUTSTANDING; i++) begin
                sl_countdown[i] <= '0;
                sl_data[i]      <= '0;
                sl_id[i]        <= '0;
            end
        end else begin
            // Allocate new slot on AR accept
            if (ar_accepted && free_found && det_enable) begin
                sl_valid[free_idx]      <= 1'b1;
                sl_data_ready[free_idx] <= 1'b0;
                sl_countdown[free_idx]  <= (FIXED_LATENCY - 1);
                sl_data[free_idx]       <= '0;
                sl_id[free_idx]         <= ar_id;
            end

            // Capture data when it arrives
            if (dram_rvalid && match_found && det_enable) begin
                sl_data_ready[match_idx] <= 1'b1;
                sl_data[match_idx]       <= dram_rdata;
            end

            // Decrement countdowns
            for (int i = 0; i < MAX_OUTSTANDING; i++) begin
                if (sl_valid[i] && sl_countdown[i] != '0)
                    sl_countdown[i] <= sl_countdown[i] - 1'b1;
            end

            // Release firing slot
            if (fire_found && det_enable) begin
                sl_valid[fire_idx] <= 1'b0;
            end
        end
    end

    // Output MUX
    always_comb begin
        if (!det_enable) begin
            // Passthrough mode
            det_rvalid = dram_rvalid;
            det_rdata  = dram_rdata;
            det_rid    = dram_rid;
            det_rlast  = dram_rvalid;
            err_deadline_miss = 1'b0;
        end else begin
            det_rvalid = fire_found && sl_data_ready[fire_idx];
            det_rdata  = fire_found ? sl_data[fire_idx] : '0;
            det_rid    = fire_found ? sl_id[fire_idx]   : '0;
            det_rlast  = det_rvalid;
            // Deadline miss: countdown hit 0 but data not ready
            err_deadline_miss = fire_found && !sl_data_ready[fire_idx];
        end
    end

endmodule
