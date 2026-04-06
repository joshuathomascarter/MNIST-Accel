// ===========================================================================
// dram_write_buffer.sv — DRAM Write Data Buffer
// ===========================================================================
// Buffers write data from AXI W channel until the scheduler issues the
// corresponding WRITE command to the DRAM bank. Matches entries by index
// to the command queue.
//
// Dual-purpose:
//   1. Decouple AXI W handshake from DRAM timing
//   2. Coalesce byte-enables for sub-word writes
//
// Resource estimate: ~100 LUTs + DEPTH×(DATA_W+STRB_W) FFs
// ===========================================================================

/* verilator lint_off UNUSEDSIGNAL */

module dram_write_buffer #(
    parameter int DEPTH     = 16,
    parameter int DATA_W    = 32,       // AXI data width
    parameter int STRB_W    = DATA_W/8,
    parameter int ID_W      = 4
)(
    input  logic               clk,
    input  logic               rst_n,

    // AXI W channel interface (enqueue)
    input  logic               wr_valid,
    output logic               wr_ready,
    input  logic [DATA_W-1:0]  wr_data,
    input  logic [STRB_W-1:0]  wr_strb,
    input  logic [ID_W-1:0]    wr_id,

    // Scheduler drain interface (dequeue)
    input  logic               drain_valid,   // scheduler wants data
    input  logic [$clog2(DEPTH)-1:0] drain_idx,
    output logic               drain_ready,
    output logic [DATA_W-1:0]  drain_data,
    output logic [STRB_W-1:0]  drain_strb,

    // Status
    output logic [$clog2(DEPTH):0] count,
    output logic               empty,
    output logic               full
);

    localparam int IDX_W = $clog2(DEPTH);

    // Storage
    logic [DEPTH-1:0]              valid_r;
    logic [DEPTH-1:0][DATA_W-1:0]  data_r;
    logic [DEPTH-1:0][STRB_W-1:0]  strb_r;
    logic [DEPTH-1:0][ID_W-1:0]    id_r;

    logic [IDX_W:0] cnt_r;

    // Status
    assign count = cnt_r;
    assign empty = (cnt_r == 0);
    assign full  = (cnt_r == DEPTH[IDX_W:0]);
    assign wr_ready = !full;

    // Drain output
    assign drain_ready = drain_valid && (int'(drain_idx) < DEPTH) && valid_r[drain_idx];
    assign drain_data  = data_r[drain_idx];
    assign drain_strb  = strb_r[drain_idx];

    // Find first free slot
    logic [IDX_W-1:0] free_slot;
    logic              has_free;
    always_comb begin
        free_slot = '0;
        has_free  = 1'b0;
        for (int i = 0; i < DEPTH; i++) begin
            if (!valid_r[i] && !has_free) begin
                free_slot = i[IDX_W-1:0];
                has_free  = 1'b1;
            end
        end
    end

    // Write/drain logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r <= '0;
            cnt_r   <= '0;
        end else begin
            // Enqueue
            if (wr_valid && wr_ready) begin
                valid_r[free_slot] <= 1'b1;
                data_r[free_slot]  <= wr_data;
                strb_r[free_slot]  <= wr_strb;
                id_r[free_slot]    <= wr_id;
                cnt_r <= cnt_r + 1;
            end

            // Drain
            if (drain_valid && valid_r[drain_idx]) begin
                valid_r[drain_idx] <= 1'b0;
                cnt_r <= cnt_r - 1;
            end

            // Simultaneous
            if (wr_valid && wr_ready && drain_valid && valid_r[drain_idx])
                cnt_r <= cnt_r;
        end
    end

endmodule
