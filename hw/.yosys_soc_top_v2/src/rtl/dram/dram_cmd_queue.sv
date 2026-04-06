// ===========================================================================
// dram_cmd_queue.sv — DRAM Command Queue (FIFO with age tracking)
// ===========================================================================
// Stores pending read/write requests from AXI interface, tracks age for
// FR-FCFS scheduler priority.
//
// Each entry: {rw, addr[ROW:COL:BANK], burst_len, id, age_counter}
// FIFO discipline with age-based priority output for scheduler.
//
// Resource estimate: ~200 LUTs + entry storage
// ===========================================================================

module dram_cmd_queue #(
    parameter int DEPTH     = 16,       // queue entries
    parameter int ADDR_W    = 28,       // bank+row+col address
    parameter int ID_W      = 4,
    parameter int BLEN_W    = 4         // burst length field
)(
    input  logic                clk,
    input  logic                rst_n,

    // Enqueue interface (from AXI front-end)
    input  logic                enq_valid,
    output logic                enq_ready,
    input  logic                enq_rw,         // 0=read, 1=write
    input  logic [ADDR_W-1:0]   enq_addr,
    input  logic [ID_W-1:0]     enq_id,
    input  logic [BLEN_W-1:0]   enq_blen,

    // Dequeue interface (from scheduler)
    input  logic                deq_valid,      // scheduler picks an entry
    input  logic [$clog2(DEPTH)-1:0] deq_idx,   // which entry to dequeue
    output logic                deq_ready,      // entry was actually removed

    // Queue status
    output logic [$clog2(DEPTH):0] count,       // entries currently occupied
    output logic                   empty,
    output logic                   full,

    // Entry peek ports (scheduler reads all entries)
    output logic [DEPTH-1:0]            entry_valid,
    output logic [DEPTH-1:0]            entry_rw,
    output logic [DEPTH-1:0][ADDR_W-1:0] entry_addr,
    output logic [DEPTH-1:0][ID_W-1:0]   entry_id,
    output logic [DEPTH-1:0][BLEN_W-1:0] entry_blen,
    output logic [DEPTH-1:0][7:0]        entry_age      // saturating age counter
);

    localparam int IDX_W = $clog2(DEPTH);

    // -----------------------------------------------------------------------
    // Storage
    // -----------------------------------------------------------------------
    logic [DEPTH-1:0]            valid_r;
    logic [DEPTH-1:0]            rw_r;
    logic [DEPTH-1:0][ADDR_W-1:0] addr_r;
    logic [DEPTH-1:0][ID_W-1:0]   id_r;
    logic [DEPTH-1:0][BLEN_W-1:0] blen_r;
    logic [DEPTH-1:0][7:0]        age_r;

    logic [IDX_W:0] cnt_r;

    // -----------------------------------------------------------------------
    // Output assignments
    // -----------------------------------------------------------------------
    assign entry_valid = valid_r;
    assign entry_rw    = rw_r;
    assign entry_addr  = addr_r;
    assign entry_id    = id_r;
    assign entry_blen  = blen_r;
    assign entry_age   = age_r;

    assign count = cnt_r;
    assign empty = (cnt_r == 0);
    assign full  = (cnt_r == DEPTH[IDX_W:0]);

    assign enq_ready = !full;
    assign deq_ready = deq_valid && (int'(deq_idx) < DEPTH) && valid_r[deq_idx];

    // -----------------------------------------------------------------------
    // Find first free slot
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Queue logic
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r <= '0;
            rw_r    <= '0;
            age_r   <= '0;
            cnt_r   <= '0;
        end else begin
            // Age increment (saturating at 255)
            for (int i = 0; i < DEPTH; i++) begin
                if (valid_r[i] && age_r[i] < 8'hFF)
                    age_r[i] <= age_r[i] + 1;
            end

            // Enqueue
            if (enq_valid && enq_ready) begin
                valid_r[free_slot] <= 1'b1;
                rw_r[free_slot]    <= enq_rw;
                addr_r[free_slot]  <= enq_addr;
                id_r[free_slot]    <= enq_id;
                blen_r[free_slot]  <= enq_blen;
                age_r[free_slot]   <= '0;
                cnt_r <= cnt_r + 1;
            end

            // Dequeue
            if (deq_valid && valid_r[deq_idx]) begin
                valid_r[deq_idx] <= 1'b0;
                cnt_r <= cnt_r - 1;
            end

            // Handle simultaneous enq+deq
            if (enq_valid && enq_ready && deq_valid && valid_r[deq_idx])
                cnt_r <= cnt_r;  // net zero change
        end
    end

endmodule
