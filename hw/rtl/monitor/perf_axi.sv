// perf_axi.sv — AXI-Lite mapped Performance Counter array.
// NUM_COUNTERS free-running 32-bit counters, each incremented when the
// corresponding event_valid bit is high.  Read counters via AXI-Lite slave.
// Write any address to clear the corresponding counter.
// Address map: counter[i] @ byte offset i*4.

`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
module perf_axi #(
    parameter int NUM_COUNTERS   = 6,
    parameter int COUNTER_WIDTH  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // -------------------------------------------------------------------
    // Event inputs (one bit per counter; increment when asserted)
    // -------------------------------------------------------------------
    input  wire [NUM_COUNTERS-1:0]     event_valid,

    // -------------------------------------------------------------------
    // AXI-Lite slave (read counters / write-to-clear)
    // -------------------------------------------------------------------
    input  wire                        s_axi_awvalid,
    output logic                       s_axi_awready,
    input  wire [7:0]                  s_axi_awaddr,
    input  wire                        s_axi_wvalid,
    output logic                       s_axi_wready,
    input  wire [31:0]                 s_axi_wdata,
    input  wire [3:0]                  s_axi_wstrb,
    output logic                       s_axi_bvalid,
    input  wire                        s_axi_bready,
    output logic [1:0]                 s_axi_bresp,
    input  wire                        s_axi_arvalid,
    output logic                       s_axi_arready,
    input  wire [7:0]                  s_axi_araddr,
    output logic                       s_axi_rvalid,
    input  wire                        s_axi_rready,
    output logic [31:0]                s_axi_rdata,
    output logic [1:0]                 s_axi_rresp
);

    // -----------------------------------------------------------------------
    // Counter storage
    // -----------------------------------------------------------------------
    logic [COUNTER_WIDTH-1:0] counters [0:NUM_COUNTERS-1];

    // -----------------------------------------------------------------------
    // AXI-Lite channel registers
    // -----------------------------------------------------------------------
    int axi_widx;
    int axi_ridx;

    always_comb begin
        axi_widx = int'(s_axi_awaddr[7:2]);
        axi_ridx = int'(s_axi_araddr[7:2]);
    end

    logic [NUM_COUNTERS-1:0] clear_en;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clear_en        <= '0;
            s_axi_awready   <= 1'b0;
            s_axi_wready    <= 1'b0;
            s_axi_bvalid    <= 1'b0;
            s_axi_bresp     <= 2'b00;
            s_axi_arready   <= 1'b0;
            s_axi_rvalid    <= 1'b0;
            s_axi_rdata     <= '0;
            s_axi_rresp     <= 2'b00;
        end else begin
            // --- Write channel ---
            clear_en <= '0;
            if (s_axi_awvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
                if (s_axi_wvalid) begin
                    if (axi_widx >= 0 && axi_widx < NUM_COUNTERS)
                        clear_en[axi_widx] <= 1'b1;
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp  <= 2'b00;
                end
            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;

            // --- Read channel ---
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                s_axi_rresp   <= 2'b00;
                s_axi_rdata   <= (axi_ridx >= 0 && axi_ridx < NUM_COUNTERS) ?
                                  counters[axi_ridx][31:0] : 32'b0;
            end else begin
                s_axi_arready <= 1'b0;
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        end
    end

    // -----------------------------------------------------------------------
    // Counter update: clear takes priority over increment
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_COUNTERS; i++)
                counters[i] <= '0;
        end else begin
            for (int i = 0; i < NUM_COUNTERS; i++) begin
                if (clear_en[i])
                    counters[i] <= '0;
                else if (event_valid[i])
                    counters[i] <= counters[i] + 1;
            end
        end
    end

endmodule

`default_nettype wire
