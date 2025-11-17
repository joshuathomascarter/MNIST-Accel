`timescale 1ns/1ps
`default_nettype none
// tb_scheduler.sv - Unit testbench for scheduler (RS / ping-pong behavior)
// This TB is written to instantiate existing RTL named `schedular` (note spelling)
module tb_scheduler;
    // Clock / reset
    reg clk = 0;
    always #5 clk = ~clk; // 10ns period -> 100MHz

    reg rst_n;
    reg start;
    reg abort;

    // Config from CSR (narrow widths in DUT; use full 32-bit values then truncate)
    reg [9:0]  M;   // matches M_W=10 in scheduler
    reg [9:0]  N;
    reg [11:0] K;
    reg [5:0]  Tm;
    reg [5:0]  Tn;
    reg [5:0]  Tk;

    // Optional CSR-provided tile counts (not used in this TB)
    reg [9:0]  MT_csr;
    reg [9:0]  NT_csr;
    reg [11:0] KT_csr;

    // Bank readiness inputs (driven by RS engine in real design)
    reg valid_A_ping, valid_A_pong, valid_B_ping, valid_B_pong;

    // DUT outputs
    wire rd_en;
    wire [5:0] k_idx; // TK_W=6
    wire bank_sel_rd_A, bank_sel_rd_B;
    wire clr, en;
    // Concrete mask sizes derived from log2 parameters (number of PEs per dim)
    localparam int MAX_TM = (1 << TM_W);
    localparam int MAX_TN = (1 << TN_W);

    wire [MAX_TM-1:0] en_mask_row;
    wire [MAX_TN-1:0] en_mask_col;
    wire busy;
    wire done_tile;
    wire [9:0] m_tile;
    wire [9:0] n_tile;
    wire [11:0] k_tile;
    wire [31:0] cycles_tile;
    wire [31:0] stall_cycles;

    // Parameter widths must match DUT -- duplicate the scheduler parameters here
    localparam int M_W  = 10;
    localparam int N_W  = 10;
    localparam int K_W  = 12;
    localparam int TM_W = 6;
    localparam int TN_W = 6;
    localparam int TK_W = 6;
    // Match PREPRIME used to synthesize DUT (update if you changed scheduler)
    localparam bit PREPRIME = 0;

    // Instantiate DUT -- note: repo file uses module name `schedular` (misspelling)
    scheduler #(
        .M_W(M_W), .N_W(N_W), .K_W(K_W),
        .TM_W(TM_W), .TN_W(TN_W), .TK_W(TK_W),
        .PREPRIME(PREPRIME),
        .USE_CSR_COUNTS(1)
     ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .abort(abort),
        .M(M), .N(N), .K(K), .Tm(Tm), .Tn(Tn), .Tk(Tk),
    .MT_csr(MT_csr), .NT_csr(NT_csr), .KT_csr(KT_csr),
        .valid_A_ping(valid_A_ping), .valid_A_pong(valid_A_pong),
        .valid_B_ping(valid_B_ping), .valid_B_pong(valid_B_pong),
        .rd_en(rd_en), .k_idx(k_idx),
        .bank_sel_rd_A(bank_sel_rd_A), .bank_sel_rd_B(bank_sel_rd_B),
        .clr(clr), .en(en), .en_mask_row(en_mask_row), .en_mask_col(en_mask_col),
        .busy(busy), .done_tile(done_tile),
        .m_tile(m_tile), .n_tile(n_tile), .k_tile(k_tile),
        .cycles_tile(cycles_tile), .stall_cycles(stall_cycles)
    );

    integer errors;
    integer tile_count;
    integer MT, NT, KT;
    integer expected_cycles_per_tile;
    integer tolerance = 3; // allow +/- tolerance cycles
    integer rs_total_injected; // sum of latencies RS emulator inserted
    integer stall_cycles_at_tile_start;
    integer rs_injected_at_tile_start;
    bit expecting_first_rd;

    // Helper: compute ceil div in TB
    function int ceil_div(input int a, input int b);
        if (b == 0) return 0;
        return (a + b - 1) / b;
    endfunction

    // Stimulus / monitors
    initial begin
        $display("\n--- tb_scheduler: BEGIN ---");
        // Default values
        rst_n = 0; start = 0; abort = 0;
        M = 4; N = 4; K = 8; Tm = 2; Tn = 2; Tk = 4;
    MT_csr = 0; NT_csr = 0; KT_csr = 0;
        valid_A_ping = 0; valid_A_pong = 0; valid_B_ping = 0; valid_B_pong = 0;
        errors = 0;
        tile_count = 0;

        // Reset pulse
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // Compute expected tile counts
        MT = ceil_div(M, Tm);
        NT = ceil_div(N, Tn);
        KT = ceil_div(K, Tk);

        $display("Config: M=%0d N=%0d K=%0d  Tm=%0d Tn=%0d Tk=%0d  => MT=%0d NT=%0d KT=%0d",
                  M, N, K, Tm, Tn, Tk, MT, NT, KT);

    // Initialize RS emulator tracking and provide bank readiness emulation:
        // Initially mark ping banks ready (bank 0) and delay pong readiness to emulate preload
        valid_A_ping = 1;
        valid_B_ping = 1;
        valid_A_pong = 0;
        valid_B_pong = 0;
    rs_total_injected = 0;
    stall_cycles_at_tile_start = 0;
    rs_injected_at_tile_start = 0;
    expecting_first_rd = 0;

        // Pulse start
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // RS emulator: whenever scheduler switches to require pong bank, assert pong ready after latency
        fork
            begin : rs_emul
                forever begin
                    @(posedge clk);
                    // If scheduler requests k_tile odd (pong) but valid_pong not set, emulate preload latency
                    // Determine the bank the RS should *prepare next* (opposite of current read parity)
                    int next_bank = ~k_tile[0]; // 0 -> ping next, 1 -> pong next
                    // If next bank ping and not valid, inject random latency then assert
                    if (next_bank == 0 && !(valid_A_ping && valid_B_ping)) begin
                        int lat = $urandom_range(0,5);
                        repeat (lat) @(posedge clk);
                        rs_total_injected += lat;
                        valid_A_ping <= 1;
                        valid_B_ping <= 1;
                    end
                    // If next bank pong and not valid, inject random latency then assert
                    if (next_bank == 1 && !(valid_A_pong && valid_B_pong)) begin
                        int lat = $urandom_range(0,5);
                        repeat (lat) @(posedge clk);
                        rs_total_injected += lat;
                        valid_A_pong <= 1;
                        valid_B_pong <= 1;
                    end
                    // Keep current read bank valid asserted (if read bank is ping when k_tile[0]==0)
                    if (k_tile[0] == 0) begin
                        valid_A_ping <= 1;
                        valid_B_ping <= 1;
                    end else begin
                        valid_A_pong <= valid_A_pong;
                        valid_B_pong <= valid_B_pong;
                    end
                end
            end
        join_none

        // Monitor tile completion events and check cycles_tile (uses per-tile effective sizes)
        fork
            begin : monitor_tiles
                integer total_tiles = MT * NT;
                int m_off_tb, n_off_tb, k_off_tb;
                int m_rem_tb, n_rem_tb, k_rem_tb;
                int Tm_eff_tb, Tn_eff_tb, Tk_eff_tb;
                while (tile_count < total_tiles) begin
                    @(posedge clk);
                    // Check: DUT should not assert rd_en unless selected banks are valid
                    if (rd_en) begin
                        if (bank_sel_rd_A == 0 && !(valid_A_ping)) begin
                            $display("ASSERTION FAIL: rd_en asserted while A ping not valid at time %0t", $time);
                            errors = errors + 1;
                        end
                        if (bank_sel_rd_A == 1 && !(valid_A_pong)) begin
                            $display("ASSERTION FAIL: rd_en asserted while A pong not valid at time %0t", $time);
                            errors = errors + 1;
                        end
                    end

                    if (done_tile) begin
                        tile_count = tile_count + 1;

                        // Compute effective sizes for this just-completed tile using DUT-reported m_tile/n_tile/k_tile
                        m_off_tb = m_tile * Tm;
                        n_off_tb = n_tile * Tn;
                        k_off_tb = k_tile * Tk;
                        m_rem_tb = (M > m_off_tb) ? (M - m_off_tb) : 0;
                        n_rem_tb = (N > n_off_tb) ? (N - n_off_tb) : 0;
                        k_rem_tb = (K > k_off_tb) ? (K - k_off_tb) : 0;
                        Tm_eff_tb = (m_rem_tb > Tm) ? Tm : m_rem_tb;
                        Tn_eff_tb = (n_rem_tb > Tn) ? Tn : n_rem_tb;
                        Tk_eff_tb = (k_rem_tb > Tk) ? Tk : k_rem_tb;

                        expected_cycles_per_tile = Tk_eff_tb + (Tm_eff_tb - 1) + (Tn_eff_tb - 1) + (PREPRIME ? 0 : 1);
                        // Check stall_cycles delta equals RS injected latency during this tile
                        int stall_delta = stall_cycles - stall_cycles_at_tile_start;
                        int injected_delta = rs_total_injected - rs_injected_at_tile_start;
                        if (stall_delta != injected_delta) begin
                            $display("ERROR: stall_cycles delta (%0d) != RS injected delta (%0d) for tile %0d", stall_delta, injected_delta, tile_count);
                            errors = errors + 1;
                        end

                        // PREPRIME check: identify first STREAM_K cycle by watching rd_en/en sequence
                        // If PREPRIME==1 then first STREAM_K cycle should have en==1; else en==0 then en==1 next
                        // We can't directly capture STREAM_K start here easily; do a simple waveform check using expecting_first_rd
                        if (expecting_first_rd) begin
                            // expecting_first_rd was set at tile start; check recent rd/en samples
                            // Simple heuristic: if PREPRIME==0 we expect the first observed rd_en had en==0
                            // If PREPRIME==1, the first observed rd_en should have en==1
                            // This relies on monitor observing rd_en/en transitions; if ambiguous, increase TB sampling resolution.
                            // (No-op here; leave future detailed sampling for stricter checks.)
                        end

                        $display("Tile %0d done: m_tile=%0d n_tile=%0d k_tile=%0d cycles_tile=%0d stall_cycles=%0d expected≈%0d (Tm_eff=%0d Tn_eff=%0d Tk_eff=%0d)",
                                 tile_count, m_tile, n_tile, k_tile, cycles_tile, stall_cycles, expected_cycles_per_tile, Tm_eff_tb, Tn_eff_tb, Tk_eff_tb);

                        if ( (cycles_tile < expected_cycles_per_tile - tolerance) ||
                             (cycles_tile > expected_cycles_per_tile + tolerance) ) begin
                            $display("ERROR: cycles_tile out of expected range for tile %0d", tile_count);
                            errors = errors + 1;
                        end
                    end
                end
            end
        join_none

        // Wait until full sweep completes or timeout using a watchdog and cleanly disable background forks
        int MAX_CYCLES = 2000;
        bit timed_out = 0;
        fork
            begin : watchdog_f
                repeat (MAX_CYCLES) @(posedge clk);
                timed_out = 1;
            end
        join_none

        // Wait for either completion or timeout
        wait (tile_count == MT*NT || timed_out);
        // Stop background processes
        disable rs_emul;
        disable monitor_tiles;

        if (timed_out) begin
            $display("ERROR: timeout after %0d cycles", MAX_CYCLES);
            errors = errors + 1;
        end

        // Give simulator a couple cycles to settle
        repeat (5) @(posedge clk);

        if (errors == 0) $display("tb_scheduler: PASS");
        else $display("tb_scheduler: FAIL - %0d errors", errors);

        $finish;
    end

endmodule
`default_nettype wire
