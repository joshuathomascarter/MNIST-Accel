`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_csr.sv - Unit testbench for csr.v
// Verifies byte-exact RW parity and field mapping
// -----------------------------------------------------------------------------
module tb_csr;
    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;

    reg wr_en, rd_en;
    reg [7:0] addr;
    reg [31:0] wdata;
    wire [31:0] rdata;

    // Exposed fields
    wire [2:0] ctrl;
    wire [7:0] M, N, K, Tm, Tn, Tk, Sa, Sw;
    wire [15:0] m_idx, n_idx, k_idx, pkt_len_max;
    wire bank_sel_wr_A, bank_sel_wr_B, bank_sel_rd_A, bank_sel_rd_B, crc_en;
    wire [2:0] status;
    
    // Performance counter inputs (simulated)
    reg [31:0] perf_total_cycles;
    reg [31:0] perf_active_cycles;
    reg [31:0] perf_idle_cycles;
    reg [31:0] perf_cache_hits;
    reg [31:0] perf_cache_misses;
    reg [31:0] perf_decode_count;
    reg [127:0] result_data;
    reg core_busy, core_done_tile_pulse;
    reg core_bank_sel_rd_A, core_bank_sel_rd_B;
    reg rx_crc_error, rx_illegal_cmd;

    csr #(.ADDR_W(8)) dut (
        .clk(clk), .rst_n(rst_n),
        .csr_wen(wr_en), .csr_ren(rd_en), .csr_addr(addr), 
        .csr_wdata(wdata), .csr_rdata(rdata),
        .core_busy(core_busy), .core_done_tile_pulse(core_done_tile_pulse),
        .core_bank_sel_rd_A(core_bank_sel_rd_A), .core_bank_sel_rd_B(core_bank_sel_rd_B),
        .rx_crc_error(rx_crc_error), .rx_illegal_cmd(rx_illegal_cmd),
        .perf_total_cycles(perf_total_cycles),
        .perf_active_cycles(perf_active_cycles),
        .perf_idle_cycles(perf_idle_cycles),
        .perf_cache_hits(perf_cache_hits),
        .perf_cache_misses(perf_cache_misses),
        .perf_decode_count(perf_decode_count),
        .result_data(result_data),
        .start_pulse(), .abort_pulse(), .irq_en(),
        .M(), .N(), .K(), .Tm(), .Tn(), .Tk(),
        .m_idx(), .n_idx(), .k_idx(),
        .bank_sel_wr_A(), .bank_sel_wr_B(),
        .bank_sel_rd_A(), .bank_sel_rd_B(),
        .Sa_bits(), .Sw_bits(),
        .uart_len_max(), .uart_crc_en()
    );

    integer errors;
    initial begin
        $display("\n--- tb_csr: BEGIN ---");
        rst_n = 0; wr_en = 0; rd_en = 0; addr = 0; wdata = 0; errors = 0;
        perf_total_cycles = 32'd1000;
        perf_active_cycles = 32'd800;
        perf_idle_cycles = 32'd200;
        perf_cache_hits = 32'd450;
        perf_cache_misses = 32'd50;
        perf_decode_count = 32'd100;
        result_data = 128'h0;
        core_busy = 0; core_done_tile_pulse = 0;
        core_bank_sel_rd_A = 0; core_bank_sel_rd_B = 0;
        rx_crc_error = 0; rx_illegal_cmd = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // CTRL (byte address 0x00)
        addr = 8'h00; wdata = 3'b101; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata[2:0] !== 3'b101) begin $display("CTRL RW error"); errors = errors + 1; end

        // DIMS_M (byte address 0x04)
        addr = 8'h04; wdata = 32'd7; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd7) begin $display("DIMS_M RW error"); errors = errors + 1; end

        // DIMS_N (byte address 0x08)
        addr = 8'h08; wdata = 32'd11; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd11) begin $display("DIMS_N RW error"); errors = errors + 1; end

        // DIMS_K (byte address 0x0C)
        addr = 8'h0C; wdata = 32'd22; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd22) begin $display("DIMS_K RW error"); errors = errors + 1; end

        // TILES_Tm (byte address 0x10)
        addr = 8'h10; wdata = 32'd1; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd1) begin $display("TILES_Tm RW error"); errors = errors + 1; end

        // TILES_Tn (byte address 0x14)
        addr = 8'h14; wdata = 32'd2; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd2) begin $display("TILES_Tn RW error"); errors = errors + 1; end

        // TILES_Tk (byte address 0x18)
        addr = 8'h18; wdata = 32'd3; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd3) begin $display("TILES_Tk RW error"); errors = errors + 1; end

        // INDEX_m (byte address 0x1C)
        addr = 8'h1C; wdata = 32'd9; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd9) begin $display("INDEX_m RW error"); errors = errors + 1; end

        // INDEX_n (byte address 0x20)
        addr = 8'h20; wdata = 32'd5; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd5) begin $display("INDEX_n RW error"); errors = errors + 1; end

        // BUFF (byte address 0x28)
        addr = 8'h28; wdata = 4'b1101; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata[3:0] !== 4'b1101) begin $display("BUFF RW error"); errors = errors + 1; end

        // SCALE_Sa (byte address 0x2C)
        addr = 8'h2C; wdata = 32'h42AA0000; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'h42AA0000) begin $display("SCALE_Sa RW error"); errors = errors + 1; end

        // SCALE_Sw (byte address 0x30)
        addr = 8'h30; wdata = 32'h41BB0000; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'h41BB0000) begin $display("SCALE_Sw RW error"); errors = errors + 1; end

        // UART_len_max (byte address 0x34)
        addr = 8'h34; wdata = 32'd1234; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== 32'd1234) begin $display("UART_len_max RW error"); errors = errors + 1; end

        // UART_crc_en (byte address 0x38)
        addr = 8'h38; wdata = 32'b1; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata[0] !== 1'b1) begin $display("UART_crc_en RW error"); errors = errors + 1; end

        // STATUS (byte address 0x3C)
        addr = 8'h3C; wdata = 3'b110; wr_en = 1; @(posedge clk); wr_en = 0;
        rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata[2:0] !== 3'b110) begin $display("STATUS RW error"); errors = errors + 1; end
        
        // PERF_TOTAL (byte address 0x40) - Read-only
        addr = 8'h40; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_total_cycles) begin $display("PERF_TOTAL read error"); errors = errors + 1; end
        
        // PERF_ACTIVE (byte address 0x44) - Read-only
        addr = 8'h44; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_active_cycles) begin $display("PERF_ACTIVE read error"); errors = errors + 1; end
        
        // PERF_IDLE (byte address 0x48) - Read-only
        addr = 8'h48; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_idle_cycles) begin $display("PERF_IDLE read error"); errors = errors + 1; end
        
        // PERF_CACHE_HITS (byte address 0x4C) - Read-only
        addr = 8'h4C; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_cache_hits) begin $display("PERF_CACHE_HITS read error"); errors = errors + 1; end
        
        // PERF_CACHE_MISSES (byte address 0x50) - Read-only
        addr = 8'h50; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_cache_misses) begin $display("PERF_CACHE_MISSES read error"); errors = errors + 1; end
        
        // PERF_DECODE_COUNT (byte address 0x54) - Read-only
        addr = 8'h54; rd_en = 1; @(posedge clk); rd_en = 0;
        if (rdata !== perf_decode_count) begin $display("PERF_DECODE_COUNT read error"); errors = errors + 1; end

        if (errors == 0)
            $display("tb_csr: PASS");
        else
            $display("tb_csr: FAIL - %0d errors", errors);
        $finish;
    end
endmodule
`default_nettype wire
