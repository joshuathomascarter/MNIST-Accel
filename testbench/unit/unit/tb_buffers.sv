`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_buffers.sv - Unit testbench for act_buffer and wgt_buffer
// Verifies write→read ordering, 1-cycle latency, and ping/pong bank swap logic.
// -----------------------------------------------------------------------------
module tb_buffers;
    parameter TM = 4;
    parameter TN = 4;
    parameter ADDR_WIDTH = 3; // 8-depth for quick test

    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;

    // DUTs
    reg we, rd_en, bank_sel_wr, bank_sel_rd;
    reg [ADDR_WIDTH-1:0] waddr, k_idx;
    reg [TM*8-1:0] wdata_a;
    reg [TN*8-1:0] wdata_b;
    wire [TM*8-1:0] a_vec;
    wire [TN*8-1:0] b_vec;

    act_buffer #(.TM(TM), .ADDR_WIDTH(ADDR_WIDTH)) u_act (
        .clk(clk), .rst_n(rst_n),
        .we(we), .waddr(waddr), .wdata(wdata_a), .bank_sel_wr(bank_sel_wr),
        .rd_en(rd_en), .k_idx(k_idx), .bank_sel_rd(bank_sel_rd), .a_vec(a_vec)
    );
    wgt_buffer #(.TN(TN), .ADDR_WIDTH(ADDR_WIDTH)) u_wgt (
        .clk(clk), .rst_n(rst_n),
        .we(we), .waddr(waddr), .wdata(wdata_b), .bank_sel_wr(bank_sel_wr),
        .rd_en(rd_en), .k_idx(k_idx), .bank_sel_rd(bank_sel_rd), .b_vec(b_vec)
    );

    integer i, errors;
    reg [TM*8-1:0] golden_a [0:(1<<ADDR_WIDTH)-1];
    reg [TN*8-1:0] golden_b [0:(1<<ADDR_WIDTH)-1];

    initial begin
        $display("\n--- tb_buffers: BEGIN ---");
        rst_n = 0; we = 0; rd_en = 0; bank_sel_wr = 0; bank_sel_rd = 1; waddr = 0; k_idx = 0;
        wdata_a = 0; wdata_b = 0; errors = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // Write to bank 0
        bank_sel_wr = 0; bank_sel_rd = 1; // write bank 0, read bank 1 (empty)
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            waddr = i;
            wdata_a = {TM{8'hA0 + i}}; // pattern
            wdata_b = {TN{8'hB0 + i}};
            golden_a[i] = wdata_a;
            golden_b[i] = wdata_b;
            we = 1;
            @(posedge clk);
            we = 0;
        end

        // Swap banks: now read from bank 0, write to bank 1
        bank_sel_wr = 1; bank_sel_rd = 0;
        // Read out all entries, check 1-cycle latency
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            k_idx = i;
            rd_en = 1;
            @(posedge clk);
            rd_en = 0;
            @(posedge clk); // wait for 1-cycle latency
            if (a_vec !== golden_a[i]) begin
                $display("ERROR: act_buffer mismatch at %0d: got %h exp %h", i, a_vec, golden_a[i]);
                errors = errors + 1;
            end
            if (b_vec !== golden_b[i]) begin
                $display("ERROR: wgt_buffer mismatch at %0d: got %h exp %h", i, b_vec, golden_b[i]);
                errors = errors + 1;
            end
        end

        // Write to bank 1 with new data
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            waddr = i;
            wdata_a = {TM{8'hC0 + i}};
            wdata_b = {TN{8'hD0 + i}};
            golden_a[i] = wdata_a;
            golden_b[i] = wdata_b;
            we = 1;
            @(posedge clk);
            we = 0;
        end

        // Swap banks again: read from bank 1, write to bank 0
        bank_sel_wr = 0; bank_sel_rd = 1;
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            k_idx = i;
            rd_en = 1;
            @(posedge clk);
            rd_en = 0;
            @(posedge clk);
            if (a_vec !== golden_a[i]) begin
                $display("ERROR: act_buffer mismatch at %0d: got %h exp %h", i, a_vec, golden_a[i]);
                errors = errors + 1;
            end
            if (b_vec !== golden_b[i]) begin
                $display("ERROR: wgt_buffer mismatch at %0d: got %h exp %h", i, b_vec, golden_b[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("tb_buffers: PASS");
        else
            $display("tb_buffers: FAIL - %0d errors", errors);
        $finish;
    end
endmodule
`default_nettype wire
