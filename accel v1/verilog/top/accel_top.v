// accel_top.v
// Top-level integration for ACCEL-v1 systolic array accelerator
// Integrates systolic array, buffers, scheduler, CSR, and UART modules

`default_nettype none

module accel_top #(
    parameter N_ROWS = 2,
    parameter N_COLS = 2,
    parameter TM = 8,
    parameter TN = 8, 
    parameter TK = 8,
    parameter CLK_HZ = 50_000_000,
    parameter BAUD = 115_200,
    parameter ADDR_WIDTH = 6
)(
    input  wire clk,
    input  wire rst_n,
    // UART interface
    input  wire uart_rx,
    output wire uart_tx,
    // Status LEDs or debug outputs
    output wire busy,
    output wire done_tile
);

    // Internal signals
    wire csr_wen, csr_ren;
    wire [7:0] csr_addr;
    wire [31:0] csr_wdata, csr_rdata;
    wire start_pulse, abort_pulse, irq_en;
    wire core_busy, core_done_tile_pulse;
    
    // CSR configuration outputs
    wire [31:0] M, N, K;
    wire [31:0] Tm, Tn, Tk;
    wire [31:0] m_idx, n_idx, k_idx;
    wire bank_sel_wr_A, bank_sel_wr_B;
    wire bank_sel_rd_A, bank_sel_rd_B;
    wire [31:0] Sa_bits, Sw_bits;
    wire [31:0] uart_len_max;
    wire uart_crc_en;
    
    // Systolic array signals
    wire array_en, array_clr;
    wire [N_ROWS*8-1:0] a_in_flat;
    wire [N_COLS*8-1:0] b_in_flat;
    wire [N_ROWS*N_COLS*32-1:0] c_out_flat;
    
    // Buffer control signals
    wire act_we, wgt_we;
    wire [TM*8-1:0] act_wdata, wgt_wdata;
    wire [TM*8-1:0] a_vec;
    wire [TN*8-1:0] b_vec;
    wire [ADDR_WIDTH-1:0] act_waddr, wgt_waddr;
    wire [ADDR_WIDTH-1:0] act_k_idx, wgt_k_idx;
    wire act_rd_en, wgt_rd_en;
    
    // UART data interface
    wire [7:0] uart_rx_data, uart_tx_data;
    wire uart_rx_valid, uart_tx_ready, uart_tx_valid;
    wire uart_rx_frm_err, uart_rx_par_err;

    // Scheduler outputs
    wire sched_busy, sched_done_tile;
    wire [9:0] sched_m_tile, sched_n_tile;
    wire [11:0] sched_k_tile;
    wire [31:0] cycles_tile, stall_cycles;
    wire [TK-1:0] k_idx_sched;
    wire [TM-1:0] en_mask_row;
    wire [TN-1:0] en_mask_col;

    // Status outputs
    assign busy = core_busy;
    assign done_tile = core_done_tile_pulse;

    // CSR (Control/Status Register) module
    csr #(
        .ADDR_W(8)
    ) csr_inst (
        .clk(clk),
        .rst_n(rst_n),
        .csr_wen(csr_wen),
        .csr_ren(csr_ren),
        .csr_addr(csr_addr),
        .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata),
        .core_busy(core_busy),
        .core_done_tile_pulse(core_done_tile_pulse),
        .core_bank_sel_rd_A(bank_sel_rd_A),
        .core_bank_sel_rd_B(bank_sel_rd_B),
        .rx_crc_error(uart_rx_par_err),
        .rx_illegal_cmd(uart_rx_frm_err),
        .start_pulse(start_pulse),
        .abort_pulse(abort_pulse),
        .irq_en(irq_en),
        .M(M), .N(N), .K(K),
        .Tm(Tm), .Tn(Tn), .Tk(Tk),
        .m_idx(m_idx), .n_idx(n_idx), .k_idx(k_idx),
        .bank_sel_wr_A(bank_sel_wr_A), .bank_sel_wr_B(bank_sel_wr_B),
        .bank_sel_rd_A(bank_sel_rd_A), .bank_sel_rd_B(bank_sel_rd_B),
        .Sa_bits(Sa_bits), .Sw_bits(Sw_bits),
        .uart_len_max(uart_len_max),
        .uart_crc_en(uart_crc_en)
    );

    // Systolic Array
    systolic_array #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .PIPE(1),
        .SAT(0)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(array_en),
        .clr(array_clr),
        .a_in_flat(a_in_flat),
        .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );

    // Activation Buffer
    act_buffer #(
        .TM(TM),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) act_buf_inst (
        .clk(clk),
        .rst_n(rst_n),
        .we(act_we),
        .waddr(act_waddr),
        .wdata(act_wdata),
        .bank_sel_wr(bank_sel_wr_A),
        .rd_en(act_rd_en),
        .k_idx(act_k_idx),
        .bank_sel_rd(bank_sel_rd_A),
        .a_vec(a_vec)
    );

    // Weight Buffer  
    wgt_buffer #(
        .TN(TN),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) wgt_buf_inst (
        .clk(clk),
        .rst_n(rst_n),
        .we(wgt_we),
        .waddr(wgt_waddr),
        .wdata(wgt_wdata),
        .bank_sel_wr(bank_sel_wr_B),
        .rd_en(wgt_rd_en),
        .k_idx(wgt_k_idx),
        .bank_sel_rd(bank_sel_rd_B),
        .b_vec(b_vec)
    );

    // Scheduler
    scheduler #(
        .M_W(10),
        .N_W(10),
        .K_W(12),
        .TM_W(6),
        .TN_W(6),
        .TK_W(6),
        .PREPRIME(0),
        .USE_CSR_COUNTS(1)
    ) scheduler_inst (
        .clk(clk),
        .rst_n(rst_n),
        // CSR interface 
        .start(start_pulse),
        .abort(abort_pulse),
        .M(M[9:0]),
        .N(N[9:0]),
        .K(K[11:0]),
        .Tm(Tm[5:0]),
        .Tn(Tn[5:0]),
        .Tk(Tk[5:0]),
        .MT_csr(10'd1),  // Simplified - should come from CSR
        .NT_csr(10'd1),
        .KT_csr(12'd1),
        // Bank readiness (simplified)
        .valid_A_ping(1'b1),
        .valid_A_pong(1'b1),
        .valid_B_ping(1'b1),
        .valid_B_pong(1'b1),
        // Status outputs
        .busy(sched_busy),
        .done_tile(sched_done_tile),
        .m_tile(sched_m_tile),
        .n_tile(sched_n_tile),
        .k_tile(sched_k_tile),
        .cycles_tile(cycles_tile),
        .stall_cycles(stall_cycles),
        // Buffer/Array control
        .rd_en(act_rd_en),  // Same for both buffers
        .k_idx(k_idx_sched),
        .bank_sel_rd_A(bank_sel_rd_A),
        .bank_sel_rd_B(bank_sel_rd_B),
        .clr(array_clr),
        .en(array_en),
        .en_mask_row(en_mask_row),
        .en_mask_col(en_mask_col)
    );

    // UART RX
    uart_rx #(
        .DATA_BITS(8),
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .OVERSAMPLE(16),
        .PARITY(0),
        .STOP_BITS(1),
        .USE_NCO(0),
        .ACCW(32),
        .MAJORITY3(0)
    ) uart_rx_inst (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_rx(uart_rx),
        .o_data(uart_rx_data),
        .o_valid(uart_rx_valid),
        .o_frm_err(uart_rx_frm_err),
        .o_par_err(uart_rx_par_err)
    );

    // UART TX  
    uart_tx #(
        .DATA_BITS(8),
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .OVERSAMPLE(16),
        .PARITY(0),
        .STOP_BITS(1),
        .USE_NCO(0),
        .ACCW(32)
    ) uart_tx_inst (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_data(uart_tx_data),
        .i_valid(uart_tx_valid),
        .i_ready(uart_tx_ready),
        .o_tx(uart_tx)
    );

    // Signal assignments and connections
    assign core_busy = sched_busy;
    assign core_done_tile_pulse = sched_done_tile;
    
    // Connect systolic array inputs (map from buffer outputs to array size)
    assign a_in_flat = a_vec[N_ROWS*8-1:0];
    assign b_in_flat = b_vec[N_COLS*8-1:0];
    
    // Buffer controls
    assign wgt_rd_en = act_rd_en;  // Same read enable for both buffers
    assign act_k_idx = k_idx_sched[ADDR_WIDTH-1:0];
    assign wgt_k_idx = k_idx_sched[ADDR_WIDTH-1:0];

    // UART protocol handler (simplified - needs proper packet handling)
    assign csr_wen = uart_rx_valid;
    assign csr_ren = 1'b0;
    assign csr_addr = uart_rx_data;
    assign csr_wdata = {24'b0, uart_rx_data};
    assign uart_tx_data = csr_rdata[7:0];
    assign uart_tx_valid = 1'b0; // Simplified - needs proper response logic

    // Buffer write controls (simplified - needs proper UART data handling)
    assign act_we = 1'b0;
    assign wgt_we = 1'b0;
    assign act_wdata = {TM*8{1'b0}};
    assign wgt_wdata = {TN*8{1'b0}};
    assign act_waddr = {ADDR_WIDTH{1'b0}};
    assign wgt_waddr = {ADDR_WIDTH{1'b0}};

endmodule

`default_nettype wire
