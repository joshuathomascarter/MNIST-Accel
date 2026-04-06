// accel_top_tb.sv - SystemVerilog Testbench for accel_top

`timescale 1ns/1ps

module accel_top_tb;

    // Parameters
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter AXI_DATA_W = 64;
    parameter AXI_ADDR_W = 32;
    parameter AXI_ID_W   = 4;

    // CSR Address Map (matches csr.sv)
    localparam CSR_CTRL        = 8'h00;  // Start/Abort
    localparam CSR_DIMS_M      = 8'h04;  // Matrix dimension M
    localparam CSR_DIMS_N      = 8'h08;  // Matrix dimension N
    localparam CSR_DIMS_K      = 8'h0C;  // Matrix dimension K
    localparam CSR_TILES_Tm    = 8'h10;  // Tile size Tm
    localparam CSR_TILES_Tn    = 8'h14;  // Tile size Tn
    localparam CSR_TILES_Tk    = 8'h18;  // Tile size Tk
    localparam CSR_SCALE_Sa    = 8'h2C;  // Scaling factor Sa
    localparam CSR_SCALE_Sw    = 8'h30;  // Scaling factor Sw
    localparam CSR_STATUS      = 8'h3C;  // Status (read-only)
    localparam CSR_PERF_TOTAL  = 8'h40;  // Performance: total cycles
    localparam CSR_PERF_ACTIVE = 8'h44;  // Performance: active cycles
    localparam CSR_PERF_IDLE   = 8'h48;  // Performance: idle cycles
    localparam CSR_DMA_SRC_ADDR     = 8'h90;  // BSR DMA source address
    localparam CSR_DMA_XFER_LEN     = 8'h98;  // BSR DMA transfer length
    localparam CSR_DMA_CTRL         = 8'h9C;  // BSR DMA control
    localparam CSR_ACT_DMA_SRC_ADDR = 8'hA0;  // Activation DMA source address
    localparam CSR_ACT_DMA_LEN      = 8'hA4;  // Activation DMA length
    localparam CSR_ACT_DMA_CTRL     = 8'hA8;  // Activation DMA control

    // Signals
    reg         clk;
    reg         rst_n;
    
    // AXI4 Master (to DDR)
    wire [AXI_ID_W-1:0]   m_axi_arid;
    wire [AXI_ADDR_W-1:0] m_axi_araddr;
    wire [7:0]            m_axi_arlen;
    wire [2:0]            m_axi_arsize;
    wire [1:0]            m_axi_arburst;
    wire                  m_axi_arvalid;
    reg                   m_axi_arready;
    reg  [AXI_ID_W-1:0]   m_axi_rid;
    reg  [AXI_DATA_W-1:0] m_axi_rdata;
    reg  [1:0]            m_axi_rresp;
    reg                   m_axi_rlast;
    reg                   m_axi_rvalid;
    wire                  m_axi_rready;
    
    // AXI-Lite Slave (CSR)
    reg  [AXI_ADDR_W-1:0] s_axi_awaddr;
    reg  [2:0]            s_axi_awprot;
    reg                   s_axi_awvalid;
    wire                  s_axi_awready;
    reg  [31:0]           s_axi_wdata;
    reg  [3:0]            s_axi_wstrb;
    reg                   s_axi_wvalid;
    wire                  s_axi_wready;
    wire [1:0]            s_axi_bresp;
    wire                  s_axi_bvalid;
    reg                   s_axi_bready;
    reg  [AXI_ADDR_W-1:0] s_axi_araddr;
    reg  [2:0]            s_axi_arprot;
    reg                   s_axi_arvalid;
    wire                  s_axi_arready;
    wire [31:0]           s_axi_rdata;
    wire [1:0]            s_axi_rresp;
    wire                  s_axi_rvalid;
    reg                   s_axi_rready;
    
    // Status
    wire busy, done, error;

    // Test control
    integer test_num = 0;
    integer errors = 0;
    reg [31:0] read_data;

    // DUT Instantiation
    accel_top #(
        .N_ROWS     (8),
        .N_COLS     (8),
        .DATA_W     (8),
        .ACC_W      (32),
        .AXI_ADDR_W (AXI_ADDR_W),
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ID_W   (AXI_ID_W),
        .CSR_ADDR_W (8),
        .BRAM_ADDR_W(10)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // AXI4 Master
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        
        // AXI-Lite Slave
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awprot   (s_axi_awprot),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arprot   (s_axi_arprot),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),
        
        // Status
        .busy           (busy),
        .done           (done),
        .error          (error)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Memory Model (Simplified)
    reg [63:0] mem [0:16383];  // 128KB simulated memory
    
    initial begin
        // Initialize memory with test pattern
        for (int i = 0; i < 16384; i++) begin
            mem[i] = {32'hDEAD0000 + i[15:0], 32'hBEEF0000 + i[15:0]};
        end
    end
    
    // Simple memory read responder
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid  <= 1'b0;
            m_axi_rlast   <= 1'b0;
            m_axi_rresp   <= 2'b00;
            m_axi_rid     <= '0;
            m_axi_rdata   <= '0;
        end else begin
            // Accept read requests
            if (m_axi_arvalid && m_axi_arready) begin
                // Start responding
                m_axi_rdata  <= mem[m_axi_araddr[16:3]];
                m_axi_rid    <= m_axi_arid;
                m_axi_rresp  <= 2'b00;  // OKAY
                m_axi_rlast  <= 1'b1;   // Single beat for simplicity
                m_axi_rvalid <= 1'b1;
            end
            
            // Clear valid after accepted
            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast  <= 1'b0;
            end
        end
    end

    // AXI-Lite Write Task
    // Simple protocol: assert address+data, wait for bvalid response
    task automatic axi_lite_write(input [31:0] addr, input [31:0] data);
        integer timeout;
        reg aw_done, w_done;
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awprot  <= 3'b000;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;
            aw_done = 0;
            w_done = 0;
            
            // Wait for both handshakes to complete, then wait for bvalid
            timeout = 0;
            while (timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
                
                // Track address handshake
                if (s_axi_awvalid && s_axi_awready) begin
                    aw_done = 1;
                    s_axi_awvalid <= 1'b0;
                end
                
                // Track data handshake
                if (s_axi_wvalid && s_axi_wready) begin
                    w_done = 1;
                    s_axi_wvalid <= 1'b0;
                end
                
                // Exit when we get bvalid
                if (s_axi_bvalid) begin
                    @(posedge clk);  // Wait one more cycle for slave to clear
                    s_axi_bready <= 1'b0;
                    @(posedge clk);  // Ensure bready deasserts before returning
                    return;
                end
            end
            
            $display("  [DEBUG] Timeout waiting for bvalid at addr 0x%02h (aw=%0d w=%0d)", 
                     addr, aw_done, w_done);
        end
    endtask
    
    // AXI-Lite Read Task
    task automatic axi_lite_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arprot  <= 3'b000;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;
            
            // Wait for address ready
            while (!s_axi_arready) @(posedge clk);
            @(posedge clk);
            s_axi_arvalid <= 1'b0;
            
            // Wait for data valid
            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
            s_axi_rready <= 1'b0;
        end
    endtask

    // Check utility
    task automatic check(input [31:0] got, input [31:0] expected, input [255:0] msg);
        if (got !== expected) begin
            $display("  [FAIL] %s: got 0x%08h, expected 0x%08h", msg, got, expected);
            errors = errors + 1;
        end else begin
            $display("  [PASS] %s: 0x%08h", msg, got);
        end
    endtask
    
    // Reset task
    task automatic reset_dut();
        begin
            rst_n = 0;
            s_axi_awaddr  = 0;
            s_axi_awprot  = 0;
            s_axi_awvalid = 0;
            s_axi_wdata   = 0;
            s_axi_wstrb   = 0;
            s_axi_wvalid  = 0;
            s_axi_bready  = 0;
            s_axi_araddr  = 0;
            s_axi_arprot  = 0;
            s_axi_arvalid = 0;
            s_axi_rready  = 0;
            repeat (10) @(posedge clk);
            rst_n = 1;
            repeat (5) @(posedge clk);
        end
    endtask

    // Main Test Sequence
    initial begin
        // Waveform dump
        $dumpfile("accel_top_tb.vcd");
        $dumpvars(0, accel_top_tb);
        
        $display("");
        $display("==============================================");
        $display("  ACCEL_TOP Testbench");
        $display("==============================================");
        
        // Test 1: Reset
        test_num = 1;
        $display("\n[TEST %0d] Reset behavior", test_num);
        reset_dut();
        check(busy, 0, "busy after reset");
        check(done, 0, "done after reset");
        check(error, 0, "error after reset");
        
        // Test 2: CSR Write/Read
        test_num = 2;
        $display("\n[TEST %0d] CSR Write/Read", test_num);
        
        axi_lite_write(CSR_DIMS_M, 32'd64);
        axi_lite_write(CSR_DIMS_N, 32'd64);
        axi_lite_write(CSR_DIMS_K, 32'd128);
        axi_lite_write(CSR_ACT_DMA_SRC_ADDR, 32'h1000_0000);
        axi_lite_write(CSR_DMA_SRC_ADDR, 32'h2000_0000);
        axi_lite_write(CSR_ACT_DMA_LEN, 32'd512);
        
        axi_lite_read(CSR_DIMS_M, read_data);
        check(read_data, 64, "DIMS_M");
        
        axi_lite_read(CSR_DIMS_N, read_data);
        check(read_data, 64, "DIMS_N");
        
        axi_lite_read(CSR_DIMS_K, read_data);
        check(read_data, 128, "DIMS_K");
        
        axi_lite_read(CSR_ACT_DMA_SRC_ADDR, read_data);
        check(read_data, 32'h1000_0000, "ACT_DMA_SRC_ADDR");
        
        axi_lite_read(CSR_DMA_SRC_ADDR, read_data);
        check(read_data, 32'h2000_0000, "DMA_SRC_ADDR");
        
        axi_lite_read(CSR_ACT_DMA_LEN, read_data);
        check(read_data, 512, "ACT_DMA_LEN");
        
        // Test 3: Status Register
        test_num = 3;
        $display("\n[TEST %0d] Status Register", test_num);
        
        axi_lite_read(CSR_STATUS, read_data);
        $display("  STATUS = 0x%08h", read_data);
        check(read_data & 32'h7, 0, "STATUS (idle)");
        
        // Test 4: Start Pulse
        test_num = 4;
        $display("\n[TEST %0d] Start Pulse", test_num);
        axi_lite_write(CSR_CTRL, 32'h0000_0001);
        repeat (5) @(posedge clk);
        axi_lite_read(CSR_STATUS, read_data);
        $display("  STATUS after start = 0x%08h", read_data);
        
        // Test 5: Abort Pulse
        test_num = 5;
        $display("\n[TEST %0d] Abort Pulse", test_num);
        axi_lite_write(CSR_CTRL, 32'h0000_0002);
        repeat (5) @(posedge clk);
        axi_lite_read(CSR_STATUS, read_data);
        $display("  STATUS after abort = 0x%08h", read_data);
        
        // Test 6: Performance Counters
        test_num = 6;
        $display("\n[TEST %0d] Performance Counters", test_num);
        axi_lite_read(CSR_PERF_TOTAL, read_data);
        $display("  PERF_TOTAL = %0d", read_data);
        axi_lite_read(CSR_PERF_ACTIVE, read_data);
        $display("  PERF_ACTIVE = %0d", read_data);
        
        // Test 7: AXI Master Idle
        test_num = 7;
        $display("\n[TEST %0d] AXI Master Idle Check", test_num);
        reset_dut();
        repeat (20) @(posedge clk);
        check(m_axi_arvalid, 0, "m_axi_arvalid idle");
        
        // Test 8: Rapid CSR Writes
        test_num = 8;
        $display("\n[TEST %0d] Rapid CSR Writes", test_num);
        for (int i = 0; i < 10; i++) begin
            axi_lite_write(CSR_DIMS_M, i * 8);
            axi_lite_write(CSR_DIMS_N, i * 8 + 1);
            axi_lite_write(CSR_DIMS_K, i * 8 + 2);
        end
        axi_lite_read(CSR_DIMS_M, read_data);
        check(read_data, 72, "DIMS_M final");
        axi_lite_read(CSR_DIMS_N, read_data);
        check(read_data, 73, "DIMS_N final");
        axi_lite_read(CSR_DIMS_K, read_data);
        check(read_data, 74, "DIMS_K final");
        
        // Summary
        $display("\n==============================================");
        if (errors == 0) begin
            $display("  ALL TESTS PASSED!");
        end else begin
            $display("  FAILED: %0d errors", errors);
        end
        $display("==============================================\n");
        
        repeat (10) @(posedge clk);
        $finish;
    end

    // Timeout Watchdog
    initial begin
        #100000;  // 100us timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end

endmodule
