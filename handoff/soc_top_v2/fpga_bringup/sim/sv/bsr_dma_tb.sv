// =============================================================================
// BSR DMA Unit Testbench for Coverage
// Tests the BSR DMA engine with proper AXI responses
// =============================================================================

`timescale 1ns/1ps

module bsr_dma_tb;
    // Parameters
    localparam AXI_ADDR_W = 32;
    localparam AXI_DATA_W = 64;
    localparam AXI_ID_W = 4;
    localparam BRAM_ADDR_W = 10;
    
    // Clock and Reset
    reg clk;
    reg rst_n;
    
    // Control
    reg                       start;
    reg  [AXI_ADDR_W-1:0]     src_addr;
    wire                      done;
    wire                      busy;
    wire                      error;
    
    // AXI Master Read Interface
    wire [AXI_ID_W-1:0]       m_axi_arid;
    wire [AXI_ADDR_W-1:0]     m_axi_araddr;
    wire [7:0]                m_axi_arlen;
    wire [2:0]                m_axi_arsize;
    wire [1:0]                m_axi_arburst;
    wire                      m_axi_arvalid;
    reg                       m_axi_arready;
    
    reg  [AXI_ID_W-1:0]       m_axi_rid;
    reg  [AXI_DATA_W-1:0]     m_axi_rdata;
    reg  [1:0]                m_axi_rresp;
    reg                       m_axi_rlast;
    reg                       m_axi_rvalid;
    wire                      m_axi_rready;
    
    // BRAM Write Interfaces
    wire                      row_ptr_we;
    wire [BRAM_ADDR_W-1:0]    row_ptr_addr;
    wire [31:0]               row_ptr_wdata;
    
    wire                      col_idx_we;
    wire [BRAM_ADDR_W-1:0]    col_idx_addr;
    wire [15:0]               col_idx_wdata;
    
    wire                      wgt_we;
    wire [BRAM_ADDR_W+6:0]    wgt_addr;
    wire [63:0]               wgt_wdata;
    
    // Test Memory (byte-addressed)
    reg [7:0] memory [0:65535];
    
    // AXI read state machine
    reg [31:0] axi_rd_addr;
    reg [7:0]  axi_rd_len;
    reg [7:0]  axi_rd_cnt;
    reg [1:0]  axi_rd_state;
    localparam AXI_IDLE = 0, AXI_DATA = 1;
    
    // DUT
    bsr_dma #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .AXI_ID_W(AXI_ID_W),
        .STREAM_ID(0),
        .BRAM_ADDR_W(BRAM_ADDR_W),
        .BURST_LEN(8'd15)
    ) u_bsr_dma (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .src_addr(src_addr),
        .done(done),
        .busy(busy),
        .error(error),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .row_ptr_we(row_ptr_we),
        .row_ptr_addr(row_ptr_addr),
        .row_ptr_wdata(row_ptr_wdata),
        .col_idx_we(col_idx_we),
        .col_idx_addr(col_idx_addr),
        .col_idx_wdata(col_idx_wdata),
        .wgt_we(wgt_we),
        .wgt_addr(wgt_addr),
        .wgt_wdata(wgt_wdata)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // AXI Read Response Model - combinational data/last, registered handshake
    // Uses combinational logic to present rlast correctly on the same cycle
    wire [15:0] mem_addr = axi_rd_addr[15:0] + axi_rd_cnt * 8;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rid <= 4'd0;
            m_axi_rresp <= 2'b00;
            axi_rd_state <= AXI_IDLE;
            axi_rd_addr <= 0;
            axi_rd_len <= 0;
            axi_rd_cnt <= 0;
        end else begin
            case (axi_rd_state)
                AXI_IDLE: begin
                    m_axi_arready <= 1'b1;
                    m_axi_rvalid <= 1'b0;
                    
                    if (m_axi_arvalid && m_axi_arready) begin
                        $display("  [AXI] AR: addr=0x%08X len=%0d", m_axi_araddr, m_axi_arlen);
                        axi_rd_addr <= m_axi_araddr;
                        axi_rd_len <= m_axi_arlen;
                        axi_rd_cnt <= 0;
                        m_axi_arready <= 1'b0;
                        axi_rd_state <= AXI_DATA;
                    end
                end
                
                AXI_DATA: begin
                    // Present data on every cycle
                    m_axi_rvalid <= 1'b1;
                    m_axi_rid <= 4'd0;
                    m_axi_rresp <= 2'b00;
                    
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (axi_rd_cnt == axi_rd_len) begin
                            axi_rd_state <= AXI_IDLE;
                            m_axi_rvalid <= 1'b0;
                        end else begin
                            axi_rd_cnt <= axi_rd_cnt + 1;
                        end
                    end
                end
            endcase
        end
    end
    
    // Combinational data and last signals - presented immediately
    always @(*) begin
        // 64-bit read from byte-addressed memory (little-endian)
        m_axi_rdata = {
            memory[mem_addr + 7],
            memory[mem_addr + 6],
            memory[mem_addr + 5],
            memory[mem_addr + 4],
            memory[mem_addr + 3],
            memory[mem_addr + 2],
            memory[mem_addr + 1],
            memory[mem_addr + 0]
        };
        
        // Last signal is combinational based on current count
        m_axi_rlast = (axi_rd_state == AXI_DATA) && (axi_rd_cnt == axi_rd_len);
    end
    
    // Initialize BSR test data with EXACT memory layout matching bsr_dma.sv
    // Memory Layout (from bsr_dma.sv comments):
    //   Header (16 bytes = 2 beats @ 64-bit):
    //     Beat 0: [num_rows (32-bit), num_cols (32-bit)]
    //     Beat 1: [total_blocks (32-bit), padding (32-bit)]
    //   Row_Ptr Array: (num_rows + 1) x 32-bit words, 64-bit aligned
    //   Col_Idx Array: total_blocks x 16-bit words, 64-bit aligned
    //   Weight Blocks: total_blocks x 64 bytes each
    task automatic init_bsr_memory(input integer num_rows, input integer num_cols, input integer total_blocks);
        integer i, j, addr, row_ptr_start, col_idx_start, weights_start;
        integer row_ptr_bytes, col_idx_bytes;
        begin
            // Clear memory
            for (i = 0; i < 65536; i = i + 1) memory[i] = 8'h00;
            
            // ===== HEADER (16 bytes at offset 0) =====
            // Beat 0 (bytes 0-7): num_rows[31:0], num_cols[31:0]
            memory[0] = num_rows[7:0];
            memory[1] = num_rows[15:8];
            memory[2] = num_rows[23:16];
            memory[3] = num_rows[31:24];
            memory[4] = num_cols[7:0];
            memory[5] = num_cols[15:8];
            memory[6] = num_cols[23:16];
            memory[7] = num_cols[31:24];
            
            // Beat 1 (bytes 8-15): total_blocks[31:0], padding[31:0]
            memory[8] = total_blocks[7:0];
            memory[9] = total_blocks[15:8];
            memory[10] = total_blocks[23:16];
            memory[11] = total_blocks[31:24];
            memory[12] = 8'h00; memory[13] = 8'h00; memory[14] = 8'h00; memory[15] = 8'h00;
            
            // ===== ROW POINTERS (starts at offset 16) =====
            // (num_rows + 1) x 32-bit words
            row_ptr_start = 16;
            for (i = 0; i <= num_rows; i = i + 1) begin
                addr = row_ptr_start + i * 4;
                // Simple cumulative: row i starts at block i (for diagonal sparse)
                j = (i < total_blocks) ? i : total_blocks;
                memory[addr + 0] = j[7:0];
                memory[addr + 1] = j[15:8];
                memory[addr + 2] = j[23:16];
                memory[addr + 3] = j[31:24];
            end
            
            // ===== COL INDICES (64-bit aligned after row_ptr) =====
            row_ptr_bytes = (num_rows + 1) * 4;
            col_idx_start = row_ptr_start + ((row_ptr_bytes + 7) & ~7); // 64-bit align
            for (i = 0; i < total_blocks; i = i + 1) begin
                addr = col_idx_start + i * 2;
                // Column index = block number (diagonal pattern)
                memory[addr + 0] = i[7:0];
                memory[addr + 1] = i[15:8];
            end
            
            // ===== WEIGHT BLOCKS (64-bit aligned after col_idx) =====
            col_idx_bytes = total_blocks * 2;
            weights_start = col_idx_start + ((col_idx_bytes + 7) & ~7); // 64-bit align
            for (i = 0; i < total_blocks; i = i + 1) begin
                for (j = 0; j < 64; j = j + 1) begin  // 64 bytes per 8x8 block
                    addr = weights_start + i * 64 + j;
                    memory[addr] = ((i * 64 + j + 1) & 8'hFF);  // Test pattern
                end
            end
            
            $display("  BSR Memory Layout:");
            $display("    Header:    0x%04X - 0x%04X (16 bytes)", 0, 15);
            $display("    Row_Ptr:   0x%04X - 0x%04X (%0d bytes)", row_ptr_start, col_idx_start-1, row_ptr_bytes);
            $display("    Col_Idx:   0x%04X - 0x%04X (%0d bytes)", col_idx_start, weights_start-1, col_idx_bytes);
            $display("    Weights:   0x%04X onwards (%0d blocks x 64 bytes)", weights_start, total_blocks);
        end
    endtask
    
    // Wait for DMA with timeout
    task automatic wait_for_done(input integer timeout_cycles);
        integer i;
        begin
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (done || error) i = timeout_cycles; // Exit early
            end
        end
    endtask
    
    // Main test sequence
    initial begin
        $dumpfile("bsr_dma_tb.vcd");
        $dumpvars(0, bsr_dma_tb);
        
        // Initialize
        rst_n = 0;
        start = 0;
        src_addr = 0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        $display("\n=== BSR DMA Testbench Starting ===\n");
        
        // =========================================================================
        // Test 1: Basic functionality - 1 row, 1 block
        // =========================================================================
        $display("Test 1: Minimal BSR (1 row, 1 block)");
        init_bsr_memory(1, 1, 1);
        
        // Debug: Show header bytes
        $display("  Header bytes: %02X %02X %02X %02X | %02X %02X %02X %02X | %02X %02X %02X %02X | %02X %02X %02X %02X",
                 memory[0], memory[1], memory[2], memory[3],
                 memory[4], memory[5], memory[6], memory[7],
                 memory[8], memory[9], memory[10], memory[11],
                 memory[12], memory[13], memory[14], memory[15]);
        
        src_addr = 32'h0000_0000;
        start = 1;
        @(posedge clk);
        start = 0;
        
        $display("  Started DMA: busy=%b done=%b error=%b", busy, done, error);
        
        wait_for_done(5000);
        
        if (done) $display("  PASS: DMA completed successfully");
        else if (error) $display("  FAIL: DMA error");
        else $display("  FAIL: DMA timeout after 5000 cycles");
        
        repeat(20) @(posedge clk);
        
        // =========================================================================
        // Test 2: Medium-size - 2 rows, 2 blocks
        // =========================================================================
        $display("\nTest 2: 2x2 BSR (2 rows, 2 blocks)");
        init_bsr_memory(2, 2, 2);
        
        src_addr = 32'h0000_0000;
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_done(50000);
        
        if (done) $display("  PASS: DMA completed");
        else if (error) $display("  FAIL: DMA error");
        else $display("  FAIL: DMA timeout");
        
        repeat(20) @(posedge clk);
        
        // =========================================================================
        // Test 3: Reset during operation
        // =========================================================================
        $display("\nTest 3: Reset during transfer");
        init_bsr_memory(8, 8, 8);
        
        src_addr = 32'h0000_0000;
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for transfer to start
        repeat(50) @(posedge clk);
        
        // Apply reset
        rst_n = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
        
        $display("  After reset: busy=%b done=%b error=%b", busy, done, error);
        if (!busy && !done) $display("  PASS: Reset cleared state");
        else $display("  FAIL: Reset did not clear state properly");
        
        repeat(20) @(posedge clk);
        
        // =========================================================================
        // Test 4: Single block minimal transfer
        // =========================================================================
        $display("\nTest 4: Single block DMA");
        init_bsr_memory(1, 1, 1);
        
        src_addr = 32'h0000_0000;
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait_for_done(500);
        
        if (done) $display("  PASS: Single block DMA completed");
        else $display("  FAIL: DMA not complete (timeout or error)");
        
        repeat(20) @(posedge clk);
        
        $display("\n=== BSR DMA Test Complete ===");
        $finish;
    end
    
endmodule
