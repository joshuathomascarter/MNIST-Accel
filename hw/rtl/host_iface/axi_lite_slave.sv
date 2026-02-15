// axi_lite_slave.sv — AXI4-Lite Slave Interface for CSR Access
// Converts AXI4-Lite transactions from Zynq PS into CSR read/write strobes.
// Write: AW+W channels both required, then pulse csr_wen
// Read: AR channel triggers csr_ren, respond next cycle

module axi_lite_slave #(
    parameter CSR_ADDR_WIDTH = 8,   // 256-byte address space
    parameter CSR_DATA_WIDTH = 32    // AXI4-Lite fixed width
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        s_axi_awvalid, s_axi_arvalid, s_axi_wvalid,
    output logic                        s_axi_awready, s_axi_arready, s_axi_wready,
    input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_awaddr, s_axi_araddr,
    input  logic [2:0]                  s_axi_awprot, s_axi_arprot,
    input  logic [CSR_DATA_WIDTH-1:0]   s_axi_wdata,
    input  logic [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    output logic                        s_axi_bvalid, s_axi_rvalid,
    input  logic                        s_axi_bready, s_axi_rready,
    output logic [1:0]                  s_axi_bresp, s_axi_rresp,
    output logic [CSR_DATA_WIDTH-1:0]   s_axi_rdata,
    output logic                        csr_wen, csr_ren,
    output logic [CSR_ADDR_WIDTH-1:0]   csr_addr,
    output logic [CSR_DATA_WIDTH-1:0]   csr_wdata,
    input  logic [CSR_DATA_WIDTH-1:0]   csr_rdata,
    output logic                        axi_error
);

    logic [CSR_ADDR_WIDTH-1:0] waddr_latch;
    logic [CSR_DATA_WIDTH-1:0] wdata_latch;
    logic                      got_waddr, got_wdata;
    assign axi_error = 1'b0;

    // AXI4-Lite protocol ports — present per spec, unused by this implementation:
    //   s_axi_awprot/arprot: Protection encoding (no privilege checks).
    //   s_axi_wstrb: Byte strobes (always full-word writes in this design).
    wire _unused_axi_proto = &{1'b0, s_axi_awprot, s_axi_arprot, s_axi_wstrb};

    // Write requires both address and data channels
    assign s_axi_awready = !got_waddr && !s_axi_bvalid;
    assign s_axi_wready  = !got_wdata && !s_axi_bvalid;
    wire aw_handshake = s_axi_awvalid && s_axi_awready;
    wire w_handshake  = s_axi_wvalid && s_axi_wready;
    wire both_complete = (got_waddr || aw_handshake) && (got_wdata || w_handshake);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: Clear all latches and response
            got_waddr    <= 1'b0;
            got_wdata    <= 1'b0;
            waddr_latch  <= '0;
            wdata_latch  <= '0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;  // OKAY
        end else begin
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0; got_waddr <= 1'b0; got_wdata <= 1'b0;
            end else begin
                if (aw_handshake) begin waddr_latch <= s_axi_awaddr; got_waddr <= 1'b1; end
                if (w_handshake)  begin wdata_latch <= s_axi_wdata;  got_wdata <= 1'b1; end
                if (both_complete && !s_axi_bvalid) begin
                    s_axi_bvalid <= 1'b1; s_axi_bresp <= 2'b00;
                end
            end
        end
    end

    // Read: AR triggers response next cycle
    assign s_axi_arready = !s_axi_rvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
            s_axi_rresp  <= 2'b00;  // OKAY
        end else begin
            // Accept read request: Capture address and data, assert rvalid
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;  // Always OKAY
                s_axi_rdata  <= csr_rdata;  // Combinational from CSR
            end
            
            // Response accepted: Clear for next transaction
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    assign csr_wen   = both_complete && !s_axi_bvalid;
    assign csr_addr  = s_axi_arvalid ? s_axi_araddr : (aw_handshake ? s_axi_awaddr : waddr_latch);
    assign csr_wdata = w_handshake ? s_axi_wdata : wdata_latch;
    assign csr_ren   = s_axi_arvalid && s_axi_arready;

endmodule
