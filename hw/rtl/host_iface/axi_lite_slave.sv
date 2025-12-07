//==============================================================================
// AXI4-Lite Slave for CSR Access
// Simple design - passes through to external CSR module
//==============================================================================

module axi_lite_slave #(
    parameter CSR_ADDR_WIDTH = 8,
    parameter CSR_DATA_WIDTH = 32
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // AXI4-Lite Write Address Channel
    input  logic                        s_axi_awvalid,
    output logic                        s_axi_awready,
    input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  logic [2:0]                  s_axi_awprot,

    // AXI4-Lite Write Data Channel
    input  logic                        s_axi_wvalid,
    output logic                        s_axi_wready,
    input  logic [CSR_DATA_WIDTH-1:0]   s_axi_wdata,
    input  logic [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb,

    // AXI4-Lite Write Response Channel
    output logic                        s_axi_bvalid,
    input  logic                        s_axi_bready,
    output logic [1:0]                  s_axi_bresp,

    // AXI4-Lite Read Address Channel
    input  logic                        s_axi_arvalid,
    output logic                        s_axi_arready,
    input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_araddr,
    input  logic [2:0]                  s_axi_arprot,

    // AXI4-Lite Read Data Channel
    output logic                        s_axi_rvalid,
    input  logic                        s_axi_rready,
    output logic [CSR_DATA_WIDTH-1:0]   s_axi_rdata,
    output logic [1:0]                  s_axi_rresp,

    // CSR Interface (to external CSR module)
    output logic                        csr_wen,
    output logic [CSR_ADDR_WIDTH-1:0]   csr_addr,
    output logic [CSR_DATA_WIDTH-1:0]   csr_wdata,
    input  logic [CSR_DATA_WIDTH-1:0]   csr_rdata,
    output logic                        csr_ren,
    output logic                        axi_error
);

    // Latched write address and data
    logic [CSR_ADDR_WIDTH-1:0] waddr_latch;
    logic [CSR_DATA_WIDTH-1:0] wdata_latch;
    logic                      got_waddr;
    logic                      got_wdata;

    // Error tracking
    assign axi_error = 1'b0;

    //--------------------------------------------------------------------------
    // WRITE CHANNEL
    //--------------------------------------------------------------------------
    
    // Ready when not holding pending data and not responding
    assign s_axi_awready = !got_waddr && !s_axi_bvalid;
    assign s_axi_wready  = !got_wdata && !s_axi_bvalid;

    // Track when handshakes occur this cycle
    wire aw_handshake = s_axi_awvalid && s_axi_awready;
    wire w_handshake  = s_axi_wvalid && s_axi_wready;
    
    // Both channels complete this cycle OR have completed previously
    wire both_complete = (got_waddr || aw_handshake) && (got_wdata || w_handshake);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            got_waddr    <= 1'b0;
            got_wdata    <= 1'b0;
            waddr_latch  <= '0;
            wdata_latch  <= '0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;
        end else begin
            // Default: clear bvalid when accepted
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
                got_waddr    <= 1'b0;
                got_wdata    <= 1'b0;
            end else begin
                // Capture write address when valid
                if (aw_handshake) begin
                    waddr_latch <= s_axi_awaddr;
                    got_waddr   <= 1'b1;
                end
                
                // Capture write data when valid
                if (w_handshake) begin
                    wdata_latch <= s_axi_wdata;
                    got_wdata   <= 1'b1;
                end
                
                // When we have both address and data, assert bvalid
                if (both_complete && !s_axi_bvalid) begin
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp  <= 2'b00;
                end
            end
        end
    end

    //--------------------------------------------------------------------------
    // READ CHANNEL
    //--------------------------------------------------------------------------
    
    assign s_axi_arready = !s_axi_rvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
            s_axi_rresp  <= 2'b00;
        end else begin
            // Accept read request and respond
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                s_axi_rdata  <= csr_rdata;
            end
            
            // Clear rvalid when response is accepted
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    //--------------------------------------------------------------------------
    // CSR Interface Outputs
    //--------------------------------------------------------------------------
    
    // Write enable when both channels complete and we're about to assert bvalid
    assign csr_wen   = both_complete && !s_axi_bvalid;
    assign csr_addr  = s_axi_arvalid ? s_axi_araddr : 
                       (aw_handshake ? s_axi_awaddr : waddr_latch);
    assign csr_wdata = w_handshake ? s_axi_wdata : wdata_latch;
    assign csr_ren   = s_axi_arvalid && s_axi_arready;

endmodule
