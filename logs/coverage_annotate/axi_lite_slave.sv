//      // verilator_coverage annotation
        //==============================================================================
        // AXI4-Lite Slave for CSR Access
        // Simple design - passes through to external CSR module
        //==============================================================================
        
        module axi_lite_slave #(
            parameter CSR_ADDR_WIDTH = 8,
            parameter CSR_DATA_WIDTH = 32
        )(
 012713     input  logic                        clk,
%000007     input  logic                        rst_n,
        
            // AXI4-Lite Write Address Channel
 000048     input  logic                        s_axi_awvalid,
 000096     output logic                        s_axi_awready,
~000028     input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_awaddr,
%000000     input  logic [2:0]                  s_axi_awprot,
        
            // AXI4-Lite Write Data Channel
 000048     input  logic                        s_axi_wvalid,
 000096     output logic                        s_axi_wready,
~000014     input  logic [CSR_DATA_WIDTH-1:0]   s_axi_wdata,
%000007     input  logic [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb,
        
            // AXI4-Lite Write Response Channel
 000096     output logic                        s_axi_bvalid,
 000052     input  logic                        s_axi_bready,
%000000     output logic [1:0]                  s_axi_bresp,
        
            // AXI4-Lite Read Address Channel
 000152     input  logic                        s_axi_arvalid,
 000153     output logic                        s_axi_arready,
~000021     input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_araddr,
%000000     input  logic [2:0]                  s_axi_arprot,
        
            // AXI4-Lite Read Data Channel
 000152     output logic                        s_axi_rvalid,
 000152     input  logic                        s_axi_rready,
~000011     output logic [CSR_DATA_WIDTH-1:0]   s_axi_rdata,
%000000     output logic [1:0]                  s_axi_rresp,
        
            // CSR Interface (to external CSR module)
 000141     output logic                        csr_wen,
~000094     output logic [CSR_ADDR_WIDTH-1:0]   csr_addr,
~000023     output logic [CSR_DATA_WIDTH-1:0]   csr_wdata,
~000075     input  logic [CSR_DATA_WIDTH-1:0]   csr_rdata,
 000152     output logic                        csr_ren,
%000000     output logic                        axi_error
        );
        
            // Latched write address and data
~000012     logic [CSR_ADDR_WIDTH-1:0] waddr_latch;
~000012     logic [CSR_DATA_WIDTH-1:0] wdata_latch;
 000096     logic                      got_waddr;
 000096     logic                      got_wdata;
        
            // Error tracking
            assign axi_error = 1'b0;
        
            //--------------------------------------------------------------------------
            // WRITE CHANNEL
            //--------------------------------------------------------------------------
            
            // Ready when not holding pending data and not responding
            assign s_axi_awready = !got_waddr && !s_axi_bvalid;
            assign s_axi_wready  = !got_wdata && !s_axi_bvalid;
        
            // Track when handshakes occur this cycle
 000141     wire aw_handshake = s_axi_awvalid && s_axi_awready;
 000141     wire w_handshake  = s_axi_wvalid && s_axi_wready;
            
            // Both channels complete this cycle OR have completed previously
 000048     wire both_complete = (got_waddr || aw_handshake) && (got_wdata || w_handshake);
        
 012713     always_ff @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             got_waddr    <= 1'b0;
 000069             got_wdata    <= 1'b0;
 000069             waddr_latch  <= '0;
 000069             wdata_latch  <= '0;
 000069             s_axi_bvalid <= 1'b0;
 000069             s_axi_bresp  <= 2'b00;
 012644         end else begin
                    // Default: clear bvalid when accepted
 012551             if (s_axi_bvalid && s_axi_bready) begin
 000093                 s_axi_bvalid <= 1'b0;
 000093                 got_waddr    <= 1'b0;
 000093                 got_wdata    <= 1'b0;
 012551             end else begin
                        // Capture write address when valid
 012455                 if (aw_handshake) begin
 000096                     waddr_latch <= s_axi_awaddr;
 000096                     got_waddr   <= 1'b1;
                        end
                        
                        // Capture write data when valid
 012455                 if (w_handshake) begin
 000096                     wdata_latch <= s_axi_wdata;
 000096                     got_wdata   <= 1'b1;
                        end
                        
                        // When we have both address and data, assert bvalid
 012455                 if (both_complete && !s_axi_bvalid) begin
 000096                     s_axi_bvalid <= 1'b1;
 000096                     s_axi_bresp  <= 2'b00;
                        end
                    end
                end
            end
        
            //--------------------------------------------------------------------------
            // READ CHANNEL
            //--------------------------------------------------------------------------
            
            assign s_axi_arready = !s_axi_rvalid;
        
 012713     always_ff @(posedge clk or negedge rst_n) begin
 012644         if (!rst_n) begin
 000069             s_axi_rvalid <= 1'b0;
 000069             s_axi_rdata  <= '0;
 000069             s_axi_rresp  <= 2'b00;
 012644         end else begin
                    // Accept read request and respond
 012492             if (s_axi_arvalid && s_axi_arready) begin
 000152                 s_axi_rvalid <= 1'b1;
 000152                 s_axi_rresp  <= 2'b00;
 000152                 s_axi_rdata  <= csr_rdata;
                    end
                    
                    // Clear rvalid when response is accepted
 012492             if (s_axi_rvalid && s_axi_rready) begin
 000152                 s_axi_rvalid <= 1'b0;
                    end
                end
            end
        
            //--------------------------------------------------------------------------
            // CSR Interface Outputs
            //--------------------------------------------------------------------------
            
            // Write enable when both channels complete and we're about to assert bvalid
            assign csr_wen   = both_complete && !s_axi_bvalid;
 025123     assign csr_addr  = s_axi_arvalid ? s_axi_araddr : 
 024934                        (aw_handshake ? s_axi_awaddr : waddr_latch);
 025238     assign csr_wdata = w_handshake ? s_axi_wdata : wdata_latch;
            assign csr_ren   = s_axi_arvalid && s_axi_arready;
        
        endmodule
        
