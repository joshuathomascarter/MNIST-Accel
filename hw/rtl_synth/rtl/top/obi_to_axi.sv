// OBI to AXI-Lite Bridge
// Converts RISC-V OBI protocol to AXI-Lite

module obi_to_axi #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned ID_WIDTH = 4
) (
  input  logic              clk,
  input  logic              rst_n,
  
  // OBI Slave (from CPU)
  output logic              obi_gnt,
  input  logic              obi_req,
  input  logic [ADDR_WIDTH-1:0] obi_addr,
  input  logic              obi_we,
  input  logic [DATA_WIDTH/8-1:0] obi_be,
  input  logic [DATA_WIDTH-1:0] obi_wdata,
  input  logic [1:0]        obi_burst,
  
  output logic              obi_rvalid,
  output logic [DATA_WIDTH-1:0] obi_rdata,
  output logic              obi_err,
  
  // AXI-Lite Master (to crossbar)
  output logic              axi_awvalid,
  input  logic              axi_awready,
  output logic [ADDR_WIDTH-1:0] axi_awaddr,
  output logic [ID_WIDTH-1:0] axi_awid,
  
  output logic              axi_wvalid,
  input  logic              axi_wready,
  output logic [DATA_WIDTH-1:0] axi_wdata,
  output logic [DATA_WIDTH/8-1:0] axi_wstrb,
  output logic              axi_wlast,
  
  input  logic              axi_bvalid,
  output logic              axi_bready,
  input  logic [1:0]        axi_bresp,
  input  logic [ID_WIDTH-1:0] axi_bid,
  
  output logic              axi_arvalid,
  input  logic              axi_arready,
  output logic [ADDR_WIDTH-1:0] axi_araddr,
  output logic [ID_WIDTH-1:0] axi_arid,
  
  input  logic              axi_rvalid,
  output logic              axi_rready,
  input  logic [DATA_WIDTH-1:0] axi_rdata,
  input  logic [1:0]        axi_rresp,
  input  logic [ID_WIDTH-1:0] axi_rid
);

  // State machine for read/write transactions
  typedef enum logic [2:0] {
    IDLE,
    WRITE_ADDR,
    WRITE_DATA,
    WRITE_RESP,
    READ_ADDR,
    READ_DATA
  } state_e;

  state_e state, next_state;
  logic [ADDR_WIDTH-1:0] addr_reg;
  logic [DATA_WIDTH-1:0] wdata_reg;
  logic [DATA_WIDTH/8-1:0] strb_reg;
  logic we_reg;

  // OBI grant is combinational from request
  assign obi_gnt = obi_req;

  // ===== FSM =====
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      addr_reg <= '0;
      wdata_reg <= '0;
      strb_reg <= '0;
      we_reg <= 1'b0;
    end else begin
      state <= next_state;
      if (obi_req && obi_gnt) begin
        addr_reg <= obi_addr;
        wdata_reg <= obi_wdata;
        strb_reg <= obi_be;
        we_reg <= obi_we;
      end
    end
  end

  always_comb begin
    next_state = state;
    
    case (state)
      IDLE: begin
        if (obi_req) begin
          if (obi_we) begin
            next_state = WRITE_ADDR;
          end else begin
            next_state = READ_ADDR;
          end
        end
      end
      
      WRITE_ADDR: begin
        if (axi_awready) begin
          next_state = WRITE_DATA;
        end
      end
      
      WRITE_DATA: begin
        if (axi_wready) begin
          next_state = WRITE_RESP;
        end
      end
      
      WRITE_RESP: begin
        if (axi_bvalid) begin
          next_state = IDLE;
        end
      end
      
      READ_ADDR: begin
        if (axi_arready) begin
          next_state = READ_DATA;
        end
      end
      
      READ_DATA: begin
        if (axi_rvalid) begin
          next_state = IDLE;
        end
      end
      
      default: next_state = IDLE;
    endcase
  end

  // ===== AXI WRITE PATH =====
  
  assign axi_awvalid = (state == WRITE_ADDR);
  assign axi_awaddr  = addr_reg;
  assign axi_awid    = '0;
  
  assign axi_wvalid  = (state == WRITE_DATA);
  assign axi_wdata   = wdata_reg;
  assign axi_wstrb   = strb_reg;
  assign axi_wlast   = 1'b1;
  
  assign axi_bready  = (state == WRITE_RESP);
  
  // ===== AXI READ PATH =====
  
  assign axi_arvalid = (state == READ_ADDR);
  assign axi_araddr  = addr_reg;
  assign axi_arid    = '0;
  
  assign axi_rready  = (state == READ_DATA);
  
  // ===== OBI RESPONSE =====
  
  assign obi_rvalid = axi_rvalid;
  assign obi_rdata  = axi_rdata;
  assign obi_err    = (state == WRITE_RESP && axi_bresp != 2'b00) ||
                      (state == READ_DATA && axi_rresp != 2'b00);

endmodule : obi_to_axi
