// meta_decode.sv - Simplified BSR Metadata Decoder (No Cache)
`timescale 1ns / 1ps
`default_nettype none

module meta_decode #(
    parameter DATA_WIDTH = 32,
    parameter CACHE_DEPTH = 64
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     req_valid,
    input  wire [31:0]              req_addr,
    output wire                     req_ready,
    output wire                     mem_en,
    output wire [31:0]              mem_addr,
    input  wire [DATA_WIDTH-1:0]    mem_rdata,
    output wire                     meta_valid,
    output wire [DATA_WIDTH-1:0]    meta_rdata,
    input  wire                     meta_ready
);

    reg [1:0] state;
    localparam S_IDLE = 2'd0, S_WAIT = 2'd1, S_VALID = 2'd2;
    
    reg [31:0] addr_r;
    reg [DATA_WIDTH-1:0] data_r;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            addr_r <= 0;
            data_r <= 0;
        end else begin
            case (state)
                S_IDLE: if (req_valid) begin addr_r <= req_addr; state <= S_WAIT; end
                S_WAIT: begin data_r <= mem_rdata; state <= S_VALID; end
                S_VALID: if (meta_ready) begin
                    if (req_valid) begin addr_r <= req_addr; state <= S_WAIT; end
                    else state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
    
    assign req_ready   = (state == S_IDLE) || (state == S_VALID && meta_ready);
    assign mem_en      = (state == S_IDLE && req_valid) || (state == S_WAIT);
    assign mem_addr    = (state == S_IDLE) ? req_addr : addr_r;
    assign meta_valid  = (state == S_VALID);
    assign meta_rdata  = data_r;

endmodule
`default_nettype wire
