`default_nettype none

// sram_1rw_wrapper.sv — Technology-neutral synchronous 1RW SRAM wrapper
//
// This module is the abstraction barrier for future foundry SRAM macro
// binding. The current implementation is a generic synchronous behavioral
// model so simulation and generic synthesis continue to work before macro
// views are integrated.

module sram_1rw_wrapper #(
    parameter DATA_W = 32,
    parameter ADDR_W = 10,
    parameter DEPTH  = (1 << ADDR_W)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              en,
    input  wire              we,
    input  wire [ADDR_W-1:0] addr,
    input  wire [DATA_W-1:0] wdata,
    output reg  [DATA_W-1:0] rdata
);

    reg [DATA_W-1:0] mem [0:DEPTH-1] /* verilator public */;


    generate
        if (DEPTH < (1 << ADDR_W)) begin : gen_addr_assert
            localparam [ADDR_W-1:0] LAST_ADDR = DEPTH - 1;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rdata <= {DATA_W{1'b0}};
        else if (en) begin
            if (we)
                mem[addr] <= wdata;
            else
                rdata <= mem[addr];
        end
    end

endmodule

`default_nettype wire
