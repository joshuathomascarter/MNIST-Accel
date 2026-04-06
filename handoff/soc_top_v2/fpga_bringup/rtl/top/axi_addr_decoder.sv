// ===========================================================================
// axi_addr_decoder.sv — AXI Address Decoder
// ===========================================================================
// Pure combinational address decoder. Maps an AXI address to a one-hot
// slave select vector based on the upper 4 bits of the address.
//
// Extracted from axi_crossbar.sv for standalone testability and reuse
// (NoC crossbar in Month 3 can instantiate the same decoder).
//
// Memory map:
//   4'h0 → Slave 0  (Boot ROM   @ 0x0000_0000)
//   4'h1 → Slave 1  (SRAM       @ 0x1000_0000)
//   4'h2 → Slave 2  (Peripherals@ 0x2000_0000)
//   4'h3 → Slave 3  (Accelerator@ 0x3000_0000)
//   4'h4 → Slave 4  (DRAM       @ 0x4000_0000)
//   other→ Slave 7  (Decode error)
//
// Resource estimate: ~15 LUTs
// ===========================================================================

module axi_addr_decoder #(
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned NUM_SLAVES  = 8,
    parameter int unsigned DECODE_BITS = 4,      // how many upper bits to decode
    parameter int unsigned DECODE_MSB  = 31      // top bit of decode field
)(
    input  logic [ADDR_WIDTH-1:0]   addr,
    output logic [NUM_SLAVES-1:0]   slave_sel,   // one-hot slave select
    output logic                    decode_error  // no valid slave matched
);

    localparam int unsigned DECODE_LSB = DECODE_MSB - DECODE_BITS + 1;

    logic [DECODE_BITS-1:0] region;
    assign region = addr[DECODE_MSB:DECODE_LSB];

    // Lower address bits are not needed for region decode — suppress lint
    wire _unused_addr_lsbs = &{1'b0, addr[DECODE_LSB-1:0]};

    always_comb begin
        slave_sel    = '0;
        decode_error = 1'b0;

        case (region)
            4'h0:    slave_sel[0] = 1'b1;  // Boot ROM
            4'h1:    slave_sel[1] = 1'b1;  // SRAM
            4'h2:    slave_sel[2] = 1'b1;  // Peripherals
            4'h3:    slave_sel[3] = 1'b1;  // Accelerator
            4'h4:    slave_sel[4] = 1'b1;  // DRAM Controller
            default: begin
                slave_sel[NUM_SLAVES-1] = 1'b1;  // Route to last slave (error sink)
                decode_error = 1'b1;
            end
        endcase
    end

endmodule : axi_addr_decoder
