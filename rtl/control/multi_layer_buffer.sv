`timescale 1ns / 1ps

// Simple multi-layer buffer manager
// Partitioned BRAM address space for multiple layers. Host writes layer data
// into assigned base regions; scheduler reads using active_layer selection.

module multi_layer_buffer #(
    parameter NUM_LAYERS = 8,
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,

    // Host load interface
    input  logic                host_wr_en,
    input  logic [2:0]          host_layer_id,
    input  logic [ADDR_WIDTH-1:0] host_addr,
    input  logic [DATA_WIDTH-1:0] host_wdata,

    // Scheduler read interface
    input  logic [2:0]          active_layer,
    input  logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [DATA_WIDTH-1:0] rd_data
);

    // Base addresses (simple round-robin partitioning for demo)
    localparam LAYER_STRIDE = (1 << (ADDR_WIDTH-3)); // divide address space

    // Underlying unified memory
    logic [DATA_WIDTH-1:0] unified_mem [0:(1<<ADDR_WIDTH)-1];

    // Compute absolute address
    wire [ADDR_WIDTH-1:0] absolute_addr = (active_layer * LAYER_STRIDE) + rd_addr;
    wire [ADDR_WIDTH-1:0] host_absolute_addr = (host_layer_id * LAYER_STRIDE) + host_addr;

    // Read (combinational)
    assign rd_data = unified_mem[absolute_addr];

    // Host write (synchronous)
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // optionally clear memory on reset (omitted for size)
        end else begin
            if (host_wr_en) begin
                unified_mem[host_absolute_addr] <= host_wdata;
            end
        end
    end

endmodule
