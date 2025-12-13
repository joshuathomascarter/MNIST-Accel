//------------------------------------------------------------------------------
// output_accumulator.sv
// Output Accumulator with Double-Buffering for Sparse Systolic Array
//
// Features:
//  - Accumulates partial sums from systolic array across K-tiles
//  - Double-buffered: one bank accumulates while other drains to DMA
//  - Supports 14x14 output tiles (196 × 32-bit accumulators per bank)
//  - ReLU activation option before output
//  - Quantization/scaling support for INT8 output
//
// Operation:
//  1. Scheduler signals acc_valid when systolic outputs are ready
//  2. Accumulator adds new values to current bank
//  3. When tile_done, banks swap (ping-pong)
//  4. DMA reads from inactive bank while active bank accumulates
//------------------------------------------------------------------------------

`default_nettype none

module output_accumulator #(
    parameter N_ROWS    = 14,           // Systolic array rows
    parameter N_COLS    = 14,           // Systolic array columns  
    parameter ACC_W     = 32,           // Accumulator width (INT32)
    parameter OUT_W     = 8,            // Output data width (INT8)
    parameter ADDR_W    = 10            // Address width for output buffer
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // =========================================================================
    // Control Interface (from Scheduler/CSR)
    // =========================================================================
    input  wire                         acc_valid,      // Systolic output valid
    input  wire                         acc_clear,      // Clear accumulators (new tile)
    input  wire                         tile_done,      // Current tile complete, swap banks
    input  wire                         relu_en,        // Enable ReLU activation
    input  wire [31:0]                  scale_factor,   // Quantization scale (Q16.16 fixed-point)
    
    // =========================================================================
    // Systolic Array Input (from systolic_array_sparse)
    // =========================================================================
    input  wire [N_ROWS*N_COLS*ACC_W-1:0] systolic_out,  // Flattened output

    // =========================================================================
    // DMA Read Interface (to output DMA)
    // =========================================================================
    input  wire                         dma_rd_en,      // DMA read enable
    input  wire [ADDR_W-1:0]            dma_rd_addr,    // DMA read address
    output reg  [63:0]                  dma_rd_data,    // 64-bit read data (8 INT8s)
    output wire                         dma_ready,      // Bank ready for DMA

    // =========================================================================
    // Status
    // =========================================================================
    output reg                          busy,           // Accumulator busy
    output reg                          bank_sel,       // Current active bank (0/1)
    output wire [31:0]                  acc_debug       // Debug: first accumulator value
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam NUM_ACCS = N_ROWS * N_COLS;  // 196 for 14x14
    localparam BANK_DEPTH = NUM_ACCS;       // One entry per output element
    
    // =========================================================================
    // Double-Buffered Accumulator Banks
    // =========================================================================
    // Bank 0
    reg signed [ACC_W-1:0] acc_bank0 [0:BANK_DEPTH-1];
    // Bank 1  
    reg signed [ACC_W-1:0] acc_bank1 [0:BANK_DEPTH-1];
    
    // Bank ready flags (for DMA)
    reg bank0_ready;
    reg bank1_ready;
    
    assign dma_ready = bank_sel ? bank0_ready : bank1_ready;

    // =========================================================================
    // Unpack Systolic Array Output
    // =========================================================================
    wire signed [ACC_W-1:0] sys_out [0:NUM_ACCS-1];
    
    genvar i;
    generate
        for (i = 0; i < NUM_ACCS; i = i + 1) begin : UNPACK
            assign sys_out[i] = systolic_out[i*ACC_W +: ACC_W];
        end
    endgenerate

    // =========================================================================
    // Accumulation Logic
    // =========================================================================
    integer j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_sel <= 1'b0;
            bank0_ready <= 1'b0;
            bank1_ready <= 1'b0;
            busy <= 1'b0;
            
            // Clear both banks
            for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                acc_bank0[j] <= 32'sd0;
                acc_bank1[j] <= 32'sd0;
            end
        end else begin
            // Clear accumulators on new tile
            if (acc_clear) begin
                if (bank_sel == 1'b0) begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank0[j] <= 32'sd0;
                    end
                end else begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank1[j] <= 32'sd0;
                    end
                end
                busy <= 1'b1;
            end
            
            // Accumulate when valid
            if (acc_valid) begin
                if (bank_sel == 1'b0) begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank0[j] <= acc_bank0[j] + sys_out[j];
                    end
                end else begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank1[j] <= acc_bank1[j] + sys_out[j];
                    end
                end
            end
            
            // Swap banks on tile completion
            if (tile_done) begin
                bank_sel <= ~bank_sel;
                busy <= 1'b0;
                
                // Mark completed bank as ready for DMA
                if (bank_sel == 1'b0) begin
                    bank0_ready <= 1'b1;
                end else begin
                    bank1_ready <= 1'b1;
                end
            end
            
            // Clear ready flag when DMA starts reading
            if (dma_rd_en) begin
                if (bank_sel == 1'b1) begin  // DMA reads from bank0
                    bank0_ready <= 1'b0;
                end else begin               // DMA reads from bank1
                    bank1_ready <= 1'b0;
                end
            end
        end
    end

    // =========================================================================
    // DMA Read Path with Optional ReLU and Quantization
    // =========================================================================
    // Read from inactive bank, apply ReLU, quantize to INT8
    
    // Pipeline stage 1: Read from bank
    reg signed [ACC_W-1:0] rd_acc_0, rd_acc_1, rd_acc_2, rd_acc_3;
    reg signed [ACC_W-1:0] rd_acc_4, rd_acc_5, rd_acc_6, rd_acc_7;
    reg [ADDR_W-1:0] rd_addr_d1;
    reg rd_valid_d1;
    
    always @(posedge clk) begin
        rd_valid_d1 <= dma_rd_en;
        rd_addr_d1 <= dma_rd_addr;
        
        if (dma_rd_en) begin
            // Read 8 consecutive accumulators (for 64-bit output)
            // DMA address is in 64-bit words, so multiply by 8 for accumulator index
            if (bank_sel == 1'b1) begin  // Read from bank 0
                rd_acc_0 <= acc_bank0[{dma_rd_addr, 3'b000}];
                rd_acc_1 <= acc_bank0[{dma_rd_addr, 3'b001}];
                rd_acc_2 <= acc_bank0[{dma_rd_addr, 3'b010}];
                rd_acc_3 <= acc_bank0[{dma_rd_addr, 3'b011}];
                rd_acc_4 <= acc_bank0[{dma_rd_addr, 3'b100}];
                rd_acc_5 <= acc_bank0[{dma_rd_addr, 3'b101}];
                rd_acc_6 <= acc_bank0[{dma_rd_addr, 3'b110}];
                rd_acc_7 <= acc_bank0[{dma_rd_addr, 3'b111}];
            end else begin               // Read from bank 1
                rd_acc_0 <= acc_bank1[{dma_rd_addr, 3'b000}];
                rd_acc_1 <= acc_bank1[{dma_rd_addr, 3'b001}];
                rd_acc_2 <= acc_bank1[{dma_rd_addr, 3'b010}];
                rd_acc_3 <= acc_bank1[{dma_rd_addr, 3'b011}];
                rd_acc_4 <= acc_bank1[{dma_rd_addr, 3'b100}];
                rd_acc_5 <= acc_bank1[{dma_rd_addr, 3'b101}];
                rd_acc_6 <= acc_bank1[{dma_rd_addr, 3'b110}];
                rd_acc_7 <= acc_bank1[{dma_rd_addr, 3'b111}];
            end
        end
    end

    // Pipeline stage 2: Apply ReLU and Quantization
    function automatic [OUT_W-1:0] quantize_relu;
        input signed [ACC_W-1:0] acc_val;
        input [31:0] scale;
        input relu;
        
        reg signed [63:0] scaled;
        reg signed [ACC_W-1:0] relu_val;
        reg signed [15:0] quant_val;
    begin
        // ReLU: max(0, x)
        if (relu && acc_val < 0)
            relu_val = 32'sd0;
        else
            relu_val = acc_val;
        
        // Scale (Q16.16 fixed-point multiply, then shift right by 16)
        scaled = (relu_val * $signed({1'b0, scale[15:0]})) >>> 16;
        
        // Saturate to INT8 range [-128, 127]
        if (scaled > 127)
            quant_val = 127;
        else if (scaled < -128)
            quant_val = -128;
        else
            quant_val = scaled[15:0];
        
        quantize_relu = quant_val[OUT_W-1:0];
    end
    endfunction

    always @(posedge clk) begin
        if (rd_valid_d1) begin
            dma_rd_data <= {
                quantize_relu(rd_acc_7, scale_factor, relu_en),
                quantize_relu(rd_acc_6, scale_factor, relu_en),
                quantize_relu(rd_acc_5, scale_factor, relu_en),
                quantize_relu(rd_acc_4, scale_factor, relu_en),
                quantize_relu(rd_acc_3, scale_factor, relu_en),
                quantize_relu(rd_acc_2, scale_factor, relu_en),
                quantize_relu(rd_acc_1, scale_factor, relu_en),
                quantize_relu(rd_acc_0, scale_factor, relu_en)
            };
        end
    end

    // =========================================================================
    // Debug Output
    // =========================================================================
    assign acc_debug = bank_sel ? acc_bank1[0] : acc_bank0[0];

endmodule

`default_nettype wire
