//------------------------------------------------------------------------------
// systolic_array_sparse.sv
// Sparse-Aware Systolic Array
//
// Features:
//  - Supports "Skip-Zero" logic via block_valid
//  - Gates clock/enable when block_valid is low
//  - Correctly chains activation pipeline (a_out -> a_in)
//------------------------------------------------------------------------------
module systolic_array_sparse #(
  parameter N_ROWS = 8,
  parameter N_COLS = 8,
  parameter DATA_W = 8,
  parameter ACC_W  = 32
)(
  input  wire                     clk,
  input  wire                     rst_n,
  
  // Sparse Control
  input  wire                     block_valid, // From scheduler (pe_en)
  input  wire                     load_weight, // From scheduler
  
  // Data Inputs
  input  wire [N_ROWS*DATA_W-1:0] a_in_flat,   // Activations (from act_buffer)
  input  wire [N_COLS*DATA_W-1:0] b_in_flat,   // Weights (from wgt_buffer)
  
  // Outputs
  output wire [N_ROWS*N_COLS*ACC_W-1:0] c_out_flat
);

  // Unpack inputs
  wire signed [DATA_W-1:0] a_in [0:N_ROWS-1];
  wire signed [DATA_W-1:0] b_in [0:N_COLS-1];
  
  genvar i;
  generate
    for (i = 0; i < N_ROWS; i = i + 1) assign a_in[i] = a_in_flat[i*DATA_W +: DATA_W];
    for (i = 0; i < N_COLS; i = i + 1) assign b_in[i] = b_in_flat[i*DATA_W +: DATA_W];
  endgenerate

  //----------------------------------------------------------------------------
  // Weight Loading Logic (Row-by-Row)
  //----------------------------------------------------------------------------
  // We load 64 weights over 8 cycles.
  // load_weight is held high by scheduler for 8 cycles.
  // We use a counter to enable one row at a time.
  
  reg [$clog2(N_ROWS)-1:0] load_ptr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      load_ptr <= 0;
    end else begin
      if (load_weight) begin
        load_ptr <= load_ptr + 1;
      end else begin
        load_ptr <= 0;
      end
    end
  end

  // PE Array
  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL
        
        // Horizontal activation forwarding (The Pipeline)
        // If col 0, take from input. Else take from neighbor's a_out (a_fwd).
        wire signed [DATA_W-1:0] a_fwd;
        wire signed [DATA_W-1:0] a_src = (c == 0) ? a_in[r] : ROW[r].COL[c-1].a_fwd;
        
        // Weight broadcast (Vertical/Direct)
        // In Weight Stationary, we load the weight once and hold it.
        wire signed [DATA_W-1:0] b_src = b_in[c];

        // PE Instantiation
        // Note: 'en' is gated by block_valid. If 0, PE holds state (saves power).
        // Load is enabled only for the current row in the loading sequence.
        pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
          .clk      (clk),
          .rst_n    (rst_n),
          .en       (block_valid),    
          .load_weight(load_weight && (load_ptr == r)), 
          .a_in     (a_src),
          .b_in     (b_src),
          .a_out    (a_fwd), // Output to next PE in row
          .c_out    (c_out_flat[(r*N_COLS + c)*ACC_W +: ACC_W])
        );
      end
    end
  endgenerate

endmodule
