//------------------------------------------------------------------------------
// systolic_array_sparse.sv
// Sparse-Aware Systolic Array
//
// Features:
//  - Supports "Skip-Zero" logic via block_valid
//  - Gates clock/enable when block_valid is low
//  - Correctly chains activation pipeline (a_out -> a_in)
//  - Parameterized for 8x8 or 16x16 configurations
//------------------------------------------------------------------------------
`default_nettype none

module systolic_array_sparse #(
  parameter N_ROWS = 16,  // Default to 16x16 for higher throughput
  parameter N_COLS = 16,
  parameter DATA_W = 8,
  parameter ACC_W  = 32
)(
  input  wire                     clk,
  input  wire                     rst_n,
  
  // Sparse Control
  input  wire                     block_valid, // From scheduler (pe_en)
  input  wire                     load_weight, // From scheduler
  input  wire [N_ROWS*N_COLS-1:0] bypass_flat, // NEW: Per-PE bypass signal for residual mode
  
  // Data Inputs
  input  wire [N_ROWS*DATA_W-1:0] a_in_flat,   // Activations (from act_buffer)
  input  wire [N_COLS*DATA_W-1:0] b_in_flat,   // Weights (from wgt_buffer)
  
  // Outputs
  output wire [N_ROWS*N_COLS*ACC_W-1:0] c_out_flat
);

  // Unpack inputs
  wire signed [DATA_W-1:0] a_in [0:N_ROWS-1];
  wire signed [DATA_W-1:0] b_in [0:N_COLS-1];
  wire bypass_mat [0:N_ROWS-1][0:N_COLS-1];  // NEW: 2D bypass matrix
  
  genvar i;
  generate
    for (i = 0; i < N_ROWS; i = i + 1) assign a_in[i] = a_in_flat[i*DATA_W +: DATA_W];
    for (i = 0; i < N_COLS; i = i + 1) assign b_in[i] = b_in_flat[i*DATA_W +: DATA_W];
    
    // NEW: Unpack bypass signals
    for (i = 0; i < N_ROWS*N_COLS; i = i + 1) begin : UNPACK_BYPASS
      integer r_idx, c_idx;
      initial begin
        r_idx = i / N_COLS;
        c_idx = i % N_COLS;
      end
      assign bypass_mat[r_idx][c_idx] = bypass_flat[i];
    end
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

  //----------------------------------------------------------------------------
  // 2D Wire Arrays for PE Interconnect
  //----------------------------------------------------------------------------
  wire signed [DATA_W-1:0] a_fwd [0:N_ROWS-1][0:N_COLS-1];
  wire load_weight_fwd [0:N_ROWS-1][0:N_COLS-1];

  // PE Array
  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL
        
        // Activation source: from input (col 0) or left neighbor
        wire signed [DATA_W-1:0] a_src;
        if (c == 0) begin : gen_a_input
          assign a_src = a_in[r];
        end else begin : gen_a_chain
          assign a_src = a_fwd[r][c-1];
        end
        
        // Weight broadcast (Vertical/Direct)
        wire signed [DATA_W-1:0] b_src = b_in[c];
        
        // Load weight source - from scheduler (col 0) or neighbor
        wire load_weight_src;
        if (c == 0) begin : gen_lw_input
          assign load_weight_src = load_weight && (load_ptr == r[$clog2(N_ROWS)-1:0]);
        end else begin : gen_lw_chain
          assign load_weight_src = load_weight_fwd[r][c-1];
        end

        // PE Instantiation
        pe #(.PIPE(1)) u_pe (
          .clk            (clk),
          .rst_n          (rst_n),
          .en             (block_valid),
          .clr            (1'b0),
          .load_weight    (load_weight_src), 
          .a_in           (a_src),
          .b_in           (b_src),
          .bypass         (bypass_mat[r][c]),  // NEW: Bypass signal for residual mode
          .a_out          (a_fwd[r][c]),
          .load_weight_out(load_weight_fwd[r][c]),
          .acc            (c_out_flat[(r*N_COLS + c)*ACC_W +: ACC_W])
        );
      end
    end
  endgenerate

endmodule

`default_nettype wire
