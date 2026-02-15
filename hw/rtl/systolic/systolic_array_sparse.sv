// systolic_array_sparse.sv — Sparse-aware weight-stationary 14×14 INT8 systolic array
// Skips zero BSR blocks via block_valid gating. 196 DSPs on Zynq-7020.
// Row-by-row weight loading (14 cycles), all columns latch simultaneously per row.

`default_nettype none

module systolic_array_sparse #(
    parameter N_ROWS = 14,
    parameter N_COLS = 14,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    // Sparse control from scheduler
    input  wire                     block_valid,    // 0 = zero block, skip compute
    input  wire                     load_weight,    // Assert 14 cycles to load weights
    input  wire                     clr,            // Sync clear all PE accumulators
    // Data
    input  wire [N_ROWS*DATA_W-1:0] a_in_flat,     // 14 × INT8 activations
    input  wire [N_COLS*DATA_W-1:0] b_in_flat,     // 14 × INT8 weights
    output wire [N_ROWS*N_COLS*ACC_W-1:0] c_out_flat // 196 × INT32 accumulators
);

  // ---------- 1. Input Unpacking + Triangular Activation Skew ----------
  wire signed [DATA_W-1:0] a_in_raw [0:N_ROWS-1];  // Raw from flat bus
  wire signed [DATA_W-1:0] a_in     [0:N_ROWS-1];  // Skewed to PE grid
  wire signed [DATA_W-1:0] b_in     [0:N_COLS-1];

  genvar i;
  generate
    for (i = 0; i < N_ROWS; i = i + 1)
      assign a_in_raw[i] = a_in_flat[i*DATA_W +: DATA_W];
    for (i = 0; i < N_COLS; i = i + 1)
      assign b_in[i] = b_in_flat[i*DATA_W +: DATA_W];

    // Triangular skew: row i gets i delay stages (gated by block_valid)
    // Row 0: straight through. Row 13: 13-deep shift register.
    // Compensates for the 1-cycle PE pipeline delay per column.
    for (i = 0; i < N_ROWS; i = i + 1)
      if (i == 0) begin : SKEW_ROW0
        assign a_in[i] = a_in_raw[i];
      end else begin : SKEW_ROW
        reg signed [DATA_W-1:0] delay_regs [0:i-1];
        integer k;
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (k = 0; k < i; k = k + 1) delay_regs[k] <= {DATA_W{1'b0}};
          end else if (block_valid) begin
            delay_regs[0] <= a_in_raw[i];
            for (k = 1; k < i; k = k + 1)
              delay_regs[k] <= delay_regs[k-1];
          end
        end
        assign a_in[i] = delay_regs[i-1];
      end
  endgenerate

  // ---------- 2. Row-by-Row Weight Loading ----------
  // load_ptr selects which row receives load_weight each cycle (0..N_ROWS-1)
  reg [$clog2(N_ROWS)-1:0] load_ptr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      load_ptr <= 0;
    else if (load_weight)
      load_ptr <= load_ptr + 1;
    else
      load_ptr <= 0;
  end

  // ---------- 3. PE Interconnection ----------
  wire signed [DATA_W-1:0] a_fwd [0:N_ROWS-1][0:N_COLS-1];

  // ---------- 4. PE Grid ----------
  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL

        wire signed [DATA_W-1:0] a_src;
        if (c == 0) begin : gen_a_input
          assign a_src = a_in[r];
        end else begin : gen_a_chain
          assign a_src = a_fwd[r][c-1];
        end

        wire signed [DATA_W-1:0] b_src = b_in[c];

        // Sparse direct-addressing: ALL columns in the selected row latch
        // simultaneously via load_ptr match.  This differs from the dense
        // systolic_array.sv which chains load_weight_out PE-to-PE across
        // columns (load_weight_src = prev PE's load_weight_out).  Because
        // every column latches in the same cycle here, the chain output
        // from the PE is architecturally vestigial.
        wire load_weight_src = load_weight && (load_ptr == r[$clog2(N_ROWS)-1:0]);

        wire _unused_lw_out;  // PE chain output — unused in direct-addressing mode

        pe #(.PIPE(1)) u_pe (
          .clk            (clk),
          .rst_n          (rst_n),
          .en             (block_valid),
          .clr            (clr),
          .load_weight    (load_weight_src),
          .a_in           (a_src),
          .b_in           (b_src),
          .a_out          (a_fwd[r][c]),
          .load_weight_out(_unused_lw_out),
          .acc            (c_out_flat[(r*N_COLS + c)*ACC_W +: ACC_W])
        );
      end
    end
  endgenerate

endmodule

`default_nettype wire
