//------------------------------------------------------------------------------
// bsr_scheduler.sv
// Block-Sparse Row (BSR) Scheduler for Sparse Systolic Array
//
// Logic:
//  1. Iterate k_tile (Rows of B).
//  2. Read row_ptr to find non-zero blocks in this k-row.
//  3. For each non-zero block (which gives us n_tile):
//     - Load the Weight Block (B_k,n) -> Weight Stationary
//     - Loop m_tile (0..MT):
//       - Load Activation Tile (A_m,k)
//       - Execute MAC (accumulate into C_m,n)
//
//  This skips all (k,n) tiles where B is zero.
//------------------------------------------------------------------------------

`default_nettype none

module bsr_scheduler #(
  parameter M_W  = 10,
  parameter N_W  = 10,
  parameter K_W  = 12,
  parameter ADDR_W = 32
)(
  input  wire                 clk,
  input  wire                 rst_n,

  // Control
  input  wire                 start,
  input  wire                 abort,
  output reg                  busy,
  output reg                  done,

  // Configuration
  input  wire [M_W-1:0]       MT, // Number of M tiles
  input  wire [K_W-1:0]       KT, // Number of K tiles (Rows of B)

  // Metadata Interface (to meta_decode.sv)
  output reg  [31:0]          meta_raddr,
  output reg                  meta_ren,
  input  wire [31:0]          meta_rdata,
  input  wire                 meta_rvalid,
  output wire                 meta_ready,

  // Buffer Interfaces
  output reg                  wgt_rd_en,
  output reg  [ADDR_W-1:0]    wgt_addr,
  output reg                  act_rd_en,
  output reg  [ADDR_W-1:0]    act_addr,

  // Systolic Array Control
  output reg                  load_weight, // Latch weight into PE
  output reg                  pe_en,       // Enable MAC (block_valid)
  output reg                  accum_en     // Enable accumulation
);

  // ---------------------------------------------------------------------------
  // 1. One-Hot FSM Encoding
  // ---------------------------------------------------------------------------
  localparam [8:0] 
    S_IDLE        = 9'b000000001,
    S_FETCH_PTR1  = 9'b000000010, // Get row_ptr[k]
    S_FETCH_PTR2  = 9'b000000100, // Get row_ptr[k+1]
    S_CALC_LEN    = 9'b000001000, // Calc block count
    S_FETCH_COL   = 9'b000010000, // Get col_idx[blk] -> n_tile
    S_LOAD_WGT    = 9'b000100000, // Load Weight Block
    S_STREAM_ACT  = 9'b001000000, // Stream Activations (Loop M)
    S_NEXT_BLK    = 9'b010000000, // Next block in k-row
    S_NEXT_K      = 9'b100000000; // Next k-row

  reg [8:0] state, state_n;

  // ---------------------------------------------------------------------------
  // 2. Internal Registers
  // ---------------------------------------------------------------------------
  reg [K_W-1:0] k_idx;      // Current K tile (Row of B)
  reg [M_W-1:0] m_idx;      // Current M tile (Row of A)
  reg [2:0]     load_cnt;   // Counter for loading 8 rows of weights
  
  reg [31:0]    blk_ptr;    // Current index in compressed arrays
  reg [31:0]    blk_end;    // End index for current k-row
  reg [31:0]    n_idx;      // Decoded N tile (Col of B)

  // Metadata Latches
  reg [31:0] ptr_start_reg;

  // ---------------------------------------------------------------------------
  // 3. Next State Logic
  // ---------------------------------------------------------------------------
  always @(*) begin
    state_n = state;
    case (state)
      S_IDLE:       if (start) state_n = S_FETCH_PTR1;
      
      S_FETCH_PTR1: if (meta_rvalid) state_n = S_FETCH_PTR2;
      
      S_FETCH_PTR2: if (meta_rvalid) state_n = S_CALC_LEN;
      
      S_CALC_LEN:   state_n = (ptr_start_reg == meta_rdata) ? S_NEXT_K : S_FETCH_COL; 
                    // If ptr_start == ptr_end, row is empty -> Next K
      
      S_FETCH_COL:  if (meta_rvalid) state_n = S_LOAD_WGT;
      
      S_LOAD_WGT:   state_n = (load_cnt == 3'd7) ? S_STREAM_ACT : S_LOAD_WGT;
      
      S_STREAM_ACT: if (m_idx == MT - 1) state_n = S_NEXT_BLK;
                    // Stream all M tiles against this Weight Block
      
      S_NEXT_BLK:   state_n = (blk_ptr < blk_end) ? S_FETCH_COL : S_NEXT_K;
      
      S_NEXT_K:     state_n = (k_idx < KT - 1) ? S_FETCH_PTR1 : S_IDLE; // Done if k==KT-1
      
      default:      state_n = S_IDLE;
    endcase
  end

  // ---------------------------------------------------------------------------
  // 4. Datapath & Output Logic
  // ---------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      k_idx <= 0;
      m_idx <= 0;
      load_cnt <= 0;
      blk_ptr <= 0;
      blk_end <= 0;
      
      // Outputs
      meta_ren <= 0;
      wgt_rd_en <= 0;
      act_rd_en <= 0;
      load_weight <= 0;
      pe_en <= 0;
      busy <= 0;
      done <= 0;
    end else begin
      state <= state_n;
      
      // Default Pulses
      meta_ren <= 0;
      wgt_rd_en <= 0;
      act_rd_en <= 0;
      load_weight <= 0;
      pe_en <= 0;
      done <= 0;

      case (state)
        S_IDLE: begin
          busy <= 0;
          k_idx <= 0;
          load_cnt <= 0;
          if (start) busy <= 1;
        end

        S_FETCH_PTR1: begin
          // Read row_ptr[k]
          meta_ren <= 1;
          meta_raddr <= k_idx; 
          if (meta_rvalid) ptr_start_reg <= meta_rdata;
        end

        S_FETCH_PTR2: begin
          // Read row_ptr[k+1]
          meta_ren <= 1;
          meta_raddr <= k_idx + 1;
          // meta_rdata will be ptr_end
        end

        S_CALC_LEN: begin
          // Setup block loop
          blk_ptr <= ptr_start_reg;
          blk_end <= meta_rdata; // ptr_end from previous state
        end

        S_FETCH_COL: begin
          // Read col_idx[blk_ptr]
          meta_ren <= 1;
          meta_raddr <= 128 + blk_ptr; // Offset 128 for col_idx table
          if (meta_rvalid) n_idx <= meta_rdata;
        end

        S_LOAD_WGT: begin
          // Load Weight Block (B_k,n) - 8 cycles
          wgt_rd_en <= 1;
          wgt_addr <= (blk_ptr << 3) + load_cnt; 
          load_weight <= 1;    
          load_cnt <= load_cnt + 1;
          
          if (load_cnt == 3'd7) begin
             m_idx <= 0;
             load_cnt <= 0; 
          end
        end

        S_STREAM_ACT: begin
          // Stream Activation Tile (A_m,k)
          act_rd_en <= 1;
          // Address logic: A is stored row-major (M, K). 
          // We need tile at (m_idx, k_idx).
          // Assuming linear addressing: addr = m_idx * KT + k_idx
          act_addr <= m_idx * KT + k_idx; 
          
          pe_en <= 1; // Enable MAC (block_valid = 1)
          
          if (m_idx < MT - 1) m_idx <= m_idx + 1;
        end

        S_NEXT_BLK: begin
          blk_ptr <= blk_ptr + 1;
        end

        S_NEXT_K: begin
          k_idx <= k_idx + 1;
          if (k_idx == KT - 1) begin
            done <= 1;
            busy <= 0;
          end
        end
      endcase
      
      if (abort) begin
        state <= S_IDLE;
        busy <= 0;
      end
    end
  end

  // Always ready to accept metadata when requested
  assign meta_ready = 1'b1;

endmodule
