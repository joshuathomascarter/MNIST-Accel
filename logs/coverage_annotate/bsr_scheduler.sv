//      // verilator_coverage annotation
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
          parameter ADDR_W = 32,
          parameter BLOCK_SIZE = 16  // Block dimension (8x8 or 16x16)
        )(
 012713   input  wire                 clk,
%000007   input  wire                 rst_n,
        
          // Control
%000005   input  wire                 start,
%000001   input  wire                 abort,
%000004   output reg                  busy,
%000004   output reg                  done,
        
          // Configuration
%000007   input  wire [M_W-1:0]       MT, // Number of M tiles
%000004   input  wire [K_W-1:0]       KT, // Number of K tiles (Rows of B)
        
          // Metadata Interface (to meta_decode.sv)
%000004   output reg  [31:0]          meta_raddr,
%000005   output reg                  meta_ren,
%000000   input  wire [31:0]          meta_rdata,
 000014   input  wire                 meta_rvalid,
%000001   output wire                 meta_ready,
        
          // Buffer Interfaces
%000000   output reg                  wgt_rd_en,
%000000   output reg  [ADDR_W-1:0]    wgt_addr,
%000000   output reg                  act_rd_en,
%000000   output reg  [ADDR_W-1:0]    act_addr,
        
          // Systolic Array Control
%000000   output wire                 load_weight, // Latch weight into PE (Registered)
%000000   output reg                  pe_en,       // Enable MAC (block_valid)
%000000   output reg                  accum_en     // Enable accumulation
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
        
%000006   (* fsm_encoding = "one_hot"*) reg [8:0] state, state_n;
        
          // ---------------------------------------------------------------------------
          // 2. Internal Registers
          // ---------------------------------------------------------------------------
%000004   reg [K_W-1:0] k_idx;      // Current K tile (Row of B)
%000000   reg [M_W-1:0] m_idx;      // Current M tile (Row of A)
%000000   reg [4:0]     load_cnt;   // Counter for loading BLOCK_SIZE rows of weights
          
          // Derived parameter for load counter comparison
          localparam LOAD_CNT_MAX = BLOCK_SIZE - 1;
          
%000000   reg [31:0]    blk_ptr;    // Current index in compressed arrays
%000000   reg [31:0]    blk_end;    // End index for current k-row
%000000   reg [31:0]    n_idx;      // Decoded N tile (Col of B)
        
          // Pipeline register for load_weight (SRAM Latency Fix)
%000000   reg load_weight_r;
          assign load_weight = load_weight_r;
        
          // Metadata Latches
%000000   reg [31:0] ptr_start_reg;
        
          // ---------------------------------------------------------------------------
          // 3. Next State Logic
          // ---------------------------------------------------------------------------
 025523   always @(*) begin
 025523     state_n = state;
 025523     case (state)
~025433       S_IDLE:       if (start) state_n = S_FETCH_PTR1;
              
 000046       S_FETCH_PTR1: if (meta_rvalid) state_n = S_FETCH_PTR2;
              
 000022       S_FETCH_PTR2: if (meta_rvalid) state_n = S_CALC_LEN;
              
~025523       S_CALC_LEN:   state_n = (ptr_start_reg == meta_rdata) ? S_NEXT_K : S_FETCH_COL; 
                            // If ptr_start == ptr_end, row is empty -> Next K
              
%000000       S_FETCH_COL:  if (meta_rvalid) state_n = S_LOAD_WGT;
              
~025523       S_LOAD_WGT:   state_n = (load_cnt == LOAD_CNT_MAX[4:0]) ? S_STREAM_ACT : S_LOAD_WGT;
              
%000000       S_STREAM_ACT: if (m_idx == MT - 1) state_n = S_NEXT_BLK;
                            // Stream all M tiles against this Weight Block
              
~025523       S_NEXT_BLK:   state_n = (blk_ptr < blk_end) ? S_FETCH_COL : S_NEXT_K;
              
~020716       S_NEXT_K:     state_n = (k_idx < KT - 1) ? S_FETCH_PTR1 : S_IDLE; // Done if k==KT-1
              
%000002       default:      state_n = S_IDLE;
            endcase
          end
        
          // ---------------------------------------------------------------------------
          // 4. Datapath & Output Logic
          // ---------------------------------------------------------------------------
 012713   always @(posedge clk or negedge rst_n) begin
 012644     if (!rst_n) begin
 000069       state <= S_IDLE;
 000069       k_idx <= 0;
 000069       m_idx <= 0;
 000069       load_cnt <= 0;
 000069       blk_ptr <= 0;
 000069       blk_end <= 0;
              
              // Outputs
 000069       meta_ren <= 0;
 000069       wgt_rd_en <= 0;
 000069       act_rd_en <= 0;
 000069       load_weight_r <= 0;
 000069       pe_en <= 0;
 000069       busy <= 0;
 000069       done <= 0;
 012644     end else begin
 012644       state <= state_n;
              
              // Default Pulses
 012644       meta_ren <= 0;
 012644       wgt_rd_en <= 0;
 012644       act_rd_en <= 0;
              // load_weight_r <= 0; // Don't clear here, controlled explicitly in states
 012644       pe_en <= 0;
 012644       done <= 0;
        
 012644       case (state)
 012602         S_IDLE: begin
 012602           busy <= 0;
 012602           k_idx <= 0;
 012602           load_cnt <= 0;
 012602           load_weight_r <= 0;
~012598           if (start) busy <= 1;
                end
        
 000021         S_FETCH_PTR1: begin
                  // Read row_ptr[k]
 000021           meta_ren <= 1;
 000021           meta_raddr <= k_idx; 
~000016           if (meta_rvalid) ptr_start_reg <= meta_rdata;
                end
        
 000011         S_FETCH_PTR2: begin
                  // Read row_ptr[k+1]
 000011           meta_ren <= 1;
 000011           meta_raddr <= k_idx + 1;
                  // meta_rdata will be ptr_end
                end
        
%000005         S_CALC_LEN: begin
                  // Setup block loop
%000005           blk_ptr <= ptr_start_reg;
%000005           blk_end <= meta_rdata; // ptr_end from previous state
                end
        
%000000         S_FETCH_COL: begin
                  // Read col_idx[blk_ptr]
%000000           meta_ren <= 1;
%000000           meta_raddr <= 128 + blk_ptr; // Offset 128 for col_idx table
%000000           if (meta_rvalid) n_idx <= meta_rdata;
                end
        
%000000         S_LOAD_WGT: begin
                  // Load Weight Block (B_k,n) - BLOCK_SIZE cycles
%000000           wgt_rd_en <= 1;
                  // Address = (Block Index * Block Size) + Row Offset
                  // We use a shift if BLOCK_SIZE is power of 2, or multiply otherwise.
                  // For 16x16, this is (blk_ptr << 4).
%000000           wgt_addr <= (blk_ptr * BLOCK_SIZE) + load_cnt; 
                  
                  // FIX: Pipeline load_weight to match SRAM latency
%000000           load_weight_r <= 1;    
                  
%000000           load_cnt <= load_cnt + 1;
                  
%000000           if (load_cnt == LOAD_CNT_MAX[4:0]) begin
%000000              m_idx <= 0;
%000000              load_cnt <= 0; 
                  end
                end
        
%000000         S_STREAM_ACT: begin
                  // Stop loading weights (delayed by 1 cycle effectively)
%000000           load_weight_r <= 0;
        
                  // Stream Activation Tile (A_m,k)
%000000           act_rd_en <= 1;
                  // Address logic: A is stored row-major (M, K). 
                  // We need tile at (m_idx, k_idx).
                  // Assuming linear addressing: addr = m_idx * KT + k_idx
%000000           act_addr <= m_idx * KT + k_idx; 
                  
%000000           pe_en <= 1; // Enable MAC (block_valid = 1)
                  
%000000           if (m_idx < MT - 1) m_idx <= m_idx + 1;
                end
        
%000000         S_NEXT_BLK: begin
%000000           blk_ptr <= blk_ptr + 1;
                end
        
%000005         S_NEXT_K: begin
%000005           k_idx <= k_idx + 1;
%000004           if (k_idx == KT - 1) begin
%000004             done <= 1;
%000004             busy <= 0;
                  end
                end
              endcase
              
~012644       if (abort) begin
%000000         state <= S_IDLE;
%000000         busy <= 0;
              end
            end
          end
        
          // Always ready to accept metadata when requested
          assign meta_ready = 1'b1;
        
        endmodule
        
