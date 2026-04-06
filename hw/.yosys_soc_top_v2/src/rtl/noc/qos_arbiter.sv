// =============================================================================
// qos_arbiter.sv — Priority-Based AXI QoS Arbiter
// =============================================================================
// Arbitrates between multiple AXI masters with priority + bandwidth allocation.
//
// Features:
//   - Configurable priority per master (higher number = higher priority)
//   - Token-bucket bandwidth limiter per master
//   - Strict priority with round-robin tie-breaking
//   - Starvation prevention via age-based promotion

module qos_arbiter #(
  parameter int NUM_MASTERS   = 4,
  parameter int ADDR_WIDTH    = 32,
  parameter int DATA_WIDTH    = 32,
  parameter int ID_WIDTH      = 4,
  parameter int TOKEN_DEPTH   = 16,    // Token bucket depth
  parameter int REFILL_PERIOD = 64     // Cycles between token refills
) (
  input  logic              clk,
  input  logic              rst_n,

  // --- Per-master priority (CSR-programmed) ---
  input  logic [2:0]        priority_i  [NUM_MASTERS],
  input  logic [7:0]        bw_alloc_i  [NUM_MASTERS],  // Tokens replenished per period

  // --- AXI AR requests from masters ---
  input  logic              ar_valid_i  [NUM_MASTERS],
  output logic              ar_ready_o  [NUM_MASTERS],
  input  logic [ADDR_WIDTH-1:0] ar_addr_i [NUM_MASTERS],
  input  logic [ID_WIDTH-1:0]   ar_id_i   [NUM_MASTERS],
  input  logic [7:0]        ar_len_i    [NUM_MASTERS],
  input  logic [2:0]        ar_size_i   [NUM_MASTERS],
  input  logic [1:0]        ar_burst_i  [NUM_MASTERS],

  // --- AXI AW requests from masters ---
  input  logic              aw_valid_i  [NUM_MASTERS],
  output logic              aw_ready_o  [NUM_MASTERS],
  input  logic [ADDR_WIDTH-1:0] aw_addr_i [NUM_MASTERS],
  input  logic [ID_WIDTH-1:0]   aw_id_i   [NUM_MASTERS],
  input  logic [7:0]        aw_len_i    [NUM_MASTERS],
  input  logic [2:0]        aw_size_i   [NUM_MASTERS],
  input  logic [1:0]        aw_burst_i  [NUM_MASTERS],

  // --- Granted read channel (to downstream slave) ---
  output logic              ar_valid_o,
  input  logic              ar_ready_i,
  output logic [ADDR_WIDTH-1:0] ar_addr_o,
  output logic [ID_WIDTH-1:0]   ar_id_o,
  output logic [7:0]        ar_len_o,
  output logic [2:0]        ar_size_o,
  output logic [1:0]        ar_burst_o,
  output logic [$clog2(NUM_MASTERS)-1:0] ar_grant_id_o,

  // --- Granted write channel (to downstream slave) ---
  output logic              aw_valid_o,
  input  logic              aw_ready_i,
  output logic [ADDR_WIDTH-1:0] aw_addr_o,
  output logic [ID_WIDTH-1:0]   aw_id_o,
  output logic [7:0]        aw_len_o,
  output logic [2:0]        aw_size_o,
  output logic [1:0]        aw_burst_o,
  output logic [$clog2(NUM_MASTERS)-1:0] aw_grant_id_o,

  // --- Status ---
  output logic [NUM_MASTERS-1:0] throttled_o   // Master has no tokens
);

  localparam int M_BITS = $clog2(NUM_MASTERS);

  // =========================================================================
  // Token bucket per master — bandwidth limiter
  // =========================================================================
  logic [$clog2(TOKEN_DEPTH):0] tokens [NUM_MASTERS];
  logic [$clog2(REFILL_PERIOD)-1:0] refill_cnt;

  // Refill counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      refill_cnt <= '0;
    else if (refill_cnt == REFILL_PERIOD - 1)
      refill_cnt <= '0;
    else
      refill_cnt <= refill_cnt + 1;
  end

  logic refill_tick;
  assign refill_tick = (refill_cnt == '0);

  // Token management
  generate
    for (genvar m = 0; m < NUM_MASTERS; m++) begin : g_tokens
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          tokens[m] <= TOKEN_DEPTH;
        end else begin
          // Refill
          if (refill_tick) begin
            if (tokens[m] + bw_alloc_i[m] > TOKEN_DEPTH)
              tokens[m] <= TOKEN_DEPTH;
            else
              tokens[m] <= tokens[m] + bw_alloc_i[m];
          end

          // Consume on grant
          if ((ar_ready_o[m] && ar_valid_i[m]) ||
              (aw_ready_o[m] && aw_valid_i[m])) begin
            if (tokens[m] > 0)
              tokens[m] <= tokens[m] - 1;
          end
        end
      end

      assign throttled_o[m] = (tokens[m] == 0);
    end
  endgenerate

  // =========================================================================
  // Age counter per master — starvation prevention
  // =========================================================================
  logic [7:0] age [NUM_MASTERS];

  generate
    for (genvar m = 0; m < NUM_MASTERS; m++) begin : g_age
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          age[m] <= '0;
        else begin
          if (ar_valid_i[m] || aw_valid_i[m]) begin
            if (age[m] < 8'hFF)
              age[m] <= age[m] + 1;
          end else begin
            age[m] <= '0;
          end

          // Reset on grant
          if (ar_ready_o[m] || aw_ready_o[m])
            age[m] <= '0;
        end
      end
    end
  endgenerate

  // =========================================================================
  // Effective priority = base priority + age promotion
  // =========================================================================
  logic [3:0] eff_priority [NUM_MASTERS];

  always_comb begin
    for (int m = 0; m < NUM_MASTERS; m++) begin
      eff_priority[m] = {1'b0, priority_i[m]};
      // Promote after 128 cycles of waiting (prevent starvation)
      if (age[m] >= 8'd128)
        eff_priority[m] = 4'hF;
    end
  end

  // =========================================================================
  // Read arbitration — strict priority with round-robin tie-break
  // =========================================================================
  logic [M_BITS-1:0] ar_winner;
  logic              ar_winner_valid;
  logic [M_BITS-1:0] ar_rr_ptr;
  // Hoisted combinational temporaries (no inline decls after statements)
  logic [3:0]        arb_best_pri;
  int                arb_m;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      ar_rr_ptr <= '0;
    else if (ar_valid_o && ar_ready_i)
      ar_rr_ptr <= ar_winner + 1;
  end

  always_comb begin
    ar_winner       = '0;
    ar_winner_valid = 1'b0;

    // Find highest effective priority among valid, non-throttled requestors
    arb_best_pri = '0;
    for (int pass = 0; pass < NUM_MASTERS; pass++) begin
      arb_m = (ar_rr_ptr + pass) % NUM_MASTERS;
      if (ar_valid_i[arb_m] && !throttled_o[arb_m]) begin
        if (!ar_winner_valid || eff_priority[arb_m] > arb_best_pri) begin
          arb_best_pri    = eff_priority[arb_m];
          ar_winner       = arb_m[M_BITS-1:0];
          ar_winner_valid = 1'b1;
        end
      end
    end
  end

  // =========================================================================
  // Write arbitration — same scheme
  // =========================================================================
  logic [M_BITS-1:0] aw_winner;
  logic              aw_winner_valid;
  logic [M_BITS-1:0] aw_rr_ptr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      aw_rr_ptr <= '0;
    else if (aw_valid_o && aw_ready_i)
      aw_rr_ptr <= aw_winner + 1;
  end

  always_comb begin
    aw_winner       = '0;
    aw_winner_valid = 1'b0;

    // Find highest effective priority among valid, non-throttled requestors
    arb_best_pri = '0;
    for (int pass = 0; pass < NUM_MASTERS; pass++) begin
      arb_m = (aw_rr_ptr + pass) % NUM_MASTERS;
      if (aw_valid_i[arb_m] && !throttled_o[arb_m]) begin
        if (!aw_winner_valid || eff_priority[arb_m] > arb_best_pri) begin
          arb_best_pri    = eff_priority[arb_m];
          aw_winner       = arb_m[M_BITS-1:0];
          aw_winner_valid = 1'b1;
        end
      end
    end
  end

  // =========================================================================
  // Output muxing — read channel
  // =========================================================================
  assign ar_valid_o    = ar_winner_valid;
  assign ar_addr_o     = ar_addr_i[ar_winner];
  assign ar_id_o       = ar_id_i[ar_winner];
  assign ar_len_o      = ar_len_i[ar_winner];
  assign ar_size_o     = ar_size_i[ar_winner];
  assign ar_burst_o    = ar_burst_i[ar_winner];
  assign ar_grant_id_o = ar_winner;

  // Output muxing — write channel
  assign aw_valid_o    = aw_winner_valid;
  assign aw_addr_o     = aw_addr_i[aw_winner];
  assign aw_id_o       = aw_id_i[aw_winner];
  assign aw_len_o      = aw_len_i[aw_winner];
  assign aw_size_o     = aw_size_i[aw_winner];
  assign aw_burst_o    = aw_burst_i[aw_winner];
  assign aw_grant_id_o = aw_winner;

  // =========================================================================
  // Per-master ready signals
  // =========================================================================
  generate
    for (genvar m = 0; m < NUM_MASTERS; m++) begin : g_ready
      assign ar_ready_o[m] = ar_winner_valid && (ar_winner == m[M_BITS-1:0]) && ar_ready_i;
      assign aw_ready_o[m] = aw_winner_valid && (aw_winner == m[M_BITS-1:0]) && aw_ready_i;
    end
  endgenerate

endmodule
