// =============================================================================
// barrier_sync.sv — Global Barrier Synchronization
// =============================================================================
// Centralized barrier for multi-tile synchronization. Tiles signal arrival
// at barrier; once all participating tiles check in, all are released.
//
// Typically placed at mesh node 0 (master tile) or as a standalone unit.
// Tiles communicate barrier_req/barrier_ack through the NoC or direct wires.

module barrier_sync #(
  parameter int NUM_TILES = 16
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Per-tile barrier request (asserted when tile reaches barrier) ---
  input  logic [NUM_TILES-1:0]   tile_barrier_req,

  // --- Configuration: which tiles participate (bitmask) ---
  input  logic [NUM_TILES-1:0]   participant_mask,

  // --- Global barrier release (asserted for 1 cycle when all arrived) ---
  output logic                   barrier_release,

  // --- Status ---
  output logic [NUM_TILES-1:0]   arrived_mask,
  output logic                   barrier_active
);

  logic [NUM_TILES-1:0] arrived;
  logic all_arrived;

  // Check if all participants have arrived
  assign all_arrived = ((arrived & participant_mask) == participant_mask) &&
                       (participant_mask != '0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      arrived         <= '0;
      barrier_release <= 1'b0;
    end else begin
      barrier_release <= 1'b0;

      // Latch arrivals
      for (int i = 0; i < NUM_TILES; i++) begin
        if (tile_barrier_req[i])
          arrived[i] <= 1'b1;
      end

      // Release and reset when all have arrived
      if (all_arrived) begin
        barrier_release <= 1'b1;
        arrived         <= '0;  // Reset for next barrier
      end
    end
  end

  assign arrived_mask   = arrived;
  assign barrier_active = (arrived != '0) && !all_arrived;

endmodule
