// =============================================================================
// tile_reduce_consumer.sv — Root-Side Reduce Consumer for Local Tile Port
// =============================================================================
// Consumes single-flit MSG_REDUCE packets delivered to a tile's local port,
// accumulates partial sums per reduce_id, and commits completed results into
// tile-local result storage via a commit pulse.

/* verilator lint_off IMPORTSTAR */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off UNUSEDSIGNAL */
import noc_pkg::*;

module tile_reduce_consumer #(
  parameter int NUM_VCS     = noc_pkg::NUM_VCS,
  parameter int ENTRY_DEPTH = noc_pkg::INNET_SP_DEPTH
) (
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  enable,
  input  flit_t                 flit_in,
  input  logic                  valid_in,

  output logic [NUM_VCS-1:0]    credit_out,

  output logic                  commit_valid,
  output logic [7:0]            commit_id,
  output logic [31:0]           commit_value,
  output logic [15:0]           packets_consumed,
  output logic [15:0]           groups_completed
);

  typedef struct packed {
    logic        valid;
    logic [7:0]  reduce_id;
    logic [3:0]  expected;
    logic [3:0]  contrib_count;
    logic [31:0] acc_val;
    logic [7:0]  age;
  } entry_t;

  localparam int HIT_IDX_W = $clog2(ENTRY_DEPTH + 1);

  entry_t entries [ENTRY_DEPTH];

  logic is_reduce_single;
  logic [7:0]  in_reduce_id;
  logic [3:0]  in_expected;
  logic [3:0]  in_contrib_count;
  logic [31:0] in_reduce_val;

  logic [HIT_IDX_W-1:0] hit_idx;
  logic                         hit;
  logic [$clog2(ENTRY_DEPTH)-1:0] free_idx;
  logic                         has_free;
  logic                         can_accept;
  logic [3:0]                   next_contrib;
  logic [31:0]                  next_value;

  assign is_reduce_single = enable
                         && valid_in
                         && (flit_in.msg_type == MSG_REDUCE)
                         && (flit_in.flit_type == FLIT_HEADTAIL);

  assign in_reduce_id     = flit_in.payload[REDUCE_ID_HI     : REDUCE_ID_LO];
  assign in_expected      = flit_in.payload[REDUCE_EXPECT_HI : REDUCE_EXPECT_LO];
  assign in_reduce_val    = flit_in.payload[REDUCE_VAL_HI    : REDUCE_VAL_LO];
  assign in_contrib_count = (flit_in.payload[REDUCE_COUNT_HI : REDUCE_COUNT_LO] == 4'h0)
                          ? 4'h1
                          : flit_in.payload[REDUCE_COUNT_HI : REDUCE_COUNT_LO];

  always_comb begin
    hit_idx = HIT_IDX_W'(ENTRY_DEPTH);
    hit     = 1'b0;
    for (int e = 0; e < ENTRY_DEPTH; e++) begin
      if (entries[e].valid && (entries[e].reduce_id == in_reduce_id)) begin
        hit_idx = HIT_IDX_W'(e);
        hit     = 1'b1;
      end
    end
  end

  always_comb begin
    free_idx = '0;
    has_free = 1'b0;
    for (int e = ENTRY_DEPTH - 1; e >= 0; e--) begin
      if (!entries[e].valid) begin
        free_idx = $clog2(ENTRY_DEPTH)'(e);
        has_free = 1'b1;
      end
    end
  end

  assign next_contrib = hit ? (entries[int'(hit_idx)].contrib_count + in_contrib_count)
                            : in_contrib_count;
  assign next_value   = hit ? (entries[int'(hit_idx)].acc_val + in_reduce_val)
                            : in_reduce_val;

  assign can_accept = is_reduce_single &&
                      (hit || has_free || (in_contrib_count >= in_expected) || (in_expected <= 4'd1));

  always_comb begin
    credit_out = '0;
    if (can_accept)
      credit_out[flit_in.vc_id] = 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      packets_consumed <= '0;
      groups_completed <= '0;
      commit_valid     <= 1'b0;
      commit_id        <= '0;
      commit_value     <= '0;
      for (int e = 0; e < ENTRY_DEPTH; e++) begin
        entries[e].valid         <= 1'b0;
        entries[e].reduce_id     <= '0;
        entries[e].expected      <= '0;
        entries[e].contrib_count <= '0;
        entries[e].acc_val       <= '0;
        entries[e].age           <= '0;
      end
    end else begin
      commit_valid <= 1'b0;

      for (int e = 0; e < ENTRY_DEPTH; e++) begin
        if (entries[e].valid) begin
          if (entries[e].age == 8'hFF) begin
            entries[e].valid <= 1'b0;
            entries[e].age   <= '0;
          end else begin
            entries[e].age <= entries[e].age + 8'h1;
          end
        end
      end

      if (can_accept) begin
        packets_consumed <= packets_consumed + 16'd1;

        if (next_contrib >= in_expected) begin
          commit_valid     <= 1'b1;
          commit_id        <= in_reduce_id;
          commit_value     <= next_value;
          groups_completed <= groups_completed + 16'd1;

          if (hit)
            entries[int'(hit_idx)].valid <= 1'b0;
        end else if (hit) begin
          entries[int'(hit_idx)].contrib_count <= next_contrib;
          entries[int'(hit_idx)].acc_val       <= next_value;
          entries[int'(hit_idx)].age           <= '0;
        end else begin
          entries[free_idx].valid         <= 1'b1;
          entries[free_idx].reduce_id     <= in_reduce_id;
          entries[free_idx].expected      <= in_expected;
          entries[free_idx].contrib_count <= in_contrib_count;
          entries[free_idx].acc_val       <= in_reduce_val;
          entries[free_idx].age           <= '0;
        end
      end
    end
  end

endmodule
