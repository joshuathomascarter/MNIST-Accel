module barrier_sync (
	clk,
	rst_n,
	tile_barrier_req,
	participant_mask,
	barrier_release,
	arrived_mask,
	barrier_active
);
	parameter signed [31:0] NUM_TILES = 16;
	input wire clk;
	input wire rst_n;
	input wire [NUM_TILES - 1:0] tile_barrier_req;
	input wire [NUM_TILES - 1:0] participant_mask;
	output reg barrier_release;
	output wire [NUM_TILES - 1:0] arrived_mask;
	output wire barrier_active;
	reg [NUM_TILES - 1:0] arrived;
	wire all_arrived;
	assign all_arrived = ((arrived & participant_mask) == participant_mask) && (participant_mask != '0);
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			arrived <= '0;
			barrier_release <= 1'b0;
		end
		else begin
			barrier_release <= 1'b0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < NUM_TILES; i = i + 1)
					if (tile_barrier_req[i])
						arrived[i] <= 1'b1;
			end
			if (all_arrived) begin
				barrier_release <= 1'b1;
				arrived <= '0;
			end
		end
	assign arrived_mask = arrived;
	assign barrier_active = (arrived != '0) && !all_arrived;
endmodule
