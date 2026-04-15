`default_nettype none
module perf (
	clk,
	rst_n,
	start_pulse,
	done_pulse,
	busy_signal,
	pe_en_signal,
	sched_busy_signal,
	dma_beat_valid,
	block_done_pulse,
	total_cycles_count,
	active_cycles_count,
	idle_cycles_count,
	dma_bytes_count,
	blocks_processed_count,
	stall_cycles_count,
	measurement_done
);
	parameter COUNTER_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire start_pulse;
	input wire done_pulse;
	input wire busy_signal;
	input wire pe_en_signal;
	input wire sched_busy_signal;
	input wire dma_beat_valid;
	input wire block_done_pulse;
	output reg [COUNTER_WIDTH - 1:0] total_cycles_count;
	output reg [COUNTER_WIDTH - 1:0] active_cycles_count;
	output reg [COUNTER_WIDTH - 1:0] idle_cycles_count;
	output reg [31:0] dma_bytes_count;
	output reg [31:0] blocks_processed_count;
	output reg [31:0] stall_cycles_count;
	output reg measurement_done;
	localparam S_IDLE = 1'b0;
	localparam S_MEASURING = 1'b1;
	reg state_reg;
	reg prev_state;
	reg [COUNTER_WIDTH - 1:0] run_total;
	reg [COUNTER_WIDTH - 1:0] run_active;
	reg [COUNTER_WIDTH - 1:0] run_idle;
	reg [31:0] run_dma_bytes;
	reg [31:0] run_blocks;
	reg [31:0] run_stalls;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state_reg <= S_IDLE;
			prev_state <= S_IDLE;
			run_total <= {COUNTER_WIDTH {1'b0}};
			run_active <= {COUNTER_WIDTH {1'b0}};
			run_idle <= {COUNTER_WIDTH {1'b0}};
			run_dma_bytes <= 32'd0;
			run_blocks <= 32'd0;
			run_stalls <= 32'd0;
			total_cycles_count <= {COUNTER_WIDTH {1'b0}};
			active_cycles_count <= {COUNTER_WIDTH {1'b0}};
			idle_cycles_count <= {COUNTER_WIDTH {1'b0}};
			dma_bytes_count <= 32'd0;
			blocks_processed_count <= 32'd0;
			stall_cycles_count <= 32'd0;
			measurement_done <= 1'b0;
		end
		else begin
			prev_state <= state_reg;
			measurement_done <= 1'b0;
			case (state_reg)
				S_IDLE:
					if (start_pulse) begin
						state_reg <= S_MEASURING;
						run_total <= {COUNTER_WIDTH {1'b0}};
						run_active <= {COUNTER_WIDTH {1'b0}};
						run_idle <= {COUNTER_WIDTH {1'b0}};
						run_dma_bytes <= 32'd0;
						run_blocks <= 32'd0;
						run_stalls <= 32'd0;
					end
				S_MEASURING: begin
					run_total <= run_total + 1;
					if (busy_signal)
						run_active <= run_active + 1;
					else
						run_idle <= run_idle + 1;
					if (dma_beat_valid)
						run_dma_bytes <= run_dma_bytes + 32'd8;
					if (block_done_pulse)
						run_blocks <= run_blocks + 1;
					if ((sched_busy_signal && !pe_en_signal) && !dma_beat_valid)
						run_stalls <= run_stalls + 1;
					if (done_pulse)
						state_reg <= S_IDLE;
				end
				default: state_reg <= S_IDLE;
			endcase
			if ((prev_state == S_MEASURING) && (state_reg == S_IDLE)) begin
				total_cycles_count <= run_total;
				active_cycles_count <= run_active;
				idle_cycles_count <= run_idle;
				dma_bytes_count <= run_dma_bytes;
				blocks_processed_count <= run_blocks;
				stall_cycles_count <= run_stalls;
				measurement_done <= 1'b1;
			end
		end
endmodule
`default_nettype wire
