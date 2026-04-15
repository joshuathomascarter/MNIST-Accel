module axi_arbiter (
	clk,
	rst_n,
	req,
	handshake_done,
	grant,
	grant_idx
);
	reg _sv2v_0;
	parameter [31:0] NUM_MASTERS = 2;
	input wire clk;
	input wire rst_n;
	input wire [NUM_MASTERS - 1:0] req;
	input wire handshake_done;
	output wire [NUM_MASTERS - 1:0] grant;
	output wire [$clog2(NUM_MASTERS) - 1:0] grant_idx;
	localparam [31:0] IDX_W = $clog2(NUM_MASTERS);
	reg [IDX_W - 1:0] priority_r;
	reg [NUM_MASTERS - 1:0] req_masked;
	wire [NUM_MASTERS - 1:0] grant_masked;
	wire [NUM_MASTERS - 1:0] grant_unmasked;
	wire masked_has_req;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_MASTERS; i = i + 1)
				req_masked[i] = req[i] && (i >= priority_r);
		end
	end
	assign masked_has_req = |req_masked;
	function [NUM_MASTERS - 1:0] find_first;
		input [NUM_MASTERS - 1:0] vec;
		reg [NUM_MASTERS - 1:0] result;
		integer i;
		begin
			result = {NUM_MASTERS {1'b0}};
			for (i = 0; i < NUM_MASTERS; i = i + 1)
				if (vec[i] && (result == {NUM_MASTERS {1'b0}}))
					result[i] = 1'b1;
			find_first = result;
		end
	endfunction
	function automatic signed [IDX_W - 1:0] sv2v_cast_9B931_signed;
		input reg signed [IDX_W - 1:0] inp;
		sv2v_cast_9B931_signed = inp;
	endfunction
	function [IDX_W - 1:0] onehot_to_idx;
		input [NUM_MASTERS - 1:0] oh;
		reg [IDX_W - 1:0] idx;
		integer i;
		begin
			idx = {IDX_W {1'b0}};
			for (i = 0; i < NUM_MASTERS; i = i + 1)
				if (oh[i])
					idx = sv2v_cast_9B931_signed(i);
			onehot_to_idx = idx;
		end
	endfunction
	assign grant_masked = find_first(req_masked);
	assign grant_unmasked = find_first(req);
	wire [NUM_MASTERS - 1:0] grant_raw;
	wire [IDX_W - 1:0] grant_raw_idx;
	assign grant_raw = (masked_has_req ? grant_masked : grant_unmasked);
	assign grant_raw_idx = onehot_to_idx(grant_raw);
	reg [NUM_MASTERS - 1:0] grant_locked;
	wire [IDX_W - 1:0] grant_locked_idx;
	reg lock_active;
	assign grant_locked_idx = onehot_to_idx(grant_locked);
	function automatic [IDX_W - 1:0] sv2v_cast_9B931;
		input reg [IDX_W - 1:0] inp;
		sv2v_cast_9B931 = inp;
	endfunction
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			priority_r <= '0;
			grant_locked <= '0;
			lock_active <= 1'b0;
		end
		else if (lock_active) begin
			if (handshake_done) begin
				lock_active <= 1'b0;
				if (grant_locked_idx == sv2v_cast_9B931(NUM_MASTERS - 1))
					priority_r <= '0;
				else
					priority_r <= grant_locked_idx + 1'b1;
			end
		end
		else if (|grant_raw) begin
			if (handshake_done) begin
				if (grant_raw_idx == sv2v_cast_9B931(NUM_MASTERS - 1))
					priority_r <= '0;
				else
					priority_r <= grant_raw_idx + 1'b1;
			end
			else begin
				grant_locked <= grant_raw;
				lock_active <= 1'b1;
			end
		end
	assign grant = (lock_active ? grant_locked : grant_raw);
	assign grant_idx = (lock_active ? grant_locked_idx : grant_raw_idx);
	initial _sv2v_0 = 0;
endmodule
