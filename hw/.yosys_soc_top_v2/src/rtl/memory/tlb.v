module tlb (
	clk,
	rst_n,
	lookup_valid,
	lookup_va,
	lookup_asid,
	lookup_is_store,
	lookup_is_exec,
	lookup_hit,
	lookup_pa,
	lookup_fault,
	fill_valid,
	fill_va,
	fill_asid,
	fill_ppn,
	fill_superpage,
	fill_dirty,
	fill_accessed,
	fill_global,
	fill_user,
	fill_exec,
	fill_write,
	fill_read,
	sfence_valid,
	sfence_all,
	sfence_va,
	sfence_asid
);
	reg _sv2v_0;
	parameter signed [31:0] NUM_ENTRIES = 16;
	parameter signed [31:0] VA_WIDTH = 32;
	parameter signed [31:0] PA_WIDTH = 34;
	parameter signed [31:0] ASID_WIDTH = 9;
	input wire clk;
	input wire rst_n;
	input wire lookup_valid;
	input wire [VA_WIDTH - 1:0] lookup_va;
	input wire [ASID_WIDTH - 1:0] lookup_asid;
	input wire lookup_is_store;
	input wire lookup_is_exec;
	output reg lookup_hit;
	output reg [PA_WIDTH - 1:0] lookup_pa;
	output reg lookup_fault;
	input wire fill_valid;
	input wire [VA_WIDTH - 1:0] fill_va;
	input wire [ASID_WIDTH - 1:0] fill_asid;
	input wire [21:0] fill_ppn;
	input wire fill_superpage;
	input wire fill_dirty;
	input wire fill_accessed;
	input wire fill_global;
	input wire fill_user;
	input wire fill_exec;
	input wire fill_write;
	input wire fill_read;
	input wire sfence_valid;
	input wire sfence_all;
	input wire [VA_WIDTH - 1:0] sfence_va;
	input wire [ASID_WIDTH - 1:0] sfence_asid;
	localparam signed [31:0] ENTRY_IDX_BITS = $clog2(NUM_ENTRIES);
	reg entries_valid [0:NUM_ENTRIES - 1];
	reg [ASID_WIDTH - 1:0] entries_asid [0:NUM_ENTRIES - 1];
	reg [9:0] entries_vpn1 [0:NUM_ENTRIES - 1];
	reg [9:0] entries_vpn0 [0:NUM_ENTRIES - 1];
	reg [21:0] entries_ppn [0:NUM_ENTRIES - 1];
	reg entries_superpage [0:NUM_ENTRIES - 1];
	reg entries_dirty [0:NUM_ENTRIES - 1];
	reg entries_accessed [0:NUM_ENTRIES - 1];
	reg entries_global [0:NUM_ENTRIES - 1];
	reg entries_user [0:NUM_ENTRIES - 1];
	reg entries_exec [0:NUM_ENTRIES - 1];
	reg entries_write [0:NUM_ENTRIES - 1];
	reg entries_read [0:NUM_ENTRIES - 1];
	reg [$clog2(NUM_ENTRIES) - 1:0] plru_ptr;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			plru_ptr <= '0;
		else if (fill_valid)
			plru_ptr <= plru_ptr + 1;
	wire [VA_WIDTH - 1:0] va;
	wire [9:0] vpn1;
	wire [9:0] vpn0;
	wire [11:0] page_offset;
	assign va = lookup_va;
	assign vpn1 = va[31:22];
	assign vpn0 = va[21:12];
	assign page_offset = va[11:0];
	reg [NUM_ENTRIES - 1:0] match_vec;
	reg [ENTRY_IDX_BITS - 1:0] match_idx;
	reg [21:0] matched_ppn;
	reg matched_superpage;
	reg matched_dirty;
	reg matched_exec;
	reg matched_write;
	reg matched_read;
	reg [ENTRY_IDX_BITS - 1:0] fill_victim_c;
	reg fill_found_invalid_c;
	function automatic signed [ENTRY_IDX_BITS - 1:0] sv2v_cast_7AA98_signed;
		input reg signed [ENTRY_IDX_BITS - 1:0] inp;
		sv2v_cast_7AA98_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		match_vec = '0;
		match_idx = '0;
		matched_ppn = '0;
		matched_superpage = 1'b0;
		matched_dirty = 1'b0;
		matched_exec = 1'b0;
		matched_write = 1'b0;
		matched_read = 1'b0;
		lookup_hit = 1'b0;
		lookup_pa = '0;
		lookup_fault = 1'b0;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				match_vec[i] = ((entries_valid[i] && (entries_vpn1[i] == vpn1)) && (entries_superpage[i] || (entries_vpn0[i] == vpn0))) && (entries_global[i] || (entries_asid[i] == lookup_asid));
		end
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				if (match_vec[i] && !lookup_hit) begin
					lookup_hit = 1'b1;
					match_idx = sv2v_cast_7AA98_signed(i);
					matched_ppn = entries_ppn[i];
					matched_superpage = entries_superpage[i];
					matched_dirty = entries_dirty[i];
					matched_exec = entries_exec[i];
					matched_write = entries_write[i];
					matched_read = entries_read[i];
				end
		end
		if (lookup_hit) begin
			if (matched_superpage)
				lookup_pa = {matched_ppn[21:10], vpn0, page_offset};
			else
				lookup_pa = {matched_ppn, page_offset};
			if (lookup_is_store && !matched_write)
				lookup_fault = 1'b1;
			else if (lookup_is_exec && !matched_exec)
				lookup_fault = 1'b1;
			else if ((!lookup_is_store && !lookup_is_exec) && !matched_read)
				lookup_fault = 1'b1;
			else if (lookup_is_store && !matched_dirty)
				lookup_fault = 1'b1;
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		fill_found_invalid_c = 1'b0;
		fill_victim_c = plru_ptr;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				if (!entries_valid[i] && !fill_found_invalid_c) begin
					fill_victim_c = sv2v_cast_7AA98_signed(i);
					fill_found_invalid_c = 1'b1;
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_4
			reg signed [31:0] i;
			for (i = 0; i < NUM_ENTRIES; i = i + 1)
				begin
					entries_valid[i] <= 1'b0;
					entries_asid[i] <= '0;
					entries_vpn1[i] <= '0;
					entries_vpn0[i] <= '0;
					entries_ppn[i] <= '0;
					entries_superpage[i] <= 1'b0;
					entries_dirty[i] <= 1'b0;
					entries_accessed[i] <= 1'b0;
					entries_global[i] <= 1'b0;
					entries_user[i] <= 1'b0;
					entries_exec[i] <= 1'b0;
					entries_write[i] <= 1'b0;
					entries_read[i] <= 1'b0;
				end
		end
		else begin
			if (sfence_valid) begin : sv2v_autoblock_5
				reg signed [31:0] i;
				for (i = 0; i < NUM_ENTRIES; i = i + 1)
					if (sfence_all)
						entries_valid[i] <= 1'b0;
					else if (((entries_vpn1[i] == sfence_va[31:22]) && ((sfence_asid == '0) || (entries_asid[i] == sfence_asid))) && !entries_global[i])
						entries_valid[i] <= 1'b0;
			end
			if (fill_valid) begin
				entries_valid[fill_victim_c] <= 1'b1;
				entries_asid[fill_victim_c] <= fill_asid;
				entries_vpn1[fill_victim_c] <= fill_va[31:22];
				entries_vpn0[fill_victim_c] <= fill_va[21:12];
				entries_ppn[fill_victim_c] <= fill_ppn;
				entries_superpage[fill_victim_c] <= fill_superpage;
				entries_dirty[fill_victim_c] <= fill_dirty;
				entries_accessed[fill_victim_c] <= fill_accessed;
				entries_global[fill_victim_c] <= fill_global;
				entries_user[fill_victim_c] <= fill_user;
				entries_exec[fill_victim_c] <= fill_exec;
				entries_write[fill_victim_c] <= fill_write;
				entries_read[fill_victim_c] <= fill_read;
			end
		end
	initial _sv2v_0 = 0;
endmodule
