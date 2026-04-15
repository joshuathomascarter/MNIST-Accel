module page_table_walker (
	clk,
	rst_n,
	walk_req_valid,
	walk_req_ready,
	walk_va,
	walk_asid,
	walk_is_store,
	walk_is_exec,
	walk_done,
	walk_fault,
	walk_fault_cause,
	walk_result_va,
	walk_result_asid,
	walk_result_ppn,
	walk_result_superpage,
	walk_result_dirty,
	walk_result_accessed,
	walk_result_global,
	walk_result_user,
	walk_result_exec,
	walk_result_write,
	walk_result_read,
	mem_req_valid,
	mem_req_ready,
	mem_req_addr,
	mem_resp_valid,
	mem_resp_data,
	satp_ppn,
	satp_mode
);
	reg _sv2v_0;
	parameter signed [31:0] VA_WIDTH = 32;
	parameter signed [31:0] PA_WIDTH = 34;
	parameter signed [31:0] PTE_WIDTH = 32;
	parameter signed [31:0] ASID_WIDTH = 9;
	input wire clk;
	input wire rst_n;
	input wire walk_req_valid;
	output wire walk_req_ready;
	input wire [VA_WIDTH - 1:0] walk_va;
	input wire [ASID_WIDTH - 1:0] walk_asid;
	input wire walk_is_store;
	input wire walk_is_exec;
	output wire walk_done;
	output wire walk_fault;
	output reg [1:0] walk_fault_cause;
	output wire [VA_WIDTH - 1:0] walk_result_va;
	output wire [ASID_WIDTH - 1:0] walk_result_asid;
	output wire [21:0] walk_result_ppn;
	output wire walk_result_superpage;
	output wire walk_result_dirty;
	output wire walk_result_accessed;
	output wire walk_result_global;
	output wire walk_result_user;
	output wire walk_result_exec;
	output wire walk_result_write;
	output wire walk_result_read;
	output wire mem_req_valid;
	input wire mem_req_ready;
	output wire [PA_WIDTH - 1:0] mem_req_addr;
	input wire mem_resp_valid;
	input wire [PTE_WIDTH - 1:0] mem_resp_data;
	input wire [21:0] satp_ppn;
	input wire satp_mode;
	reg [2:0] state;
	reg [2:0] state_next;
	reg [VA_WIDTH - 1:0] va_reg;
	reg [ASID_WIDTH - 1:0] asid_reg;
	reg is_store_reg;
	reg is_exec_reg;
	wire [9:0] vpn1;
	wire [9:0] vpn0;
	reg [PTE_WIDTH - 1:0] pte_reg;
	reg [21:0] l1_ppn;
	assign vpn1 = va_reg[31:22];
	assign vpn0 = va_reg[21:12];
	wire pte_v;
	wire pte_r;
	wire pte_w;
	wire pte_x;
	wire pte_u;
	wire pte_g;
	wire pte_a;
	wire pte_d;
	wire [21:0] pte_ppn;
	wire [11:0] pte_ppn1;
	wire [9:0] pte_ppn0;
	assign pte_v = pte_reg[0];
	assign pte_r = pte_reg[1];
	assign pte_w = pte_reg[2];
	assign pte_x = pte_reg[3];
	assign pte_u = pte_reg[4];
	assign pte_g = pte_reg[5];
	assign pte_a = pte_reg[6];
	assign pte_d = pte_reg[7];
	assign pte_ppn0 = pte_reg[19:10];
	assign pte_ppn1 = pte_reg[31:20];
	assign pte_ppn = {pte_ppn1, pte_ppn0};
	wire pte_is_leaf;
	assign pte_is_leaf = pte_r || pte_x;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			state <= 3'd0;
		else
			state <= state_next;
	always @(*) begin
		if (_sv2v_0)
			;
		state_next = state;
		case (state)
			3'd0:
				if (walk_req_valid && satp_mode)
					state_next = 3'd1;
			3'd1:
				if (mem_req_ready)
					state_next = 3'd2;
			3'd2:
				if (mem_resp_valid) begin
					if (!mem_resp_data[0])
						state_next = 3'd6;
					else if (mem_resp_data[1] || mem_resp_data[3])
						state_next = 3'd5;
					else
						state_next = 3'd3;
				end
			3'd3:
				if (mem_req_ready)
					state_next = 3'd4;
			3'd4:
				if (mem_resp_valid) begin
					if (!mem_resp_data[0])
						state_next = 3'd6;
					else
						state_next = 3'd5;
				end
			3'd5: state_next = 3'd0;
			3'd6: state_next = 3'd0;
			default: state_next = 3'd0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			va_reg <= '0;
			asid_reg <= '0;
			is_store_reg <= 1'b0;
			is_exec_reg <= 1'b0;
			pte_reg <= '0;
			l1_ppn <= '0;
		end
		else
			case (state)
				3'd0:
					if (walk_req_valid && satp_mode) begin
						va_reg <= walk_va;
						asid_reg <= walk_asid;
						is_store_reg <= walk_is_store;
						is_exec_reg <= walk_is_exec;
					end
				3'd2:
					if (mem_resp_valid) begin
						pte_reg <= mem_resp_data;
						l1_ppn <= {mem_resp_data[31:20], mem_resp_data[19:10]};
					end
				3'd4:
					if (mem_resp_valid)
						pte_reg <= mem_resp_data;
				default:
					;
			endcase
	wire [PA_WIDTH - 1:0] l1_pte_addr;
	wire [PA_WIDTH - 1:0] l0_pte_addr;
	assign l1_pte_addr = {satp_ppn, 12'b000000000000} + {{PA_WIDTH - 12 {1'b0}}, vpn1, 2'b00};
	assign l0_pte_addr = {l1_ppn, 12'b000000000000} + {{PA_WIDTH - 12 {1'b0}}, vpn0, 2'b00};
	assign mem_req_valid = (state == 3'd1) || (state == 3'd3);
	assign mem_req_addr = (state == 3'd1 ? l1_pte_addr : l0_pte_addr);
	assign walk_req_ready = (state == 3'd0) && satp_mode;
	reg perm_fault;
	always @(*) begin
		if (_sv2v_0)
			;
		perm_fault = 1'b0;
		if (state == 3'd5) begin
			if (!pte_a)
				perm_fault = 1'b1;
			if (is_store_reg && !pte_w)
				perm_fault = 1'b1;
			if (is_store_reg && !pte_d)
				perm_fault = 1'b1;
			if (is_exec_reg && !pte_x)
				perm_fault = 1'b1;
			if ((!is_store_reg && !is_exec_reg) && !pte_r)
				perm_fault = 1'b1;
		end
	end
	wire superpage_misaligned;
	assign superpage_misaligned = (((state == 3'd5) && pte_is_leaf) && (l1_ppn == pte_ppn)) && (pte_ppn0 != '0);
	assign walk_done = (state == 3'd5) || (state == 3'd6);
	assign walk_fault = ((state == 3'd6) || perm_fault) || superpage_misaligned;
	always @(*) begin
		if (_sv2v_0)
			;
		walk_fault_cause = 2'b00;
		if (state == 3'd6)
			walk_fault_cause = 2'b00;
		if (perm_fault)
			walk_fault_cause = 2'b01;
		if (superpage_misaligned)
			walk_fault_cause = 2'b10;
	end
	wire is_superpage;
	assign is_superpage = (state == 3'd5) && (l1_ppn == pte_ppn);
	assign walk_result_va = va_reg;
	assign walk_result_asid = asid_reg;
	assign walk_result_ppn = pte_ppn;
	assign walk_result_superpage = is_superpage;
	assign walk_result_dirty = pte_d;
	assign walk_result_accessed = pte_a;
	assign walk_result_global = pte_g;
	assign walk_result_user = pte_u;
	assign walk_result_exec = pte_x;
	assign walk_result_write = pte_w;
	assign walk_result_read = pte_r;
	initial _sv2v_0 = 0;
endmodule
