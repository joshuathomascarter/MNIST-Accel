module simple_cpu (
	clk,
	rst_n,
	cpu_reset,
	irq_external,
	irq_timer,
	req,
	gnt,
	addr,
	we,
	be,
	wdata,
	rvalid,
	rdata,
	err
);
	reg _sv2v_0;
	parameter [31:0] ADDR_WIDTH = 32;
	parameter [31:0] DATA_WIDTH = 32;
	parameter [31:0] ID_WIDTH = 4;
	input wire clk;
	input wire rst_n;
	input wire cpu_reset;
	input wire irq_external;
	input wire irq_timer;
	output reg req;
	input wire gnt;
	output reg [ADDR_WIDTH - 1:0] addr;
	output reg we;
	output reg [(DATA_WIDTH / 8) - 1:0] be;
	output reg [DATA_WIDTH - 1:0] wdata;
	input wire rvalid;
	input wire [DATA_WIDTH - 1:0] rdata;
	input wire err;
	reg [2:0] state;
	reg [31:0] pc;
	reg [31:0] instr;
	reg [31:0] rf [1:31];
	reg [4:0] rf_wa;
	reg [31:0] rf_wd;
	reg rf_we;
	always @(posedge clk)
		if (rf_we && (rf_wa != 5'd0))
			rf[rf_wa] <= rf_wd;
	wire [4:0] rs1 = instr[19:15];
	wire [31:0] rf_rs1 = (rs1 == 5'd0 ? 32'h00000000 : rf[rs1]);
	wire [4:0] rs2 = instr[24:20];
	wire [31:0] rf_rs2 = (rs2 == 5'd0 ? 32'h00000000 : rf[rs2]);
	reg [31:0] csr_mtvec;
	reg [31:0] csr_mcause;
	reg [31:0] csr_mstatus;
	reg [31:0] csr_mie;
	reg [31:0] csr_mepc;
	reg [31:0] mem_addr_r;
	reg [31:0] mem_wdata_r;
	reg [3:0] mem_be_r;
	reg mem_we_r;
	reg [4:0] mem_rd_r;
	reg [2:0] mem_funct3_r;
	wire irq_pending;
	assign irq_pending = csr_mstatus[3] && ((irq_external && csr_mie[11]) || (irq_timer && csr_mie[7]));
	wire [6:0] opcode = instr[6:0];
	wire [4:0] rd = instr[11:7];
	wire [2:0] funct3 = instr[14:12];
	wire funct7_5 = instr[30];
	wire [31:0] imm_i = {{20 {instr[31]}}, instr[31:20]};
	wire [31:0] imm_s = {{20 {instr[31]}}, instr[31:25], instr[11:7]};
	wire [31:0] imm_b = {{19 {instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
	wire [31:0] imm_u = {instr[31:12], 12'b000000000000};
	wire [31:0] imm_j = {{11 {instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
	wire [31:0] rs1_val = rf_rs1;
	wire [31:0] rs2_val = rf_rs2;
	reg [31:0] alu_result;
	reg branch_taken;
	always @(*) begin
		if (_sv2v_0)
			;
		alu_result = 32'h00000000;
		branch_taken = 1'b0;
		case (opcode)
			7'b0110111: alu_result = imm_u;
			7'b0010111: alu_result = pc + imm_u;
			7'b1101111: alu_result = pc + 4;
			7'b1100111: alu_result = pc + 4;
			7'b1100011:
				case (funct3)
					3'b000: branch_taken = rs1_val == rs2_val;
					3'b001: branch_taken = rs1_val != rs2_val;
					3'b100: branch_taken = $signed(rs1_val) < $signed(rs2_val);
					3'b101: branch_taken = $signed(rs1_val) >= $signed(rs2_val);
					3'b110: branch_taken = rs1_val < rs2_val;
					3'b111: branch_taken = rs1_val >= rs2_val;
					default:
						;
				endcase
			7'b0010011:
				case (funct3)
					3'b000: alu_result = rs1_val + imm_i;
					3'b010: alu_result = {31'b0000000000000000000000000000000, $signed(rs1_val) < $signed(imm_i)};
					3'b011: alu_result = {31'b0000000000000000000000000000000, rs1_val < $unsigned(imm_i)};
					3'b100: alu_result = rs1_val ^ imm_i;
					3'b110: alu_result = rs1_val | imm_i;
					3'b111: alu_result = rs1_val & imm_i;
					3'b001: alu_result = rs1_val << instr[24:20];
					3'b101: alu_result = (funct7_5 ? $signed(rs1_val) >>> instr[24:20] : rs1_val >> instr[24:20]);
				endcase
			7'b0110011:
				case ({funct7_5, funct3})
					4'b0000: alu_result = rs1_val + rs2_val;
					4'b1000: alu_result = rs1_val - rs2_val;
					4'b0001: alu_result = rs1_val << rs2_val[4:0];
					4'b0010: alu_result = {31'b0000000000000000000000000000000, $signed(rs1_val) < $signed(rs2_val)};
					4'b0011: alu_result = {31'b0000000000000000000000000000000, rs1_val < rs2_val};
					4'b0100: alu_result = rs1_val ^ rs2_val;
					4'b0101: alu_result = rs1_val >> rs2_val[4:0];
					4'b1101: alu_result = $signed(rs1_val) >>> rs2_val[4:0];
					4'b0110: alu_result = rs1_val | rs2_val;
					4'b0111: alu_result = rs1_val & rs2_val;
					default:
						;
				endcase
			default:
				;
		endcase
	end
	reg [31:0] store_data;
	reg [3:0] store_be;
	reg [31:0] load_addr;
	always @(*) begin
		if (_sv2v_0)
			;
		load_addr = rs1_val + imm_s;
		store_data = 32'h00000000;
		store_be = 4'b0000;
		case (funct3[1:0])
			2'b00: begin
				store_data = {4 {rs2_val[7:0]}};
				store_be = 4'b0001 << load_addr[1:0];
			end
			2'b01: begin
				store_data = {2 {rs2_val[15:0]}};
				store_be = (load_addr[1] ? 4'b1100 : 4'b0011);
			end
			2'b10: begin
				store_data = rs2_val;
				store_be = 4'b1111;
			end
			default: begin
				store_data = rs2_val;
				store_be = 4'b1111;
			end
		endcase
	end
	reg [31:0] load_result;
	always @(*) begin
		if (_sv2v_0)
			;
		load_result = 32'h00000000;
		case (mem_funct3_r)
			3'b000:
				case (mem_addr_r[1:0])
					2'b00: load_result = {{24 {rdata[7]}}, rdata[7:0]};
					2'b01: load_result = {{24 {rdata[15]}}, rdata[15:8]};
					2'b10: load_result = {{24 {rdata[23]}}, rdata[23:16]};
					2'b11: load_result = {{24 {rdata[31]}}, rdata[31:24]};
				endcase
			3'b001: load_result = (mem_addr_r[1] ? {{16 {rdata[31]}}, rdata[31:16]} : {{16 {rdata[15]}}, rdata[15:0]});
			3'b010: load_result = rdata;
			3'b100:
				case (mem_addr_r[1:0])
					2'b00: load_result = {24'h000000, rdata[7:0]};
					2'b01: load_result = {24'h000000, rdata[15:8]};
					2'b10: load_result = {24'h000000, rdata[23:16]};
					2'b11: load_result = {24'h000000, rdata[31:24]};
				endcase
			3'b101: load_result = (mem_addr_r[1] ? {16'h0000, rdata[31:16]} : {16'h0000, rdata[15:0]});
			default: load_result = rdata;
		endcase
	end
	reg [31:0] csr_rval;
	wire [11:0] csr_addr;
	assign csr_addr = instr[31:20];
	always @(*) begin
		if (_sv2v_0)
			;
		case (csr_addr)
			12'h300: csr_rval = csr_mstatus;
			12'h304: csr_rval = csr_mie;
			12'h305: csr_rval = csr_mtvec;
			12'h341: csr_rval = csr_mepc;
			12'h342: csr_rval = csr_mcause;
			default: csr_rval = 32'h00000000;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			pc <= 32'h00000000;
			instr <= 32'h00000013;
			csr_mtvec <= 32'h00000000;
			csr_mcause <= 32'h00000000;
			csr_mstatus <= 32'h00000000;
			csr_mie <= 32'h00000000;
			csr_mepc <= 32'h00000000;
			mem_addr_r <= '0;
			mem_wdata_r <= '0;
			mem_be_r <= '0;
			mem_we_r <= 1'b0;
			mem_rd_r <= '0;
			mem_funct3_r <= '0;
			rf_we <= 1'b0;
			rf_wa <= '0;
			rf_wd <= '0;
		end
		else begin
			rf_we <= 1'b0;
			case (state)
				3'd0:
					if (gnt)
						state <= 3'd1;
				3'd1:
					if (rvalid) begin
						instr <= rdata;
						if (0)
;
						state <= 3'd2;
					end
				3'd2:
					if (irq_pending) begin
						csr_mepc <= pc;
						csr_mcause <= (irq_external ? 32'h8000000b : 32'h80000007);
						csr_mstatus[3] <= 1'b0;
						csr_mstatus[7] <= csr_mstatus[3];
						pc <= csr_mtvec;
						state <= 3'd0;
					end
					else
						case (opcode)
							7'b0110111: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= pc + 4;
								state <= 3'd0;
							end
							7'b0010111: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= pc + 4;
								state <= 3'd0;
							end
							7'b1101111: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= pc + imm_j;
								state <= 3'd0;
							end
							7'b1100111: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= (rs1_val + imm_i) & ~32'h00000001;
								state <= 3'd0;
							end
							7'b1100011: begin
								pc <= (branch_taken ? pc + imm_b : pc + 4);
								state <= 3'd0;
							end
							7'b0000011: begin
								mem_addr_r <= rs1_val + imm_i;
								mem_we_r <= 1'b0;
								mem_rd_r <= rd;
								mem_funct3_r <= funct3;
								mem_be_r <= 4'b1111;
								state <= 3'd3;
							end
							7'b0100011: begin
								mem_addr_r <= load_addr;
								mem_wdata_r <= store_data;
								mem_be_r <= store_be;
								mem_we_r <= 1'b1;
								state <= 3'd3;
							end
							7'b0010011: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= pc + 4;
								state <= 3'd0;
							end
							7'b0110011: begin
								rf_we <= 1'b1;
								rf_wa <= rd;
								rf_wd <= alu_result;
								pc <= pc + 4;
								state <= 3'd0;
							end
							7'b1110011:
								case (funct3)
									3'b000:
										case (instr[31:20])
											12'h000: begin
												pc <= pc + 4;
												state <= 3'd0;
											end
											12'h001: begin
;
												pc <= pc + 4;
												state <= 3'd0;
											end
											12'h302: begin
												pc <= csr_mepc;
												csr_mstatus[3] <= csr_mstatus[7];
												state <= 3'd0;
											end
											12'h105:
												if (irq_pending) begin
													pc <= pc + 4;
													state <= 3'd0;
												end
												else
													state <= 3'd5;
											default: begin
												pc <= pc + 4;
												state <= 3'd0;
											end
										endcase
									3'b001: begin
										rf_we <= 1'b1;
										rf_wa <= rd;
										rf_wd <= csr_rval;
										case (csr_addr)
											12'h300: csr_mstatus <= rs1_val;
											12'h304: csr_mie <= rs1_val;
											12'h305: csr_mtvec <= rs1_val;
											12'h341: csr_mepc <= rs1_val;
											12'h342: csr_mcause <= rs1_val;
											default:
												;
										endcase
										pc <= pc + 4;
										state <= 3'd0;
									end
									3'b010: begin
										rf_we <= 1'b1;
										rf_wa <= rd;
										rf_wd <= csr_rval;
										if (rs1 != 0)
											case (csr_addr)
												12'h300: csr_mstatus <= csr_mstatus | rs1_val;
												12'h304: csr_mie <= csr_mie | rs1_val;
												12'h305: csr_mtvec <= csr_mtvec | rs1_val;
												12'h341: csr_mepc <= csr_mepc | rs1_val;
												12'h342: csr_mcause <= csr_mcause | rs1_val;
												default:
													;
											endcase
										pc <= pc + 4;
										state <= 3'd0;
									end
									3'b011: begin
										rf_we <= 1'b1;
										rf_wa <= rd;
										rf_wd <= csr_rval;
										if (rs1 != 0)
											case (csr_addr)
												12'h300: csr_mstatus <= csr_mstatus & ~rs1_val;
												12'h304: csr_mie <= csr_mie & ~rs1_val;
												default:
													;
											endcase
										pc <= pc + 4;
										state <= 3'd0;
									end
									default: begin
										pc <= pc + 4;
										state <= 3'd0;
									end
								endcase
							default: begin
								pc <= pc + 4;
								state <= 3'd0;
							end
						endcase
				3'd3:
					if (gnt) begin
						if (0)
;
						state <= 3'd4;
					end
				3'd4:
					if (rvalid) begin
						if (!mem_we_r) begin
							rf_we <= 1'b1;
							rf_wa <= mem_rd_r;
							rf_wd <= load_result;
						end
						pc <= pc + 4;
						state <= 3'd0;
					end
				3'd5:
					if (irq_pending) begin
						pc <= pc + 4;
						state <= 3'd0;
					end
				default: state <= 3'd0;
			endcase
		end
	always @(*) begin
		if (_sv2v_0)
			;
		req = 1'b0;
		addr = pc;
		we = 1'b0;
		be = 4'b1111;
		wdata = 32'h00000000;
		case (state)
			3'd0: begin
				req = 1'b1;
				addr = pc;
				we = 1'b0;
				be = 4'b1111;
			end
			3'd3: begin
				req = 1'b1;
				addr = {mem_addr_r[31:2], 2'b00};
				we = mem_we_r;
				be = mem_be_r;
				wdata = mem_wdata_r;
			end
			default:
				;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
