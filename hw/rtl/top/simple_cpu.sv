// simple_cpu.sv — Minimal multi-cycle RV32I CPU
// Synthesizable: uses explicit 2R1W register file (no behavioral array).
// Supports: RV32I base (no M/C) + mtvec/mcause/mstatus/mie CSRs + WFI/MRET/EBREAK

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */

module simple_cpu #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned ID_WIDTH = 4
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              cpu_reset,
  input  logic              irq_external,
  input  logic              irq_timer,

  output logic              req,
  input  logic              gnt,
  output logic [ADDR_WIDTH-1:0] addr,
  output logic              we,
  output logic [DATA_WIDTH/8-1:0] be,
  output logic [DATA_WIDTH-1:0] wdata,

  input  logic              rvalid,
  input  logic [DATA_WIDTH-1:0] rdata,
  input  logic              err
);

  // -----------------------------------------------------------------------
  // State machine: FETCH → EXEC → MEM_RD/MEM_WR → WB → FETCH
  // -----------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_FETCH,        // request instruction read
    S_FETCH_WAIT,   // wait for rvalid
    S_EXEC,         // decode + execute (1 cycle)
    S_MEM,          // load/store: issue request
    S_MEM_WAIT,     // load/store: wait for response
    S_WFI           // wait for interrupt
  } state_e;

  state_e state;

  logic [31:0] pc;
  logic [31:0] instr;

  // -----------------------------------------------------------------------
  // Synthesizable 2R1W Register File (x0 hardwired to 0)
  // -----------------------------------------------------------------------
  logic [31:0] rf [1:31];  // x1..x31; x0 is always 0
  logic [4:0]  rf_wa;      // write address
  logic [31:0] rf_wd;      // write data
  logic        rf_we;      // write enable

  always_ff @(posedge clk) begin
    if (rf_we && rf_wa != 5'd0)
      rf[rf_wa] <= rf_wd;
  end

  // Synchronous read (combinational for multi-cycle — reads registered values)
  wire [31:0] rf_rs1 = (rs1 == 5'd0) ? 32'h0 : rf[rs1];
  wire [31:0] rf_rs2 = (rs2 == 5'd0) ? 32'h0 : rf[rs2];

  // CSRs (Machine mode subset)
  logic [31:0] csr_mtvec;
  logic [31:0] csr_mcause;
  logic [31:0] csr_mstatus;
  logic [31:0] csr_mie;
  logic [31:0] csr_mepc;

  // Memory access staging
  logic [31:0] mem_addr_r, mem_wdata_r;
  logic [3:0]  mem_be_r;
  logic        mem_we_r;
  logic [4:0]  mem_rd_r;     // destination register for loads
  logic [2:0]  mem_funct3_r; // for load sign/zero extension

  // Interrupt pending
  logic irq_pending;
  assign irq_pending = (csr_mstatus[3]) &&   // MIE bit
                       (  (irq_external && csr_mie[11])   // MEIE
                       || (irq_timer    && csr_mie[7]) ); // MTIE

  // -----------------------------------------------------------------------
  // Instruction decode helpers
  // -----------------------------------------------------------------------
  wire [6:0]  opcode  = instr[6:0];
  wire [4:0]  rd      = instr[11:7];
  wire [2:0]  funct3  = instr[14:12];
  wire [4:0]  rs1     = instr[19:15];
  wire [4:0]  rs2     = instr[24:20];
  wire        funct7_5 = instr[30];

  // Immediates
  wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
  wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
  wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
  wire [31:0] imm_u = {instr[31:12], 12'b0};
  wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

  // Register read
  wire [31:0] rs1_val = rf_rs1;
  wire [31:0] rs2_val = rf_rs2;

  // ALU
  logic [31:0] alu_result;
  logic        branch_taken;

  always_comb begin
    alu_result   = 32'h0;
    branch_taken = 1'b0;

    case (opcode)
      7'b0110111: alu_result = imm_u;                          // LUI
      7'b0010111: alu_result = pc + imm_u;                     // AUIPC
      7'b1101111: alu_result = pc + 4;                         // JAL (link)
      7'b1100111: alu_result = pc + 4;                         // JALR (link)

      7'b1100011: begin // Branch
        case (funct3)
          3'b000: branch_taken = (rs1_val == rs2_val);                          // BEQ
          3'b001: branch_taken = (rs1_val != rs2_val);                          // BNE
          3'b100: branch_taken = ($signed(rs1_val) <  $signed(rs2_val));        // BLT
          3'b101: branch_taken = ($signed(rs1_val) >= $signed(rs2_val));        // BGE
          3'b110: branch_taken = (rs1_val < rs2_val);                           // BLTU
          3'b111: branch_taken = (rs1_val >= rs2_val);                          // BGEU
          default: ;
        endcase
      end

      7'b0010011: begin // OP-IMM (arith with imm)
        case (funct3)
          3'b000: alu_result = rs1_val + imm_i;                                // ADDI
          3'b010: alu_result = {31'b0, $signed(rs1_val) < $signed(imm_i)};     // SLTI
          3'b011: alu_result = {31'b0, rs1_val < $unsigned(imm_i)};            // SLTIU
          3'b100: alu_result = rs1_val ^ imm_i;                                // XORI
          3'b110: alu_result = rs1_val | imm_i;                                // ORI
          3'b111: alu_result = rs1_val & imm_i;                                // ANDI
          3'b001: alu_result = rs1_val << instr[24:20];                         // SLLI
          3'b101: alu_result = funct7_5
                    ? ($signed(rs1_val) >>> instr[24:20])                       // SRAI
                    : (rs1_val >> instr[24:20]);                                // SRLI
        endcase
      end

      7'b0110011: begin // OP (register-register)
        case ({funct7_5, funct3})
          4'b0_000: alu_result = rs1_val + rs2_val;                             // ADD
          4'b1_000: alu_result = rs1_val - rs2_val;                             // SUB
          4'b0_001: alu_result = rs1_val << rs2_val[4:0];                       // SLL
          4'b0_010: alu_result = {31'b0, $signed(rs1_val) < $signed(rs2_val)};  // SLT
          4'b0_011: alu_result = {31'b0, rs1_val < rs2_val};                    // SLTU
          4'b0_100: alu_result = rs1_val ^ rs2_val;                             // XOR
          4'b0_101: alu_result = rs1_val >> rs2_val[4:0];                       // SRL
          4'b1_101: alu_result = $signed(rs1_val) >>> rs2_val[4:0];             // SRA
          4'b0_110: alu_result = rs1_val | rs2_val;                             // OR
          4'b0_111: alu_result = rs1_val & rs2_val;                             // AND
          default:  ;
        endcase
      end

      default: ;
    endcase
  end

  // -----------------------------------------------------------------------
  // Byte/Halfword/Word alignment for stores
  // -----------------------------------------------------------------------
  logic [31:0] store_data;
  logic [3:0]  store_be;
  logic [31:0] load_addr;

  always_comb begin
    load_addr  = rs1_val + imm_s;  // also used for loads (imm_i has same bits for [31:20])
    store_data = 32'h0;
    store_be   = 4'b0000;
    case (funct3[1:0])
      2'b00: begin // SB
        store_data = {4{rs2_val[7:0]}};
        store_be   = 4'b0001 << load_addr[1:0];
      end
      2'b01: begin // SH
        store_data = {2{rs2_val[15:0]}};
        store_be   = load_addr[1] ? 4'b1100 : 4'b0011;
      end
      2'b10: begin // SW
        store_data = rs2_val;
        store_be   = 4'b1111;
      end
      default: begin
        store_data = rs2_val;
        store_be   = 4'b1111;
      end
    endcase
  end

  // -----------------------------------------------------------------------
  // Load result extraction
  // -----------------------------------------------------------------------
  logic [31:0] load_result;

  always_comb begin
    load_result = 32'h0;
    case (mem_funct3_r)
      3'b000: begin // LB
        case (mem_addr_r[1:0])
          2'b00: load_result = {{24{rdata[7]}},  rdata[7:0]};
          2'b01: load_result = {{24{rdata[15]}}, rdata[15:8]};
          2'b10: load_result = {{24{rdata[23]}}, rdata[23:16]};
          2'b11: load_result = {{24{rdata[31]}}, rdata[31:24]};
        endcase
      end
      3'b001: begin // LH
        load_result = mem_addr_r[1]
          ? {{16{rdata[31]}}, rdata[31:16]}
          : {{16{rdata[15]}}, rdata[15:0]};
      end
      3'b010: load_result = rdata; // LW
      3'b100: begin // LBU
        case (mem_addr_r[1:0])
          2'b00: load_result = {24'h0, rdata[7:0]};
          2'b01: load_result = {24'h0, rdata[15:8]};
          2'b10: load_result = {24'h0, rdata[23:16]};
          2'b11: load_result = {24'h0, rdata[31:24]};
        endcase
      end
      3'b101: begin // LHU
        load_result = mem_addr_r[1]
          ? {16'h0, rdata[31:16]}
          : {16'h0, rdata[15:0]};
      end
      default: load_result = rdata;
    endcase
  end

  // -----------------------------------------------------------------------
  // CSR read
  // -----------------------------------------------------------------------
  logic [31:0] csr_rval;
  logic [11:0] csr_addr;
  assign csr_addr = instr[31:20];

  always_comb begin
    case (csr_addr)
      12'h300: csr_rval = csr_mstatus;
      12'h304: csr_rval = csr_mie;
      12'h305: csr_rval = csr_mtvec;
      12'h341: csr_rval = csr_mepc;
      12'h342: csr_rval = csr_mcause;
      default: csr_rval = 32'h0;
    endcase
  end

  // -----------------------------------------------------------------------
  // Main FSM
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || cpu_reset) begin
      state        <= S_FETCH;
      pc           <= 32'h0;
      instr        <= 32'h00000013; // NOP
      csr_mtvec    <= 32'h0;
      csr_mcause   <= 32'h0;
      csr_mstatus  <= 32'h0;
      csr_mie      <= 32'h0;
      csr_mepc     <= 32'h0;
      mem_addr_r   <= '0;
      mem_wdata_r  <= '0;
      mem_be_r     <= '0;
      mem_we_r     <= 1'b0;
      mem_rd_r     <= '0;
      mem_funct3_r <= '0;
      rf_we        <= 1'b0;
      rf_wa        <= '0;
      rf_wd        <= '0;
    end else begin
      // Default: no register write
      rf_we <= 1'b0;

      case (state)
        // ---- FETCH: issue read request for pc ----
        S_FETCH: begin
          if (gnt) begin
            state <= S_FETCH_WAIT;
          end
        end

        // ---- FETCH_WAIT: capture instruction ----
        S_FETCH_WAIT: begin
          if (rvalid) begin
            instr <= rdata;
            // synthesis translate_off
            if ($test$plusargs("CPU_TRACE"))
              $display("[CPU] PC=%h instr=%h", pc, rdata);
            // synthesis translate_on
            state <= S_EXEC;
          end
        end

        // ---- EXEC: decode + execute + writeback (or start mem access) ----
        S_EXEC: begin
          // Check for pending interrupt before executing
          if (irq_pending) begin
            csr_mepc   <= pc;
            csr_mcause <= irq_external ? 32'h8000000B : 32'h80000007;
            csr_mstatus[3] <= 1'b0; // clear MIE
            csr_mstatus[7] <= csr_mstatus[3]; // save MIE to MPIE
            pc    <= csr_mtvec;
            state <= S_FETCH;
          end else begin
            case (opcode)
              7'b0110111: begin // LUI
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= pc + 4; state <= S_FETCH;
              end
              7'b0010111: begin // AUIPC
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= pc + 4; state <= S_FETCH;
              end
              7'b1101111: begin // JAL
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= pc + imm_j; state <= S_FETCH;
              end
              7'b1100111: begin // JALR
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= (rs1_val + imm_i) & ~32'h1; state <= S_FETCH;
              end
              7'b1100011: begin // Branches
                pc <= branch_taken ? (pc + imm_b) : (pc + 4);
                state <= S_FETCH;
              end
              7'b0000011: begin // Loads (LB/LH/LW/LBU/LHU)
                mem_addr_r   <= rs1_val + imm_i;
                mem_we_r     <= 1'b0;
                mem_rd_r     <= rd;
                mem_funct3_r <= funct3;
                mem_be_r     <= 4'b1111; // always read full word
                state        <= S_MEM;
              end
              7'b0100011: begin // Stores (SB/SH/SW)
                mem_addr_r   <= load_addr;
                mem_wdata_r  <= store_data;
                mem_be_r     <= store_be;
                mem_we_r     <= 1'b1;
                state        <= S_MEM;
              end
              7'b0010011: begin // OP-IMM
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= pc + 4; state <= S_FETCH;
              end
              7'b0110011: begin // OP
                rf_we <= 1'b1; rf_wa <= rd; rf_wd <= alu_result;
                pc <= pc + 4; state <= S_FETCH;
              end
              7'b1110011: begin // SYSTEM
                case (funct3)
                  3'b000: begin // ECALL/EBREAK/MRET/WFI
                    case (instr[31:20])
                      12'h000: begin // ECALL
                        pc <= pc + 4; state <= S_FETCH;
                      end
                      12'h001: begin // EBREAK
                        // synthesis translate_off
                        $display("[CPU] EBREAK at PC=%h", pc);
                        // synthesis translate_on
                        pc <= pc + 4; state <= S_FETCH;
                      end
                      12'h302: begin // MRET
                        pc <= csr_mepc;
                        csr_mstatus[3] <= csr_mstatus[7]; // restore MIE from MPIE
                        state <= S_FETCH;
                      end
                      12'h105: begin // WFI
                        if (irq_pending) begin
                          pc <= pc + 4;
                          state <= S_FETCH;
                        end else begin
                          state <= S_WFI;
                        end
                      end
                      default: begin
                        pc <= pc + 4; state <= S_FETCH;
                      end
                    endcase
                  end
                  3'b001: begin // CSRRW
                    rf_we <= 1'b1; rf_wa <= rd; rf_wd <= csr_rval;
                    case (csr_addr)
                      12'h300: csr_mstatus <= rs1_val;
                      12'h304: csr_mie     <= rs1_val;
                      12'h305: csr_mtvec   <= rs1_val;
                      12'h341: csr_mepc    <= rs1_val;
                      12'h342: csr_mcause  <= rs1_val;
                      default: ;
                    endcase
                    pc <= pc + 4; state <= S_FETCH;
                  end
                  3'b010: begin // CSRRS
                    rf_we <= 1'b1; rf_wa <= rd; rf_wd <= csr_rval;
                    if (rs1 != 0) begin
                      case (csr_addr)
                        12'h300: csr_mstatus <= csr_mstatus | rs1_val;
                        12'h304: csr_mie     <= csr_mie     | rs1_val;
                        12'h305: csr_mtvec   <= csr_mtvec   | rs1_val;
                        12'h341: csr_mepc    <= csr_mepc    | rs1_val;
                        12'h342: csr_mcause  <= csr_mcause  | rs1_val;
                        default: ;
                      endcase
                    end
                    pc <= pc + 4; state <= S_FETCH;
                  end
                  3'b011: begin // CSRRC
                    rf_we <= 1'b1; rf_wa <= rd; rf_wd <= csr_rval;
                    if (rs1 != 0) begin
                      case (csr_addr)
                        12'h300: csr_mstatus <= csr_mstatus & ~rs1_val;
                        12'h304: csr_mie     <= csr_mie     & ~rs1_val;
                        default: ;
                      endcase
                    end
                    pc <= pc + 4; state <= S_FETCH;
                  end
                  default: begin
                    pc <= pc + 4; state <= S_FETCH;
                  end
                endcase
              end
              default: begin // Unknown opcode — skip
                pc <= pc + 4; state <= S_FETCH;
              end
            endcase
          end
        end

        // ---- MEM: issue load/store request ----
        S_MEM: begin
          if (gnt) begin
            // synthesis translate_off
            if ($test$plusargs("CPU_TRACE"))
              $display("[CPU-MEM] %s addr=%h data=%h be=%b", mem_we_r ? "ST" : "LD", {mem_addr_r[31:2],2'b00}, mem_wdata_r, mem_be_r);
            // synthesis translate_on
            state <= S_MEM_WAIT;
          end
        end

        // ---- MEM_WAIT: capture response ----
        S_MEM_WAIT: begin
          if (rvalid) begin
            if (!mem_we_r) begin
              rf_we <= 1'b1;
              rf_wa <= mem_rd_r;
              rf_wd <= load_result;
            end
            pc    <= pc + 4;
            state <= S_FETCH;
          end
        end

        // ---- WFI: wait for interrupt ----
        S_WFI: begin
          if (irq_pending) begin
            pc    <= pc + 4;  // advance past WFI
            state <= S_FETCH;
          end
        end

        default: state <= S_FETCH;
      endcase
    end
  end

  // -----------------------------------------------------------------------
  // Memory port mux
  // -----------------------------------------------------------------------
  always_comb begin
    req   = 1'b0;
    addr  = pc;
    we    = 1'b0;
    be    = 4'b1111;
    wdata = 32'h0;

    case (state)
      S_FETCH: begin
        req  = 1'b1;
        addr = pc;
        we   = 1'b0;
        be   = 4'b1111;
      end
      S_MEM: begin
        req   = 1'b1;
        addr  = {mem_addr_r[31:2], 2'b00}; // word-aligned
        we    = mem_we_r;
        be    = mem_be_r;
        wdata = mem_wdata_r;
      end
      default: ;
    endcase
  end

  `ifdef SIMULATION
  initial $display("[CPU-INIT] simple_cpu module loaded");
  always @(posedge clk) begin
    if (rst_n && !cpu_reset && $test$plusargs("CPU_TRACE_FULL")) begin
      if (state == S_FETCH && gnt)
        $display("[CPU] FETCH pc=%h t=%0t", pc, $time);
      if (state == S_FETCH_WAIT && rvalid)
        $display("[CPU] INSTR pc=%h data=%h", pc, rdata);
      if (state == S_MEM && gnt)
        $display("[CPU] MEM %s addr=%h data=%h be=%b", mem_we_r?"ST":"LD", {mem_addr_r[31:2],2'b00}, mem_wdata_r, mem_be_r);
    end
  end
  `endif

endmodule : simple_cpu
