// Simple CPU Wrapper - Mimics RISC-V CPU behavior
// In real implementation, replace with lowRISC Ibex
// This is a test stub that fetches from boot ROM and runs basic code

module simple_cpu #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned ID_WIDTH = 4
) (
  input  logic              clk,
  input  logic              rst_n,
  
  // CPU Reset & Interrupt
  input  logic              cpu_reset,
  input  logic              irq_external,
  input  logic              irq_timer,
  
  // OBI-like Master interface (instruction and data) 
  // For simplicity, we'll use a single-port interface
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

  // Simple program counter and instruction register
  logic [ADDR_WIDTH-1:0] pc;
  logic [31:0] instr;
  logic [31:0] registers [0:31];
  logic halted;

  // Fetch phase: request instruction from address pointed by PC
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc <= 32'h0000_0000;  // Boot from address 0
      instr <= 32'h0000_0013;  // NOP (addi x0, x0, 0)
      halted <= 1'b0;
    end else if (gnt && req) begin
      // Instruction returned after grant
      if (rvalid) begin
        instr <= rdata;
        pc <= pc + 4;
      end
    end
  end

  // Memory request: fetch next instruction
  assign req = !halted;
  assign addr = pc;
  assign we = 1'b0;  // CPU is always reading in this simple stub
  assign be = 4'b1111;
  assign wdata = '0;

  // Simple decoder - catch li (load immediate) and ebreak (halt)
  always_comb begin
    // Detect ebreak (0x00100073) or an infinite loop to stop simulation
    if (instr == 32'h0010_0073) begin
      halted = 1'b1;
    end else begin
      halted = 1'b0;
    end
  end

endmodule : simple_cpu
