// =============================================================================
// page_table_walker.sv — Sv32 Hardware Page Table Walker
// =============================================================================
// Two-level radix page table walker for RISC-V Sv32.
//   Level 1: PTE at satp.ppn × 4096 + VPN[1] × 4
//   Level 0: PTE at PPN_from_L1 × 4096 + VPN[0] × 4
//
// On TLB miss, this FSM reads PTEs from memory via a simple request port
// (intended to go through the AXI crossbar to SRAM/DRAM).
//
// Outputs a fill request to the TLB on successful walk, or a page fault.

/* verilator lint_off UNUSEDSIGNAL */

module page_table_walker #(
  parameter int VA_WIDTH   = 32,
  parameter int PA_WIDTH   = 34,
  parameter int PTE_WIDTH  = 32,
  parameter int ASID_WIDTH = 9
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Walk request (from TLB miss) ---
  input  logic                   walk_req_valid,
  output logic                   walk_req_ready,
  input  logic [VA_WIDTH-1:0]    walk_va,
  input  logic [ASID_WIDTH-1:0]  walk_asid,
  input  logic                   walk_is_store,
  input  logic                   walk_is_exec,

  // --- Walk result (to TLB fill port) ---
  output logic                   walk_done,
  output logic                   walk_fault,
  output logic [1:0]             walk_fault_cause,  // 0=invalid, 1=perm, 2=misaligned
  output logic [VA_WIDTH-1:0]    walk_result_va,
  output logic [ASID_WIDTH-1:0]  walk_result_asid,
  output logic [21:0]            walk_result_ppn,
  output logic                   walk_result_superpage,
  output logic                   walk_result_dirty,
  output logic                   walk_result_accessed,
  output logic                   walk_result_global,
  output logic                   walk_result_user,
  output logic                   walk_result_exec,
  output logic                   walk_result_write,
  output logic                   walk_result_read,

  // --- Memory read port (to crossbar or L1 cache) ---
  output logic                   mem_req_valid,
  input  logic                   mem_req_ready,
  output logic [PA_WIDTH-1:0]    mem_req_addr,

  input  logic                   mem_resp_valid,
  input  logic [PTE_WIDTH-1:0]   mem_resp_data,

  // --- CSR inputs ---
  input  logic [21:0]            satp_ppn,     // Page table root PPN
  input  logic                   satp_mode     // 0=bare (no translation), 1=Sv32
);

  // =========================================================================
  // FSM states
  // =========================================================================
  typedef enum logic [2:0] {
    S_IDLE,
    S_L1_REQ,       // Request level-1 PTE
    S_L1_WAIT,      // Wait for level-1 PTE response
    S_L0_REQ,       // Request level-0 PTE
    S_L0_WAIT,      // Wait for level-0 PTE response
    S_DONE,         // Walk complete (fill TLB or fault)
    S_FAULT
  } ptw_state_e;

  ptw_state_e state, state_next;

  // =========================================================================
  // Registers
  // =========================================================================
  logic [VA_WIDTH-1:0]    va_reg;
  logic [ASID_WIDTH-1:0]  asid_reg;
  logic                   is_store_reg, is_exec_reg;
  logic [9:0]             vpn1, vpn0;
  logic [PTE_WIDTH-1:0]   pte_reg;
  logic [21:0]            l1_ppn;       // PPN from level-1 PTE

  assign vpn1 = va_reg[31:22];
  assign vpn0 = va_reg[21:12];

  // =========================================================================
  // PTE field extraction
  // =========================================================================
  // Sv32 PTE: PPN[1] (12b) | PPN[0] (10b) | RSW (2b) | D|A|G|U|X|W|R|V
  logic        pte_v, pte_r, pte_w, pte_x, pte_u, pte_g, pte_a, pte_d;
  logic [21:0] pte_ppn;
  logic [11:0] pte_ppn1;
  logic [9:0]  pte_ppn0;

  assign pte_v    = pte_reg[0];
  assign pte_r    = pte_reg[1];
  assign pte_w    = pte_reg[2];
  assign pte_x    = pte_reg[3];
  assign pte_u    = pte_reg[4];
  assign pte_g    = pte_reg[5];
  assign pte_a    = pte_reg[6];
  assign pte_d    = pte_reg[7];
  assign pte_ppn0 = pte_reg[19:10];
  assign pte_ppn1 = pte_reg[31:20];
  assign pte_ppn  = {pte_ppn1, pte_ppn0};

  // Is this a leaf PTE? (R or X set means leaf)
  logic pte_is_leaf;
  assign pte_is_leaf = pte_r || pte_x;

  // =========================================================================
  // State machine
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= S_IDLE;
    else
      state <= state_next;
  end

  always_comb begin
    state_next = state;

    case (state)
      S_IDLE: begin
        if (walk_req_valid && satp_mode)
          state_next = S_L1_REQ;
      end

      S_L1_REQ: begin
        if (mem_req_ready)
          state_next = S_L1_WAIT;
      end

      S_L1_WAIT: begin
        if (mem_resp_valid) begin
          // Check PTE validity
          if (!mem_resp_data[0])  // V bit = 0 → invalid
            state_next = S_FAULT;
          else if (mem_resp_data[1] || mem_resp_data[3])
            // Leaf at L1 → superpage
            state_next = S_DONE;
          else
            // Non-leaf → walk to L0
            state_next = S_L0_REQ;
        end
      end

      S_L0_REQ: begin
        if (mem_req_ready)
          state_next = S_L0_WAIT;
      end

      S_L0_WAIT: begin
        if (mem_resp_valid) begin
          if (!mem_resp_data[0])
            state_next = S_FAULT;
          else
            state_next = S_DONE;
        end
      end

      S_DONE:  state_next = S_IDLE;
      S_FAULT: state_next = S_IDLE;
      default: state_next = S_IDLE;
    endcase
  end

  // =========================================================================
  // Datapath
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      va_reg       <= '0;
      asid_reg     <= '0;
      is_store_reg <= 1'b0;
      is_exec_reg  <= 1'b0;
      pte_reg      <= '0;
      l1_ppn       <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (walk_req_valid && satp_mode) begin
            va_reg       <= walk_va;
            asid_reg     <= walk_asid;
            is_store_reg <= walk_is_store;
            is_exec_reg  <= walk_is_exec;
          end
        end

        S_L1_WAIT: begin
          if (mem_resp_valid) begin
            pte_reg <= mem_resp_data;
            l1_ppn  <= {mem_resp_data[31:20], mem_resp_data[19:10]};
          end
        end

        S_L0_WAIT: begin
          if (mem_resp_valid)
            pte_reg <= mem_resp_data;
        end

        default: ;
      endcase
    end
  end

  // =========================================================================
  // Memory request address generation
  // =========================================================================
  // Level 1: satp_ppn × 4096 + VPN[1] × 4
  // Level 0: L1_PPN × 4096 + VPN[0] × 4
  logic [PA_WIDTH-1:0] l1_pte_addr, l0_pte_addr;
  assign l1_pte_addr = {satp_ppn, 12'b0} + {{(PA_WIDTH-12){1'b0}}, vpn1, 2'b0};
  assign l0_pte_addr = {l1_ppn, 12'b0}   + {{(PA_WIDTH-12){1'b0}}, vpn0, 2'b0};

  assign mem_req_valid = (state == S_L1_REQ) || (state == S_L0_REQ);
  assign mem_req_addr  = (state == S_L1_REQ) ? l1_pte_addr : l0_pte_addr;

  // =========================================================================
  // Walk request handshake
  // =========================================================================
  assign walk_req_ready = (state == S_IDLE) && satp_mode;

  // =========================================================================
  // Permission check (at leaf PTE)
  // =========================================================================
  logic perm_fault;
  always_comb begin
    perm_fault = 1'b0;
    if (state == S_DONE) begin
      if (!pte_a)                        perm_fault = 1'b1;  // Accessed bit not set
      if (is_store_reg && !pte_w)        perm_fault = 1'b1;  // Store to read-only
      if (is_store_reg && !pte_d)        perm_fault = 1'b1;  // Dirty bit not set
      if (is_exec_reg && !pte_x)         perm_fault = 1'b1;  // Exec on non-exec page
      if (!is_store_reg && !is_exec_reg && !pte_r) perm_fault = 1'b1;  // Load from non-readable
    end
  end

  // Superpage misalignment check: L1 leaf with PPN[0] != 0
  logic superpage_misaligned;
  assign superpage_misaligned = (state == S_DONE) && pte_is_leaf &&
                                 (l1_ppn == pte_ppn) && (pte_ppn0 != '0);

  // =========================================================================
  // Walk result outputs
  // =========================================================================
  assign walk_done  = (state == S_DONE) || (state == S_FAULT);
  assign walk_fault = (state == S_FAULT) || perm_fault || superpage_misaligned;

  always_comb begin
    walk_fault_cause = 2'b00;
    if (state == S_FAULT)          walk_fault_cause = 2'b00;  // Invalid PTE
    if (perm_fault)                walk_fault_cause = 2'b01;  // Permission
    if (superpage_misaligned)      walk_fault_cause = 2'b10;  // Misaligned superpage
  end

  // Determine if this is a superpage (leaf found at level 1)
  // We know it's a superpage if we went IDLE→L1→DONE without visiting L0.
  logic is_superpage;
  assign is_superpage = (state == S_DONE) && (l1_ppn == pte_ppn);

  assign walk_result_va        = va_reg;
  assign walk_result_asid      = asid_reg;
  assign walk_result_ppn       = pte_ppn;
  assign walk_result_superpage = is_superpage;
  assign walk_result_dirty     = pte_d;
  assign walk_result_accessed  = pte_a;
  assign walk_result_global    = pte_g;
  assign walk_result_user      = pte_u;
  assign walk_result_exec      = pte_x;
  assign walk_result_write     = pte_w;
  assign walk_result_read      = pte_r;

endmodule
