// =============================================================================
// tlb.sv — Translation Lookaside Buffer (Sv32 for Ibex)
// =============================================================================
// Fully-associative TLB for Sv32 (RISC-V 32-bit virtual memory).
// Caches page table translations to avoid page table walks on every access.
//
// Sv32 specifics:
//   - 32-bit virtual address: VPN[1] (10b) | VPN[0] (10b) | Offset (12b)
//   - 34-bit physical address: PPN[1] (12b) | PPN[0] (10b) | Offset (12b)
//   - 4KB base page, 4MB superpage
//   - PTE: PPN[1:0] (22b) | RSW (2b) | D | A | G | U | X | W | R | V

/* verilator lint_off UNUSEDSIGNAL */
module tlb #(
  parameter int NUM_ENTRIES = 16,     // TLB entries (fully associative)
  parameter int VA_WIDTH    = 32,
  parameter int PA_WIDTH    = 34,
  parameter int ASID_WIDTH  = 9       // Sv32 ASID
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // --- Lookup interface ---
  input  logic                   lookup_valid,
  input  logic [VA_WIDTH-1:0]    lookup_va,
  input  logic [ASID_WIDTH-1:0]  lookup_asid,
  input  logic                   lookup_is_store,  // For dirty bit check
  input  logic                   lookup_is_exec,   // For execute permission

  output logic                   lookup_hit,
  output logic [PA_WIDTH-1:0]    lookup_pa,
  output logic                   lookup_fault,     // Permission violation

  // --- Fill interface (from PTW) ---
  input  logic                   fill_valid,
  input  logic [VA_WIDTH-1:0]    fill_va,
  input  logic [ASID_WIDTH-1:0]  fill_asid,
  input  logic [21:0]            fill_ppn,         // PPN[1:0]
  input  logic                   fill_superpage,   // 4MB superpage
  input  logic                   fill_dirty,
  input  logic                   fill_accessed,
  input  logic                   fill_global,
  input  logic                   fill_user,
  input  logic                   fill_exec,
  input  logic                   fill_write,
  input  logic                   fill_read,

  // --- Invalidation ---
  input  logic                   sfence_valid,     // SFENCE.VMA
  input  logic                   sfence_all,       // Invalidate all
  input  logic [VA_WIDTH-1:0]    sfence_va,        // Specific VA (if !sfence_all)
  input  logic [ASID_WIDTH-1:0]  sfence_asid       // Specific ASID (0=all ASIDs)
);

  // =========================================================================
  // TLB entry storage
  // =========================================================================
  localparam int ENTRY_IDX_BITS = $clog2(NUM_ENTRIES);

  logic                       entries_valid [NUM_ENTRIES];
  logic [ASID_WIDTH-1:0]      entries_asid [NUM_ENTRIES];
  logic [9:0]                 entries_vpn1 [NUM_ENTRIES];
  logic [9:0]                 entries_vpn0 [NUM_ENTRIES];
  logic [21:0]                entries_ppn [NUM_ENTRIES];
  logic                       entries_superpage [NUM_ENTRIES];
  logic                       entries_dirty [NUM_ENTRIES];
  logic                       entries_accessed [NUM_ENTRIES];
  logic                       entries_global [NUM_ENTRIES];
  logic                       entries_user [NUM_ENTRIES];
  logic                       entries_exec [NUM_ENTRIES];
  logic                       entries_write [NUM_ENTRIES];
  logic                       entries_read [NUM_ENTRIES];

  // =========================================================================
  // PLRU replacement
  // =========================================================================
  logic [$clog2(NUM_ENTRIES)-1:0] plru_ptr;

  // Simple counter-based replacement (simplified from tree-PLRU)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      plru_ptr <= '0;
    else if (fill_valid)
      plru_ptr <= plru_ptr + 1;
  end

  // =========================================================================
  // Lookup (combinational)
  // =========================================================================
  logic [VA_WIDTH-1:0] va;
  logic [9:0] vpn1, vpn0;
  logic [11:0] page_offset;

  assign va          = lookup_va;
  assign vpn1        = va[31:22];
  assign vpn0        = va[21:12];
  assign page_offset = va[11:0];

  logic [NUM_ENTRIES-1:0] match_vec;
  logic [ENTRY_IDX_BITS-1:0] match_idx;
  logic [21:0]               matched_ppn;
  logic                      matched_superpage;
  logic                      matched_dirty;
  logic                      matched_exec;
  logic                      matched_write;
  logic                      matched_read;
  logic [ENTRY_IDX_BITS-1:0] fill_victim_c;
  logic                      fill_found_invalid_c;

  always_comb begin
    match_vec    = '0;
    match_idx    = '0;
    matched_ppn = '0;
    matched_superpage = 1'b0;
    matched_dirty = 1'b0;
    matched_exec = 1'b0;
    matched_write = 1'b0;
    matched_read = 1'b0;
    lookup_hit   = 1'b0;
    lookup_pa    = '0;
    lookup_fault = 1'b0;

    for (int i = 0; i < NUM_ENTRIES; i++) begin
      match_vec[i] = entries_valid[i] &&
                     (entries_vpn1[i] == vpn1) &&
                     (entries_superpage[i] || (entries_vpn0[i] == vpn0)) &&
                     (entries_global[i] || (entries_asid[i] == lookup_asid));
    end

    // Priority encoder: pick first match
    for (int i = 0; i < NUM_ENTRIES; i++) begin
      if (match_vec[i] && !lookup_hit) begin
        lookup_hit    = 1'b1;
        match_idx     = ENTRY_IDX_BITS'(i);
        matched_ppn = entries_ppn[i];
        matched_superpage = entries_superpage[i];
        matched_dirty = entries_dirty[i];
        matched_exec = entries_exec[i];
        matched_write = entries_write[i];
        matched_read = entries_read[i];
      end
    end

    if (lookup_hit) begin
      // Construct physical address
      if (matched_superpage) begin
        // 4MB superpage: PA = PPN[1] | VPN[0] | offset
        lookup_pa = {matched_ppn[21:10], vpn0, page_offset};
      end else begin
        // 4KB page: PA = PPN[1:0] | offset
        lookup_pa = {matched_ppn, page_offset};
      end

      // Permission check
      if (lookup_is_store && !matched_write)
        lookup_fault = 1'b1;
      else if (lookup_is_exec && !matched_exec)
        lookup_fault = 1'b1;
      else if (!lookup_is_store && !lookup_is_exec && !matched_read)
        lookup_fault = 1'b1;
      else if (lookup_is_store && !matched_dirty)
        lookup_fault = 1'b1;  // Dirty bit not set, need PTW to set it
    end
  end

  always_comb begin
    fill_found_invalid_c = 1'b0;
    fill_victim_c = plru_ptr;
    for (int i = 0; i < NUM_ENTRIES; i++) begin
      if (!entries_valid[i] && !fill_found_invalid_c) begin
        fill_victim_c = ENTRY_IDX_BITS'(i);
        fill_found_invalid_c = 1'b1;
      end
    end
  end

  // =========================================================================
  // Fill (on PTW completion)
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_ENTRIES; i++) begin
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
    end else begin
      // SFENCE.VMA invalidation
      if (sfence_valid) begin
        for (int i = 0; i < NUM_ENTRIES; i++) begin
          if (sfence_all) begin
            entries_valid[i] <= 1'b0;
          end else begin
            if ((entries_vpn1[i] == sfence_va[31:22]) &&
                ((sfence_asid == '0) || (entries_asid[i] == sfence_asid)) &&
                !entries_global[i])
              entries_valid[i] <= 1'b0;
          end
        end
      end

      // Fill new entry
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
  end

endmodule
