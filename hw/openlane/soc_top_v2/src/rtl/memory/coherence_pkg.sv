`timescale 1ns/1ps

// =============================================================================
// coherence_pkg.sv — Cache Coherence Message Types and Structures
// =============================================================================
// Standalone coherence protocol definitions for the MESI/MOESI demo.
// NOT integrated into the main SoC (per architecture decision).

package coherence_pkg;

  // =========================================================================
  // MESI states (4 classic states)
  // =========================================================================
  typedef enum logic [1:0] {
    MESI_I = 2'b00,  // Invalid
    MESI_S = 2'b01,  // Shared (clean, may be in other caches)
    MESI_E = 2'b10,  // Exclusive (clean, only copy)
    MESI_M = 2'b11   // Modified (dirty, only copy)
  } mesi_state_e;

  // =========================================================================
  // Coherence message types
  // =========================================================================
  typedef enum logic [3:0] {
    COH_GET_S      = 4'h0,  // Read request (want Shared)
    COH_GET_M      = 4'h1,  // Write request (want Modified)
    COH_PUT_M      = 4'h2,  // Writeback of Modified line
    COH_PUT_E      = 4'h3,  // Writeback of Exclusive line (silent drop OK)
    COH_PUT_S      = 4'h4,  // Eviction of Shared line
    COH_INV        = 4'h5,  // Invalidate (directory → cache)
    COH_INV_ACK    = 4'h6,  // Invalidate acknowledge
    COH_DATA       = 4'h7,  // Data response
    COH_DATA_E     = 4'h8,  // Data response (grant Exclusive)
    COH_FWD_GET_S  = 4'h9,  // Forward: owner send data to requester (Shared)
    COH_FWD_GET_M  = 4'hA,  // Forward: owner send data to requester (Modified)
    COH_WB_ACK     = 4'hB   // Writeback acknowledge from directory
  } coh_msg_e;

  // =========================================================================
  // Coherence request structure (on snoopy bus or directory message)
  // =========================================================================
  parameter int COH_ADDR_W = 32;
  parameter int COH_DATA_W = 256;  // Cache line width (32 bytes)
  parameter int COH_NODE_W = 4;    // Supports up to 16 nodes

  typedef struct packed {
    coh_msg_e                    msg_type;
    logic [COH_NODE_W-1:0]      src;
    logic [COH_NODE_W-1:0]      dst;
    logic [COH_ADDR_W-1:0]      addr;
    logic [COH_DATA_W-1:0]      data;       // Data payload (for DATA messages)
    logic                        has_data;   // Message carries data
  } coh_req_t;

  // =========================================================================
  // Directory entry (for directory-based protocol)
  // =========================================================================
  parameter int MAX_SHARERS = 16;

  typedef struct packed {
    mesi_state_e                 state;
    logic [COH_NODE_W-1:0]      owner;      // Owner node (for M/E states)
    logic [MAX_SHARERS-1:0]     sharer_vec; // Bit vector of sharers (for S state)
  } dir_entry_t;

endpackage
