`timescale 1ns / 1ps

//=============================================================================
// Pulse Synchronizer for Clock Domain Crossing
//=============================================================================
// 
// PURPOSE:
// --------
// Safely transfer single-cycle pulses from one clock domain to another.
// Common use case: Control signals (start, abort) crossing from 50 MHz to 200 MHz.
//
// OPERATION:
// ----------
// 1. Input pulse (src_clk domain) sets toggle flip-flop
// 2. Toggle crosses via 2-FF synchronizer (metastability protection)
// 3. Edge detector in dst_clk domain generates single-cycle pulse
//
// GUARANTEES:
// -----------
// - No pulse loss (as long as input pulses separated by ≥ 3 dst_clk cycles)
// - Metastability-safe (uses 2-FF synchronizer with ASYNC_REG attribute)
// - Single-cycle output pulse in destination domain
//
// LIMITATIONS:
// ------------
// - Input pulses must be separated by ≥ (2 × dst_period + setup_time)
// - For 50 MHz → 200 MHz: min separation = 2 × 5ns + 1ns = 11ns (220 MHz OK)
//
//=============================================================================

module pulse_sync (
    input  wire src_clk,      // Source clock domain
    input  wire src_rst_n,    // Source reset (async, active-low)
    input  wire src_pulse,    // Input pulse (single cycle in src_clk)
    
    input  wire dst_clk,      // Destination clock domain
    input  wire dst_rst_n,    // Destination reset (async, active-low)
    output reg  dst_pulse     // Output pulse (single cycle in dst_clk)
);

    // ========================================================================
    // Stage 1: Toggle flip-flop in source domain
    // ========================================================================
    reg src_toggle;
    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n)
            src_toggle <= 1'b0;
        else if (src_pulse)
            src_toggle <= ~src_toggle;
    end

    // ========================================================================
    // Stage 2: 2-FF synchronizer in destination domain (metastability protection)
    // ========================================================================
    (* ASYNC_REG = "TRUE" *) reg dst_toggle_meta;
    (* ASYNC_REG = "TRUE" *) reg dst_toggle_sync;
    
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            dst_toggle_meta <= 1'b0;
            dst_toggle_sync <= 1'b0;
        end else begin
            dst_toggle_meta <= src_toggle;      // Metastability stage
            dst_toggle_sync <= dst_toggle_meta; // Stable output
        end
    end

    // ========================================================================
    // Stage 3: Edge detector in destination domain
    // ========================================================================
    reg dst_toggle_d1;
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            dst_toggle_d1 <= 1'b0;
            dst_pulse     <= 1'b0;
        end else begin
            dst_toggle_d1 <= dst_toggle_sync;
            dst_pulse     <= dst_toggle_sync ^ dst_toggle_d1; // XOR = edge detect
        end
    end

endmodule
