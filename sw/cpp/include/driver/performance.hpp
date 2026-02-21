// performance.hpp — Performance counter readout and roofline analysis
// =============================================================================
//
// Reads hardware performance counters from CSR registers and computes
// derived metrics: compute utilisation, bandwidth, arithmetic intensity,
// and roofline position.
//
// Hardware counters (from csr.sv / address_map.hpp):
//   REG_PERF_TOTAL   — total cycles from start to done
//   REG_PERF_ACTIVE  — cycles where systolic PEs were computing
//   REG_PERF_IDLE    — cycles where pipeline was idle/stalled
//   REG_PERF_DMA_BYTES — total DMA bytes transferred
//   REG_PERF_BLOCKS  — number of BSR blocks processed
//   REG_PERF_STALL   — stall cycles (scheduler busy, PEs idle)
//
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <ostream>
#include <vector>

namespace accel {
namespace driver {

// =============================================================================
// PerfSnapshot — Raw hardware counter snapshot
// =============================================================================
struct PerfSnapshot {
    uint32_t total_cycles    = 0;   // REG_PERF_TOTAL
    uint32_t active_cycles   = 0;   // REG_PERF_ACTIVE
    uint32_t idle_cycles     = 0;   // REG_PERF_IDLE
    uint32_t dma_bytes       = 0;   // REG_PERF_DMA_BYTES
    uint32_t blocks_computed = 0;   // REG_PERF_BLOCKS
    uint32_t stall_cycles    = 0;   // REG_PERF_STALL
};

// =============================================================================
// PerfMetrics — Derived performance metrics
// =============================================================================
struct PerfMetrics {
    // Basic utilisation
    double compute_utilisation;   // active_cycles / total_cycles (0.0–1.0)
    double stall_ratio;           // stall_cycles / total_cycles
    double idle_ratio;            // idle_cycles / total_cycles

    // Throughput
    double gops;                  // INT8 operations per second (×10^9)
    double effective_gops;        // Adjusted for utilisation
    uint64_t total_ops;           // Total INT8 MACs

    // Bandwidth
    double dma_bandwidth_mbps;    // DMA bytes / time (MB/s)
    double memory_bandwidth_utilisation; // vs theoretical max

    // Roofline model
    double arithmetic_intensity;  // ops / byte (ops:byte ratio)
    double peak_gops;             // Theoretical max (196 PEs × freq)
    double peak_bandwidth_mbps;   // AXI HP port theoretical max

    // Timing
    double wall_time_ms;          // Total wall-clock time
    double hw_time_ms;            // total_cycles / freq

    // Layer-level
    uint32_t tiles_processed;     // Number of tiles computed
    double   cycles_per_tile;     // Average cycles per tile

    // Print formatted report
    void print(std::ostream& os) const;
};

// =============================================================================
// PerfAnalyser — Computes derived metrics from raw counters
// =============================================================================
class PerfAnalyser {
public:
    /// @param clock_freq_hz  Accelerator clock frequency (default: 100 MHz for PYNQ-Z2)
    explicit PerfAnalyser(double clock_freq_hz = 100e6);

    /// Set the theoretical peak bandwidth (AXI HP port)
    /// Default: 1200 MB/s (32-bit @ 150 MHz AXI)
    void setPeakBandwidth(double mbps) { peak_bw_mbps_ = mbps; }

    /// Compute derived metrics from a raw counter snapshot.
    /// @param snap         Raw counter values
    /// @param total_ops    Total INT8 MACs for the workload
    /// @param wall_time_ms Wall-clock time in milliseconds
    PerfMetrics analyse(const PerfSnapshot& snap,
                        uint64_t total_ops = 0,
                        double wall_time_ms = 0.0) const;

    /// Compute metrics for a tiling plan
    PerfMetrics analyseLayer(const PerfSnapshot& snap,
                             uint32_t M, uint32_t N, uint32_t K,
                             double wall_time_ms = 0.0) const;

    /// Generate a text-based roofline chart (ASCII art)
    std::string rooflinePlot(const PerfMetrics& metrics) const;

    /// Compare two snapshots (before/after) and compute delta
    PerfSnapshot delta(const PerfSnapshot& before,
                       const PerfSnapshot& after) const;

    /// Print a comparison table of multiple layer metrics
    static void printComparison(const std::vector<std::pair<std::string, PerfMetrics>>& layers,
                                std::ostream& os);

private:
    double clock_freq_hz_;
    double peak_bw_mbps_;
};

// =============================================================================
// PerfTimer — RAII wall-clock timer
// =============================================================================
class PerfTimer {
public:
    PerfTimer();

    /// Reset the timer
    void reset();

    /// Elapsed time in milliseconds since construction or last reset
    double elapsedMs() const;

    /// Elapsed time in microseconds
    uint64_t elapsedUs() const;

private:
    uint64_t start_us_;

    static uint64_t nowUs();
};

} // namespace driver
} // namespace accel
