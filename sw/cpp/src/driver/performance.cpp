// performance.cpp — Performance counter analysis implementation
// =============================================================================
#include "driver/performance.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace accel {
namespace driver {

// =============================================================================
// PerfMetrics::print
// =============================================================================

void PerfMetrics::print(std::ostream& os) const {
    os << "\n┌──────────────────────────────────────────────┐\n";
    os <<   "│         PERFORMANCE METRICS                   │\n";
    os <<   "├──────────────────────────────────────────────┤\n";

    os << std::fixed << std::setprecision(2);
    os << "│  Compute utilisation:  " << (compute_utilisation * 100.0) << "%\n";
    os << "│  Stall ratio:          " << (stall_ratio * 100.0) << "%\n";
    os << "│  Idle ratio:           " << (idle_ratio * 100.0) << "%\n";
    os << "│\n";
    os << "│  Total ops:            " << total_ops << "\n";
    os << "│  Peak GOPS:            " << peak_gops << "\n";
    os << "│  Effective GOPS:       " << effective_gops << "\n";
    os << "│\n";
    os << "│  DMA bandwidth:        " << dma_bandwidth_mbps << " MB/s\n";
    os << "│  BW utilisation:       " << (memory_bandwidth_utilisation * 100.0) << "%\n";
    os << "│  Arithmetic intensity: " << arithmetic_intensity << " ops/byte\n";
    os << "│\n";
    os << "│  Wall time:            " << wall_time_ms << " ms\n";
    os << "│  HW time:              " << hw_time_ms << " ms\n";
    os << "│  Tiles processed:      " << tiles_processed << "\n";
    os << "│  Cycles/tile:          " << cycles_per_tile << "\n";
    os << "└──────────────────────────────────────────────┘\n";
}

// =============================================================================
// PerfAnalyser
// =============================================================================

PerfAnalyser::PerfAnalyser(double clock_freq_hz)
    : clock_freq_hz_(clock_freq_hz)
    , peak_bw_mbps_(1200.0)  // 32-bit AXI HP @ 150 MHz theoretical
{
}

PerfMetrics PerfAnalyser::analyse(const PerfSnapshot& snap,
                                   uint64_t total_ops,
                                   double wall_time_ms) const {
    PerfMetrics m{};

    // Utilisation ratios
    if (snap.total_cycles > 0) {
        m.compute_utilisation = static_cast<double>(snap.active_cycles) /
                                snap.total_cycles;
        m.stall_ratio         = static_cast<double>(snap.stall_cycles) /
                                snap.total_cycles;
        m.idle_ratio          = static_cast<double>(snap.idle_cycles) /
                                snap.total_cycles;
    }

    // Timing
    m.hw_time_ms    = (snap.total_cycles / clock_freq_hz_) * 1000.0;
    m.wall_time_ms  = (wall_time_ms > 0) ? wall_time_ms : m.hw_time_ms;

    // Throughput
    // 196 PEs (14×14), each does 1 MAC/cycle = 196 ops/cycle
    constexpr double PE_COUNT = 196.0;
    m.peak_gops = (PE_COUNT * clock_freq_hz_) / 1e9;

    m.total_ops = total_ops;
    if (m.hw_time_ms > 0 && total_ops > 0) {
        m.gops = (static_cast<double>(total_ops) / (m.hw_time_ms / 1000.0)) / 1e9;
        m.effective_gops = m.gops * m.compute_utilisation;
    }

    // Bandwidth
    if (m.hw_time_ms > 0) {
        m.dma_bandwidth_mbps = static_cast<double>(snap.dma_bytes) /
                               (m.hw_time_ms / 1000.0) / 1e6;
    }
    m.peak_bandwidth_mbps = peak_bw_mbps_;
    if (peak_bw_mbps_ > 0) {
        m.memory_bandwidth_utilisation = m.dma_bandwidth_mbps / peak_bw_mbps_;
    }

    // Arithmetic intensity
    if (snap.dma_bytes > 0 && total_ops > 0) {
        m.arithmetic_intensity = static_cast<double>(total_ops) / snap.dma_bytes;
    }

    // Tile-level
    m.tiles_processed = snap.blocks_computed;
    if (snap.blocks_computed > 0) {
        m.cycles_per_tile = static_cast<double>(snap.total_cycles) /
                            snap.blocks_computed;
    }

    return m;
}

PerfMetrics PerfAnalyser::analyseLayer(const PerfSnapshot& snap,
                                        uint32_t M, uint32_t N, uint32_t K,
                                        double wall_time_ms) const {
    // Total ops = 2 * M * N * K (multiply + accumulate)
    uint64_t total_ops = 2ULL * M * N * K;
    return analyse(snap, total_ops, wall_time_ms);
}

PerfSnapshot PerfAnalyser::delta(const PerfSnapshot& before,
                                  const PerfSnapshot& after) const {
    PerfSnapshot d;
    d.total_cycles    = after.total_cycles    - before.total_cycles;
    d.active_cycles   = after.active_cycles   - before.active_cycles;
    d.idle_cycles     = after.idle_cycles     - before.idle_cycles;
    d.dma_bytes       = after.dma_bytes       - before.dma_bytes;
    d.blocks_computed = after.blocks_computed - before.blocks_computed;
    d.stall_cycles    = after.stall_cycles    - before.stall_cycles;
    return d;
}

std::string PerfAnalyser::rooflinePlot(const PerfMetrics& metrics) const {
    std::ostringstream os;
    os << "\n  ROOFLINE MODEL (ASCII)\n";
    os << "  ───────────────────────────────────────\n";

    // Simple textual roofline representation
    double ridge_point = metrics.peak_gops / (metrics.peak_bandwidth_mbps / 1000.0);
    os << std::fixed << std::setprecision(2);
    os << "  Peak compute:  " << metrics.peak_gops << " GOPS\n";
    os << "  Peak bandwidth: " << metrics.peak_bandwidth_mbps << " MB/s\n";
    os << "  Ridge point:   " << ridge_point << " ops/byte\n";
    os << "  Your AI:       " << metrics.arithmetic_intensity << " ops/byte\n";
    os << "  Your perf:     " << metrics.effective_gops << " GOPS\n";

    if (metrics.arithmetic_intensity < ridge_point) {
        os << "  → MEMORY BOUND (left of ridge)\n";
    } else {
        os << "  → COMPUTE BOUND (right of ridge)\n";
    }

    // ASCII bar chart for utilisation
    os << "\n  Utilisation:\n";
    int compute_bar = static_cast<int>(metrics.compute_utilisation * 40);
    int stall_bar   = static_cast<int>(metrics.stall_ratio * 40);
    int idle_bar    = static_cast<int>(metrics.idle_ratio * 40);

    os << "  Compute: [";
    for (int i = 0; i < 40; ++i) os << (i < compute_bar ? '#' : '.');
    os << "] " << std::setprecision(1) << (metrics.compute_utilisation * 100) << "%\n";

    os << "  Stall:   [";
    for (int i = 0; i < 40; ++i) os << (i < stall_bar ? '=' : '.');
    os << "] " << (metrics.stall_ratio * 100) << "%\n";

    os << "  Idle:    [";
    for (int i = 0; i < 40; ++i) os << (i < idle_bar ? '-' : '.');
    os << "] " << (metrics.idle_ratio * 100) << "%\n";

    return os.str();
}

void PerfAnalyser::printComparison(
    const std::vector<std::pair<std::string, PerfMetrics>>& layers,
    std::ostream& os)
{
    os << "\n┌────────────┬────────┬─────────┬───────────┬──────────┐\n";
    os <<   "│ Layer      │ Util % │ GOPS    │ BW MB/s   │ cy/tile  │\n";
    os <<   "├────────────┼────────┼─────────┼───────────┼──────────┤\n";

    for (const auto& [name, m] : layers) {
        os << "│ " << std::setw(10) << std::left << name
           << " │ " << std::setw(5) << std::right << std::fixed
           << std::setprecision(1) << (m.compute_utilisation * 100.0)
           << "% │ " << std::setw(7) << std::setprecision(2)
           << m.effective_gops
           << " │ " << std::setw(9) << std::setprecision(1)
           << m.dma_bandwidth_mbps
           << " │ " << std::setw(8) << std::setprecision(0)
           << m.cycles_per_tile << " │\n";
    }

    os << "└────────────┴────────┴─────────┴───────────┴──────────┘\n";
}

// =============================================================================
// PerfTimer
// =============================================================================

uint64_t PerfTimer::nowUs() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

PerfTimer::PerfTimer() : start_us_(nowUs()) {}

void PerfTimer::reset() {
    start_us_ = nowUs();
}

double PerfTimer::elapsedMs() const {
    return static_cast<double>(nowUs() - start_us_) / 1000.0;
}

uint64_t PerfTimer::elapsedUs() const {
    return nowUs() - start_us_;
}

} // namespace driver
} // namespace accel
