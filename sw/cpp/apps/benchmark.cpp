// benchmark.cpp — Performance analysis tool for memory architecture metrics
// =============================================================================
//
// Runs configurable GEMM workloads through the software pipeline and measures:
//   - Bandwidth utilisation (DMA throughput)
//   - Compute efficiency (PE utilisation, MAC/cycle)
//   - Stall analysis (idle cycles, pipeline bubbles)
//   - Roofline model positioning
//
// Modes:
//   1. Golden sweep:  software-only, sweeps M×K×N dimensions, measures tiling
//   2. Layer profile:  per-MNIST-layer analysis (tiling, BSR density, MACs)
//   3. Hardware bench: actual accelerator measurements (requires PYNQ-Z2)
//
// Usage:
//   ./benchmark                     # Default: golden sweep + layer profile
//   ./benchmark --sweep             # Dimension sweep only
//   ./benchmark --layers            # MNIST layer profiling only
//   ./benchmark --hardware          # Hardware benchmarking (on PYNQ-Z2)
//   ./benchmark --weights data/int8 # Weight dir for BSR density analysis
//
// =============================================================================
#include "driver/accelerator.hpp"
#include "driver/performance.hpp"
#include "compute/golden_model.hpp"
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"
#include "memory/dma_controller.hpp"
#include "memory/address_map.hpp"
#include "utils/npy_loader.hpp"
#include "utils/logging.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>

using namespace accel;

// =============================================================================
// Options
// =============================================================================
struct BenchOptions {
    bool sweep     = true;
    bool layers    = true;
    bool hardware  = false;
    bool batch     = false;        // Batched FC roofline comparison
    bool verbose   = false;
    std::string weights_dir = "data/int8";
    std::string bsr_dir     = "data/bsr_export_14x14";
    uint32_t num_iterations = 5;   // For averaging
    uint32_t batch_size     = 28;  // Default: 2 N-tiles for weight-stationary reuse
};

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --sweep       Run dimension sweep benchmark\n"
        "  --layers      Run per-layer MNIST profiling\n"
        "  --batch [N]   Batched FC roofline comparison (default N=14)\n"
        "  --hardware    Run hardware benchmarks (requires PYNQ-Z2)\n"
        "  --all         Run all benchmarks (default)\n"
        "  --weights DIR INT8 weight directory\n"
        "  --bsr DIR     BSR export directory\n"
        "  --iters N     Number of iterations for averaging (default: 5)\n"
        "  --verbose     Enable debug logging\n"
        "  --help        Show this help\n"
        "\n", prog);
}

static BenchOptions parseArgs(int argc, char* argv[]) {
    BenchOptions opts;
    bool explicit_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sweep")    { opts.sweep = true;  opts.layers = false; explicit_mode = true; }
        else if (arg == "--layers")   { opts.layers = true; opts.sweep = false; explicit_mode = true; }
        else if (arg == "--batch") {
            opts.batch = true; explicit_mode = true;
            // Optional numeric argument
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.batch_size = std::atoi(argv[++i]);
            }
        }
        else if (arg == "--hardware") { opts.hardware = true; explicit_mode = true; }
        else if (arg == "--all")      { opts.sweep = true; opts.layers = true; opts.batch = true; }
        else if (arg == "--weights" && i + 1 < argc) opts.weights_dir = argv[++i];
        else if (arg == "--bsr" && i + 1 < argc)     opts.bsr_dir = argv[++i];
        else if (arg == "--iters" && i + 1 < argc)   opts.num_iterations = std::atoi(argv[++i]);
        else if (arg == "--verbose") opts.verbose = true;
        else if (arg == "--help") { printUsage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown option: %s\n", arg.c_str()); printUsage(argv[0]); std::exit(1); }
    }
    if (!explicit_mode) { opts.sweep = true; opts.layers = true; }
    return opts;
}

// =============================================================================
// Dimension Sweep Benchmark
// =============================================================================
struct SweepResult {
    uint32_t M, N, K;
    uint32_t tiles;
    uint64_t macs;
    double   golden_ms;      // Golden model GEMM time
    double   tiled_ms;       // Tiled GEMM time
    double   tiling_overhead;// Percentage overhead from tiling
    double   bsr_density;    // BSR block density (1.0 = fully dense)
    double   effective_gops; // GOPs for tiled path
};

static void runDimensionSweep(const BenchOptions& opts) {
    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "==========================================================\n");
    std::fprintf(stdout, "  DIMENSION SWEEP BENCHMARK\n");
    std::fprintf(stdout, "==========================================================\n\n");

    // Test configurations: (M, N, K) — representative workloads
    struct Config { uint32_t M, N, K; const char* desc; };
    std::vector<Config> configs = {
        {14,  1,  14,   "1 tile (minimal)"},
        {14,  1,  28,   "1×1×2 tiles (K reduction)"},
        {28,  1,  14,   "2×1×1 tiles (M spatial)"},
        {28,  1,  28,   "2×1×2 tiles"},
        {32,  1,  9,    "conv1 weight shape"},
        {64,  1,  288,  "conv2 weight shape"},
        {128, 1,  9216, "fc1 weight shape"},
        {10,  1,  128,  "fc2 weight shape"},
        {56,  1,  56,   "4×1×4 tiles (medium)"},
        {140, 1,  140,  "10×1×10 tiles (large)"},
    };

    std::vector<SweepResult> results;

    std::fprintf(stdout, "%-25s %6s %6s %6s %6s %12s %12s %12s %8s %10s\n",
                 "Config", "M", "N", "K", "Tiles", "MACs",
                 "Golden(ms)", "Tiled(ms)", "Ohead%", "GOPS");
    std::fprintf(stdout, "%s\n", std::string(110, '-').c_str());

    for (auto& cfg : configs) {
        SweepResult res;
        res.M = cfg.M; res.N = cfg.N; res.K = cfg.K;

        // Generate random data
        std::vector<int8_t> A(cfg.M * cfg.K), B(cfg.K * cfg.N);
        for (auto& v : A) v = static_cast<int8_t>((std::rand() % 256) - 128);
        for (auto& v : B) v = static_cast<int8_t>((std::rand() % 256) - 128);

        // Tiling plan
        auto plan = compute::planDenseGEMM(compute::GEMMShape{cfg.M, cfg.N, cfg.K}, cfg.desc);
        res.tiles = plan.totalTiles();
        res.macs  = plan.totalMACs();

        // BSR density
        compute::BSREncoder encoder;
        auto bsr = encoder.encode(A.data(), cfg.M, cfg.K);
        res.bsr_density = bsr.density();

        // Benchmark golden GEMM
        double golden_total = 0;
        for (uint32_t iter = 0; iter < opts.num_iterations; ++iter) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto C = compute::GoldenModel::gemmINT8(A.data(), B.data(), cfg.M, cfg.N, cfg.K);
            auto t1 = std::chrono::high_resolution_clock::now();
            golden_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        res.golden_ms = golden_total / opts.num_iterations;

        // Benchmark tiled GEMM
        double tiled_total = 0;
        for (uint32_t iter = 0; iter < opts.num_iterations; ++iter) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto C = compute::GoldenModel::tiledGemmINT8(A.data(), B.data(), cfg.M, cfg.N, cfg.K);
            auto t1 = std::chrono::high_resolution_clock::now();
            tiled_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        res.tiled_ms = tiled_total / opts.num_iterations;

        // Compute overhead and GOPS
        res.tiling_overhead = (res.golden_ms > 0)
            ? 100.0 * (res.tiled_ms - res.golden_ms) / res.golden_ms
            : 0.0;
        res.effective_gops = (res.tiled_ms > 0)
            ? (double)res.macs / (res.tiled_ms * 1e6)
            : 0.0;

        results.push_back(res);

        std::fprintf(stdout, "%-25s %6u %6u %6u %6u %12llu %12.4f %12.4f %7.1f%% %10.2f\n",
                     cfg.desc, cfg.M, cfg.N, cfg.K, res.tiles,
                     (unsigned long long)res.macs,
                     res.golden_ms, res.tiled_ms,
                     res.tiling_overhead, res.effective_gops);
    }

    // Summary statistics
    double avg_overhead = 0;
    for (auto& r : results) avg_overhead += r.tiling_overhead;
    avg_overhead /= results.size();

    std::fprintf(stdout, "\n--- Summary ---\n");
    std::fprintf(stdout, "Average tiling overhead: %.1f%%\n", avg_overhead);
    std::fprintf(stdout, "Iterations per config:   %u\n", opts.num_iterations);
}

// =============================================================================
// MNIST Layer Profiling
// =============================================================================
static void runLayerProfile(const BenchOptions& opts) {
    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "==========================================================\n");
    std::fprintf(stdout, "  MNIST LAYER PROFILING\n");
    std::fprintf(stdout, "==========================================================\n\n");

    // Get tiling plans
    auto plans = compute::mnist::planAllLayers();

    std::fprintf(stdout, "%-8s %6s %6s %6s %6s %6s %12s %12s %12s %12s\n",
                 "Layer", "M", "N", "K", "M_pad", "K_pad",
                 "Tiles", "MACs", "Wgt(KB)", "Out(KB)");
    std::fprintf(stdout, "%s\n", std::string(100, '-').c_str());

    uint64_t total_macs = 0;
    uint32_t total_tiles = 0;

    for (auto& p : plans) {
        total_macs  += p.totalMACs();
        total_tiles += p.totalTiles();

        std::fprintf(stdout, "%-8s %6u %6u %6u %6u %6u %12u %12llu %12.1f %12.1f\n",
                     p.layer_name.c_str(),
                     p.original.M, p.original.N, p.original.K,
                     p.padded.M_padded, p.padded.K_padded,
                     p.totalTiles(),
                     (unsigned long long)p.totalMACs(),
                     p.totalWeightBytes() / 1024.0,
                     p.totalOutputBytes() / 1024.0);
    }

    std::fprintf(stdout, "%s\n", std::string(100, '-').c_str());
    std::fprintf(stdout, "%-8s %6s %6s %6s %6s %6s %12u %12llu\n",
                 "TOTAL", "", "", "", "", "",
                 total_tiles, (unsigned long long)total_macs);

    // BSR density analysis (if weight files are available)
    std::fprintf(stdout, "\n--- BSR Density Analysis ---\n\n");

    struct LayerInfo {
        const char* name;
        uint32_t rows, cols;
        const char* weight_file;
    };
    std::vector<LayerInfo> layer_info = {
        {"conv1",  32,   9,    "conv1_weight_int8.npy"},
        {"conv2",  64,   288,  "conv2_weight_int8.npy"},
        {"fc1",    128,  9216, "fc1_weight_int8.npy"},
        {"fc2",    10,   128,  "fc2_weight_int8.npy"},
    };

    std::fprintf(stdout, "%-8s %8s %8s %10s %10s %10s %8s\n",
                 "Layer", "Rows", "Cols", "NNZ Blks", "Tot Blks", "Savings", "Density");
    std::fprintf(stdout, "%s\n", std::string(70, '-').c_str());

    for (auto& li : layer_info) {
        std::string path = opts.weights_dir + "/" + li.weight_file;
        try {
            auto arr = utils::NpyLoader::loadInt8(path);
            compute::BSREncoder encoder;
            auto bsr = encoder.encode(arr.data.data(), li.rows, li.cols);

            uint32_t total_blocks = bsr.num_block_rows * bsr.num_block_cols;
            double savings = 100.0 * (1.0 - bsr.density());

            std::fprintf(stdout, "%-8s %8u %8u %10u %10u %9.1f%% %7.2f%%\n",
                         li.name, li.rows, li.cols,
                         bsr.nnz_blocks, total_blocks,
                         savings, 100.0 * bsr.density());
        } catch (const std::exception& e) {
            std::fprintf(stdout, "%-8s  (weights not found: %s)\n", li.name, e.what());
        }
    }

    // Roofline analysis (theoretical)
    std::fprintf(stdout, "\n--- Theoretical Roofline Analysis ---\n\n");

    driver::PerfAnalyser analyser;

    std::vector<std::pair<std::string, driver::PerfMetrics>> layer_metrics;

    for (size_t i = 0; i < plans.size(); ++i) {
        auto& p = plans[i];
        // Simulate counters: assume 100% utilization as baseline
        driver::PerfSnapshot snap;
        snap.total_cycles   = p.totalTiles() * 196;  // 196 cycles per tile (14×14)
        snap.active_cycles  = snap.total_cycles;
        snap.idle_cycles    = 0;
        snap.stall_cycles   = 0;
        snap.blocks_computed = p.totalTiles();
        snap.dma_bytes      = p.totalWeightBytes() + p.totalActivationBytes();

        auto metrics = analyser.analyseLayer(snap,
                                              p.original.M, p.original.N, p.original.K,
                                              snap.total_cycles / 100e3);  // ms at 100MHz
        layer_metrics.push_back({p.layer_name, metrics});
    }

    driver::PerfAnalyser::printComparison(layer_metrics, std::cout);

    // Per-layer roofline plots
    for (auto& [name, metrics] : layer_metrics) {
        std::fprintf(stdout, "\n--- %s Roofline ---\n", name.c_str());
        std::fprintf(stdout, "%s\n", analyser.rooflinePlot(metrics).c_str());
    }
}

// =============================================================================
// Batched FC Roofline Comparison
// =============================================================================
static void runBatchComparison(const BenchOptions& opts) {
    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "==========================================================\n");
    std::fprintf(stdout, "  BATCHED FC ROOFLINE COMPARISON\n");
    std::fprintf(stdout, "==========================================================\n\n");

    const uint32_t B = opts.batch_size;

    // Get unbatched (N=1) and batched plans
    auto unbatched = compute::mnist::planAllLayers();
    auto batched   = compute::mnist::planAllLayersBatched(B, /* cache_fc2_weights */ true);

    // Roofline constants
    constexpr double PEAK_GOPS = 19.6;   // 196 PEs × 100 MHz
    constexpr double PEAK_BW   = 1200.0; // 32-bit AXI HP @ 150 MHz (MB/s)
    const double ridge_point   = PEAK_GOPS / (PEAK_BW / 1000.0);

    std::fprintf(stdout, "Architecture: 14×14 weight-stationary systolic array\n");
    std::fprintf(stdout, "Peak compute:  %.1f GOPS  |  Peak BW: %.0f MB/s  |  Ridge: %.2f ops/byte\n",
                 PEAK_GOPS, PEAK_BW, ridge_point);
    std::fprintf(stdout, "Batch size:    N=%u (%u N-tiles → weight reuse in M→K→N scheduling)\n",
                 B, (B + 13) / 14);
    std::fprintf(stdout, "FC2 caching:   ON (all 1,960 B weight tiles fit in BRAM)\n\n");

    // Side-by-side comparison table
    std::fprintf(stdout, "┌──────────┬──────────────────────────────┬──────────────────────────────┬──────────────┐\n");
    std::fprintf(stdout, "│          │     UNBATCHED  (N=1)         │     BATCHED  (N=%-2u)          │              │\n", B);
    std::fprintf(stdout, "│  Layer   │  DMA(KB)  MACs    AI(op/B)   │  DMA(KB)  MACs    AI(op/B)   │   Result     │\n");
    std::fprintf(stdout, "├──────────┼──────────────────────────────┼──────────────────────────────┼──────────────┤\n");

    for (size_t i = 0; i < unbatched.size() && i < batched.size(); ++i) {
        auto& u = unbatched[i];
        auto& b = batched[i];

        double u_dma_kb = u.totalDMABytes() / 1024.0;
        double b_dma_kb = b.totalDMABytes() / 1024.0;
        double u_ai     = u.arithmeticIntensity();
        double b_ai     = b.arithmeticIntensity();
        // Useful ops from unpadded shape
        uint64_t u_ops = 2ULL * u.original.M * u.original.N * u.original.K;
        uint64_t b_ops = 2ULL * b.original.M * b.original.N * b.original.K;

        const char* result;
        if (u_ai >= ridge_point && b_ai >= ridge_point) {
            result = "COMPUTE ✓";
        } else if (u_ai < ridge_point && b_ai >= ridge_point) {
            result = "MEM→COMP ★";
        } else if (b_ai >= ridge_point) {
            result = "COMPUTE ✓";
        } else {
            result = "MEM BOUND";
        }

        std::fprintf(stdout, "│  %-7s │  %7.1f  %7llu  %8.2f   │  %7.1f  %7llu  %8.2f   │  %-11s│\n",
                     u.layer_name.c_str(),
                     u_dma_kb, (unsigned long long)u_ops,  u_ai,
                     b_dma_kb, (unsigned long long)b_ops,  b_ai,
                     result);
    }

    std::fprintf(stdout, "└──────────┴──────────────────────────────┴──────────────────────────────┴──────────────┘\n");

    // Attainable GOPS per layer
    std::fprintf(stdout, "\n--- Attainable Performance (Roofline Bound) ---\n\n");
    std::fprintf(stdout, "%-8s  %12s  %12s  %8s\n",
                 "Layer", "Unbatched", "Batched", "Speedup");
    std::fprintf(stdout, "%s\n", std::string(50, '-').c_str());

    for (size_t i = 0; i < unbatched.size() && i < batched.size(); ++i) {
        auto& u = unbatched[i];
        auto& b = batched[i];

        double u_ai     = u.arithmeticIntensity();
        double b_ai     = b.arithmeticIntensity();

        // Roofline bound: min(peak_compute, AI * peak_bw)
        double u_gops = std::min(PEAK_GOPS, u_ai * (PEAK_BW / 1000.0));
        double b_gops = std::min(PEAK_GOPS, b_ai * (PEAK_BW / 1000.0));
        double speedup = (u_gops > 0) ? b_gops / u_gops : 0;

        std::fprintf(stdout, "%-8s  %10.2f GOPS  %10.2f GOPS  %6.1f×\n",
                     u.layer_name.c_str(), u_gops, b_gops, speedup);
    }

    // Total DMA savings
    uint64_t u_total_dma = 0, b_total_dma = 0;
    for (auto& u : unbatched) u_total_dma += u.totalDMABytes();
    for (auto& b : batched)   b_total_dma += b.totalDMABytes();

    std::fprintf(stdout, "\n--- DMA Traffic Summary ---\n");
    std::fprintf(stdout, "Unbatched total DMA: %llu bytes (%.1f KB)\n",
                 (unsigned long long)u_total_dma, u_total_dma / 1024.0);
    std::fprintf(stdout, "Batched total DMA:   %llu bytes (%.1f KB)\n",
                 (unsigned long long)b_total_dma, b_total_dma / 1024.0);
    std::fprintf(stdout, "Weight cache saving: FC2 weights loaded once → saves %u×1,280 = %u bytes/batch\n",
                 B - 1, (B - 1) * 1280);

    // ASCII roofline chart
    std::fprintf(stdout, "\n--- ASCII Roofline (log₂ scale) ---\n\n");

    // Vertical axis: GOPS, horizontal: AI
    // Use markers for each layer
    std::fprintf(stdout, "  GOPS ^\n");
    std::fprintf(stdout, "  19.6 |");
    // Print the ceiling
    for (int col = 0; col < 60; ++col) std::fprintf(stdout, "~");
    std::fprintf(stdout, "  (peak)\n");

    // Print roofline: slope region then flat region
    // log2(ridge) ≈ 4.03, so ridge at column ~20 of 60
    const int WIDTH = 60;
    const double LOG2_MIN = -1.0;  // AI = 0.5
    const double LOG2_MAX = 5.5;   // AI = 45
    auto aiToCol = [&](double ai) -> int {
        if (ai <= 0) return 0;
        double log2_ai = std::log2(ai);
        int col = static_cast<int>((log2_ai - LOG2_MIN) / (LOG2_MAX - LOG2_MIN) * WIDTH);
        return std::max(0, std::min(WIDTH - 1, col));
    };
    auto aiToRow = [&](double ai) -> double {
        // roofline-bound GOPS
        return std::min(PEAK_GOPS, ai * (PEAK_BW / 1000.0));
    };

    // We'll print key rows from top (19.6) to bottom (0)
    const int HEIGHT = 12;
    struct Marker { int col; int row; char ch; std::string label; };
    std::vector<Marker> markers;

    auto addMarker = [&](const compute::TilingPlan& plan, char ch) {
        double ai = plan.arithmeticIntensity();
        double gops = aiToRow(ai);
        int col = aiToCol(ai);
        int row = static_cast<int>((1.0 - gops / PEAK_GOPS) * HEIGHT);
        row = std::max(0, std::min(HEIGHT - 1, row));
        markers.push_back({col, row, ch, plan.layer_name});
    };

    // Markers: uppercase = unbatched, lowercase = batched
    for (size_t i = 0; i < unbatched.size(); ++i) {
        char uch = "ABCD"[i % 4]; // conv1=A, conv2=B, fc1=C, fc2=D
        char bch = "abcd"[i % 4];
        addMarker(unbatched[i], uch);
        addMarker(batched[i], bch);
    }

    // Ridge column
    int ridge_col = aiToCol(ridge_point);

    for (int row = 0; row < HEIGHT; ++row) {
        double gops = PEAK_GOPS * (1.0 - static_cast<double>(row) / HEIGHT);
        std::fprintf(stdout, " %5.1f |", gops);

        // Build the row character by character
        std::string line(WIDTH, ' ');

        // Draw roofline: slope region
        for (int col = 0; col < WIDTH; ++col) {
            double ai = std::pow(2.0, LOG2_MIN + (col + 0.5) / WIDTH * (LOG2_MAX - LOG2_MIN));
            double roof_gops = aiToRow(ai);
            int roof_row = static_cast<int>((1.0 - roof_gops / PEAK_GOPS) * HEIGHT);
            if (roof_row == row) line[col] = '.';
        }

        // Draw ridge line
        if (ridge_col >= 0 && ridge_col < WIDTH) {
            if (row < HEIGHT) line[ridge_col] = '|';
        }

        // Draw markers on top
        for (auto& m : markers) {
            if (m.row == row) line[m.col] = m.ch;
        }

        std::fprintf(stdout, "%s\n", line.c_str());
    }

    std::fprintf(stdout, "   0.0 +");
    for (int i = 0; i < WIDTH; ++i) std::fprintf(stdout, "-");
    std::fprintf(stdout, "> AI (ops/byte)\n");
    std::fprintf(stdout, "        0.5       1        2     4    %5.1f  16   32\n", ridge_point);
    std::fprintf(stdout, "                                       ↑ ridge\n");

    std::fprintf(stdout, "\n  Legend: A/a=conv1  B/b=conv2  C/c=fc1  D/d=fc2\n");
    std::fprintf(stdout, "          UPPER=unbatched  lower=batched\n");

    // Analysis summary
    std::fprintf(stdout, "\n--- Analysis ---\n");
    std::fprintf(stdout, "FC1: batch N=%u + weight-stationary scheduling → AI crosses ridge.\n", B);
    std::fprintf(stdout, "     Weight tiles loaded once per (m,k), reused across %u N-tiles.\n", (B + 13) / 14);
    std::fprintf(stdout, "     On-chip BRAM accumulators hold partial output sums (%.1f KB).\n",
                 (double)((batched[2].grid.num_m_tiles * batched[2].grid.num_n_tiles * 784)) / 1024.0);
    std::fprintf(stdout, "FC2: AI=%.2f still below ridge (%.2f).\n",
                 batched[3].arithmeticIntensity(), ridge_point);
    std::fprintf(stdout, "     FC2 is only %u tiles × 196 cycles = %u cycles (%.1f µs @ 100 MHz).\n",
                 batched[3].totalTiles(),
                 batched[3].totalTiles() * 196,
                 batched[3].totalTiles() * 196 / 100.0);
    std::fprintf(stdout, "     Negligible runtime — optimising FC2 has no measurable impact.\n");
}

// =============================================================================
// Hardware Benchmark (requires PYNQ-Z2)
// =============================================================================
static void runHardwareBench(const BenchOptions& opts) {
    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "==========================================================\n");
    std::fprintf(stdout, "  HARDWARE BENCHMARK (PYNQ-Z2)\n");
    std::fprintf(stdout, "==========================================================\n\n");

    driver::Accelerator accel;
    driver::AcceleratorConfig config;
    config.enable_logging = opts.verbose;
    config.enable_perf    = true;

    std::fprintf(stdout, "Initialising accelerator...\n");
    accel.init(config);

    std::fprintf(stdout, "Loading BSR weights from %s...\n", opts.bsr_dir.c_str());
    accel.loadMNISTWeights(opts.bsr_dir);

    // Run each MNIST layer individually and collect metrics
    driver::PerfAnalyser analyser;
    std::vector<std::pair<std::string, driver::PerfMetrics>> layer_metrics;

    struct LayerConfig {
        const char* name;
        uint32_t M, N, K;
    };
    std::vector<LayerConfig> layers = {
        {"conv1",  32,   576,  9},
        {"conv2",  64,   144,  288},
        {"fc1",    128,  1,    9216},
        {"fc2",    10,   1,    128},
    };

    for (auto& lc : layers) {
        std::fprintf(stdout, "\nBenchmarking layer: %s (%u x %u x %u)\n",
                     lc.name, lc.M, lc.N, lc.K);

        // Generate random activation data
        std::vector<int8_t> act(lc.K * lc.N);
        for (auto& v : act) v = static_cast<int8_t>((std::rand() % 256) - 128);

        // Multiple iterations for averaging
        double total_ms = 0;
        driver::PerfSnapshot total_snap{};

        for (uint32_t iter = 0; iter < opts.num_iterations; ++iter) {
            auto result = accel.runGEMM(act.data(), lc.name, lc.M, lc.N, lc.K);
            total_ms += result.elapsed_ms;
            total_snap.total_cycles   += result.perf.total_cycles;
            total_snap.active_cycles  += result.perf.active_cycles;
            total_snap.idle_cycles    += result.perf.idle_cycles;
            total_snap.stall_cycles   += result.perf.stall_cycles;
            total_snap.dma_bytes      += result.perf.dma_bytes;
            total_snap.blocks_computed += result.perf.blocks_computed;
        }

        // Average
        total_snap.total_cycles   /= opts.num_iterations;
        total_snap.active_cycles  /= opts.num_iterations;
        total_snap.idle_cycles    /= opts.num_iterations;
        total_snap.stall_cycles   /= opts.num_iterations;
        total_snap.dma_bytes      /= opts.num_iterations;
        total_snap.blocks_computed /= opts.num_iterations;
        double avg_ms = total_ms / opts.num_iterations;

        auto metrics = analyser.analyseLayer(total_snap, lc.M, lc.N, lc.K, avg_ms);
        layer_metrics.push_back({lc.name, metrics});

        std::fprintf(stdout, "  Avg time: %.3f ms  Utilisation: %.1f%%  BW: %.1f MB/s\n",
                     avg_ms, metrics.compute_utilisation * 100.0,
                     metrics.dma_bandwidth_mbps);
    }

    // Summary comparison table
    std::fprintf(stdout, "\n");
    driver::PerfAnalyser::printComparison(layer_metrics, std::cout);

    // Full inference benchmark
    std::fprintf(stdout, "\n--- Full MNIST Inference Benchmark ---\n");
    std::vector<int8_t> dummy_input(784);
    for (auto& v : dummy_input) v = static_cast<int8_t>((std::rand() % 256) - 128);

    double total_inf_ms = 0;
    for (uint32_t iter = 0; iter < opts.num_iterations; ++iter) {
        auto result = accel.runMNIST(dummy_input.data());
        total_inf_ms += result.elapsed_ms;
    }
    double avg_inf_ms = total_inf_ms / opts.num_iterations;
    double inf_per_sec = 1000.0 / avg_inf_ms;

    std::fprintf(stdout, "Average inference: %.3f ms (%.1f inferences/sec)\n",
                 avg_inf_ms, inf_per_sec);

    accel.shutdown();
    std::fprintf(stdout, "\nDone.\n");
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    BenchOptions opts = parseArgs(argc, argv);

    if (opts.verbose) {
        utils::Logger::instance().setLevel(utils::LogLevel::Debug);
    }

    try {
        if (opts.sweep)    runDimensionSweep(opts);
        if (opts.layers)   runLayerProfile(opts);
        if (opts.batch)    runBatchComparison(opts);
        if (opts.hardware) runHardwareBench(opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }

    return 0;
}
