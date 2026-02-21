// run_mnist_inference.cpp — Main MNIST inference application for PYNQ-Z2
// =============================================================================
//
// Loads INT8 quantised weights + a 28×28 input image, runs inference on the
// 14×14 weight-stationary systolic array, and prints the prediction.
//
// Two modes:
//   1. Hardware mode (default on PYNQ-Z2): uses Accelerator → CSR → DMA → PL
//   2. Golden mode (--golden):             pure software reference for verification
//
// Usage:
//   ./run_mnist_inference                          # HW mode, default image
//   ./run_mnist_inference --input image.npy        # HW mode, specific image
//   ./run_mnist_inference --golden                 # SW golden model only
//   ./run_mnist_inference --weights data/int8      # INT8 weight dir
//   ./run_mnist_inference --bsr data/bsr_export_14x14  # BSR export dir
//   ./run_mnist_inference --verbose                # Enable debug logging
//
// =============================================================================
#include "driver/accelerator.hpp"
#include "driver/performance.hpp"
#include "compute/golden_model.hpp"
#include "compute/tiling.hpp"
#include "utils/npy_loader.hpp"
#include "utils/logging.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

using namespace accel;

// =============================================================================
// Command-line options
// =============================================================================
struct Options {
    std::string input_path    = "data/golden/mnist_inputs.npy";
    std::string weights_dir   = "data/int8";
    std::string bsr_dir       = "data/bsr_export_14x14";
    bool golden_only          = false;
    bool verbose              = false;
    bool show_logits          = false;
    int  image_index          = 0;     // Which image from batch
};

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --input PATH      Input image .npy file (default: data/golden/mnist_inputs.npy)\n"
        "  --weights DIR     INT8 weight directory (default: data/int8)\n"
        "  --bsr DIR         BSR export directory (default: data/bsr_export_14x14)\n"
        "  --golden          Run software golden model only (no hardware)\n"
        "  --verbose         Enable debug-level logging\n"
        "  --logits          Print raw logits\n"
        "  --index N         Image index in batch (default: 0)\n"
        "  --help            Show this help\n"
        "\n", prog);
}

static Options parseArgs(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc)        opts.input_path = argv[++i];
        else if (arg == "--weights" && i + 1 < argc)  opts.weights_dir = argv[++i];
        else if (arg == "--bsr" && i + 1 < argc)      opts.bsr_dir = argv[++i];
        else if (arg == "--golden")                    opts.golden_only = true;
        else if (arg == "--verbose")                   opts.verbose = true;
        else if (arg == "--logits")                    opts.show_logits = true;
        else if (arg == "--index" && i + 1 < argc)    opts.image_index = std::atoi(argv[++i]);
        else if (arg == "--help") { printUsage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown option: %s\n", arg.c_str()); printUsage(argv[0]); std::exit(1); }
    }
    return opts;
}

// =============================================================================
// Load weights from INT8 numpy directory
// =============================================================================
struct MNISTWeights {
    utils::NpyArray<int8_t>   conv1_w, conv2_w, fc1_w, fc2_w;
    utils::NpyArray<int8_t>   conv1_b_i8, conv2_b_i8, fc1_b_i8, fc2_b_i8;
    // Widened to int32 for golden model API (accumulators are 32-bit)
    std::vector<int32_t>      conv1_b, conv2_b, fc1_b, fc2_b;
};

/// Widen int8 vector to int32
static std::vector<int32_t> widenToInt32(const std::vector<int8_t>& src) {
    std::vector<int32_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<int32_t>(src[i]);
    return dst;
}

static MNISTWeights loadWeights(const std::string& dir) {
    MNISTWeights w;
    std::fprintf(stdout, "Loading weights from %s...\n", dir.c_str());

    w.conv1_w    = utils::NpyLoader::loadInt8(dir + "/conv1_weight_int8.npy");
    w.conv1_b_i8 = utils::NpyLoader::loadInt8(dir + "/conv1_bias_int8.npy");
    w.conv2_w    = utils::NpyLoader::loadInt8(dir + "/conv2_weight_int8.npy");
    w.conv2_b_i8 = utils::NpyLoader::loadInt8(dir + "/conv2_bias_int8.npy");
    w.fc1_w      = utils::NpyLoader::loadInt8(dir + "/fc1_weight_int8.npy");
    w.fc1_b_i8   = utils::NpyLoader::loadInt8(dir + "/fc1_bias_int8.npy");
    w.fc2_w      = utils::NpyLoader::loadInt8(dir + "/fc2_weight_int8.npy");
    w.fc2_b_i8   = utils::NpyLoader::loadInt8(dir + "/fc2_bias_int8.npy");

    // Widen biases from int8 → int32 (accumulator width)
    w.conv1_b = widenToInt32(w.conv1_b_i8.data);
    w.conv2_b = widenToInt32(w.conv2_b_i8.data);
    w.fc1_b   = widenToInt32(w.fc1_b_i8.data);
    w.fc2_b   = widenToInt32(w.fc2_b_i8.data);

    std::fprintf(stdout, "  conv1_w: %llu elements, bias: %llu\n", (unsigned long long)w.conv1_w.size(), (unsigned long long)w.conv1_b.size());
    std::fprintf(stdout, "  conv2_w: %llu elements, bias: %llu\n", (unsigned long long)w.conv2_w.size(), (unsigned long long)w.conv2_b.size());
    std::fprintf(stdout, "  fc1_w:   %llu elements, bias: %llu\n", (unsigned long long)w.fc1_w.size(), (unsigned long long)w.fc1_b.size());
    std::fprintf(stdout, "  fc2_w:   %llu elements, bias: %llu\n", (unsigned long long)w.fc2_w.size(), (unsigned long long)w.fc2_b.size());

    return w;
}

// =============================================================================
// Golden model inference
// =============================================================================
static int runGolden(const Options& opts) {
    std::fprintf(stdout, "\n=== MNIST Golden Model Inference ===\n\n");

    // Load weights
    MNISTWeights w = loadWeights(opts.weights_dir);

    // Load input (supports both int8 and uint8 npy files)
    std::fprintf(stdout, "Loading input from %s (index %d)...\n",
                 opts.input_path.c_str(), opts.image_index);

    // Load as uint8 (raw pixel values 0-255), then convert to int8 [-128, 127]
    auto input_u8 = utils::NpyLoader::loadUint8(opts.input_path);
    std::vector<int8_t> input_i8(input_u8.data.size());
    for (size_t i = 0; i < input_u8.data.size(); ++i) {
        input_i8[i] = static_cast<int8_t>(static_cast<int>(input_u8.data[i]) - 128);
    }

    // Extract single 28x28 image
    const size_t image_size = 784;  // 28 x 28
    if (input_i8.size() < (size_t)(opts.image_index + 1) * image_size) {
        std::fprintf(stderr, "Error: image index %d out of bounds (file has %llu elements)\n",
                     opts.image_index, (unsigned long long)input_i8.size());
        return 1;
    }
    const int8_t* input_ptr = input_i8.data() + opts.image_index * image_size;

    // Run golden inference
    auto start = std::chrono::high_resolution_clock::now();
    auto logits = compute::GoldenModel::mnistInference(
        input_ptr,
        w.conv1_w.data.data(), w.conv1_b.data(),
        w.conv2_w.data.data(), w.conv2_b.data(),
        w.fc1_w.data.data(),   w.fc1_b.data(),
        w.fc2_w.data.data(),   w.fc2_b.data());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    uint32_t predicted = compute::GoldenModel::argmax(logits);

    // Print results
    std::fprintf(stdout, "\n--- Results ---\n");
    std::fprintf(stdout, "Predicted class: %u\n", predicted);
    std::fprintf(stdout, "Inference time:  %.3f ms\n", elapsed_ms);

    if (opts.show_logits) {
        std::fprintf(stdout, "\nLogits:\n");
        for (size_t i = 0; i < logits.size(); ++i) {
            std::fprintf(stdout, "  class %zu: %12.6f%s\n",
                         i, logits[i], (i == predicted ? "  <-- predicted" : ""));
        }
    }

    // Print tiling plan summary
    std::fprintf(stdout, "\n--- Tiling Summary ---\n");
    auto plans = compute::mnist::planAllLayers();
    for (auto& p : plans) {
        p.print(std::cout);
    }

    return 0;
}

// =============================================================================
// Hardware inference (requires PYNQ-Z2)
// =============================================================================
static int runHardware(const Options& opts) {
    std::fprintf(stdout, "\n=== MNIST Hardware Inference (14x14 Systolic Array) ===\n\n");

    // Initialise accelerator
    driver::Accelerator accel;
    driver::AcceleratorConfig config;
    config.enable_logging = opts.verbose;
    config.enable_perf    = true;

    std::fprintf(stdout, "Initialising accelerator...\n");
    accel.init(config);

    // Load BSR weights
    std::fprintf(stdout, "Loading BSR weights from %s...\n", opts.bsr_dir.c_str());
    accel.loadMNISTWeights(opts.bsr_dir);

    // Load input (uint8 → int8 conversion)
    std::fprintf(stdout, "Loading input from %s (index %d)...\n",
                 opts.input_path.c_str(), opts.image_index);
    auto input_u8 = utils::NpyLoader::loadUint8(opts.input_path);
    std::vector<int8_t> input_i8(input_u8.data.size());
    for (size_t i = 0; i < input_u8.data.size(); ++i) {
        input_i8[i] = static_cast<int8_t>(static_cast<int>(input_u8.data[i]) - 128);
    }

    const size_t image_size = 784;
    if (input_i8.size() < (size_t)(opts.image_index + 1) * image_size) {
        std::fprintf(stderr, "Error: image index %d out of bounds\n", opts.image_index);
        accel.shutdown();
        return 1;
    }
    const int8_t* input_ptr = input_i8.data() + opts.image_index * image_size;

    // Progress callback
    accel.setProgressCallback([](const std::string& layer, uint32_t tile, uint32_t total) {
        std::fprintf(stdout, "\r  [%s] tile %u/%u (%.1f%%)",
                     layer.c_str(), tile + 1, total,
                     100.0f * (tile + 1) / total);
        std::fflush(stdout);
    });

    // Run inference
    std::fprintf(stdout, "Running inference...\n");
    auto result = accel.runMNIST(input_ptr);

    std::fprintf(stdout, "\n\n--- Results ---\n");
    std::fprintf(stdout, "Predicted class: %u\n", result.predicted_class);
    std::fprintf(stdout, "Inference time:  %.3f ms\n", result.elapsed_ms);

    // Performance analysis
    if (config.enable_perf) {
        driver::PerfAnalyser analyser;
        auto metrics = analyser.analyseLayer(result.perf, 0, 0, 0, result.elapsed_ms);
        std::fprintf(stdout, "\n--- Performance ---\n");
        metrics.print(std::cout);

        std::string roofline = analyser.rooflinePlot(metrics);
        std::fprintf(stdout, "\n%s\n", roofline.c_str());
    }

    // Compare with golden model for verification
    std::fprintf(stdout, "\n--- Golden Model Verification ---\n");
    MNISTWeights w = loadWeights(opts.weights_dir);
    auto golden_logits = compute::GoldenModel::mnistInference(
        input_ptr,
        w.conv1_w.data.data(), w.conv1_b.data(),
        w.conv2_w.data.data(), w.conv2_b.data(),
        w.fc1_w.data.data(),   w.fc1_b.data(),
        w.fc2_w.data.data(),   w.fc2_b.data());
    uint32_t golden_pred = compute::GoldenModel::argmax(golden_logits);

    std::fprintf(stdout, "Golden model prediction: %u\n", golden_pred);
    if (golden_pred == result.predicted_class) {
        std::fprintf(stdout, "MATCH: Hardware and golden model agree.\n");
    } else {
        std::fprintf(stderr, "MISMATCH: Hardware predicted %u, golden predicted %u\n",
                     result.predicted_class, golden_pred);
    }

    if (opts.show_logits && !result.dequantised.empty()) {
        std::fprintf(stdout, "\nHW Logits vs Golden:\n");
        for (size_t i = 0; i < 10 && i < result.dequantised.size(); ++i) {
            std::fprintf(stdout, "  class %zu: HW=%12.6f  Golden=%12.6f  delta=%+.6f\n",
                         i, result.dequantised[i], golden_logits[i],
                         result.dequantised[i] - golden_logits[i]);
        }
    }

    accel.shutdown();
    std::fprintf(stdout, "\nDone.\n");
    return 0;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    Options opts = parseArgs(argc, argv);

    if (opts.verbose) {
        utils::Logger::instance().setLevel(utils::LogLevel::Debug);
    }

    try {
        if (opts.golden_only) {
            return runGolden(opts);
        } else {
            return runHardware(opts);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}
