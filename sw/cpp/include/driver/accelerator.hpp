// accelerator.hpp — Top-level accelerator control interface
// =============================================================================
//
// High-level API for the 14×14 INT8 weight-stationary systolic array
// accelerator on PYNQ-Z2.  Encapsulates CSR, DMA, BufferManager, Tiling,
// and BSR subsystems into a single entry point for inference.
//
// Typical usage:
//   Accelerator accel;
//   accel.init();
//   accel.loadWeights("data/bsr_export_14x14");
//   auto result = accel.runGEMM(A_data, M, N, K);
//   // -or-
//   auto logits = accel.runMNIST(input_28x28);
//   accel.shutdown();
//
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "driver/csr_interface.hpp"
#include "driver/performance.hpp"
#include "memory/dma_controller.hpp"
#include "memory/buffer_manager.hpp"
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"

namespace accel {
namespace driver {

// =============================================================================
// AcceleratorConfig — Initialisation parameters
// =============================================================================
struct AcceleratorConfig {
    uint32_t csr_base_addr   = 0x43C00000;    // AXI-Lite CSR base
    uint32_t ddr_base_addr   = 0x10000000;    // Accelerator DDR region
    size_t   ddr_map_size    = 0x00800000;    // 8 MB DDR mapping
    bool     enable_logging  = true;
    bool     enable_perf     = true;
};

// =============================================================================
// LayerWeights — Loaded weight data for one layer
// =============================================================================
struct LayerWeights {
    std::string name;
    compute::BSRMatrix bsr;
    uint32_t orig_rows;
    uint32_t orig_cols;
    uint32_t ddr_offset;        // Byte offset in DDR weight region
    std::vector<int32_t> bias;  // Quantised bias (INT32)
    float weight_scale;
    float bias_scale;
};

// =============================================================================
// InferenceResult — Output from a GEMM or inference run
// =============================================================================
struct InferenceResult {
    std::vector<int32_t> raw_int32;    // Raw INT32 accumulator output
    std::vector<float>   dequantised;  // Dequantised float output
    uint32_t             predicted_class = 0;  // argmax (for classification)
    PerfSnapshot         perf;         // Performance counters
    double               elapsed_ms = 0.0;     // Wall-clock time
};

// =============================================================================
// Accelerator — Top-level driver
// =============================================================================
class Accelerator {
public:
    Accelerator();
    ~Accelerator();

    // Non-copyable
    Accelerator(const Accelerator&) = delete;
    Accelerator& operator=(const Accelerator&) = delete;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /// Initialise hardware: open /dev/mem, map CSR + DDR, reset accelerator.
    void init(const AcceleratorConfig& config = AcceleratorConfig{});

    /// Clean shutdown: abort any in-flight op, unmap memory.
    void shutdown();

    /// Check if accelerator is initialised and ready
    bool isReady() const { return initialised_; }

    // -------------------------------------------------------------------------
    // Weight Loading
    // -------------------------------------------------------------------------

    /// Load BSR weights for one layer from exported directory.
    /// @param dir       Path to export dir (e.g. "data/bsr_export_14x14/conv1")
    /// @param name      Layer name (e.g. "conv1")
    /// @param orig_rows Original weight matrix rows
    /// @param orig_cols Original weight matrix cols
    void loadLayerWeights(const std::string& dir,
                          const std::string& name,
                          uint32_t orig_rows, uint32_t orig_cols);

    /// Load all 4 MNIST layer weights from a BSR export directory.
    /// @param base_dir  e.g. "data/bsr_export_14x14"
    void loadMNISTWeights(const std::string& base_dir);

    /// Load dense INT8 weights directly (will be BSR-encoded).
    /// @param data  Row-major INT8 weight matrix
    /// @param rows, cols  Dimensions
    /// @param name  Layer name
    void loadDenseWeights(const int8_t* data, uint32_t rows, uint32_t cols,
                          const std::string& name);

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /// Run a single tiled GEMM: C[M×N] = A[M×K] × B[K×N]
    /// Weights must already be loaded.  Activations are supplied here.
    /// @param activations  Row-major INT8 activation data
    /// @param layer_name   Which loaded layer's weights to use
    /// @param M, N, K      GEMM dimensions
    InferenceResult runGEMM(const int8_t* activations,
                            const std::string& layer_name,
                            uint32_t M, uint32_t N, uint32_t K);

    /// Run full MNIST inference (all 4 layers) on one 28×28 input image.
    /// All layer weights must be loaded via loadMNISTWeights().
    /// @param input_28x28  [784] INT8 pixel values
    InferenceResult runMNIST(const int8_t* input_28x28);

    // -------------------------------------------------------------------------
    // Status / Debug
    // -------------------------------------------------------------------------

    /// Read current performance counters
    PerfSnapshot readPerf() const;

    /// Print full accelerator state
    void dumpState() const;

    /// Get the number of loaded layers
    size_t numLoadedLayers() const { return layers_.size(); }

    /// Access subsystems directly (for advanced use / testing)
    CSRInterface&           csr()  { return *csr_; }
    memory::DMAController&  dma()  { return *dma_; }
    memory::BufferManager&  buf()  { return *buf_; }

    /// Progress callback: (layer_name, tile_idx, total_tiles)
    using ProgressCallback = std::function<void(const std::string&, uint32_t, uint32_t)>;
    void setProgressCallback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

private:
    bool initialised_ = false;

    // Hardware interface objects (heap-allocated for late init)
    std::unique_ptr<CSRInterface>           csr_;
    std::unique_ptr<memory::DMAController>  dma_;
    std::unique_ptr<memory::BufferManager>  buf_;

    // Loaded layer data
    std::vector<LayerWeights> layers_;

    // Configuration
    AcceleratorConfig config_;

    // Callback
    ProgressCallback progress_cb_;

    // Internal helpers
    LayerWeights* findLayer(const std::string& name);
    const LayerWeights* findLayer(const std::string& name) const;
    void stageWeightsToDDR(LayerWeights& layer);
    void stageActivationsToDDR(const int8_t* data, size_t bytes, uint32_t offset);
    void readOutputFromDDR(int32_t* dst, size_t count, uint32_t offset);
};

} // namespace driver
} // namespace accel
