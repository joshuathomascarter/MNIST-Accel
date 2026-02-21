// accelerator.cpp — Top-level accelerator driver implementation
// =============================================================================
//
// Ties together CSR, DMA, BufferManager, Tiling, and BSR encoding into a
// single coherent inference driver for the 14×14 INT8 systolic array on
// PYNQ-Z2.
//
// =============================================================================
#include "driver/accelerator.hpp"
#include "memory/address_map.hpp"
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace accel {
namespace driver {

using namespace memory;

// =============================================================================
// Constructor / Destructor
// =============================================================================

Accelerator::Accelerator() = default;

Accelerator::~Accelerator() {
    if (initialised_) {
        try {
            shutdown();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

// =============================================================================
// Lifecycle
// =============================================================================

void Accelerator::init(const AcceleratorConfig& config) {
    if (initialised_) {
        throw CSRException("Accelerator already initialised");
    }

    config_ = config;

    // 1. Map CSR registers
    csr_ = std::make_unique<CSRInterface>(config.csr_base_addr);

    // 2. Create DMA controller and map DDR
    dma_ = std::make_unique<DMAController>(*csr_, config.ddr_base_addr);
    dma_->mapDDR(config.ddr_map_size);

    // 3. Create buffer manager
    buf_ = std::make_unique<BufferManager>(*dma_, *csr_);

    // 4. Reset hardware
    csr_->abort();
    csr_->waitDone(100000);  // Wait for any in-flight op
    buf_->reset();

    initialised_ = true;

    if (config.enable_logging) {
        std::cout << "[ACCEL] Initialised: CSR=0x" << std::hex
                  << config.csr_base_addr << " DDR=0x"
                  << config.ddr_base_addr << std::dec << "\n";
    }
}

void Accelerator::shutdown() {
    if (!initialised_) return;

    // Abort any running operation
    if (csr_) {
        csr_->abort();
    }

    // Release resources in reverse order
    buf_.reset();
    dma_.reset();
    csr_.reset();

    layers_.clear();
    initialised_ = false;

    if (config_.enable_logging) {
        std::cout << "[ACCEL] Shutdown complete\n";
    }
}

// =============================================================================
// Weight Loading
// =============================================================================

LayerWeights* Accelerator::findLayer(const std::string& name) {
    for (auto& l : layers_) {
        if (l.name == name) return &l;
    }
    return nullptr;
}

const LayerWeights* Accelerator::findLayer(const std::string& name) const {
    for (auto& l : layers_) {
        if (l.name == name) return &l;
    }
    return nullptr;
}

void Accelerator::stageWeightsToDDR(LayerWeights& layer) {
    // Pack BSR data
    compute::BSREncoder encoder;
    auto packed = encoder.pack(layer.bsr);

    // Stage weight block values to DDR weight region
    dma_->loadWeights(packed.weights.data(), packed.weights.size());

    // Stage BSR metadata (row_ptr + col_idx) to DDR BSR region
    if (!packed.metadata.empty()) {
        // Split metadata into row_ptr and col_idx
        dma_->loadBSRRowPtr(layer.bsr.row_ptr.data(), layer.bsr.rowPtrBytes());
        dma_->loadBSRColIdx(layer.bsr.col_idx.data(), layer.bsr.colIdxBytes());
    }

    // Configure BSR scheduler CSR registers
    csr_->setBSRPointers(dma_->bsrPtrAddr(), dma_->bsrIdxAddr());
    csr_->setBSRConfig(layer.bsr.nnz_blocks,
                       layer.bsr.num_block_rows,
                       layer.bsr.num_block_cols);
}

void Accelerator::stageActivationsToDDR(const int8_t* data, size_t bytes,
                                         uint32_t offset) {
    (void)offset;  // DMA controller handles offset internally
    dma_->loadActivations(data, bytes);
}

void Accelerator::readOutputFromDDR(int32_t* dst, size_t count, uint32_t offset) {
    (void)offset;
    dma_->readOutput(dst, count * sizeof(int32_t));
}

void Accelerator::loadLayerWeights(const std::string& dir,
                                    const std::string& name,
                                    uint32_t orig_rows, uint32_t orig_cols) {
    assert(initialised_);

    LayerWeights lw;
    lw.name = name;
    lw.orig_rows = orig_rows;
    lw.orig_cols = orig_cols;

    // Load BSR from export directory
    compute::BSREncoder encoder;
    lw.bsr = encoder.loadFromExport(dir);

    // Compute DDR offset (sequential packing)
    uint32_t offset = ddr_layout::WEIGHTS_OFFSET;
    for (const auto& existing : layers_) {
        offset += static_cast<uint32_t>(existing.bsr.valuesBytes());
    }
    lw.ddr_offset = offset;

    lw.weight_scale = 1.0f;
    lw.bias_scale   = 1.0f;

    layers_.push_back(std::move(lw));

    if (config_.enable_logging) {
        std::cout << "[ACCEL] Loaded layer '" << name << "' from " << dir
                  << " (" << layers_.back().bsr.nnz_blocks << " BSR blocks)\n";
    }
}

void Accelerator::loadMNISTWeights(const std::string& base_dir) {
    struct LayerSpec {
        const char* name;
        uint32_t orig_rows;
        uint32_t orig_cols;
    };

    const LayerSpec specs[] = {
        {"conv1", 32,  9},
        {"conv2", 64,  288},
        {"fc1",   128, 9216},
        {"fc2",   10,  128}
    };

    for (const auto& spec : specs) {
        std::string dir = base_dir + "/" + spec.name;
        loadLayerWeights(dir, spec.name, spec.orig_rows, spec.orig_cols);
    }

    if (config_.enable_logging) {
        std::cout << "[ACCEL] All 4 MNIST layers loaded. Total BSR blocks: ";
        uint32_t total = 0;
        for (const auto& l : layers_) total += l.bsr.nnz_blocks;
        std::cout << total << "\n";
    }
}

void Accelerator::loadDenseWeights(const int8_t* data,
                                    uint32_t rows, uint32_t cols,
                                    const std::string& name) {
    assert(initialised_);

    LayerWeights lw;
    lw.name = name;
    lw.orig_rows = rows;
    lw.orig_cols = cols;

    compute::BSREncoder encoder;
    lw.bsr = encoder.encode(data, rows, cols);

    uint32_t offset = ddr_layout::WEIGHTS_OFFSET;
    for (const auto& existing : layers_) {
        offset += static_cast<uint32_t>(existing.bsr.valuesBytes());
    }
    lw.ddr_offset = offset;
    lw.weight_scale = 1.0f;
    lw.bias_scale   = 1.0f;

    layers_.push_back(std::move(lw));
}

// =============================================================================
// Execution
// =============================================================================

InferenceResult Accelerator::runGEMM(const int8_t* activations,
                                      const std::string& layer_name,
                                      uint32_t M, uint32_t N, uint32_t K) {
    assert(initialised_);

    auto* layer = findLayer(layer_name);
    if (!layer) {
        throw CSRException("Layer not found: " + layer_name);
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Compute tiling plan
    compute::GEMMShape shape{M, N, K};
    auto plan = compute::planDenseGEMM(shape, layer_name,
                                        layer->ddr_offset,
                                        ddr_layout::ACTS_OFFSET,
                                        ddr_layout::OUTPUT_OFFSET);

    // 2. Programme CSR matrix/tile dimensions
    csr_->setMatrixDimensions(plan.padded.M_padded,
                              plan.padded.N_padded,
                              plan.padded.K_padded);
    csr_->setTileDimensions(plan.grid.num_m_tiles,
                            plan.grid.num_n_tiles,
                            plan.grid.num_k_tiles);

    // 3. Stage weights to DDR + configure BSR
    stageWeightsToDDR(*layer);

    // 4. Stage activations to DDR
    size_t act_bytes = static_cast<size_t>(K) * N;
    stageActivationsToDDR(activations, act_bytes, 0);

    // 5. Execute tile sequence with double buffering
    auto prog_cb = progress_cb_;
    std::string ln = layer_name;
    uint32_t total = plan.totalTiles();

    buf_->executeTileSequence(
        plan.tiles.data(), plan.tiles.size(),
        [&](const TileDescriptor& td, uint32_t idx) {
            (void)td;
            if (prog_cb) prog_cb(ln, idx, total);
        });

    // 6. Read output from DDR
    uint32_t out_elements = M * N;
    InferenceResult result;
    result.raw_int32.resize(out_elements);
    readOutputFromDDR(result.raw_int32.data(), out_elements, 0);

    // 7. Record timing
    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 8. Read perf counters
    if (config_.enable_perf) {
        result.perf = readPerf();
    }

    return result;
}

InferenceResult Accelerator::runMNIST(const int8_t* input_28x28) {
    assert(initialised_);
    assert(layers_.size() >= 4);

    auto t0 = std::chrono::steady_clock::now();

    // Use the compute::mnist namespace for tiling plans
    auto plans = compute::mnist::planAllLayers();

    // For a full hardware inference, we'd run each layer through the
    // accelerator in sequence, feeding each layer's output as the next
    // layer's activation.  Here we execute the tiling plan for each layer.

    InferenceResult final_result;

    for (size_t i = 0; i < plans.size(); ++i) {
        auto& plan = plans[i];
        auto* layer = findLayer(plan.layer_name);
        if (!layer) {
            throw CSRException("MNIST layer not loaded: " + plan.layer_name);
        }

        // Configure CSR
        csr_->setMatrixDimensions(plan.padded.M_padded,
                                  plan.padded.N_padded,
                                  plan.padded.K_padded);
        csr_->setTileDimensions(plan.grid.num_m_tiles,
                                plan.grid.num_n_tiles,
                                plan.grid.num_k_tiles);

        // Stage weights
        stageWeightsToDDR(*layer);

        // For the first layer, stage the raw input image as activations
        if (i == 0) {
            stageActivationsToDDR(input_28x28, 784, 0);
        }
        // For subsequent layers, the previous layer's DDR output is already
        // in the output region — the hardware pipeline or host re-stages it.

        // Execute tile sequence
        std::string ln = plan.layer_name;
        uint32_t total = plan.totalTiles();
        auto prog_cb = progress_cb_;

        buf_->executeTileSequence(
            plan.tiles.data(), plan.tiles.size(),
            [&](const TileDescriptor& td, uint32_t idx) {
                (void)td;
                if (prog_cb) prog_cb(ln, idx, total);
            });
    }

    // Read final output (10 logits)
    final_result.raw_int32.resize(10);
    readOutputFromDDR(final_result.raw_int32.data(), 10, 0);

    // Dequantise
    final_result.dequantised.resize(10);
    for (int i = 0; i < 10; ++i) {
        final_result.dequantised[i] = static_cast<float>(final_result.raw_int32[i]);
    }

    // Argmax
    final_result.predicted_class = static_cast<uint32_t>(
        std::distance(final_result.dequantised.begin(),
                      std::max_element(final_result.dequantised.begin(),
                                       final_result.dequantised.end())));

    auto t1 = std::chrono::steady_clock::now();
    final_result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (config_.enable_perf) {
        final_result.perf = readPerf();
    }

    return final_result;
}

// =============================================================================
// Status / Debug
// =============================================================================

PerfSnapshot Accelerator::readPerf() const {
    if (!csr_) return PerfSnapshot{};
    auto hw = csr_->readPerfCounters();
    PerfSnapshot snap;
    snap.total_cycles    = hw.total_cycles;
    snap.active_cycles   = hw.active_cycles;
    snap.idle_cycles     = hw.idle_cycles;
    snap.dma_bytes       = hw.dma_bytes;
    snap.blocks_computed = hw.blocks_processed;
    snap.stall_cycles    = hw.stall_cycles;
    return snap;
}

void Accelerator::dumpState() const {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout <<   "║         ACCELERATOR STATE                    ║\n";
    std::cout <<   "╠══════════════════════════════════════════════╣\n";
    std::cout << "║  Initialised: " << (initialised_ ? "YES" : "NO") << "\n";
    std::cout << "║  Loaded layers: " << layers_.size() << "\n";

    for (const auto& l : layers_) {
        std::cout << "║    " << l.name
                  << " (" << l.orig_rows << "×" << l.orig_cols << ")"
                  << " BSR blocks=" << l.bsr.nnz_blocks
                  << " DDR offset=0x" << std::hex << l.ddr_offset
                  << std::dec << "\n";
    }

    if (csr_) {
        std::cout << "║\n║  CSR valid: " << csr_->isValid() << "\n";
        std::cout << "║  HW busy:   " << csr_->isBusy() << "\n";
    }

    if (buf_) {
        buf_->dumpState();
    }

    std::cout << "╚══════════════════════════════════════════════╝\n\n";
}

} // namespace driver
} // namespace accel
