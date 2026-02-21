// tiling.cpp — Tile size calculation for 14×14 systolic array mapping
// =============================================================================
//
// Implements the tiling strategy that maps arbitrary GEMM operations onto a
// 14×14 weight-stationary systolic array.  Produces TileDescriptor arrays
// consumed by BufferManager::executeTileSequence().
//
// Key invariants:
//   - All dimensions are padded UP to multiples of 14 (zero-pad)
//   - Weight tiles are 14×14 INT8 → 196 bytes each
//   - Activation tiles are 14×14 INT8 → 196 bytes each
//   - Output tiles are 14×14 INT32 → 784 bytes each
//   - Loop order: M → N → K  (output-stationary for accumulator reuse)
//   - is_first_k: clear accumulators; is_last_k: drain to DDR
//
// =============================================================================
#include "compute/tiling.hpp"
#include "compute/bsr_encoder.hpp"     // BSRMatrix
#include "memory/buffer_manager.hpp"   // TileDescriptor
#include "memory/address_map.hpp"

#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace accel {
namespace compute {

// =============================================================================
// ConvParams helpers
// =============================================================================

uint32_t ConvParams::H_out() const {
    return (H_in + 2 * padding - kH) / stride + 1;
}

uint32_t ConvParams::W_out() const {
    return (W_in + 2 * padding - kW) / stride + 1;
}

GEMMShape ConvParams::toGEMM() const {
    // im2col converts convolution to GEMM:
    //   Weight matrix:     C_out × (C_in * kH * kW)       → M × K
    //   Activation matrix: (C_in * kH * kW) × (H_out * W_out) → K × N
    //   Output:            C_out × (H_out * W_out)         → M × N
    uint32_t M = C_out;
    uint32_t K = C_in * kH * kW;
    uint32_t N = H_out() * W_out();
    return GEMMShape{M, N, K};
}

// =============================================================================
// Padding & grid computation
// =============================================================================

PaddedShape computePaddedShape(const GEMMShape& shape) {
    return PaddedShape{
        padTo14(shape.M),
        padTo14(shape.N),
        padTo14(shape.K)
    };
}

TileGrid computeTileGrid(const PaddedShape& padded) {
    return TileGrid{
        padded.M_padded / TILE_DIM,
        padded.N_padded / TILE_DIM,
        padded.K_padded / TILE_DIM
    };
}

// =============================================================================
// TilingPlan aggregate metrics
// =============================================================================

float TilingPlan::weight_sparsity() const {
    uint32_t total = dense_weight_blocks();
    if (total == 0) return 0.0f;
    uint32_t nz = is_sparse ? nnz_weight_blocks : total;
    return 1.0f - static_cast<float>(nz) / static_cast<float>(total);
}

uint32_t TilingPlan::totalTiles() const {
    return static_cast<uint32_t>(tiles.size());
}

uint32_t TilingPlan::totalWeightBytes() const {
    if (is_sparse) {
        // Sparse: only NNZ blocks are stored/transferred
        return nnz_weight_blocks * wgt_tile_bytes;
    }
    // Dense: full grid
    return grid.num_m_tiles * grid.num_k_tiles * wgt_tile_bytes;
}

uint32_t TilingPlan::totalActivationBytes() const {
    // Activation layout is always dense (activations aren't pruned)
    return grid.num_k_tiles * grid.num_n_tiles * act_tile_bytes;
}

uint32_t TilingPlan::totalOutputBytes() const {
    // Output is always M × N regardless of sparsity
    return grid.num_m_tiles * grid.num_n_tiles * out_tile_bytes;
}

uint64_t TilingPlan::totalMACs() const {
    if (is_sparse) {
        // Sparse: only NNZ blocks contribute compute
        // Each NNZ weight block × num_n_tiles activation tiles = nnz × nN × 14³ MACs
        return static_cast<uint64_t>(nnz_weight_blocks)
             * static_cast<uint64_t>(grid.num_n_tiles)
             * static_cast<uint64_t>(TILE_DIM) * TILE_DIM * TILE_DIM;
    }
    // Dense: M_padded × N_padded × K_padded
    return static_cast<uint64_t>(padded.M_padded)
         * static_cast<uint64_t>(padded.N_padded)
         * static_cast<uint64_t>(padded.K_padded);
}

double TilingPlan::sparseSpeedup() const {
    if (!is_sparse || nnz_weight_blocks == 0) return 1.0;
    return static_cast<double>(dense_weight_blocks())
         / static_cast<double>(nnz_weight_blocks);
}

uint64_t TilingPlan::totalDMABytes() const {
    // Count actual DMA bytes, skipping weight tiles marked as cached
    uint64_t wgt_dma = 0;
    uint64_t act_dma = 0;
    for (const auto& td : tiles) {
        if (!td.wgt_cached) {
            wgt_dma += td.wgt_bytes;
        }
        act_dma += td.act_bytes;
    }
    // Output tiles are written on is_last_k only
    uint64_t out_dma = 0;
    for (const auto& td : tiles) {
        if (td.is_last_k) {
            out_dma += out_tile_bytes;
        }
    }
    return wgt_dma + act_dma + out_dma;
}

double TilingPlan::arithmeticIntensity() const {
    uint64_t dma = totalDMABytes();
    if (dma == 0) return 0.0;
    // Use *useful* ops from unpadded original shape, not padded MACs.
    // With N=1, padded to 14 → 13/14 of activation tiles are zero-padded
    // waste.  With N=14, every DMA byte carries real data → 14× more
    // useful work for the same DMA cost.  That's the whole point of batching.
    uint64_t useful_ops = 2ULL * original.M * original.N * original.K;
    return static_cast<double>(useful_ops) / static_cast<double>(dma);
}

void TilingPlan::print(std::ostream& os) const {
    os << "\n╔══════════════════════════════════════════════════════╗\n";
    os <<   "║  TILING PLAN: " << layer_name;
    for (size_t i = layer_name.size(); i < 40; ++i) os << ' ';
    os << "║\n";
    os <<   "╠══════════════════════════════════════════════════════╣\n";
    os << "║  Original shape:  M=" << original.M
       << "  N=" << original.N
       << "  K=" << original.K << "\n";
    os << "║  Padded shape:    M=" << padded.M_padded
       << "  N=" << padded.N_padded
       << "  K=" << padded.K_padded << "\n";
    os << "║  Tile grid:       "
       << grid.num_m_tiles << " × "
       << grid.num_n_tiles << " × "
       << grid.num_k_tiles
       << " = " << grid.totalTiles() << " tiles (dense)\n";
    if (is_sparse) {
        os << "║  ── SPARSE ─────────────────────────────────────\n";
        os << "║  NNZ weight blks: " << nnz_weight_blocks
           << " / " << dense_weight_blocks()
           << " (" << std::fixed << std::setprecision(1)
           << (weight_sparsity() * 100.0f) << "% sparse)\n";
        os << "║  Actual tiles:    " << tiles.size()
           << " (" << std::setprecision(1)
           << sparseSpeedup() << "× fewer than dense)\n";
    }
    os << "║  Weight bytes:    " << totalWeightBytes() << "\n";
    os << "║  Activation bytes:" << totalActivationBytes() << "\n";
    os << "║  Output bytes:    " << totalOutputBytes() << "\n";
    os << "║  Total MACs:      " << totalMACs() << "\n";
    os << "║  DMA bytes:       " << totalDMABytes() << "\n";
    os << "║  Arith intensity: " << std::fixed << std::setprecision(2)
       << arithmeticIntensity() << " ops/byte\n";
    os << "║  DDR offsets:     wgt=0x" << std::hex << wgt_base_offset
       << "  act=0x" << act_base_offset
       << "  out=0x" << out_base_offset << std::dec << "\n";
    os << "╚══════════════════════════════════════════════════════╝\n";
}

// =============================================================================
// DDR offset helpers
// =============================================================================

uint32_t weightTileOffset(uint32_t m_idx, uint32_t k_idx,
                          uint32_t num_k_tiles, uint32_t base_offset) {
    // Row-major tile layout: tile(m, k) is at index m * num_k_tiles + k
    return base_offset + (m_idx * num_k_tiles + k_idx) * TILE_ELEMS;
}

uint32_t activationTileOffset(uint32_t k_idx, uint32_t n_idx,
                              uint32_t num_n_tiles, uint32_t base_offset) {
    // Activation matrix tile(k, n) at index k * num_n_tiles + n
    return base_offset + (k_idx * num_n_tiles + n_idx) * TILE_ELEMS;
}

uint32_t outputTileOffset(uint32_t m_idx, uint32_t n_idx,
                          uint32_t num_n_tiles, uint32_t base_offset) {
    // Output matrix tile(m, n) — INT32 so 4 bytes per element
    return base_offset + (m_idx * num_n_tiles + n_idx) * TILE_ELEMS * sizeof(int32_t);
}

// =============================================================================
// Core: planDenseGEMM
// =============================================================================

TilingPlan planDenseGEMM(const GEMMShape& shape,
                         const std::string& layer_name,
                         uint32_t wgt_base,
                         uint32_t act_base,
                         uint32_t out_base)
{
    TilingPlan plan;
    plan.layer_name = layer_name.empty() ? "gemm" : layer_name;
    plan.original   = shape;
    plan.padded     = computePaddedShape(shape);
    plan.grid       = computeTileGrid(plan.padded);

    plan.wgt_tile_bytes = TILE_ELEMS;                          // 196 bytes INT8
    plan.act_tile_bytes = TILE_ELEMS;                          // 196 bytes INT8
    plan.out_tile_bytes = TILE_ELEMS * sizeof(int32_t);        // 784 bytes INT32

    plan.wgt_base_offset = wgt_base;
    plan.act_base_offset = act_base;
    plan.out_base_offset = out_base;

    const uint32_t nM = plan.grid.num_m_tiles;
    const uint32_t nN = plan.grid.num_n_tiles;
    const uint32_t nK = plan.grid.num_k_tiles;

    // Reserve space for all tiles
    plan.tiles.reserve(static_cast<size_t>(nM) * nN * nK);

    // Generate tile descriptors
    // Loop order: M → N → K (output stationary)
    //   For each output block (m,n), we iterate over K tiles to accumulate
    //   the full dot product.
    for (uint32_t m = 0; m < nM; ++m) {
        for (uint32_t n = 0; n < nN; ++n) {
            for (uint32_t k = 0; k < nK; ++k) {
                memory::TileDescriptor td;
                td.m_idx = m;
                td.n_idx = n;
                td.k_idx = k;

                // Weight tile: row m, col k in the tiled weight matrix
                td.wgt_offset = weightTileOffset(m, k, nK, wgt_base);
                td.wgt_bytes  = plan.wgt_tile_bytes;

                // Activation tile: row k, col n in the tiled activation matrix
                td.act_offset = activationTileOffset(k, n, nN, act_base);
                td.act_bytes  = plan.act_tile_bytes;

                // First K → clear accumulators; last K → drain output
                td.is_first_k = (k == 0);
                td.is_last_k  = (k == nK - 1);
                td.wgt_cached = false;  // Default: no weight caching

                plan.tiles.push_back(td);
            }
        }
    }

    return plan;
}

// =============================================================================
// Core: planWeightStationaryGEMM (M → K → N loop order)
// =============================================================================

TilingPlan planWeightStationaryGEMM(const GEMMShape& shape,
                                     const std::string& layer_name,
                                     uint32_t wgt_base,
                                     uint32_t act_base,
                                     uint32_t out_base)
{
    TilingPlan plan;
    plan.layer_name = layer_name.empty() ? "gemm_ws" : layer_name;
    plan.original   = shape;
    plan.padded     = computePaddedShape(shape);
    plan.grid       = computeTileGrid(plan.padded);

    plan.wgt_tile_bytes = TILE_ELEMS;                          // 196 bytes INT8
    plan.act_tile_bytes = TILE_ELEMS;                          // 196 bytes INT8
    plan.out_tile_bytes = TILE_ELEMS * sizeof(int32_t);        // 784 bytes INT32

    plan.wgt_base_offset = wgt_base;
    plan.act_base_offset = act_base;
    plan.out_base_offset = out_base;

    const uint32_t nM = plan.grid.num_m_tiles;
    const uint32_t nN = plan.grid.num_n_tiles;
    const uint32_t nK = plan.grid.num_k_tiles;

    plan.tiles.reserve(static_cast<size_t>(nM) * nN * nK);

    // Weight-stationary loop order: M → K → N
    //   For each (m, k) weight tile, stream all N activation tiles through.
    //   The weight stays in the wgt_buffer ping-pong bank while N varies.
    //   Output partials for each (m, n) accumulate in on-chip BRAM across k.
    for (uint32_t m = 0; m < nM; ++m) {
        for (uint32_t k = 0; k < nK; ++k) {
            for (uint32_t n = 0; n < nN; ++n) {
                memory::TileDescriptor td;
                td.m_idx = m;
                td.n_idx = n;
                td.k_idx = k;

                td.wgt_offset = weightTileOffset(m, k, nK, wgt_base);
                td.wgt_bytes  = plan.wgt_tile_bytes;

                td.act_offset = activationTileOffset(k, n, nN, act_base);
                td.act_bytes  = plan.act_tile_bytes;

                // Accumulation: first k clears, last k drains output
                td.is_first_k = (k == 0);
                td.is_last_k  = (k == nK - 1);

                // Weight reuse: for a given (m, k), the weight tile is
                // loaded for n=0 and stays in the buffer for n>0.
                td.wgt_cached = (n > 0);

                plan.tiles.push_back(td);
            }
        }
    }

    return plan;
}

// =============================================================================
// Core: planConvGEMM
// =============================================================================

TilingPlan planConvGEMM(const ConvParams& conv,
                        const std::string& layer_name,
                        uint32_t wgt_base,
                        uint32_t act_base,
                        uint32_t out_base)
{
    // Convert convolution to GEMM via im2col
    GEMMShape gemm = conv.toGEMM();
    return planDenseGEMM(gemm, layer_name, wgt_base, act_base, out_base);
}

// =============================================================================
// Core: planSparseGEMM — BSR-aware tiling (only non-zero weight blocks)
// =============================================================================

TilingPlan planSparseGEMM(const BSRMatrix& bsr,
                          const GEMMShape& original,
                          const std::string& layer_name,
                          uint32_t wgt_base,
                          uint32_t act_base,
                          uint32_t out_base)
{
    TilingPlan plan;
    plan.layer_name    = layer_name.empty() ? "sparse_gemm" : layer_name;
    plan.original      = original;
    plan.padded        = computePaddedShape(original);
    plan.grid          = computeTileGrid(plan.padded);

    plan.wgt_tile_bytes = TILE_ELEMS;                    // 196 bytes INT8
    plan.act_tile_bytes = TILE_ELEMS;                    // 196 bytes INT8
    plan.out_tile_bytes = TILE_ELEMS * sizeof(int32_t);  // 784 bytes INT32

    plan.wgt_base_offset = wgt_base;
    plan.act_base_offset = act_base;
    plan.out_base_offset = out_base;

    // --- Sparse metadata ---
    plan.is_sparse         = true;
    plan.nnz_weight_blocks = bsr.nnz_blocks;

    const uint32_t nN = plan.grid.num_n_tiles;

    // Sanity: BSR block rows should match padded M / 14
    assert(bsr.num_block_rows == plan.grid.num_m_tiles &&
           "BSR block rows must match padded M-tiles");

    // Reserve: each NNZ block fans out across N-tiles
    plan.tiles.reserve(static_cast<size_t>(bsr.nnz_blocks) * nN);

    // ─── Weight-stationary loop: M → K_sparse → N ───────────────────────
    //
    // For each block-row m, iterate ONLY over non-zero K columns from BSR.
    // The hardware BSR scheduler does the same thing in bsr_scheduler.sv:
    //   it reads row_ptr[m], row_ptr[m+1] and iterates col_idx between them.
    //
    // Weight layout in DDR:  BSR values are packed sequentially —
    //   block i starts at wgt_base + i * 196.
    //   (This matches bsr_dma.sv which reads blocks at base + blk_idx*196.)
    //
    // Activation layout:  standard tiled layout (same as dense).
    //   tile(k, n) at act_base + (k * num_n_tiles + n) * 196.

    for (uint32_t m = 0; m < bsr.num_block_rows; ++m) {
        const uint32_t row_start = bsr.row_ptr[m];
        const uint32_t row_end   = bsr.row_ptr[m + 1];

        // If this block-row is entirely zero (all blocks pruned), no tiles
        // are emitted.  The output region should be pre-zeroed by the host.

        for (uint32_t nz = row_start; nz < row_end; ++nz) {
            const uint32_t k = bsr.col_idx[nz];   // Block-column of NNZ block

            for (uint32_t n = 0; n < nN; ++n) {
                memory::TileDescriptor td;
                td.m_idx = m;
                td.n_idx = n;
                td.k_idx = k;

                // Weight: packed BSR values — nz-th non-zero block
                td.wgt_offset = wgt_base + nz * TILE_ELEMS;
                td.wgt_bytes  = plan.wgt_tile_bytes;

                // Activation: standard tile layout (k, n)
                td.act_offset = activationTileOffset(k, n, nN, act_base);
                td.act_bytes  = plan.act_tile_bytes;

                // Accumulation control:
                //   is_first_k = first NNZ block in this block-row → clear ACC
                //   is_last_k  = last NNZ block in this block-row → drain output
                // This is correct because zero blocks would contribute 0 to the
                // accumulator — we can skip them without affecting the result.
                td.is_first_k = (nz == row_start);
                td.is_last_k  = (nz == row_end - 1);

                // Weight reuse across N: weight stays in buffer for n > 0
                td.wgt_cached = (n > 0);

                plan.tiles.push_back(td);
            }
        }
    }

    return plan;
}

// =============================================================================
// MNIST-Specific Layer Plans
// =============================================================================

namespace mnist {

GEMMShape conv1Shape() {
    // conv1: 1 input channel, 32 output channels, 3×3 kernel
    // Input: 28×28 → Output: 26×26 = 676 spatial positions
    // Weight: (32, 1*3*3) = (32, 9)
    // Activation: (9, 676)
    // GEMM: (32) × (9) × (676) → but for weight-focused tiling:
    //   M=32, K=9, N=676
    return GEMMShape{32, 676, 9};
}

GEMMShape conv2Shape() {
    // conv2: 32 input channels, 64 output channels, 3×3 kernel
    // Input: 13×13 (after conv1 26×26 → pool 13×13)
    // Output: 11×11 = 121 spatial positions
    // Weight: (64, 32*3*3) = (64, 288)
    // GEMM: M=64, K=288, N=121
    return GEMMShape{64, 121, 288};
}

GEMMShape fc1Shape() {
    // fc1: After conv2 11×11 → pool 5×5 → flatten = 64*5*5 = 1600
    // Actually from model_summary: weight (128, 9216) → must be 64*12*12=9216
    // (conv2 output 12×12 after pool to 6×6 → flatten = 64*12*12...
    //  OR: different architecture, trust model_summary)
    // Weight: (128, 9216),  batch=1 → N=1
    return GEMMShape{128, 1, 9216};
}

GEMMShape fc2Shape() {
    // fc2: Weight (10, 128), batch=1 → N=1
    return GEMMShape{10, 1, 128};
}

GEMMShape fc1BatchedShape(uint32_t batch) {
    // fc1 with batched inference: process `batch` images simultaneously
    // Weights (128, 9216) stay the same; N = batch instead of 1
    return GEMMShape{128, batch, 9216};
}

GEMMShape fc2BatchedShape(uint32_t batch) {
    // fc2 with batched inference: process `batch` images simultaneously
    // Weights (10, 128) stay the same; N = batch instead of 1
    return GEMMShape{10, batch, 128};
}

std::vector<TilingPlan> planAllLayers() {
    using namespace memory::ddr_layout;

    std::vector<TilingPlan> plans;
    plans.reserve(4);

    // ── Compute DDR offsets for sequential layer packing ─────────────────
    // Weights: packed sequentially from WEIGHTS_OFFSET
    // Activations: packed from ACTS_OFFSET
    // Outputs: packed from OUTPUT_OFFSET

    uint32_t wgt_cursor = WEIGHTS_OFFSET;
    uint32_t act_cursor = ACTS_OFFSET;
    uint32_t out_cursor = OUTPUT_OFFSET;

    // ── conv1 ────────────────────────────────────────────────────────────
    {
        auto plan = planDenseGEMM(conv1Shape(), "conv1",
                                  wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        // For subsequent layers, the output of conv1 becomes the activation
        // of conv2 (after pooling + im2col). We'll store activations at
        // the activation region for each layer.
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    // Re-base activation cursor for conv2 (conv1 output → pool → im2col)
    act_cursor = ACTS_OFFSET + plans[0].totalActivationBytes();

    // ── conv2 ────────────────────────────────────────────────────────────
    {
        auto plan = planDenseGEMM(conv2Shape(), "conv2",
                                  wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        act_cursor += plan.totalActivationBytes();
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    // ── fc1 ──────────────────────────────────────────────────────────────
    {
        auto plan = planDenseGEMM(fc1Shape(), "fc1",
                                  wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        act_cursor += plan.totalActivationBytes();
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    // ── fc2 ──────────────────────────────────────────────────────────────
    {
        auto plan = planDenseGEMM(fc2Shape(), "fc2",
                                  wgt_cursor, act_cursor, out_cursor);
        plans.push_back(std::move(plan));
    }

    return plans;
}

std::vector<TilingPlan> planAllLayersBatched(uint32_t batch_size,
                                              bool cache_fc2_weights) {
    using namespace memory::ddr_layout;

    std::vector<TilingPlan> plans;
    plans.reserve(4);

    uint32_t wgt_cursor = WEIGHTS_OFFSET;
    uint32_t act_cursor = ACTS_OFFSET;
    uint32_t out_cursor = OUTPUT_OFFSET;

    // ── conv1 (unchanged — already has large N from spatial output) ──────
    {
        auto plan = planDenseGEMM(conv1Shape(), "conv1",
                                  wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    act_cursor = ACTS_OFFSET + plans[0].totalActivationBytes();

    // ── conv2 (unchanged) ────────────────────────────────────────────────
    {
        auto plan = planDenseGEMM(conv2Shape(), "conv2",
                                  wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        act_cursor += plan.totalActivationBytes();
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    // ── fc1 (batched + weight-stationary: M→K→N → weight reuse across N tiles) ──
    {
        auto plan = planWeightStationaryGEMM(
            fc1BatchedShape(batch_size), "fc1_B" + std::to_string(batch_size),
            wgt_cursor, act_cursor, out_cursor);
        wgt_cursor += plan.totalWeightBytes();
        act_cursor += plan.totalActivationBytes();
        out_cursor += plan.totalOutputBytes();
        plans.push_back(std::move(plan));
    }

    // ── fc2 (batched + weight-stationary + optional weight caching) ─────
    {
        auto plan = planWeightStationaryGEMM(
            fc2BatchedShape(batch_size), "fc2_B" + std::to_string(batch_size),
            wgt_cursor, act_cursor, out_cursor);

        if (cache_fc2_weights) {
            // FC2 weights: 10×128 = 1,280 bytes total.
            // With weight-stationary scheduling, wgt_cached is already set
            // for n_idx > 0 by planWeightStationaryGEMM.  Additionally, if
            // we have enough BRAM to hold ALL fc2 weight tiles (~10 tiles =
            // 1,960 bytes), we can also cache across (m,k) pairs.
            // Mark ALL weight tiles as cached except the very first one
            // (which must be DMA'd to prime the cache).
            bool first = true;
            for (auto& td : plan.tiles) {
                if (first) { first = false; continue; }
                td.wgt_cached = true;
            }
        }

        plans.push_back(std::move(plan));
    }

    return plans;
}

std::vector<TilingPlan> planAllLayersSparse(const std::string& bsr_dir,
                                             uint32_t batch_size) {
    using namespace memory::ddr_layout;

    BSREncoder encoder;
    std::vector<TilingPlan> plans;
    plans.reserve(4);

    uint32_t wgt_cursor = WEIGHTS_OFFSET;
    uint32_t act_cursor = ACTS_OFFSET;
    uint32_t out_cursor = OUTPUT_OFFSET;

    // Layer names matching BSR export subdirectories
    struct LayerSpec {
        std::string name;
        GEMMShape   shape;
        bool        use_batch;   // FC layers get batched N
    };

    std::vector<LayerSpec> layers = {
        {"conv1", conv1Shape(),                     false},
        {"conv2", conv2Shape(),                     false},
        {"fc1",   fc1BatchedShape(batch_size),      true },
        {"fc2",   fc2BatchedShape(batch_size),      true },
    };

    for (const auto& spec : layers) {
        // Attempt to load BSR from export directory
        std::string layer_dir = bsr_dir + "/" + spec.name;
        BSRMatrix bsr;
        bool has_bsr = false;

        try {
            bsr = encoder.loadFromExport(layer_dir);
            has_bsr = true;
        } catch (...) {
            // No BSR export found — fall back to dense
            has_bsr = false;
        }

        TilingPlan plan;

        if (has_bsr && bsr.sparsity() > 0.01f) {
            // Significant sparsity → use sparse scheduling
            std::string sname = spec.name + "_sparse_B" + std::to_string(
                spec.use_batch ? batch_size : spec.shape.N);
            plan = planSparseGEMM(bsr, spec.shape, sname,
                                  wgt_cursor, act_cursor, out_cursor);
        } else if (spec.use_batch) {
            // FC layer with no/little sparsity → weight-stationary batched
            std::string bname = spec.name + "_B" + std::to_string(batch_size);
            plan = planWeightStationaryGEMM(spec.shape, bname,
                                            wgt_cursor, act_cursor, out_cursor);
        } else {
            // Conv layer → dense (output-stationary)
            plan = planDenseGEMM(spec.shape, spec.name,
                                 wgt_cursor, act_cursor, out_cursor);
        }

        wgt_cursor += plan.totalWeightBytes();
        act_cursor += plan.totalActivationBytes();
        out_cursor += plan.totalOutputBytes();

        plans.push_back(std::move(plan));
    }

    return plans;
}

} // namespace mnist

} // namespace compute
} // namespace accel
