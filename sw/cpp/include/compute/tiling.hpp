// tiling.hpp — Tile size calculation for systolic array mapping
// =============================================================================
//
// Maps arbitrary M×K × K×N GEMM operations onto a 14×14 weight-stationary
// systolic array.  Produces a flat array of TileDescriptors consumed by
// BufferManager::executeTileSequence().
//
// Tiling loop order:  for m in M-tiles → for n in N-tiles → for k in K-tiles
//   ▸ Outer M-N: each (m,n) pair produces one 14×14 output block in C.
//   ▸ Inner K:   partial products are accumulated in the output accumulator.
//                 is_first_k clears the accumulator, is_last_k triggers drain.
//
// Conv layers require im2col to convert (C_out, C_in*kH*kW) into a standard
// GEMM shape before tiling.
//
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <ostream>

// Forward-declare TileDescriptor from buffer_manager.hpp so users can include
// tiling.hpp without pulling in the full buffer manager.
namespace accel { namespace memory { struct TileDescriptor; } }

// Forward-declare BSRMatrix so callers can use planSparseGEMM without
// pulling in the full BSR encoder header.
namespace accel { namespace compute { struct BSRMatrix; } }

namespace accel {
namespace compute {

// =============================================================================
// Hardware constants
// =============================================================================
constexpr uint32_t SYSTOLIC_ROWS = 14;
constexpr uint32_t SYSTOLIC_COLS = 14;
constexpr uint32_t TILE_DIM      = 14;
constexpr uint32_t TILE_ELEMS    = TILE_DIM * TILE_DIM;  // 196

// =============================================================================
// GEMMShape — Describes an M×K × K×N matrix multiply
// =============================================================================
struct GEMMShape {
    uint32_t M;   // Rows of A / rows of C    (output features / spatial)
    uint32_t N;   // Cols of B / cols of C     (output features)
    uint32_t K;   // Cols of A / rows of B     (reduction dimension)
};

// =============================================================================
// ConvParams — Parameters for a single 2-D convolution layer
// =============================================================================
struct ConvParams {
    uint32_t C_in;          // Input channels
    uint32_t C_out;         // Output channels (filters)
    uint32_t kH;            // Kernel height
    uint32_t kW;            // Kernel width
    uint32_t H_in;          // Input feature map height
    uint32_t W_in;          // Input feature map width
    uint32_t stride;        // Stride (assumed square)
    uint32_t padding;       // Padding (assumed square)

    // Computed im2col GEMM shape: C_out × (C_in*kH*kW) matmul against
    // (C_in*kH*kW) × (H_out*W_out)
    GEMMShape toGEMM() const;

    // Output spatial dimensions
    uint32_t H_out() const;
    uint32_t W_out() const;
};

// =============================================================================
// PaddedShape — Padded dimensions (multiples of 14)
// =============================================================================
struct PaddedShape {
    uint32_t M_padded;      // ceil(M / 14) * 14
    uint32_t N_padded;      // ceil(N / 14) * 14
    uint32_t K_padded;      // ceil(K / 14) * 14
};

// =============================================================================
// TileGrid — Tile counts along each dimension
// =============================================================================
struct TileGrid {
    uint32_t num_m_tiles;   // ceil(M_padded / 14)
    uint32_t num_n_tiles;   // ceil(N_padded / 14)
    uint32_t num_k_tiles;   // ceil(K_padded / 14)

    uint32_t totalTiles() const { return num_m_tiles * num_n_tiles * num_k_tiles; }
};

// =============================================================================
// TilingPlan — Complete tiling strategy for one GEMM layer
// =============================================================================
struct TilingPlan {
    std::string layer_name;         // e.g. "conv1", "fc2"
    GEMMShape   original;           // Original (unpadded) shape
    PaddedShape padded;             // Padded to tile boundaries
    TileGrid    grid;               // Tile counts per dimension

    // Flat array of tile descriptors in execution order
    std::vector<memory::TileDescriptor> tiles;

    // Per-tile byte sizes (constant for all tiles in this plan)
    uint32_t wgt_tile_bytes;        // 14*14*sizeof(int8) = 196
    uint32_t act_tile_bytes;        // 14*14*sizeof(int8) = 196
    uint32_t out_tile_bytes;        // 14*14*sizeof(int32) = 784

    // DDR base offsets for this layer's data
    uint32_t wgt_base_offset;       // Byte offset into DDR weight region
    uint32_t act_base_offset;       // Byte offset into DDR activation region
    uint32_t out_base_offset;       // Byte offset into DDR output region

    // Sparse metadata (populated by planSparseGEMM; zero for dense plans)
    bool     is_sparse        = false;   // True if generated from BSR
    uint32_t nnz_weight_blocks = 0;      // Non-zero 14×14 weight blocks
    uint32_t dense_weight_blocks() const { return grid.num_m_tiles * grid.num_k_tiles; }
    float    weight_sparsity() const;

    // Summary
    uint32_t totalTiles() const;
    uint32_t totalWeightBytes() const;
    uint32_t totalActivationBytes() const;
    uint32_t totalOutputBytes() const;
    uint64_t totalMACs() const;

    /// Total DMA bytes (weights + activations + output).
    /// Accounts for wgt_cached tiles that skip weight DMA.
    uint64_t totalDMABytes() const;

    /// Arithmetic intensity for this layer (ops / DMA byte).
    double arithmeticIntensity() const;

    /// Effective throughput multiplier from sparsity.
    /// Dense: 1.0×.  At 90% sparse: ~10× (only 10% of tiles executed).
    double sparseSpeedup() const;

    // Print summary to stream
    void print(std::ostream& os) const;
};

// =============================================================================
// Padding helpers
// =============================================================================

/// Round up x to the nearest multiple of 14
constexpr uint32_t padTo14(uint32_t x) {
    return ((x + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
}

/// Compute padded shape from raw GEMM dimensions
PaddedShape computePaddedShape(const GEMMShape& shape);

/// Compute tile grid from padded shape
TileGrid computeTileGrid(const PaddedShape& padded);

// =============================================================================
// Core Tiling Functions
// =============================================================================

/// Compute a full tiling plan for a dense GEMM.
/// @param shape       Original M×K × K×N dimensions (unpadded)
/// @param layer_name  Human-readable name  (e.g. "fc1")
/// @param wgt_base    DDR byte offset for this layer's weights
/// @param act_base    DDR byte offset for activations
/// @param out_base    DDR byte offset for output
TilingPlan planDenseGEMM(const GEMMShape& shape,
                         const std::string& layer_name = "",
                         uint32_t wgt_base = 0,
                         uint32_t act_base = 0,
                         uint32_t out_base = 0);

/// Weight-stationary tile scheduling: M → K → N loop order.
/// Each weight tile is loaded once per (m, k) and reused across all N-tiles.
/// Requires on-chip output accumulators (partials stay in BRAM across k steps).
///
/// DMA model:
///   Weight: loaded once per (m,k)    → num_m × num_k tiles
///   Act:    loaded per (m,k,n)       → num_m × num_k × num_n tiles
///   Output: written on last k only   → num_m × num_n tiles (INT32)
///
/// This ordering is ideal for FC layers where weight matrices dominate DMA.
TilingPlan planWeightStationaryGEMM(const GEMMShape& shape,
                                     const std::string& layer_name = "",
                                     uint32_t wgt_base = 0,
                                     uint32_t act_base = 0,
                                     uint32_t out_base = 0);

/// Compute a full tiling plan for a convolution layer (via im2col GEMM).
/// @param conv  Conv layer parameters
/// @param layer_name  Human-readable name
/// @param wgt_base    DDR weight offset
/// @param act_base    DDR activation offset
/// @param out_base    DDR output offset
TilingPlan planConvGEMM(const ConvParams& conv,
                        const std::string& layer_name = "",
                        uint32_t wgt_base = 0,
                        uint32_t act_base = 0,
                        uint32_t out_base = 0);

/// Plan tiles for a sparse GEMM using a BSR-encoded weight matrix.
/// Only generates tile descriptors for NON-ZERO weight blocks — the hardware
/// BSR scheduler (bsr_scheduler.sv) skips zero blocks entirely.
///
/// Uses weight-stationary (M→K_sparse→N) loop order:
///   for m in 0..num_block_rows-1:
///       for nz in row_ptr[m]..row_ptr[m+1]-1:     // only NZ blocks!
///           k = col_idx[nz]
///           for n in 0..num_n_tiles-1:
///               emit tile(m, n, k)   [weight at BSR values offset]
///
/// DMA model:
///   Weight:  loaded once per (m, k_nz)       → nnz_blocks transfers
///   Act:     loaded per (m, k_nz, n)         → nnz_blocks × num_n tiles
///   Output:  written on last k per (m, n)    → num_m × num_n tiles
///   Metadata: row_ptr + col_idx loaded once   → small, ~KB
///
/// At 90% block sparsity, this generates ~10× fewer tiles than planDenseGEMM.
///
/// @param bsr         BSR-encoded weight matrix (from BSREncoder)
/// @param original    Original unpadded GEMM shape (M, N, K)
/// @param layer_name  Human-readable name
/// @param wgt_base    DDR offset for BSR weight values (packed NZ blocks)
/// @param act_base    DDR offset for activations (standard tile layout)
/// @param out_base    DDR offset for output (standard tile layout)
TilingPlan planSparseGEMM(const BSRMatrix& bsr,
                          const GEMMShape& original,
                          const std::string& layer_name = "",
                          uint32_t wgt_base = 0,
                          uint32_t act_base = 0,
                          uint32_t out_base = 0);

// =============================================================================
// MNIST-Specific Layer Plans
// =============================================================================

/// Pre-built layer descriptors for the MNIST CNN.
/// These match the model_summary.json from BSR export.
///
///   conv1: weight (32,9)   → padded (42,14),   3 blocks
///   conv2: weight (64,288) → padded (70,294), 105 blocks
///   fc1:   weight (128,9216)→ padded (140,9226), 6590 blocks
///   fc2:   weight (10,128) → padded (14,140),  10 blocks
namespace mnist {

    /// Layer shapes (weight matrix M×K view after im2col)
    GEMMShape conv1Shape();  // M=32,  K=9,    N=576  (24×24 output)
    GEMMShape conv2Shape();  // M=64,  K=288,  N=144  (12×12 → 5×5 pool → but post-pool spatial)
    GEMMShape fc1Shape();    // M=128, K=9216, N=1
    GEMMShape fc2Shape();    // M=10,  K=128,  N=1

    /// Batched FC shapes — N=batch_size fills the systolic array's N dimension.
    /// With batch=14, each weight tile is reused 14× → arithmetic intensity
    /// increases from ~2 ops/byte (memory-bound) to ~25 ops/byte (compute-bound).
    GEMMShape fc1BatchedShape(uint32_t batch = TILE_DIM);
    GEMMShape fc2BatchedShape(uint32_t batch = TILE_DIM);

    /// Compute tiling plans for all 4 layers with sequential DDR layout.
    /// Returns a vector of 4 TilingPlans in order: conv1, conv2, fc1, fc2.
    std::vector<TilingPlan> planAllLayers();

    /// Compute tiling plans with batched FC inference.
    /// @param batch_size  Number of images to process simultaneously for FC layers.
    ///                    Default 28 gives 2 N-tiles, enabling weight reuse in
    ///                    weight-stationary (M→K→N) scheduling.
    ///                    Conv layers keep their original N (spatial output).
    /// @param cache_fc2_weights  If true, mark FC2 weight tiles as wgt_cached
    ///                           after the first N-tile (only 1,280 bytes — fits in BRAM).
    std::vector<TilingPlan> planAllLayersBatched(
        uint32_t batch_size = 2 * TILE_DIM,
        bool cache_fc2_weights = true);

    /// Plan all MNIST layers using BSR sparse weights from export directory.
    /// Loads per-layer BSR files (row_ptr.npy, col_idx.npy, weights.bsr),
    /// generates sparse tile schedules for FC layers (where sparsity is high)
    /// and dense schedules for conv layers (which are typically dense).
    ///
    /// @param bsr_dir     Path to BSR export (e.g. "data/bsr_export_14x14")
    /// @param batch_size  Batch size for FC layers (default 28 for compute-bound)
    std::vector<TilingPlan> planAllLayersSparse(
        const std::string& bsr_dir,
        uint32_t batch_size = 2 * TILE_DIM);

} // namespace mnist

// =============================================================================
// Utility
// =============================================================================

/// Compute the DDR byte offset for a specific tile within a tiled weight matrix.
/// Weight matrix is laid out in row-major tile order: tile(m,k) at
///   offset = (m * num_k_tiles + k) * 196
uint32_t weightTileOffset(uint32_t m_idx, uint32_t k_idx,
                          uint32_t num_k_tiles, uint32_t base_offset = 0);

/// Compute the DDR byte offset for a specific activation tile.
/// Activation matrix column-tiles: tile(k,n) at
///   offset = (k * num_n_tiles + n) * 196
uint32_t activationTileOffset(uint32_t k_idx, uint32_t n_idx,
                              uint32_t num_n_tiles, uint32_t base_offset = 0);

/// Compute the DDR byte offset for a specific output tile.
/// Output matrix: tile(m,n) at offset = (m * num_n_tiles + n) * 784
uint32_t outputTileOffset(uint32_t m_idx, uint32_t n_idx,
                          uint32_t num_n_tiles, uint32_t base_offset = 0);

} // namespace compute
} // namespace accel
