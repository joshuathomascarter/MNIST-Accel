// bsr_encoder.hpp — BSR (Block Sparse Row) encoder for hardware scheduler
// =============================================================================
//
// Converts dense or sparse weight matrices into BSR format for consumption
// by hw/rtl/systolic/bsr_scheduler.sv.
//
// BSR Format (14×14 block size):
//   row_ptr[num_block_rows + 1] : cumulative count of non-zero blocks per row
//   col_idx[nnz_blocks]          : column index (block column) for each NZ block
//   values[nnz_blocks × 196]     : flattened 14×14 INT8 blocks, row-major
//
// The hardware scheduler reads row_ptr and col_idx from DDR to determine which
// blocks to load, then streams the corresponding weight blocks to the PE array.
//
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <ostream>
#include <stdexcept>

namespace accel {
namespace compute {

constexpr uint32_t BSR_BLOCK_DIM = 14;
constexpr uint32_t BSR_BLOCK_SIZE = BSR_BLOCK_DIM * BSR_BLOCK_DIM;  // 196

// =============================================================================
// BSRMatrix — In-memory representation of a BSR-encoded weight matrix
// =============================================================================
struct BSRMatrix {
    uint32_t num_block_rows;    // M_padded / 14
    uint32_t num_block_cols;    // K_padded / 14
    uint32_t nnz_blocks;        // Number of non-zero 14×14 blocks

    // CSR-like structure at the block level
    std::vector<uint32_t> row_ptr;     // [num_block_rows + 1], cumulative NZ count
    std::vector<uint32_t> col_idx;     // [nnz_blocks], column index per NZ block
    std::vector<int8_t>   values;      // [nnz_blocks * 196], flattened block data

    // Density = nnz_blocks / (num_block_rows * num_block_cols)
    float density() const;
    float sparsity() const { return 1.0f - density(); }

    // Total bytes for DMA transfer
    size_t rowPtrBytes()  const { return row_ptr.size() * sizeof(uint32_t); }
    size_t colIdxBytes()  const { return col_idx.size() * sizeof(uint32_t); }
    size_t valuesBytes()  const { return values.size() * sizeof(int8_t); }
    size_t totalBytes()   const { return rowPtrBytes() + colIdxBytes() + valuesBytes(); }

    // Print summary
    void print(std::ostream& os) const;
};

// =============================================================================
// BSREncoder — Converts dense weight matrices to BSR format
// =============================================================================
class BSREncoder {
public:
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Sparsity threshold: blocks with all-zero magnitudes below this are sparse
    /// Default: 0 (only exact-zero blocks are pruned)
    void setZeroThreshold(int8_t threshold) { zero_threshold_ = threshold; }
    int8_t zeroThreshold() const { return zero_threshold_; }

    // -------------------------------------------------------------------------
    // Encoding
    // -------------------------------------------------------------------------

    /// Encode a dense INT8 weight matrix into BSR format.
    /// @param data    Pointer to row-major INT8 weight matrix
    /// @param rows    Number of rows (will be padded to multiple of 14)
    /// @param cols    Number of cols (will be padded to multiple of 14)
    /// @return BSRMatrix with row_ptr, col_idx, and values populated
    BSRMatrix encode(const int8_t* data, uint32_t rows, uint32_t cols) const;

    /// Encode from a std::vector (convenience)
    BSRMatrix encode(const std::vector<int8_t>& data,
                     uint32_t rows, uint32_t cols) const;

    // -------------------------------------------------------------------------
    // Decoding (for verification)
    // -------------------------------------------------------------------------

    /// Decode BSR back to dense matrix (original unpadded dimensions)
    /// @param bsr     BSR-encoded matrix
    /// @param orig_rows  Original (unpadded) row count
    /// @param orig_cols  Original (unpadded) col count
    /// @return Dense row-major INT8 matrix [orig_rows × orig_cols]
    std::vector<int8_t> decode(const BSRMatrix& bsr,
                               uint32_t orig_rows, uint32_t orig_cols) const;

    // -------------------------------------------------------------------------
    // File I/O — compatible with Python BSR export format
    // -------------------------------------------------------------------------

    /// Load BSR from exported numpy files (row_ptr.npy, col_idx.npy, weights.bsr)
    /// @param dir  Directory containing the exported files
    BSRMatrix loadFromExport(const std::string& dir) const;

    /// Save BSR to directory in the same export format
    void saveToExport(const BSRMatrix& bsr, const std::string& dir) const;

    // -------------------------------------------------------------------------
    // Serialisation — flat byte buffer for DMA staging
    // -------------------------------------------------------------------------

    /// Pack BSR metadata + values into a flat byte buffer suitable for
    /// DMA transfer to DDR.  Layout:
    ///   [0..rowPtrBytes-1]                       : row_ptr
    ///   [rowPtrBytes..rowPtrBytes+colIdxBytes-1] : col_idx
    /// Values are transferred separately to the weight region.
    struct PackedBuffers {
        std::vector<uint8_t> metadata;   // row_ptr + col_idx concatenated
        std::vector<int8_t>  weights;    // block values (same as bsr.values)
    };

    PackedBuffers pack(const BSRMatrix& bsr) const;

private:
    int8_t zero_threshold_ = 0;

    /// Check if a 14×14 block is "zero" (all |values| <= threshold)
    bool isBlockZero(const int8_t* block_data, uint32_t block_size) const;
};

// =============================================================================
// Exception
// =============================================================================
class BSRException : public std::runtime_error {
public:
    explicit BSRException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace compute
} // namespace accel
