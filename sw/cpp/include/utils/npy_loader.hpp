// npy_loader.hpp — Load/save NumPy .npy arrays in C++
// =============================================================================
//
// Supports the most common dtypes used in the accelerator pipeline:
//   - int8   (<i1)  — quantised weights and activations
//   - int32  (<i4)  — biases and accumulator outputs
//   - uint32 (<u4)  — BSR row_ptr, col_idx
//   - float32(<f4)  — scales, logits
//   - float64(<f8)  — some numpy defaults
//
// Supports NPY format v1.0 and v2.0, 1-D and 2-D arrays, C-order only.
//
// =============================================================================
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace accel {
namespace utils {

// =============================================================================
// NpyArray — Typed array loaded from .npy file
// =============================================================================
template <typename T>
struct NpyArray {
    std::vector<T> data;
    std::vector<uint64_t> shape;   // e.g. {128, 9216} for a 2D array

    /// Total number of elements
    size_t size() const { return data.size(); }

    /// Number of dimensions
    size_t ndim() const { return shape.size(); }

    /// Rows (dim 0) — returns 1 for 1-D arrays
    uint64_t rows() const { return shape.empty() ? 0 : shape[0]; }

    /// Cols (dim 1) — returns 1 for 1-D arrays
    uint64_t cols() const { return shape.size() >= 2 ? shape[1] : 1; }
};

// =============================================================================
// NpyLoader — Static methods for loading/saving .npy files
// =============================================================================
class NpyLoader {
public:
    // -------------------------------------------------------------------------
    // Type-specific loaders
    // -------------------------------------------------------------------------

    static NpyArray<int8_t>   loadInt8(const std::string& path);
    static NpyArray<uint8_t>  loadUint8(const std::string& path);
    static NpyArray<int32_t>  loadInt32(const std::string& path);
    static NpyArray<uint32_t> loadUint32(const std::string& path);
    static NpyArray<float>    loadFloat32(const std::string& path);
    static NpyArray<double>   loadFloat64(const std::string& path);

    // -------------------------------------------------------------------------
    // Generic loader — auto-detects dtype and casts to T
    // -------------------------------------------------------------------------
    template <typename T>
    static NpyArray<T> load(const std::string& path);

    // -------------------------------------------------------------------------
    // Type-specific savers
    // -------------------------------------------------------------------------

    static void saveInt8(const std::string& path, const int8_t* data,
                         const std::vector<uint64_t>& shape);
    static void saveInt32(const std::string& path, const int32_t* data,
                          const std::vector<uint64_t>& shape);
    static void saveUint32(const std::string& path, const uint32_t* data,
                           const std::vector<uint64_t>& shape);
    static void saveFloat32(const std::string& path, const float* data,
                            const std::vector<uint64_t>& shape);

    // -------------------------------------------------------------------------
    // Convenience: save from vector
    // -------------------------------------------------------------------------

    template <typename T>
    static void save(const std::string& path, const NpyArray<T>& arr);

    // -------------------------------------------------------------------------
    // Metadata query (without loading data)
    // -------------------------------------------------------------------------

    struct NpyInfo {
        std::string dtype;           // e.g. "<i1", "<f4"
        uint32_t    dtype_size;      // bytes per element
        bool        fortran_order;
        std::vector<uint64_t> shape;
    };

    static NpyInfo info(const std::string& path);

    // Internal header parser (public for use in implementation)
    struct Header {
        std::string dtype_str;    // e.g. "<i1"
        char dtype_char;          // 'i', 'u', 'f'
        uint32_t dtype_size;      // 1, 4, 8
        bool fortran_order;
        std::vector<uint64_t> shape;
        size_t data_offset;       // byte offset where data begins
    };

    static Header parseHeader(const std::string& path);
    static void writeHeader(FILE* f, const std::string& dtype_str,
                            const std::vector<uint64_t>& shape);

private:
};

// =============================================================================
// Exception
// =============================================================================
class NpyException : public std::runtime_error {
public:
    explicit NpyException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace utils
} // namespace accel
