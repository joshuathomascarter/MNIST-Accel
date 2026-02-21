// bsr_encoder.cpp — BSR sparse matrix encoder/decoder implementation
// =============================================================================
//
// Encodes dense INT8 weight matrices into Block Sparse Row (BSR) format for
// the hardware BSR scheduler (bsr_scheduler.sv).  Supports:
//   - Dense → BSR encoding with configurable zero-block threshold
//   - BSR → Dense decoding for golden-model verification
//   - Flat byte packing for DMA staging
//   - File I/O compatible with the Python BSR export pipeline
//
// =============================================================================
#include "compute/bsr_encoder.hpp"

#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace accel {
namespace compute {

// =============================================================================
// BSRMatrix helpers
// =============================================================================

float BSRMatrix::density() const {
    uint32_t total = num_block_rows * num_block_cols;
    if (total == 0) return 0.0f;
    return static_cast<float>(nnz_blocks) / static_cast<float>(total);
}

void BSRMatrix::print(std::ostream& os) const {
    os << "BSRMatrix: " << num_block_rows << "×" << num_block_cols
       << " blocks (" << (num_block_rows * BSR_BLOCK_DIM) << "×"
       << (num_block_cols * BSR_BLOCK_DIM) << " elements)\n";
    os << "  NNZ blocks: " << nnz_blocks
       << " / " << (num_block_rows * num_block_cols)
       << " (density=" << std::fixed << std::setprecision(2)
       << (density() * 100.0f) << "%)\n";
    os << "  row_ptr bytes:  " << rowPtrBytes() << "\n";
    os << "  col_idx bytes:  " << colIdxBytes() << "\n";
    os << "  values bytes:   " << valuesBytes() << "\n";
    os << "  total bytes:    " << totalBytes() << "\n";
}

// =============================================================================
// BSREncoder: Zero-block check
// =============================================================================

bool BSREncoder::isBlockZero(const int8_t* block_data, uint32_t block_size) const {
    for (uint32_t i = 0; i < block_size; ++i) {
        int8_t val = block_data[i];
        // Check absolute value against threshold
        if (val > zero_threshold_ || val < -zero_threshold_) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// BSREncoder: Dense → BSR encoding
// =============================================================================

BSRMatrix BSREncoder::encode(const int8_t* data, uint32_t rows, uint32_t cols) const {
    // Pad to multiples of 14
    uint32_t padded_rows = ((rows + BSR_BLOCK_DIM - 1) / BSR_BLOCK_DIM) * BSR_BLOCK_DIM;
    uint32_t padded_cols = ((cols + BSR_BLOCK_DIM - 1) / BSR_BLOCK_DIM) * BSR_BLOCK_DIM;

    uint32_t nbr = padded_rows / BSR_BLOCK_DIM;  // number of block rows
    uint32_t nbc = padded_cols / BSR_BLOCK_DIM;   // number of block cols

    BSRMatrix bsr;
    bsr.num_block_rows = nbr;
    bsr.num_block_cols = nbc;
    bsr.row_ptr.resize(nbr + 1, 0);

    // Temporary padded matrix (zero-filled for padding)
    std::vector<int8_t> padded(padded_rows * padded_cols, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        std::memcpy(&padded[r * padded_cols], &data[r * cols],
                    cols * sizeof(int8_t));
    }

    // First pass: count non-zero blocks per block-row
    // Second pass: fill col_idx and values
    // (We do it in one pass with push_back for simplicity)

    uint32_t nnz = 0;
    std::vector<int8_t> block_buf(BSR_BLOCK_SIZE);

    for (uint32_t br = 0; br < nbr; ++br) {
        bsr.row_ptr[br] = nnz;

        for (uint32_t bc = 0; bc < nbc; ++bc) {
            // Extract the 14×14 block at (br, bc)
            for (uint32_t i = 0; i < BSR_BLOCK_DIM; ++i) {
                uint32_t src_row = br * BSR_BLOCK_DIM + i;
                uint32_t src_col = bc * BSR_BLOCK_DIM;
                std::memcpy(&block_buf[i * BSR_BLOCK_DIM],
                            &padded[src_row * padded_cols + src_col],
                            BSR_BLOCK_DIM * sizeof(int8_t));
            }

            // Check if block is non-zero
            if (!isBlockZero(block_buf.data(), BSR_BLOCK_SIZE)) {
                bsr.col_idx.push_back(bc);
                bsr.values.insert(bsr.values.end(),
                                  block_buf.begin(), block_buf.end());
                ++nnz;
            }
        }
    }

    bsr.row_ptr[nbr] = nnz;
    bsr.nnz_blocks = nnz;

    return bsr;
}

BSRMatrix BSREncoder::encode(const std::vector<int8_t>& data,
                             uint32_t rows, uint32_t cols) const {
    if (data.size() < static_cast<size_t>(rows) * cols) {
        throw BSRException("Data vector too small: expected " +
                          std::to_string(static_cast<size_t>(rows) * cols) +
                          " but got " + std::to_string(data.size()));
    }
    return encode(data.data(), rows, cols);
}

// =============================================================================
// BSREncoder: BSR → Dense decoding
// =============================================================================

std::vector<int8_t> BSREncoder::decode(const BSRMatrix& bsr,
                                       uint32_t orig_rows,
                                       uint32_t orig_cols) const {
    uint32_t padded_rows = bsr.num_block_rows * BSR_BLOCK_DIM;
    uint32_t padded_cols = bsr.num_block_cols * BSR_BLOCK_DIM;

    // Allocate padded dense matrix (zeros)
    std::vector<int8_t> dense(padded_rows * padded_cols, 0);

    // Fill in non-zero blocks
    for (uint32_t br = 0; br < bsr.num_block_rows; ++br) {
        uint32_t start = bsr.row_ptr[br];
        uint32_t end   = bsr.row_ptr[br + 1];

        for (uint32_t idx = start; idx < end; ++idx) {
            uint32_t bc = bsr.col_idx[idx];
            const int8_t* block = &bsr.values[idx * BSR_BLOCK_SIZE];

            // Copy block back to dense matrix
            for (uint32_t i = 0; i < BSR_BLOCK_DIM; ++i) {
                uint32_t dst_row = br * BSR_BLOCK_DIM + i;
                uint32_t dst_col = bc * BSR_BLOCK_DIM;
                std::memcpy(&dense[dst_row * padded_cols + dst_col],
                            &block[i * BSR_BLOCK_DIM],
                            BSR_BLOCK_DIM * sizeof(int8_t));
            }
        }
    }

    // Extract original (unpadded) dimensions
    std::vector<int8_t> result(orig_rows * orig_cols, 0);
    for (uint32_t r = 0; r < orig_rows; ++r) {
        std::memcpy(&result[r * orig_cols],
                    &dense[r * padded_cols],
                    orig_cols * sizeof(int8_t));
    }

    return result;
}

// =============================================================================
// BSREncoder: File I/O (compatible with Python BSR export format)
// =============================================================================

// Minimal NPY header parser for uint32 and int8 arrays
namespace {

struct NpyHeader {
    bool     fortran_order;
    uint32_t ndim;
    uint64_t shape[4];
    char     dtype;      // 'u' for uint32, 'i' for int8
    uint32_t dtype_size;
};

NpyHeader parseNpyHeader(std::ifstream& f) {
    // Read magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw BSRException("Invalid NPY magic bytes");
    }

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        f.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    // Read header string
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    NpyHeader h{};
    h.fortran_order = false;
    h.ndim = 1;

    // Parse dtype
    auto dtype_pos = header.find("'descr'");
    if (dtype_pos == std::string::npos) dtype_pos = header.find("\"descr\"");
    if (dtype_pos != std::string::npos) {
        // Find the dtype string like '<i1' or '<u4' or '<i4'
        auto q1 = header.find('\'', dtype_pos + 7);
        if (q1 == std::string::npos) q1 = header.find('"', dtype_pos + 7);
        auto q2 = header.find('\'', q1 + 1);
        if (q2 == std::string::npos) q2 = header.find('"', q1 + 1);

        std::string dtype_str = header.substr(q1 + 1, q2 - q1 - 1);
        // e.g. "<i1", "<u4", "<i4"
        if (dtype_str.size() >= 3) {
            h.dtype = dtype_str[1];
            h.dtype_size = dtype_str[2] - '0';
        }
    }

    // Parse shape
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\"");
    if (shape_pos != std::string::npos) {
        auto paren1 = header.find('(', shape_pos);
        auto paren2 = header.find(')', paren1);
        std::string shape_str = header.substr(paren1 + 1, paren2 - paren1 - 1);

        h.ndim = 0;
        std::istringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t,") + 1);
            if (!token.empty()) {
                h.shape[h.ndim++] = std::stoull(token);
            }
        }
    }

    return h;
}

template <typename T>
std::vector<T> loadNpy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw BSRException("Cannot open NPY file: " + path);
    }

    NpyHeader hdr = parseNpyHeader(f);

    uint64_t num_elements = 1;
    for (uint32_t i = 0; i < hdr.ndim; ++i) {
        num_elements *= hdr.shape[i];
    }

    std::vector<T> data(num_elements);
    f.read(reinterpret_cast<char*>(data.data()),
           num_elements * sizeof(T));

    return data;
}

} // anonymous namespace

BSRMatrix BSREncoder::loadFromExport(const std::string& dir) const {
    BSRMatrix bsr;

    // Load row_ptr (uint32)
    bsr.row_ptr = loadNpy<uint32_t>(dir + "/row_ptr.npy");

    // Load col_idx (uint32)  — might be stored as int32
    bsr.col_idx = loadNpy<uint32_t>(dir + "/col_idx.npy");

    // Load block values (int8)
    bsr.values = loadNpy<int8_t>(dir + "/weights.bsr");

    // Derive dimensions
    bsr.num_block_rows = static_cast<uint32_t>(bsr.row_ptr.size()) - 1;
    bsr.nnz_blocks     = static_cast<uint32_t>(bsr.col_idx.size());

    // Infer num_block_cols from max col_idx + 1
    if (!bsr.col_idx.empty()) {
        bsr.num_block_cols = *std::max_element(bsr.col_idx.begin(),
                                                bsr.col_idx.end()) + 1;
    } else {
        bsr.num_block_cols = 0;
    }

    // Validate
    if (bsr.values.size() != static_cast<size_t>(bsr.nnz_blocks) * BSR_BLOCK_SIZE) {
        throw BSRException("Values size mismatch: expected " +
                          std::to_string(bsr.nnz_blocks * BSR_BLOCK_SIZE) +
                          " got " + std::to_string(bsr.values.size()));
    }

    return bsr;
}

void BSREncoder::saveToExport(const BSRMatrix& bsr, const std::string& dir) const {
    // Helper to write a simple NPY v1.0 file
    auto writeNpy = [](const std::string& path, const void* data,
                       size_t num_elements, const std::string& dtype_str,
                       size_t element_size) {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            throw BSRException("Cannot create NPY file: " + path);
        }

        // Build header dict
        std::string header = "{'descr': '" + dtype_str +
                            "', 'fortran_order': False, 'shape': (" +
                            std::to_string(num_elements) + ",), }";
        // Pad to align to 64 bytes (NPY spec: header_len + magic(10) = multiple of 64)
        size_t total_prefix = 10 + header.size();
        size_t pad = 64 - (total_prefix % 64);
        if (pad == 64) pad = 0;
        header.append(pad, ' ');
        header.back() = '\n';

        // Write magic + version
        f.write("\x93NUMPY", 6);
        uint8_t ver[2] = {1, 0};
        f.write(reinterpret_cast<char*>(ver), 2);
        uint16_t hlen = static_cast<uint16_t>(header.size());
        f.write(reinterpret_cast<char*>(&hlen), 2);
        f.write(header.data(), header.size());

        // Write data
        f.write(reinterpret_cast<const char*>(data), num_elements * element_size);
    };

    writeNpy(dir + "/row_ptr.npy", bsr.row_ptr.data(),
             bsr.row_ptr.size(), "<u4", 4);
    writeNpy(dir + "/col_idx.npy", bsr.col_idx.data(),
             bsr.col_idx.size(), "<u4", 4);
    writeNpy(dir + "/weights.bsr", bsr.values.data(),
             bsr.values.size(), "<i1", 1);
}

// =============================================================================
// BSREncoder: Pack for DMA
// =============================================================================

BSREncoder::PackedBuffers BSREncoder::pack(const BSRMatrix& bsr) const {
    PackedBuffers buf;

    // Metadata: row_ptr + col_idx concatenated
    size_t meta_size = bsr.rowPtrBytes() + bsr.colIdxBytes();
    buf.metadata.resize(meta_size);

    std::memcpy(buf.metadata.data(),
                bsr.row_ptr.data(), bsr.rowPtrBytes());
    std::memcpy(buf.metadata.data() + bsr.rowPtrBytes(),
                bsr.col_idx.data(), bsr.colIdxBytes());

    // Weights: block values directly
    buf.weights = bsr.values;

    return buf;
}

} // namespace compute
} // namespace accel
