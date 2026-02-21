// npy_loader.cpp — NumPy .npy file loader/saver implementation
// =============================================================================
//
// Implements NPY format version 1.0 and 2.0 parsing.
// NPY format specification:
//   - Magic: \x93NUMPY
//   - Version: 1 byte major, 1 byte minor
//   - Header length: 2 bytes (v1) or 4 bytes (v2)
//   - Header: Python dict literal with 'descr', 'fortran_order', 'shape'
//   - Data: raw binary, dtype-endian
//
// =============================================================================
#include "utils/npy_loader.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace accel {
namespace utils {

// =============================================================================
// Header parsing
// =============================================================================

NpyLoader::Header NpyLoader::parseHeader(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw NpyException("Cannot open file: " + path);
    }

    // Read magic: \x93NUMPY (6 bytes)
    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6) {
        std::fclose(f);
        throw NpyException("Failed to read NPY magic: " + path);
    }

    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        std::fclose(f);
        throw NpyException("Invalid NPY magic bytes: " + path);
    }

    // Version
    uint8_t major_ver = 0, minor_ver = 0;
    std::fread(&major_ver, 1, 1, f);
    std::fread(&minor_ver, 1, 1, f);

    // Header length
    uint32_t header_len = 0;
    if (major_ver == 1) {
        uint16_t len16 = 0;
        std::fread(&len16, 1, 2, f);
        header_len = len16;
    } else if (major_ver >= 2) {
        std::fread(&header_len, 1, 4, f);
    } else {
        std::fclose(f);
        throw NpyException("Unsupported NPY version: " +
                          std::to_string(major_ver) + "." +
                          std::to_string(minor_ver));
    }

    // Read header string
    std::string header(header_len, '\0');
    if (std::fread(&header[0], 1, header_len, f) != header_len) {
        std::fclose(f);
        throw NpyException("Failed to read NPY header: " + path);
    }

    // Record data offset
    size_t data_offset = static_cast<size_t>(std::ftell(f));
    std::fclose(f);

    // Parse header dict
    Header h{};
    h.data_offset = data_offset;
    h.fortran_order = false;

    // Parse 'descr': '<i1', '<u4', '<f4', etc.
    auto find_value = [&](const std::string& key) -> std::string {
        auto pos = header.find("'" + key + "'");
        if (pos == std::string::npos) {
            pos = header.find("\"" + key + "\"");
        }
        if (pos == std::string::npos) return "";

        auto colon = header.find(':', pos);
        if (colon == std::string::npos) return "";

        // Find the value start
        auto start = header.find_first_not_of(" \t", colon + 1);
        if (start == std::string::npos) return "";

        // Is it a string value (in quotes)?
        if (header[start] == '\'' || header[start] == '"') {
            char quote = header[start];
            auto end = header.find(quote, start + 1);
            if (end == std::string::npos) return "";
            return header.substr(start + 1, end - start - 1);
        }
        // Bool or tuple
        auto end = header.find_first_of(",}", start);
        return header.substr(start, end - start);
    };

    // Dtype
    h.dtype_str = find_value("descr");
    if (h.dtype_str.size() >= 3) {
        h.dtype_char = h.dtype_str[1];  // 'i', 'u', 'f', 'b'
        h.dtype_size = static_cast<uint32_t>(h.dtype_str[2] - '0');
    } else if (h.dtype_str.size() >= 2) {
        h.dtype_char = h.dtype_str[0];
        h.dtype_size = static_cast<uint32_t>(h.dtype_str[1] - '0');
    }

    // Fortran order
    std::string fo = find_value("fortran_order");
    h.fortran_order = (fo.find("True") != std::string::npos);
    if (h.fortran_order) {
        throw NpyException("Fortran order not supported: " + path);
    }

    // Shape
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\"");
    if (shape_pos != std::string::npos) {
        auto p1 = header.find('(', shape_pos);
        auto p2 = header.find(')', p1);
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::string shape_str = header.substr(p1 + 1, p2 - p1 - 1);
            std::istringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // Trim
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t,") + 1);
                if (!token.empty()) {
                    h.shape.push_back(std::stoull(token));
                }
            }
        }
    }

    return h;
}

void NpyLoader::writeHeader(FILE* f, const std::string& dtype_str,
                              const std::vector<uint64_t>& shape) {
    // Build header dict
    std::ostringstream ss;
    ss << "{'descr': '" << dtype_str
       << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size() || shape.size() == 1) ss << ",";
    }
    ss << "), }";
    std::string hdr = ss.str();

    // Pad to align data to 64 bytes total
    // Magic(6) + version(2) + header_len(2) + header = N bytes
    size_t prefix = 6 + 2 + 2;
    size_t total = prefix + hdr.size();
    size_t pad = 64 - (total % 64);
    if (pad == 64) pad = 0;
    hdr.append(pad, ' ');
    hdr.back() = '\n';

    // Write magic
    std::fwrite("\x93NUMPY", 1, 6, f);

    // Version 1.0
    uint8_t ver[2] = {1, 0};
    std::fwrite(ver, 1, 2, f);

    // Header length
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    std::fwrite(&hlen, 1, 2, f);

    // Header string
    std::fwrite(hdr.data(), 1, hdr.size(), f);
}

// =============================================================================
// Info query
// =============================================================================

NpyLoader::NpyInfo NpyLoader::info(const std::string& path) {
    auto h = parseHeader(path);
    NpyInfo info;
    info.dtype         = h.dtype_str;
    info.dtype_size    = h.dtype_size;
    info.fortran_order = h.fortran_order;
    info.shape         = h.shape;
    return info;
}

// =============================================================================
// Type-specific loaders
// =============================================================================

namespace {

template <typename T>
NpyArray<T> loadTyped(const std::string& path, char expected_char,
                      uint32_t expected_size) {
    auto hdr = NpyLoader::parseHeader(path);

    // Compute total elements
    uint64_t total = 1;
    for (auto s : hdr.shape) total *= s;

    NpyArray<T> arr;
    arr.shape = hdr.shape;
    arr.data.resize(total);

    // If dtype matches exactly, read directly
    if (hdr.dtype_char == expected_char && hdr.dtype_size == expected_size) {
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::fread(arr.data.data(), sizeof(T), total, f);
        std::fclose(f);
    }
    // If dtype is compatible, read and cast
    else if (hdr.dtype_char == 'i' && hdr.dtype_size == 4 && expected_size == 4) {
        // int32 → uint32 or vice versa
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::fread(arr.data.data(), sizeof(T), total, f);
        std::fclose(f);
    }
    else if (hdr.dtype_char == 'u' && hdr.dtype_size == 4 && expected_size == 4) {
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::fread(arr.data.data(), sizeof(T), total, f);
        std::fclose(f);
    }
    else if (hdr.dtype_char == 'u' && hdr.dtype_size == 1 && expected_char == 'i' && expected_size == 1) {
        // uint8 → int8 reinterpret (same memory layout)
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::fread(arr.data.data(), 1, total, f);
        std::fclose(f);
    }
    else if (hdr.dtype_char == 'i' && hdr.dtype_size == 1 && expected_char == 'u' && expected_size == 1) {
        // int8 → uint8 reinterpret (same memory layout)
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::fread(arr.data.data(), 1, total, f);
        std::fclose(f);
    }
    else if (hdr.dtype_char == 'f' && hdr.dtype_size == 8 && expected_size == 4) {
        // float64 → float32 (lossy downcast)
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
        std::vector<double> temp(total);
        std::fread(temp.data(), sizeof(double), total, f);
        std::fclose(f);
        for (uint64_t i = 0; i < total; ++i) {
            arr.data[i] = static_cast<T>(temp[i]);
        }
    } else {
        throw NpyException("Dtype mismatch in " + path + ": file has " +
                          hdr.dtype_str + " but expected " +
                          expected_char + std::to_string(expected_size));
    }

    return arr;
}

} // anonymous namespace

NpyArray<int8_t> NpyLoader::loadInt8(const std::string& path) {
    return loadTyped<int8_t>(path, 'i', 1);
}

NpyArray<uint8_t> NpyLoader::loadUint8(const std::string& path) {
    return loadTyped<uint8_t>(path, 'u', 1);
}

NpyArray<int32_t> NpyLoader::loadInt32(const std::string& path) {
    return loadTyped<int32_t>(path, 'i', 4);
}

NpyArray<uint32_t> NpyLoader::loadUint32(const std::string& path) {
    return loadTyped<uint32_t>(path, 'u', 4);
}

NpyArray<float> NpyLoader::loadFloat32(const std::string& path) {
    return loadTyped<float>(path, 'f', 4);
}

NpyArray<double> NpyLoader::loadFloat64(const std::string& path) {
    return loadTyped<double>(path, 'f', 8);
}

// =============================================================================
// Generic loader
// =============================================================================

template <>
NpyArray<int8_t> NpyLoader::load<int8_t>(const std::string& path) {
    return loadInt8(path);
}

template <>
NpyArray<int32_t> NpyLoader::load<int32_t>(const std::string& path) {
    return loadInt32(path);
}

template <>
NpyArray<uint32_t> NpyLoader::load<uint32_t>(const std::string& path) {
    return loadUint32(path);
}

template <>
NpyArray<float> NpyLoader::load<float>(const std::string& path) {
    return loadFloat32(path);
}

template <>
NpyArray<double> NpyLoader::load<double>(const std::string& path) {
    return loadFloat64(path);
}

// =============================================================================
// Type-specific savers
// =============================================================================

static void saveRaw(const std::string& path, const void* data,
                    size_t num_elements, size_t element_size,
                    const std::string& dtype_str,
                    const std::vector<uint64_t>& shape) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw NpyException("Cannot create file: " + path);
    }
    NpyLoader::writeHeader(f, dtype_str, shape);
    std::fwrite(data, element_size, num_elements, f);
    std::fclose(f);
}

void NpyLoader::saveInt8(const std::string& path, const int8_t* data,
                          const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (auto s : shape) total *= s;
    saveRaw(path, data, total, 1, "<i1", shape);
}

void NpyLoader::saveInt32(const std::string& path, const int32_t* data,
                           const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (auto s : shape) total *= s;
    saveRaw(path, data, total, 4, "<i4", shape);
}

void NpyLoader::saveUint32(const std::string& path, const uint32_t* data,
                            const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (auto s : shape) total *= s;
    saveRaw(path, data, total, 4, "<u4", shape);
}

void NpyLoader::saveFloat32(const std::string& path, const float* data,
                             const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (auto s : shape) total *= s;
    saveRaw(path, data, total, 4, "<f4", shape);
}

// =============================================================================
// Save from NpyArray
// =============================================================================

template <>
void NpyLoader::save<int8_t>(const std::string& path, const NpyArray<int8_t>& arr) {
    saveInt8(path, arr.data.data(), arr.shape);
}

template <>
void NpyLoader::save<int32_t>(const std::string& path, const NpyArray<int32_t>& arr) {
    saveInt32(path, arr.data.data(), arr.shape);
}

template <>
void NpyLoader::save<uint32_t>(const std::string& path, const NpyArray<uint32_t>& arr) {
    saveUint32(path, arr.data.data(), arr.shape);
}

template <>
void NpyLoader::save<float>(const std::string& path, const NpyArray<float>& arr) {
    saveFloat32(path, arr.data.data(), arr.shape);
}

} // namespace utils
} // namespace accel
