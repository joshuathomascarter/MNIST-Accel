// =============================================================================
// npy_loader.hpp — Simple NumPy .npy file loader for C++
// =============================================================================
// Loads .npy files (NumPy array format) into std::vector.
// Supports int8, int16, int32, int64, float32, float64.
//
// Usage:
//   auto row_ptr = npy::load<int32_t>("row_ptr.npy");
//   auto weights = npy::load<int8_t>("weights.npy");
// =============================================================================

#pragma once
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>

namespace npy {

// NPY file header info
struct NpyHeader {
    std::string dtype;      // '<i4', '<f4', etc.
    bool fortran_order;
    std::vector<size_t> shape;
    size_t header_size;     // Total header size (including padding)
};

inline NpyHeader parse_header(std::ifstream& file) {
    // Read magic number: 0x93 NUMPY
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Not a valid NPY file");
    }
    
    // Read version (1 byte major, 1 byte minor)
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length
    uint16_t header_len;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len32;
        file.read(reinterpret_cast<char*>(&header_len32), 4);
        header_len = static_cast<uint16_t>(header_len32);
    }
    
    // Read header string
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);
    
    NpyHeader header;
    header.header_size = 6 + 2 + (major == 1 ? 2 : 4) + header_len;
    
    // Parse dtype: 'descr': '<i4'
    std::regex dtype_re("'descr':\\s*'([^']+)'");
    std::smatch match;
    if (std::regex_search(header_str, match, dtype_re)) {
        header.dtype = match[1];
    }
    
    // Parse fortran_order
    header.fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
    
    // Parse shape: 'shape': (17,) or 'shape': (128, 9216)
    std::regex shape_re("'shape':\\s*\\(([^)]*)\\)");
    if (std::regex_search(header_str, match, shape_re)) {
        std::string shape_str = match[1];
        std::stringstream ss(shape_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                header.shape.push_back(std::stoull(item));
            }
        }
    }
    
    return header;
}

template<typename T>
std::vector<T> load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    NpyHeader header = parse_header(file);
    
    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : header.shape) {
        total_elements *= dim;
    }
    
    // Read data
    std::vector<T> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(T));
    
    return data;
}

// Load with shape info
template<typename T>
std::vector<T> load(const std::string& filename, std::vector<size_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    NpyHeader header = parse_header(file);
    shape = header.shape;
    
    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : header.shape) {
        total_elements *= dim;
    }
    
    // Read data
    std::vector<T> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(T));
    
    return data;
}

} // namespace npy
