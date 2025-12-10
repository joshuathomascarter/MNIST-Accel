/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                          TEST_UTILS.HPP                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: Testing utilities scattered across sw/tests/*.py              ║
 * ║            pytest fixtures and helpers                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Common utilities for all C++ tests in this project:                   ║
 * ║    - Random number generation with seeds for reproducibility             ║
 * ║    - Test vector generation (edge cases, random, structured)             ║
 * ║    - Result comparison with tolerance and detailed mismatch reporting    ║
 * ║    - Timing utilities for benchmarks                                     ║
 * ║    - Test result formatting and logging                                  ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTEST:                                               ║
 * ║    • Generate millions of test vectors quickly                           ║
 * ║    • Direct Verilator integration without Python FFI                     ║
 * ║    • Same test code works in CI and on FPGA                              ║
 * ║    • Consistent random seeds across platforms                            ║
 * ║                                                                           ║
 * ║  KEY UTILITIES:                                                           ║
 * ║                                                                           ║
 * ║    Random Generation:                                                    ║
 * ║      random_int8_vector(size, seed)   - Random INT8 values [-128, 127]   ║
 * ║      random_int8_matrix(M, N, seed)   - Random INT8 matrix               ║
 * ║      random_sparse_matrix(M, N, sparsity, seed) - With zero blocks       ║
 * ║                                                                           ║
 * ║    Edge Case Vectors:                                                    ║
 * ║      zeros(size)                      - All zeros                        ║
 * ║      ones(size)                       - All ones                         ║
 * ║      max_values(size)                 - All 127                          ║
 * ║      min_values(size)                 - All -128                         ║
 * ║      alternating(size)                - 127, -128, 127, -128, ...        ║
 * ║      identity_block()                 - 16x16 identity for matmul test   ║
 * ║                                                                           ║
 * ║    Comparison:                                                           ║
 * ║      compare_exact(expected, actual)  - Bit-exact comparison             ║
 * ║      compare_tolerant(exp, act, tol)  - Allow small differences          ║
 * ║      print_mismatch(exp, act, idx)    - Show first N mismatches          ║
 * ║                                                                           ║
 * ║    Timing:                                                                ║
 * ║      Timer class with start/stop/elapsed                                 ║
 * ║      benchmark() function wrapper                                        ║
 * ║                                                                           ║
 * ║    Reporting:                                                            ║
 * ║      TEST_PASS(name)                  - Green checkmark                  ║
 * ║      TEST_FAIL(name, reason)          - Red X with details               ║
 * ║      SECTION(name)                    - Section header                   ║
 * ║                                                                           ║
 * ║  USAGE EXAMPLE:                                                           ║
 * ║                                                                           ║
 * ║    #include "test_utils.hpp"                                             ║
 * ║                                                                           ║
 * ║    int main() {                                                          ║
 * ║        SECTION("Matrix Multiply Tests");                                 ║
 * ║                                                                           ║
 * ║        // Generate random test data                                      ║
 * ║        auto A = test::random_int8_matrix(16, 16, 42);                    ║
 * ║        auto B = test::random_int8_matrix(16, 16, 43);                    ║
 * ║        std::vector<int32_t> C_expected(256), C_actual(256);              ║
 * ║                                                                           ║
 * ║        // Run golden and hardware                                        ║
 * ║        golden::matmul_int8(A.data(), B.data(), C_expected.data(),        ║
 * ║                            16, 16, 16);                                  ║
 * ║        hardware_matmul(A.data(), B.data(), C_actual.data());             ║
 * ║                                                                           ║
 * ║        // Compare                                                         ║
 * ║        if (test::compare_exact(C_expected, C_actual)) {                  ║
 * ║            TEST_PASS("random_matmul");                                   ║
 * ║        } else {                                                          ║
 * ║            TEST_FAIL("random_matmul", "Output mismatch");                ║
 * ║            test::print_mismatch(C_expected, C_actual, 5);                ║
 * ║        }                                                                  ║
 * ║                                                                           ║
 * ║        return 0;                                                          ║
 * ║    }                                                                      ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace resnet_accel {
namespace test {

class RandomGenerator {
public:
    explicit RandomGenerator(std::uint32_t seed = 42) : gen_(seed) {}

    std::int8_t next_int8() {
        return static_cast<std::int8_t>(dist_int8_(gen_));
    }

    std::vector<std::int8_t> int8_vector(std::size_t size) {
        std::vector<std::int8_t> vec(size);
        for (auto& v : vec) v = next_int8();
        return vec;
    }

    std::vector<std::int8_t> int8_matrix(std::size_t rows, std::size_t cols) {
        return int8_vector(rows * cols);
    }

private:
    std::mt19937 gen_;
    std::uniform_int_distribution<int> dist_int8_{-128, 127};
};

template<typename T>
bool compare_exact(const std::vector<T>& expected, const std::vector<T>& actual) {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) return false;
    }
    return true;
}

template<typename T>
void print_mismatch(const std::vector<T>& expected, const std::vector<T>& actual, std::size_t max_show = 5) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < std::min(expected.size(), actual.size()) && count < max_show; ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "  [" << i << "] expected " << static_cast<int>(expected[i])
                     << ", got " << static_cast<int>(actual[i]) << "\n";
            ++count;
        }
    }
}

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_ - start_);
        return static_cast<double>(duration.count());
    }

private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

// Test result macros
#define TEST_PASS(name) \
    std::cout << "✓ " << (name) << " PASSED" << std::endl

#define TEST_FAIL(name, reason) \
    std::cout << "✗ " << (name) << " FAILED: " << (reason) << std::endl

#define SECTION(name) \
    std::cout << std::endl << "=== " << (name) << " ===" << std::endl

} // namespace test
} // namespace resnet_accel

#endif // TEST_UTILS_HPP
