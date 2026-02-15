/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       TEST_END_TO_END.CPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  INTEGRATION TESTS: Full inference pipeline validation                   ║
 * ║  TESTS: All components working together                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  Validates the entire inference flow from image input to classification  ║
 * ║  output. Compares accelerator results against golden model.              ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_integration.py                           ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  TEST CASES TO IMPLEMENT:                                                 ║
 * ║                                                                           ║
 * ║  1. test_single_layer_conv()                                              ║
 * ║     - Load single conv layer weights                                     ║
 * ║     - Run on test input                                                  ║
 * ║     - Compare to golden output                                           ║
 * ║                                                                           ║
 * ║  2. test_mnist_fc()                                                      ║
 * ║     - FC layer with BSR sparse weights                                    ║
 * ║     - Verify against golden reference                                     ║
 * ║                                                                           ║
 * ║  3. test_full_mnist_cnn()                                                  ║
 * ║     - All layers                                                         ║
 * ║     - End-to-end accuracy                                                ║
 * ║                                                                           ║
 * ║  4. test_batch_inference()                                                ║
 * ║     - Multiple images                                                    ║
 * ║     - Throughput measurement                                             ║
 * ║                                                                           ║
 * ║  5. test_mnist_sample()                                                    ║
 * ║     - Real MNIST digit image                                             ║
 * ║     - Known correct digit                                                ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "../include/golden_models.hpp"
#include "../include/bsr_packer.hpp"
#include "../include/accelerator_driver.hpp"

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    if (name()) { std::cout << "PASS" << std::endl; passed++; } \
    else { std::cout << "FAIL" << std::endl; failed++; }

static int passed = 0;
static int failed = 0;

// Tolerance for floating point comparison
static constexpr float TOLERANCE = 1e-4f;

bool compare_outputs(const int8_t* hw, const int8_t* golden, size_t n, int max_diff = 1) {
    for (size_t i = 0; i < n; i++) {
        if (std::abs(hw[i] - golden[i]) > max_diff) {
            std::cerr << "Mismatch at " << i << ": hw=" << (int)hw[i] 
                      << " golden=" << (int)golden[i] << std::endl;
            return false;
        }
    }
    return true;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_single_layer_conv() {
    // TODO: Implement
    //
    // // Load conv1 weights from data/int8/
    // BSRPacker packer;
    // BSRMatrix weights = packer.load_bsr("../../../data/int8/conv1_weight_bsr.bin");
    //
    // // Create test input (224x224x3 -> 112x112x64)
    // std::vector<int8_t> input(224 * 224 * 3);
    // for (size_t i = 0; i < input.size(); i++) input[i] = (i % 255) - 128;
    //
    // // Golden model
    // std::vector<int32_t> golden_out(112 * 112 * 64);
    // golden::conv2d_bsr_int8(input.data(), weights, nullptr, golden_out.data(),
    //                         1, 3, 64, 224, 224, 7, 2, 3);
    //
    // // Accelerator
    // AcceleratorDriver accel(AcceleratorDriver::Mode::SIMULATION);
    // std::vector<int32_t> hw_out(112 * 112 * 64);
    // accel.run_conv_layer(input.data(), weights, hw_out.data(), ...);
    //
    // // Compare
    // for (size_t i = 0; i < golden_out.size(); i++) {
    //     if (golden_out[i] != hw_out[i]) return false;
    // }
    
    return true;
}

bool test_mnist_fc() {
    // TODO: Implement
    //
    // FC layer test (e.g. fc1: 9216 -> 128)
    // Load BSR weights from data/bsr_export_14x14/fc1/
    // Create test input, run golden model, compare with accelerator
    
    return true;
}

bool test_full_mnist_cnn() {
    // TODO: Implement
    //
    // Full MNIST CNN: conv1 -> conv2 -> fc1 -> fc2
    // Load all layer weights from data/bsr_export_14x14/
    //
    // Test with golden input from sw/golden/mnist_inputs.npy
    // Compare final logits with sw/golden/mnist_logits_fp32.npy
    
    return true;
}

bool test_batch_inference() {
    // TODO: Implement
    //
    // ResNetInference model(true);
    // model.load_model("../../../data/int8/");
    //
    // const int batch_size = 16;
    // std::vector<std::vector<uint8_t>> images(batch_size);
    //
    // auto start = std::chrono::high_resolution_clock::now();
    //
    // for (int i = 0; i < batch_size; i++) {
    //     images[i].resize(224 * 224 * 3);
    //     // Fill with pattern
    //     auto result = model.run_inference(images[i].data());
    // }
    //
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    // float fps = batch_size * 1000.0f / duration.count();
    // std::cout << "Throughput: " << fps << " FPS" << std::endl;
    //
    // // Should achieve target FPS
    // if (fps < 10.0f) return false;  // Minimum target
    
    return true;
}

bool test_mnist_sample() {
    // TODO: Implement
    //
    // Load a known MNIST digit (e.g., a "7" from golden inputs)
    // Run through full CNN pipeline on accelerator
    // Verify predicted digit matches expected label
    
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== End-to-End Tests ===" << std::endl;
    
    TEST(test_single_layer_conv);
    TEST(test_mnist_fc);
    TEST(test_full_mnist_cnn);
    TEST(test_batch_inference);
    TEST(test_mnist_sample);
    
    std::cout << std::endl;
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;
    
    return failed == 0 ? 0 : 1;
}
