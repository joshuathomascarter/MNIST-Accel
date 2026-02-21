// golden_model.hpp — Software reference implementation for verification
// =============================================================================
//
// Pure-software golden model that replicates the exact arithmetic the hardware
// performs: INT8 × INT8 → INT32 accumulation, tiled in 14×14 blocks, with
// optional per-tensor or per-channel dequantisation back to float.
//
// Used to generate expected outputs for cocotb tests and C++ unit tests.
//
// =============================================================================
#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <ostream>

namespace accel {
namespace compute {

// =============================================================================
// QuantParams — Quantisation scales for dequantisation
// =============================================================================
struct QuantParams {
    float scale_act   = 1.0f;     // Activation scale (per-tensor)
    float scale_wgt   = 1.0f;     // Weight scale (per-tensor)
    int32_t zp_act    = 0;        // Activation zero-point
    int32_t zp_wgt    = 0;        // Weight zero-point

    // Per-channel weight scales (if non-empty, overrides scale_wgt)
    std::vector<float> per_channel_scales;

    // Combined scale for output dequantisation
    float outputScale(uint32_t channel = 0) const;
};

// =============================================================================
// GoldenModel — Software reference GEMM
// =============================================================================
class GoldenModel {
public:
    // -------------------------------------------------------------------------
    // Dense INT8 GEMM:  C[M×N] = A[M×K] × B[K×N]
    // -------------------------------------------------------------------------

    /// INT8 × INT8 → INT32 matrix multiply (no quantisation).
    /// @param A  Row-major INT8 matrix [M×K]
    /// @param B  Row-major INT8 matrix [K×N]
    /// @param M  Rows of A and C
    /// @param N  Cols of B and C
    /// @param K  Cols of A / Rows of B
    /// @return   Row-major INT32 matrix [M×N]
    static std::vector<int32_t> gemmINT8(const int8_t* A, const int8_t* B,
                                         uint32_t M, uint32_t N, uint32_t K);

    /// Tiled GEMM: same result as gemmINT8 but computed tile-by-tile in
    /// 14×14 blocks — exactly mirrors the hardware execution order.
    /// Useful for verifying that tiling doesn't introduce errors.
    static std::vector<int32_t> tiledGemmINT8(const int8_t* A, const int8_t* B,
                                              uint32_t M, uint32_t N, uint32_t K);

    // -------------------------------------------------------------------------
    // Dequantised GEMM: INT32 accumulator → float output
    // -------------------------------------------------------------------------

    /// Compute GEMM then dequantise:  out[i][j] = sum_k(A[i][k]*B[k][j]) * scale
    static std::vector<float> gemmDequant(const int8_t* A, const int8_t* B,
                                          uint32_t M, uint32_t N, uint32_t K,
                                          const QuantParams& qparams);

    // -------------------------------------------------------------------------
    // Convolution (im2col + GEMM)
    // -------------------------------------------------------------------------

    /// 2D convolution via im2col + GEMM
    /// @param input   [C_in × H_in × W_in] INT8, row-major
    /// @param weight  [C_out × C_in × kH × kW] INT8, row-major
    /// @param bias    [C_out] INT32 (can be nullptr for no bias)
    /// @param C_in, C_out, H_in, W_in, kH, kW, stride, padding
    /// @return [C_out × H_out × W_out] INT32
    static std::vector<int32_t> conv2d(const int8_t* input,
                                       const int8_t* weight,
                                       const int32_t* bias,
                                       uint32_t C_in, uint32_t C_out,
                                       uint32_t H_in, uint32_t W_in,
                                       uint32_t kH, uint32_t kW,
                                       uint32_t stride, uint32_t padding);

    // -------------------------------------------------------------------------
    // im2col helper
    // -------------------------------------------------------------------------

    /// Transform input tensor into im2col matrix for GEMM.
    /// @param input  [C_in × H_in × W_in] INT8
    /// @return [K × N] INT8 where K=C_in*kH*kW, N=H_out*W_out
    static std::vector<int8_t> im2col(const int8_t* input,
                                      uint32_t C_in, uint32_t H_in, uint32_t W_in,
                                      uint32_t kH, uint32_t kW,
                                      uint32_t stride, uint32_t padding);

    // -------------------------------------------------------------------------
    // Activation functions (post-GEMM)
    // -------------------------------------------------------------------------

    /// ReLU on INT32 accumulators (clamp negatives to 0)
    static void reluINT32(std::vector<int32_t>& data);

    /// ReLU on float outputs
    static void reluFloat(std::vector<float>& data);

    /// Max pooling on INT32 feature map [C × H × W]
    /// @return [C × H_out × W_out]
    static std::vector<int32_t> maxPool2d(const int32_t* input,
                                          uint32_t C, uint32_t H, uint32_t W,
                                          uint32_t pool_size, uint32_t pool_stride);

    // -------------------------------------------------------------------------
    // Full MNIST inference (4 layers)
    // -------------------------------------------------------------------------

    /// Run complete MNIST inference through all 4 layers.
    /// @param input_28x28  [1×28×28] INT8 input image
    /// @param conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b
    /// @return [10] float logits (after dequantisation)
    static std::vector<float> mnistInference(
        const int8_t* input_28x28,
        const int8_t* conv1_w, const int32_t* conv1_b,
        const int8_t* conv2_w, const int32_t* conv2_b,
        const int8_t* fc1_w,   const int32_t* fc1_b,
        const int8_t* fc2_w,   const int32_t* fc2_b,
        const QuantParams& qparams = QuantParams{});

    // -------------------------------------------------------------------------
    // Verification helpers
    // -------------------------------------------------------------------------

    /// Compare two INT32 arrays element-wise. Returns number of mismatches.
    static uint32_t compareINT32(const int32_t* expected,
                                 const int32_t* actual,
                                 size_t count,
                                 std::ostream* diff_log = nullptr);

    /// Compare two float arrays with tolerance. Returns number of mismatches.
    static uint32_t compareFloat(const float* expected,
                                 const float* actual,
                                 size_t count,
                                 float atol = 1e-5f,
                                 float rtol = 1e-4f,
                                 std::ostream* diff_log = nullptr);

    /// Argmax of a float vector
    static uint32_t argmax(const std::vector<float>& logits);
};

} // namespace compute
} // namespace accel
