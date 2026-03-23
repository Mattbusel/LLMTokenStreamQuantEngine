#include <gtest/gtest.h>

#ifdef LLMQUANT_QUANTIZATION_SCHEME_ENABLED

#include "QuantizationScheme.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace llmquant;

// ============================================================
// Symmetric INT8 — quantize / dequantize roundtrip
// ============================================================

TEST(QuantizationScheme, SymmetricRoundtripZeroInput) {
    std::vector<float>  data(16, 0.0f);
    std::vector<int8_t> q(16);
    float scale = 0.0f;
    QuantizationScheme::quantize_symmetric(data.data(), q.data(), scale, data.size());
    EXPECT_NEAR(scale, 1.0f, 1e-6f);
    for (auto v : q) EXPECT_EQ(v, 0);
}

TEST(QuantizationScheme, SymmetricRoundtripUniform) {
    // All values equal → single bin.
    std::vector<float>  data(8, 0.5f);
    std::vector<int8_t> q(8);
    std::vector<float>  dq(8);
    float scale = 0.0f;
    QuantizationScheme::quantize_symmetric(data.data(), q.data(), scale, data.size());
    QuantizationScheme::dequantize_symmetric(q.data(), dq.data(), scale, data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(dq[i], data[i], scale) << "at index " << i;
    }
}

TEST(QuantizationScheme, SymmetricScaleFormula) {
    // scale = max(|x|) / 127
    std::vector<float> data = {-2.54f, 0.0f, 1.27f, 2.54f};
    std::vector<int8_t> q(4);
    float scale = 0.0f;
    QuantizationScheme::quantize_symmetric(data.data(), q.data(), scale, data.size());
    float expected_scale = 2.54f / 127.0f;
    EXPECT_NEAR(scale, expected_scale, 1e-5f);
}

TEST(QuantizationScheme, SymmetricRoundtripMAESmall) {
    // For 256 values linearly spaced in [-1, 1], MAE should be < 0.5% of range.
    const size_t N = 256;
    std::vector<float>  data(N);
    for (size_t i = 0; i < N; ++i) data[i] = -1.0f + 2.0f * static_cast<float>(i) / (N - 1);

    std::vector<int8_t> q(N);
    std::vector<float>  dq(N);
    float scale = 0.0f;
    QuantizationScheme::quantize_symmetric(data.data(), q.data(), scale, N);
    QuantizationScheme::dequantize_symmetric(q.data(), dq.data(), scale, N);

    float mae = QuantizationScheme::compute_quantization_error(data.data(), dq.data(), N);
    EXPECT_LT(mae, 0.01f) << "MAE too large: " << mae;
}

// ============================================================
// Asymmetric UINT8 — range mapping
// ============================================================

TEST(QuantizationScheme, AsymmetricRoundtripRange) {
    // Values in [0, 1]: min mapped to 0, max mapped to 255.
    std::vector<float>   data = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<uint8_t> q(5);
    std::vector<float>   dq(5);
    float scale = 0.0f;
    int32_t zp  = 0;
    QuantizationScheme::quantize_asymmetric(data.data(), q.data(), scale, zp, data.size());
    QuantizationScheme::dequantize_asymmetric(q.data(), dq.data(), scale, zp, data.size());

    // Reconstructed values should be within 0.5% of the original range (1.0).
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(dq[i], data[i], 0.005f) << "at index " << i;
    }
}

TEST(QuantizationScheme, AsymmetricZeroPointFormula) {
    // For data in [-1, 1]: zero_point = round(1/scale) ≈ 128.
    std::vector<float>   data = {-1.0f, 0.0f, 1.0f};
    std::vector<uint8_t> q(3);
    float   scale = 0.0f;
    int32_t zp    = 0;
    QuantizationScheme::quantize_asymmetric(data.data(), q.data(), scale, zp, data.size());
    // scale = 2/255 ≈ 0.00784; zero_point ≈ round(1/scale) ≈ 128
    EXPECT_NEAR(scale, 2.0f / 255.0f, 1e-4f);
    EXPECT_NEAR(static_cast<float>(zp), 127.5f, 1.0f);  // rounding may vary ±1
}

TEST(QuantizationScheme, AsymmetricUniformInput) {
    std::vector<float>   data(10, 3.14f);
    std::vector<uint8_t> q(10);
    float   scale = 0.0f;
    int32_t zp    = 0;
    QuantizationScheme::quantize_asymmetric(data.data(), q.data(), scale, zp, data.size());
    // With uniform input all quantized values should be equal.
    for (size_t i = 1; i < 10; ++i) EXPECT_EQ(q[i], q[0]);
}

// ============================================================
// Per-channel quantization
// ============================================================

TEST(QuantizationScheme, PerChannelRoundtrip) {
    // 3 channels, 4 elements each.
    const int rows = 3, cols = 4;
    std::vector<float> data = {
        1.0f,  2.0f,  3.0f,  4.0f,   // channel 0: max=4
        0.1f,  0.2f,  0.3f,  0.4f,   // channel 1: max=0.4
       -5.0f, -1.0f,  2.5f,  5.0f,   // channel 2: max=5
    };
    std::vector<int8_t> q(rows * cols);
    std::vector<float>  scales(rows, 0.0f);

    QuantizationScheme::quantize_per_channel(data.data(), q.data(), scales.data(), rows, cols);

    // Dequantize each row and check MAE.
    std::vector<float> dq(rows * cols);
    for (int r = 0; r < rows; ++r) {
        QuantizationScheme::dequantize_symmetric(
            q.data()  + r * cols,
            dq.data() + r * cols,
            scales[r],
            static_cast<size_t>(cols));
    }

    float mae = QuantizationScheme::compute_quantization_error(data.data(), dq.data(),
                                                               static_cast<size_t>(rows * cols));
    EXPECT_LT(mae, 0.1f) << "Per-channel MAE too large: " << mae;

    // Each channel should have its own scale.
    EXPECT_GT(scales[2], scales[1]);  // channel 2 has larger values
}

TEST(QuantizationScheme, PerChannelScalesIndependent) {
    const int rows = 2, cols = 3;
    std::vector<float>  data   = {100.0f, 200.0f, 100.0f,   0.1f, 0.2f, 0.1f};
    std::vector<int8_t> q(rows * cols);
    std::vector<float>  scales(rows, 0.0f);
    QuantizationScheme::quantize_per_channel(data.data(), q.data(), scales.data(), rows, cols);
    // Channel 0 scale >> channel 1 scale.
    EXPECT_GT(scales[0], scales[1] * 100.0f);
}

// ============================================================
// Error computation
// ============================================================

TEST(QuantizationScheme, ComputeErrorIdentical) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    float err = QuantizationScheme::compute_quantization_error(a.data(), a.data(), a.size());
    EXPECT_NEAR(err, 0.0f, 1e-7f);
}

TEST(QuantizationScheme, ComputeErrorKnown) {
    std::vector<float> orig = {0.0f, 1.0f, 2.0f};
    std::vector<float> dq   = {0.1f, 1.1f, 1.9f};
    // MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
    float err = QuantizationScheme::compute_quantization_error(orig.data(), dq.data(), orig.size());
    EXPECT_NEAR(err, 0.1f, 1e-6f);
}

// ============================================================
// QuantizationStats
// ============================================================

TEST(QuantizationScheme, StatsSymmetricPopulated) {
    std::vector<float> data = {-3.0f, 0.0f, 1.5f, 3.0f};
    QuantizationStats  stats = QuantizationScheme::compute_stats_symmetric(data.data(), data.size());
    EXPECT_NEAR(stats.scale, 3.0f / 127.0f, 1e-5f);
    EXPECT_NEAR(stats.min_val, -3.0f, 1e-5f);
    EXPECT_NEAR(stats.max_val,  3.0f, 1e-5f);
    EXPECT_GE(stats.mae, 0.0f);
}

#endif // LLMQUANT_QUANTIZATION_SCHEME_ENABLED
