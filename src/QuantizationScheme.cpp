#include "QuantizationScheme.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Symmetric INT8
// ---------------------------------------------------------------------------

void QuantizationScheme::quantize_symmetric(
    const float* data,
    int8_t*      output,
    float&       scale,
    size_t       count)
{
    if (count == 0) {
        scale = 1.0f;
        return;
    }

    // Find max absolute value.
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float v = std::fabs(data[i]);
        if (v > max_abs) max_abs = v;
    }

    if (max_abs < std::numeric_limits<float>::epsilon()) {
        scale = 1.0f;
        for (size_t i = 0; i < count; ++i) output[i] = 0;
        return;
    }

    scale = max_abs / 127.0f;

    for (size_t i = 0; i < count; ++i) {
        float q = std::round(data[i] / scale);
        // Clamp to [-127, 127] (not -128 to keep symmetric).
        q = std::max(-127.0f, std::min(127.0f, q));
        output[i] = static_cast<int8_t>(q);
    }
}

void QuantizationScheme::dequantize_symmetric(
    const int8_t* input,
    float*        output,
    float         scale,
    size_t        count)
{
    for (size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

// ---------------------------------------------------------------------------
// Asymmetric UINT8
// ---------------------------------------------------------------------------

void QuantizationScheme::quantize_asymmetric(
    const float* data,
    uint8_t*     output,
    float&       scale,
    int32_t&     zero_point,
    size_t       count)
{
    if (count == 0) {
        scale      = 1.0f;
        zero_point = 0;
        return;
    }

    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    float range = max_val - min_val;
    if (range < std::numeric_limits<float>::epsilon()) {
        scale      = 1.0f;
        zero_point = static_cast<int32_t>(std::round(-min_val));
        zero_point = std::max(0, std::min(255, zero_point));
        for (size_t i = 0; i < count; ++i) output[i] = static_cast<uint8_t>(zero_point);
        return;
    }

    scale = range / 255.0f;
    zero_point = static_cast<int32_t>(std::round(-min_val / scale));
    zero_point = std::max(0, std::min(255, zero_point));

    for (size_t i = 0; i < count; ++i) {
        int32_t q = static_cast<int32_t>(std::round(data[i] / scale)) + zero_point;
        q = std::max(0, std::min(255, q));
        output[i] = static_cast<uint8_t>(q);
    }
}

void QuantizationScheme::dequantize_asymmetric(
    const uint8_t* input,
    float*         output,
    float          scale,
    int32_t        zero_point,
    size_t         count)
{
    for (size_t i = 0; i < count; ++i) {
        output[i] = (static_cast<int32_t>(input[i]) - zero_point) * scale;
    }
}

// ---------------------------------------------------------------------------
// Error measurement
// ---------------------------------------------------------------------------

float QuantizationScheme::compute_quantization_error(
    const float* original,
    const float* dequantized,
    size_t       count)
{
    if (count == 0) return 0.0f;

    float sum_abs_err = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum_abs_err += std::fabs(original[i] - dequantized[i]);
    }
    return sum_abs_err / static_cast<float>(count);
}

QuantizationStats QuantizationScheme::compute_stats_symmetric(
    const float* data,
    size_t       count)
{
    QuantizationStats stats{};
    if (count == 0) return stats;

    float min_val = data[0];
    float max_val = data[0];
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        float v = std::fabs(data[i]);
        if (v > max_abs) max_abs = v;
    }

    stats.min_val = min_val;
    stats.max_val = max_val;

    if (max_abs < std::numeric_limits<float>::epsilon()) {
        stats.scale = 1.0f;
        return stats;
    }

    stats.scale = max_abs / 127.0f;

    // Quantize and dequantize to compute error.
    std::vector<int8_t>  q_buf(count);
    std::vector<float>   dq_buf(count);
    float dummy_scale = stats.scale;
    quantize_symmetric(data, q_buf.data(), dummy_scale, count);
    dequantize_symmetric(q_buf.data(), dq_buf.data(), stats.scale, count);

    stats.mae = compute_quantization_error(data, dq_buf.data(), count);

    // Clamp fraction: values that hit ±127 boundary.
    size_t clamped = 0;
    for (size_t i = 0; i < count; ++i) {
        if (q_buf[i] == 127 || q_buf[i] == -127) ++clamped;
    }
    stats.clamp_fraction = static_cast<float>(clamped) / static_cast<float>(count);

    return stats;
}

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

void QuantizationScheme::quantize_per_channel(
    const float* data,
    int8_t*      output,
    float*       scales,
    int          rows,
    int          cols)
{
    for (int r = 0; r < rows; ++r) {
        const float* row_in  = data   + static_cast<size_t>(r) * static_cast<size_t>(cols);
        int8_t*      row_out = output + static_cast<size_t>(r) * static_cast<size_t>(cols);
        quantize_symmetric(row_in, row_out, scales[r], static_cast<size_t>(cols));
    }
}

} // namespace llmquant
