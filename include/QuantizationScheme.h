#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llmquant {

/**
 * @brief Statistics collected during a quantization pass.
 */
struct QuantizationStats {
    float scale{0.0f};
    int32_t zero_point{0};
    float min_val{0.0f};
    float max_val{0.0f};
    /// Fraction of values that were clamped during dequantization.
    float clamp_fraction{0.0f};
    /// Mean absolute error between original and dequantized values.
    float mae{0.0f};
};

/**
 * @brief INT8 quantization and dequantization utilities.
 *
 * Provides symmetric (signed INT8) and asymmetric (unsigned UINT8) quantization
 * schemes, per-channel quantization, and error measurement helpers.
 *
 * ## Symmetric quantization
 *
 * ```
 * scale = max(|x|) / 127
 * q_i   = round(x_i / scale)   ∈ [-127, 127]
 * x̂_i  = q_i * scale
 * ```
 *
 * ## Asymmetric quantization
 *
 * ```
 * scale      = (max(x) - min(x)) / 255
 * zero_point = round(-min(x) / scale)
 * q_i        = round(x_i / scale) + zero_point   ∈ [0, 255]
 * x̂_i       = (q_i - zero_point) * scale
 * ```
 */
class QuantizationScheme {
public:
    QuantizationScheme() = default;

    // ------------------------------------------------------------------
    // Symmetric INT8
    // ------------------------------------------------------------------

    /**
     * @brief Quantize floats to symmetric signed INT8.
     *
     * @param data      Input float array (length: count).
     * @param output    Output INT8 array (length: count, caller-allocated).
     * @param scale     Output: computed scale factor.
     * @param count     Number of elements.
     */
    static void quantize_symmetric(
        const float* data,
        int8_t*      output,
        float&       scale,
        size_t       count);

    /**
     * @brief Dequantize symmetric INT8 back to float.
     *
     * @param input     Input INT8 array (length: count).
     * @param output    Output float array (length: count, caller-allocated).
     * @param scale     Scale factor from quantize_symmetric.
     * @param count     Number of elements.
     */
    static void dequantize_symmetric(
        const int8_t* input,
        float*        output,
        float         scale,
        size_t        count);

    // ------------------------------------------------------------------
    // Asymmetric UINT8
    // ------------------------------------------------------------------

    /**
     * @brief Quantize floats to asymmetric unsigned UINT8.
     *
     * @param data        Input float array.
     * @param output      Output UINT8 array (caller-allocated).
     * @param scale       Output: computed scale.
     * @param zero_point  Output: computed zero-point.
     * @param count       Number of elements.
     */
    static void quantize_asymmetric(
        const float* data,
        uint8_t*     output,
        float&       scale,
        int32_t&     zero_point,
        size_t       count);

    /**
     * @brief Dequantize asymmetric UINT8 back to float.
     *
     * @param input       Input UINT8 array.
     * @param output      Output float array (caller-allocated).
     * @param scale       Scale from quantize_asymmetric.
     * @param zero_point  Zero-point from quantize_asymmetric.
     * @param count       Number of elements.
     */
    static void dequantize_asymmetric(
        const uint8_t* input,
        float*         output,
        float          scale,
        int32_t        zero_point,
        size_t         count);

    // ------------------------------------------------------------------
    // Error measurement
    // ------------------------------------------------------------------

    /**
     * @brief Compute the mean absolute error between original and dequantized.
     *
     * @param original     Original float array.
     * @param dequantized  Reconstructed float array.
     * @param count        Number of elements.
     * @return Mean absolute error.
     */
    static float compute_quantization_error(
        const float* original,
        const float* dequantized,
        size_t       count);

    /**
     * @brief Full stats for a symmetric quantization pass.
     */
    static QuantizationStats compute_stats_symmetric(
        const float* data,
        size_t       count);

    // ------------------------------------------------------------------
    // Per-channel quantization
    // ------------------------------------------------------------------

    /**
     * @brief Quantize a 2D weight matrix channel-by-channel (row = channel).
     *
     * Each row is quantized independently with its own scale.
     *
     * @param data    Input float array (rows × cols, row-major).
     * @param output  Output INT8 array (caller-allocated, same layout).
     * @param scales  Output scale array (length: rows, caller-allocated).
     * @param rows    Number of output channels (rows).
     * @param cols    Number of elements per channel (columns).
     */
    static void quantize_per_channel(
        const float* data,
        int8_t*      output,
        float*       scales,
        int          rows,
        int          cols);
};

} // namespace llmquant
