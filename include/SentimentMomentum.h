#pragma once

/**
 * @file include/SentimentMomentum.h
 * @brief Tracks the rolling rate of change (momentum) of sentiment scores.
 *
 * A sentiment *reversal* — positive-to-negative momentum — is a stronger
 * directional signal than the sentiment level alone.  This class maintains a
 * sliding window of sentiment samples, computes the first-order momentum
 * (rate of change) and second-order momentum (acceleration), and classifies
 * the current regime into one of four states:
 *
 *   - Accelerating Bullish  (momentum > 0, accel > 0)
 *   - Decelerating Bullish  (momentum > 0, accel < 0)  ← reversal warning
 *   - Accelerating Bearish  (momentum < 0, accel < 0)
 *   - Decelerating Bearish  (momentum < 0, accel > 0)  ← reversal warning
 *
 * The reversal condition is the most actionable: when conviction is fading
 * (decelerating) the engine should reduce position sizes or tighten stops.
 *
 * ## Algorithm
 *
 * A ring buffer of size `window` stores the last N sentiment scores.
 * Momentum is estimated as the slope of a simple linear regression over the
 * window (identical to the approach in SentimentTrajectoryAnalyzer but
 * operating on a shorter lookback and returning the raw derivative).
 * Acceleration is the slope of the momentum series over a shorter
 * sub-window of size `accel_window` (default = window / 2).
 *
 * ## Thread safety
 *
 * All public methods are thread-safe (guarded by an internal mutex).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_MOMENTUM` (default ON).
 */

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling sentiment momentum and acceleration tracker.
 */
class SentimentMomentum {
public:
    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    /**
     * @brief Momentum regime classification.
     */
    enum class Regime {
        Insufficient,        ///< Not enough samples yet.
        AcceleratingBullish, ///< Momentum positive and growing.
        DeceleratingBullish, ///< Momentum positive but fading — bullish reversal risk.
        AcceleratingBearish, ///< Momentum negative and worsening.
        DeceleratingBearish, ///< Momentum negative but fading — bearish reversal risk.
        Neutral,             ///< Momentum near zero.
    };

    /**
     * @brief Configuration parameters.
     */
    struct Config {
        /**
         * @brief Primary lookback window size for momentum (samples).
         * Default 20.
         */
        std::size_t window{20};

        /**
         * @brief Acceleration lookback window (samples).
         * Must be <= window.  Default 10.
         */
        std::size_t accel_window{10};

        /**
         * @brief Momentum values whose absolute value is below this threshold
         * are classified as Neutral.  Default 1e-4.
         */
        double neutral_threshold{1e-4};

        /**
         * @brief Minimum absolute acceleration to distinguish Accelerating
         * from Decelerating.  Default 1e-5.
         */
        double accel_threshold{1e-5};
    };

    /**
     * @brief A snapshot of the current momentum state.
     */
    struct Snapshot {
        double  momentum{0.0};           ///< First-order rate of change.
        double  acceleration{0.0};       ///< Second-order rate of change.
        Regime  regime{Regime::Insufficient};
        bool    reversal_warning{false}; ///< True when regime is Decelerating.
        uint64_t samples_seen{0};        ///< Total samples added since construction.
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SentimentMomentum(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Add a new sentiment score to the rolling window.
     *
     * Thread-safe.
     *
     * @param score  Raw sentiment score (any range; typically [-1, 1] or [0, 1]).
     */
    void add_sample(double score);

    /**
     * @brief Reset all state (window, counters, momentum history).
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Return a consistent snapshot of the current momentum state.
     *
     * Thread-safe.
     */
    [[nodiscard]] Snapshot snapshot() const;

    /**
     * @brief Current momentum (first derivative of sentiment).
     *
     * Thread-safe.
     */
    [[nodiscard]] double momentum() const;

    /**
     * @brief Current acceleration (second derivative of sentiment).
     *
     * Thread-safe.
     */
    [[nodiscard]] double acceleration() const;

    /**
     * @brief Current regime classification.
     *
     * Thread-safe.
     */
    [[nodiscard]] Regime regime() const;

    /**
     * @brief True when the regime is Decelerating (reversal warning).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool reversal_warning() const;

    /**
     * @brief Human-readable name for a Regime value.
     */
    [[nodiscard]] static const char* regime_name(Regime r) noexcept;

    /**
     * @brief Serialise current state to a compact JSON string.
     *
     * Fields: momentum, acceleration, regime, reversal_warning, samples_seen.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_json() const;

    /**
     * @brief Number of samples added since construction (or last reset).
     */
    [[nodiscard]] uint64_t samples_seen() const noexcept {
        return samples_seen_.load(std::memory_order_relaxed);
    }

private:
    // Compute linear regression slope over `data` (returns NaN if size < 2).
    static double linear_slope(const std::vector<double>& data);

    // Recompute momentum and acceleration from current ring buffer state.
    // Must be called with mutex_ held.
    void recompute_locked();

    mutable std::mutex mutex_;
    Config config_;

    // Ring buffer for raw sentiment samples.
    std::vector<double> ring_;
    std::size_t         head_{0};
    std::size_t         count_{0};

    // Ring buffer for momentum samples (for accel computation).
    std::vector<double> momentum_ring_;
    std::size_t         mom_head_{0};
    std::size_t         mom_count_{0};

    // Derived values (refreshed on every add_sample).
    double momentum_{0.0};
    double acceleration_{0.0};
    Regime regime_{Regime::Insufficient};

    std::atomic<uint64_t> samples_seen_{0};
};

} // namespace llmquant
