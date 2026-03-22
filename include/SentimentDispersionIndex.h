#pragma once

/// @file SentimentDispersionIndex.h
/// @brief Measures disagreement across sentiment dimensions (bias, volatility, confidence).
///
/// When all sentiment signals agree (high bias + high confidence + low vol),
/// the trade signal is high-conviction.  When they disagree (e.g. bullish bias
/// but low confidence and high vol), the signal may be noisy.
///
/// The Sentiment Dispersion Index (SDI) quantifies this disagreement as the
/// coefficient of variation (CV = std / mean) of the normalised signal vector
/// [|bias|, volatility_adj, confidence].  A high SDI indicates incoherent
/// signals that should be traded with reduced size or skipped.
///
/// The index is maintained as an EMA of the instantaneous CV so that it
/// adapts smoothly to changing market conditions.
///
/// Thread-safe: all public methods are thread-safe.
///
/// Feature flag: LLMQUANT_ENABLE_SENTIMENT_DISPERSION (default ON).

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SentimentDispersionIndex {
public:
    struct Config {
        /// EMA alpha for smoothing the instantaneous CV.  Default 0.15.
        double alpha{0.15};

        /// SDI level above which on_high_dispersion fires.  Default 0.8.
        double high_threshold{0.8};

        /// SDI level below which on_low_dispersion fires (coherent signals).  Default 0.2.
        double low_threshold{0.2};

        /// Fired when SDI crosses high_threshold upward.
        std::function<void(double sdi)> on_high_dispersion;

        /// Fired when SDI crosses low_threshold downward (entering coherent zone).
        std::function<void(double sdi)> on_low_dispersion;
    };

    explicit SentimentDispersionIndex(Config cfg = Config{});

    // Non-copyable.
    SentimentDispersionIndex(const SentimentDispersionIndex&)            = delete;
    SentimentDispersionIndex& operator=(const SentimentDispersionIndex&) = delete;

    /// Record a set of sentiment signals and update the dispersion index.
    ///
    /// @param abs_bias       Absolute directional bias ∈ [0, 1].
    /// @param volatility_adj Volatility adjustment ∈ [0, 1].
    /// @param confidence     Signal confidence ∈ [0, 1].
    void record(double abs_bias, double volatility_adj, double confidence) noexcept;

    /// Current smoothed SDI ∈ [0, ∞).  Values > 1 indicate very high dispersion.
    [[nodiscard]] double sdi() const noexcept;

    /// True when SDI > high_threshold (incoherent signals).
    [[nodiscard]] bool is_dispersed() const noexcept;

    /// True when SDI < low_threshold (coherent signals).
    [[nodiscard]] bool is_coherent() const noexcept;

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Total high-dispersion events fired.
    [[nodiscard]] uint64_t high_dispersion_events() const noexcept {
        return high_events_.load(std::memory_order_relaxed);
    }

    /// Total low-dispersion (coherence) events fired.
    [[nodiscard]] uint64_t low_dispersion_events() const noexcept {
        return low_events_.load(std::memory_order_relaxed);
    }

    /// Reset EMA and counters.
    void reset();

    /// Update config and reset.
    void update_config(const Config& cfg);

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    double sdi_ema_{0.0};
    bool   seeded_{false};
    bool   prev_high_{false};
    bool   prev_low_{true}; // start in "coherent" state

    std::atomic<double>   sdi_out_{0.0};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> high_events_{0};
    std::atomic<uint64_t> low_events_{0};

    mutable std::mutex mutex_;
};

} // namespace llmquant
