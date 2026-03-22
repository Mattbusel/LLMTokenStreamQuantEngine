#pragma once

/// @file SentimentAutocorrelationMeter.h
/// @brief Rolling autocorrelation of the sentiment bias series at lags 1..N.
///
/// ## Motivation
/// Whether a sentiment bias series is trending or mean-reverting is crucial for
/// signal persistence decisions.  A bias series with high lag-1 autocorrelation
/// is trending (use momentum strategies); low or negative lag-1 AC indicates
/// noise or mean reversion.  Higher lags reveal periodic patterns.
///
/// ## Algorithm
/// Maintains a circular buffer of the last `window` bias values.
/// For each lag k ∈ [1, max_lag]:
///   AC(k) = Cov(x_t, x_{t-k}) / Var(x_t)
///         = [Σ (x_i − μ)(x_{i-k} − μ)] / [Σ (x_i − μ)²]
/// Computed in O(window × max_lag) per update (typically tiny for small max_lag).
///
/// Fires `on_regime_change(lag1_ac, is_trending)` when the lag-1 sign changes.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_AUTOCORR_METER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SentimentAutocorrelationMeter {
public:
    static constexpr int kMaxLag = 10;

    struct Config {
        /// Rolling window size. Default 50.
        int window{50};

        /// Maximum lag to compute. Must be ≤ kMaxLag. Default 5.
        int max_lag{5};

        /// |lag-1 AC| threshold to be considered "trending". Default 0.2.
        double trending_threshold{0.2};

        /// Fired when trending status (lag-1 sign / magnitude) changes.
        /// Parameters: (lag1_ac, is_trending).
        std::function<void(double lag1_ac, bool is_trending)> on_regime_change;
    };

    explicit SentimentAutocorrelationMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Autocorrelation at lag k ∈ [1, max_lag]. Returns 0 if not enough data.
    [[nodiscard]] double autocorrelation(int lag) const noexcept;

    /// True when lag-1 AC exceeds the trending threshold.
    [[nodiscard]] bool is_trending() const noexcept {
        return trending_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;   // circular buffer
    int    head_{0};
    int    fill_{0};

    double ac_[kMaxLag + 1]{};  // ac_[1..max_lag]
    bool   prev_trending_{false};
    bool   trending_seeded_{false};

    std::atomic<bool>     trending_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
