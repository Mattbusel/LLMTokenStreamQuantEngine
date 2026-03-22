#pragma once

/// @file NarrativeCoherenceScorer.h
/// @brief Measures how self-consistent the LLM token stream's sentiment is —
///        low coherence reveals contradictory or confused narrative.
///
/// ## Motivation
/// A coherent narrative should have bias values that cluster around a consistent
/// sentiment.  When the LLM simultaneously produces bullish and bearish tokens in
/// the same span (high variance relative to mean magnitude), the narrative is
/// incoherent.  The coherence score distinguishes genuine conviction from hedged
/// or contradictory output.
///
/// ## Algorithm
/// Coherence = |μ| / (σ + ε)   where μ = rolling mean, σ = rolling std-dev.
/// This is a signal-to-noise ratio on the bias series:
///   - High coherence (|μ|/σ >> 1): consistent sentiment, reliable signal.
///   - Low coherence (|μ|/σ << 1): conflicting sentiment, reduce confidence.
/// Uses Welford's online algorithm for stable variance computation.
///
/// Fires `on_coherence_drop(score)` when coherence falls below `low_threshold`.
/// Fires `on_coherence_recover(score)` when coherence rises back above
/// `recover_threshold` (with hysteresis).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_COHERENCE_SCORER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeCoherenceScorer {
public:
    struct Config {
        /// Rolling window size. Default 40.
        int window{40};

        /// Minimum samples before emitting scores. Default 5.
        int min_samples{5};

        /// Coherence below this fires on_coherence_drop. Default 0.5.
        double low_threshold{0.5};

        /// Coherence above this clears the low state (hysteresis). Default 1.0.
        double recover_threshold{1.0};

        /// Small constant to avoid division by zero. Default 1e-6.
        double epsilon{1e-6};

        /// Fired when coherence crosses below low_threshold (outside lock).
        std::function<void(double score)> on_coherence_drop;

        /// Fired when coherence rises back above recover_threshold (outside lock).
        std::function<void(double score)> on_coherence_recover;
    };

    explicit NarrativeCoherenceScorer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current coherence score = |μ| / (σ + ε).
    [[nodiscard]] double coherence() const noexcept {
        return coherence_.load(std::memory_order_relaxed);
    }

    /// True when coherence < low_threshold.
    [[nodiscard]] bool is_incoherent() const noexcept {
        return incoherent_.load(std::memory_order_relaxed);
    }

    /// Rolling mean of the bias series.
    [[nodiscard]] double rolling_mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }

    /// Rolling standard deviation.
    [[nodiscard]] double rolling_stddev() const noexcept {
        return stddev_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    bool prev_incoherent_{false};

    std::atomic<double>   coherence_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<double>   stddev_{0.0};
    std::atomic<bool>     incoherent_{false};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
