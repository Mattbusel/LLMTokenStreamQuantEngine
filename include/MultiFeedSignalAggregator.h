#pragma once

/// @file MultiFeedSignalAggregator.h
/// @brief Weighted aggregation of signals from multiple concurrent LLM feeds.
///
/// ## Motivation
/// Production deployments often run several LLM models in parallel (GPT-4o,
/// Claude, Gemini, fine-tuned domain models) to improve signal robustness.
/// When all feeds agree the signal is strong; when they diverge the market is
/// ambiguous and position sizing should be reduced.
///
/// ## Algorithm
/// Each registered feed contributes a bias value and confidence score.
/// Per feed the module maintains an EWMA of recent bias values.
/// The aggregated signal is a weight-normalised average of all feed EWMAs:
///   agg_bias = Σ(w_i × ema_i) / Σ(w_i)
///
/// Divergence is measured as the normalised standard deviation across feeds:
///   divergence = stddev(ema_i) / max(|agg_bias|, eps)  ∈ [0, ∞)
///   (clamped to [0, 1] for the callback threshold comparison)
///
/// Fires `on_divergence(div, dominant_feed)` when divergence first exceeds
/// `divergence_threshold`.  Fires `on_convergence(div)` when it drops back
/// below `divergence_threshold × clear_hysteresis`.
///
/// A feed that has not recorded a value in the current epoch is treated as
/// inactive and excluded from the aggregation.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_MULTI_FEED_AGGREGATOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

class MultiFeedSignalAggregator {
public:
    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------

    struct FeedSpec {
        std::string name;
        /// Relative importance weight (positive; normalised internally). Default 1.0.
        double weight{1.0};
        /// EWMA alpha for this feed's bias smoother. Default 0.20.
        double ema_alpha{0.20};
    };

    struct Config {
        /// Registered feeds.  May be empty at construction and populated later
        /// via add_feed() / update_config().
        std::vector<FeedSpec> feeds;

        /// Normalised divergence above which on_divergence fires. Default 0.50.
        double divergence_threshold{0.50};

        /// Hysteresis: on_convergence fires when divergence drops below
        /// divergence_threshold × clear_hysteresis. Default 0.70.
        double clear_hysteresis{0.70};

        /// Minimum number of active feeds required to produce a valid aggregate.
        /// If fewer feeds have reported, aggregated_bias() returns 0. Default 2.
        int min_feeds_active{2};

        /// Fired when divergence crosses above divergence_threshold.
        /// Parameters: current divergence score, name of dominant feed.
        std::function<void(double divergence, const std::string& dominant)> on_divergence;

        /// Fired when divergence drops back below the clear threshold.
        /// Parameter: current divergence score.
        std::function<void(double divergence)> on_convergence;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit MultiFeedSignalAggregator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a bias+confidence observation from the named feed.
    /// If the feed is not registered in cfg_.feeds it is silently ignored.
    void record(const std::string& feed_name, double bias, double confidence = 1.0);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Weighted average bias across all active feeds.  0 if fewer than
    /// min_feeds_active have reported in the current window.
    [[nodiscard]] double aggregated_bias() const noexcept {
        return agg_bias_.load(std::memory_order_relaxed);
    }

    /// Weighted average confidence across active feeds.
    [[nodiscard]] double aggregated_confidence() const noexcept {
        return agg_conf_.load(std::memory_order_relaxed);
    }

    /// Normalised divergence score ∈ [0, 1].  High value = feeds disagree.
    [[nodiscard]] double divergence() const noexcept {
        return divergence_.load(std::memory_order_relaxed);
    }

    /// True when divergence > divergence_threshold.
    [[nodiscard]] bool is_diverged() const noexcept {
        return diverged_.load(std::memory_order_relaxed);
    }

    /// Name of the feed currently contributing the largest |ema_bias|.
    [[nodiscard]] std::string dominant_feed() const;

    /// Total observations recorded across all feeds.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of times divergence threshold was crossed (rising edges).
    [[nodiscard]] uint64_t divergence_events() const noexcept {
        return div_events_.load(std::memory_order_relaxed);
    }

    /// Number of currently active feeds (recorded at least once).
    [[nodiscard]] int active_feed_count() const noexcept;

    [[nodiscard]] std::string to_stats_json() const;

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    void reset();
    void update_config(const Config& cfg);

private:
    struct FeedState {
        double weight{1.0};
        double ema_alpha{0.20};
        double ema_bias{0.0};
        double ema_conf{0.0};
        bool   active{false};
    };

    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::unordered_map<std::string, FeedState> feeds_;
    bool prev_diverged_{false};

    std::atomic<double>   agg_bias_{0.0};
    std::atomic<double>   agg_conf_{0.0};
    std::atomic<double>   divergence_{0.0};
    std::atomic<bool>     diverged_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> div_events_{0};
};

} // namespace llmquant
