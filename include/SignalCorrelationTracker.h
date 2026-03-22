#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling Pearson correlation tracker across named signal sources.
 *
 * Each signal source (e.g. a subreddit name or SignalBlendLayer source ID)
 * pushes scalar bias values via `record()`.  The tracker maintains a sliding
 * window of the most recent `window_size` samples per source and computes
 * pair-wise Pearson correlation on demand.
 *
 * ## Correlation callbacks
 * After every `record()` call the correlations for all pairs involving the
 * updated source are recomputed.  If a correlation crosses the convergence
 * or divergence threshold, the corresponding callback fires (from the
 * calling thread, under no lock — keep callbacks fast).
 *
 * ## Complexity
 * - `record()`: O(P * W) where P = number of pairs, W = window_size.
 * - `get_correlation()`: O(W) for one pair.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_CORRELATION` (default ON).
 */
class SignalCorrelationTracker {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Maximum number of samples retained per source.
         *
         * Older samples are evicted when this limit is exceeded.
         * Default 100.
         */
        int window_size{100};

        /**
         * @brief Minimum number of samples in both sources before a
         *        correlation is computed and compared to thresholds.
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Correlation below this value triggers the divergence
         *        callback.  Range [−1, 1].  Default −0.3.
         */
        double diverge_threshold{-0.3};

        /**
         * @brief Correlation above this value triggers the convergence
         *        callback.  Range [−1, 1].  Default 0.7.
         */
        double converge_threshold{0.7};
    };

    /**
     * @brief Callback type for correlation events.
     *
     * Parameters: source_a name, source_b name, computed correlation in [−1,1].
     * Called from the thread that invoked record().  Must not block.
     */
    using CorrelationCallback = std::function<void(const std::string& src_a,
                                                    const std::string& src_b,
                                                    double correlation)>;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalCorrelationTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /**
     * @brief Push a new bias sample for the given source.
     *
     * Creates the source window on first call.  After updating the window,
     * all pair-wise correlations involving this source are recomputed and
     * callbacks may fire.  Thread-safe.
     *
     * @param source  Logical source name (e.g. "r/wallstreetbets").
     * @param value   Scalar bias value (any real number; typically in [−1,1]).
     */
    void record(const std::string& source, double value);

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Compute Pearson correlation between two sources.
     *
     * Uses the shorter of the two windows; returns NaN if fewer than
     * min_samples are available in either source.  Thread-safe (read lock).
     *
     * @return Pearson r in [−1, 1], or NaN if insufficient data.
     */
    [[nodiscard]] double get_correlation(const std::string& src_a,
                                          const std::string& src_b) const;

    /**
     * @brief Return all source names currently tracked.
     */
    [[nodiscard]] std::vector<std::string> source_names() const;

    /**
     * @brief Return the number of samples available for a source.
     *
     * Returns 0 if the source has not been recorded.
     */
    [[nodiscard]] int sample_count(const std::string& source) const;

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------

    /**
     * @brief Register a callback for when two sources diverge.
     *
     * Fires when the correlation drops below Config::diverge_threshold.
     * Replaces any previously registered callback.  Thread-safe.
     */
    void set_divergence_callback(CorrelationCallback cb);

    /**
     * @brief Register a callback for when two sources converge.
     *
     * Fires when the correlation rises above Config::converge_threshold.
     * Replaces any previously registered callback.  Thread-safe.
     */
    void set_convergence_callback(CorrelationCallback cb);

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total samples recorded across all sources. */
    [[nodiscard]] uint64_t total_samples() const noexcept {
        return total_samples_.load(std::memory_order_relaxed);
    }

    /** @brief Number of divergence events fired. */
    [[nodiscard]] uint64_t divergence_events() const noexcept {
        return divergence_events_.load(std::memory_order_relaxed);
    }

    /** @brief Number of convergence events fired. */
    [[nodiscard]] uint64_t convergence_events() const noexcept {
        return convergence_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise stats and all pairwise correlations to a JSON string.
     */
    [[nodiscard]] std::string to_stats_json() const;

    // -----------------------------------------------------------------------
    // Runtime configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration; does not clear existing windows.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

private:
    mutable std::mutex mutex_;
    Config             config_;

    // Per-source sliding window.
    std::unordered_map<std::string, std::deque<double>> windows_;

    // Callbacks.
    CorrelationCallback divergence_cb_;
    CorrelationCallback convergence_cb_;

    std::atomic<uint64_t> total_samples_{0};
    std::atomic<uint64_t> divergence_events_{0};
    std::atomic<uint64_t> convergence_events_{0};

    // Internal Pearson over two equal-length aligned tails.
    [[nodiscard]] static double pearson(const std::deque<double>& a,
                                         const std::deque<double>& b,
                                         int n) noexcept;

    // Recompute all pairs involving `source`; fire callbacks as needed.
    // Called with mutex_ held; callbacks fired AFTER releasing mutex_.
    struct PendingEvent {
        std::string src_a, src_b;
        double      correlation;
        bool        is_divergence;
    };

    [[nodiscard]] std::vector<PendingEvent>
    recompute_pairs_locked(const std::string& source);
};

} // namespace llmquant
