#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Measures the Shannon self-information (surprise) of each bias signal
 *        relative to its own empirical distribution.
 *
 * In information theory the self-information of an event x with probability p
 * is defined as:  I(x) = -log₂(p(x))
 *
 * Applied to the trade-signal bias stream this becomes: a bias shift that has
 * been seen many times before (high p) has low surprise (near 0 bits).
 * A bias shift that sits in an infrequently visited region of the distribution
 * has high surprise (many bits).
 *
 * ## Why this matters for trading
 * High-surprise signals are statistically anomalous and warrant heightened
 * scrutiny:
 *  - They may indicate a genuine regime shift (news-breaking event).
 *  - They may indicate a data-quality issue (hallucination, feed error).
 *  - They may indicate adversarial injection.
 *
 * The normalized surprise score ∈ [0, 1] is computed as:
 *   surprise = I(x) / log₂(n_bins)   (0 = expected, 1 = maximally surprising)
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_SURPRISE` (default ON).
 */
class SignalSurpriseIndex {
public:
    /**
     * @brief Configuration for the surprise index.
     */
    struct Config {
        /** @brief Histogram range lower bound. */
        double range_min{-3.0};

        /** @brief Histogram range upper bound. */
        double range_max{3.0};

        /** @brief Number of histogram bins. More bins = finer granularity. */
        int n_bins{64};

        /**
         * @brief Minimum number of observations before surprise estimates are
         *        meaningful.  Before this, surprise() returns 1.0 (maximally
         *        surprising — we have no prior).
         */
        int min_samples{32};

        /**
         * @brief Normalized surprise threshold for is_high_surprise().
         *
         * 0.75 means the signal falls in a region accounting for only 1/2^(0.75*log2(64))
         * ≈ bottom ~4% of the distribution.
         */
        double high_surprise_threshold{0.75};

        /**
         * @brief Callback fired when a high-surprise signal is observed.
         *
         * Parameters: (bias_shift, normalized_surprise_score).
         */
        std::function<void(double bias, double surprise)> on_high_surprise;
    };

    /** @brief Construct with default configuration. */
    SignalSurpriseIndex();

    /** @brief Construct with specified configuration. */
    explicit SignalSurpriseIndex(Config cfg);

    /**
     * @brief Record a new bias observation; updates the histogram and
     *        computes the self-information for the current observation.
     *
     * Fires on_high_surprise callback outside the internal lock if the
     * normalized surprise exceeds high_surprise_threshold.
     */
    void record(double bias) noexcept;

    /** @brief Update configuration; resets histogram. */
    void update_config(Config cfg);

    /** @brief Return a copy of the current configuration. */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Normalized surprise of the most-recently recorded observation.
     *
     * Range [0, 1].  Returns 1.0 when min_samples not yet reached.
     * Thread-safe.
     */
    [[nodiscard]] double surprise() const noexcept {
        return surprise_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns true when the last surprise exceeded high_surprise_threshold.
     * Thread-safe.
     */
    [[nodiscard]] bool is_high_surprise() const noexcept {
        return surprise_.load(std::memory_order_relaxed) >=
               threshold_cache_.load(std::memory_order_relaxed);
    }

    /**
     * @brief EMA-smoothed average surprise over recent observations.
     * Thread-safe.
     */
    [[nodiscard]] double mean_surprise() const noexcept {
        return mean_surprise_.load(std::memory_order_relaxed);
    }

    /** @brief Total observations recorded. Thread-safe. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /** @brief Total high-surprise events. Thread-safe. */
    [[nodiscard]] uint64_t high_surprise_count() const noexcept {
        return high_count_.load(std::memory_order_relaxed);
    }

    /** @brief Reset histogram and counters. */
    void reset();

    /** @brief Serialise current state to a compact JSON string. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex      mutex_;
    Config                  config_;
    std::vector<uint64_t>   hist_;  // frequency histogram

    // Lock-free outputs
    std::atomic<double>     surprise_{1.0};
    std::atomic<double>     mean_surprise_{1.0};
    std::atomic<double>     threshold_cache_{0.75};
    std::atomic<uint64_t>   total_{0};
    std::atomic<uint64_t>   high_count_{0};
};

} // namespace llmquant
