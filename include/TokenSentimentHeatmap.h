#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief 2D rolling histogram of (token_id × sentiment_bucket) counts.
 *
 * Reveals which tokens consistently carry bullish or bearish signal —
 * enabling token-level alpha attribution.  For each named token, a fixed
 * set of sentiment buckets accumulates observation counts with optional
 * exponential forgetting so the histogram adapts to regime changes.
 *
 * ## Buckets
 * The sentiment axis is divided into `num_buckets` equal-width bins
 * spanning [min_sentiment, max_sentiment].  Values outside this range
 * are clamped into the edge buckets.
 *
 * ## Forgetting
 * On each `record()` call, all bucket counts are multiplied by `forgetting`
 * before the new observation is added.  Set forgetting=1.0 for cumulative
 * counts (no decay).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_HEATMAP` (default ON).
 */
class TokenSentimentHeatmap {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Number of sentiment buckets along the Y axis.
         *
         * Default 10.
         */
        int num_buckets{10};

        /**
         * @brief Minimum sentiment value (maps to bucket 0).
         *
         * Default -1.0.
         */
        double min_sentiment{-1.0};

        /**
         * @brief Maximum sentiment value (maps to bucket num_buckets-1).
         *
         * Default +1.0.
         */
        double max_sentiment{1.0};

        /**
         * @brief Exponential forgetting factor applied before each observation.
         *
         * 1.0 = no forgetting (accumulate all history).
         * 0.99 = gentle decay.  Default 1.0.
         */
        double forgetting{1.0};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenSentimentHeatmap(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a (token_id, sentiment_value) observation.
     *
     * If the token_id has not been seen before, a new row is allocated.
     * Thread-safe.
     *
     * @param token_id   String identifier for the token (e.g. "bullish").
     * @param sentiment  Scalar sentiment value to bin.
     */
    void record(const std::string& token_id, double sentiment);

    /**
     * @brief Reset all counts.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Bucket counts for a given token.
     *
     * Returns an empty vector if the token_id has not been recorded.
     * Thread-safe.
     */
    [[nodiscard]] std::vector<double> bucket_counts(const std::string& token_id) const;

    /**
     * @brief Index of the peak (most observed) bucket for a token.
     *
     * Returns -1 if the token is unknown.
     * Thread-safe.
     */
    [[nodiscard]] int peak_bucket(const std::string& token_id) const;

    /**
     * @brief Centre sentiment value of a given bucket index.
     */
    [[nodiscard]] double bucket_centre(int bucket_idx) const noexcept;

    /**
     * @brief The dominant sentiment direction for a token: positive, negative, or neutral.
     *
     * Computes the weighted mean sentiment over all buckets and returns its sign.
     *
     * @return  +1.0 (bullish), -1.0 (bearish), or 0.0 (neutral/unknown).
     */
    [[nodiscard]] double dominant_sentiment(const std::string& token_id) const;

    /**
     * @brief Number of distinct token IDs tracked.
     *
     * Thread-safe.
     */
    [[nodiscard]] int token_count() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total observations recorded. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears all counts.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise top-N tokens (by total count) to a JSON object string.
     *
     * @param top_n  Maximum number of token rows to include.  0 = all.
     */
    [[nodiscard]] std::string to_stats_json(int top_n = 5) const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // token_id → vector of bucket counts (length = num_buckets)
    std::unordered_map<std::string, std::vector<double>> rows_;

    std::atomic<uint64_t> total_records_{0};

    [[nodiscard]] int sentiment_to_bucket(double sentiment) const noexcept;
};

} // namespace llmquant
