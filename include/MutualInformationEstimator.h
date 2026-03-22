#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Estimates the mutual information (MI) between the sentiment stream
 *        and price returns using a histogram-based estimator.
 *
 * Correlation measures only linear relationships.  Mutual information captures
 * all statistical dependencies (linear and non-linear):
 *
 *   MI(X; Y) = Σ_{x,y} P(x,y) * log[ P(x,y) / (P(x)*P(y)) ]
 *
 * A high MI indicates the sentiment stream carries predictive information about
 * price returns that may not be visible in a Pearson correlation.
 *
 * ## Algorithm
 * Both signals are discretised into configurable bins.  A 2D joint histogram
 * is maintained over a rolling window.  MI is estimated as:
 *
 *   MI ≈ Σ_{i,j} (n_{ij}/N) * log[ (n_{ij}/N) / ((n_i/N)*(n_j/N)) ]
 *
 * where n_{ij} is the joint count, n_i / n_j are the marginal counts, N total.
 *
 * ## Intended use
 * - Call record(sentiment, return) after each resolved trade.
 * - Use mi() to assess signal quality: values near 0 = noise, values > 0.1
 *   suggest meaningful nonlinear relationship.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_MUTUAL_INFO` (default ON).
 */
class MutualInformationEstimator {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Number of histogram bins for each signal.
         *
         * MI resolution improves with more bins but requires more data.
         * Default 10.
         */
        int bins{10};

        /**
         * @brief Rolling window size (number of (sentiment, return) pairs).
         *
         * Default 200.
         */
        int window_size{200};

        /**
         * @brief Minimum samples before MI is computed.
         *
         * Default 30.
         */
        int min_samples{30};

        /**
         * @brief Expected range for the sentiment signal [lo, hi].
         *
         * Values outside this range are clamped.
         * Default [-1, 1].
         */
        double sentiment_lo{-1.0};
        double sentiment_hi{ 1.0};

        /**
         * @brief Expected range for the return signal [lo, hi].
         *
         * Values outside this range are clamped.
         * Default [-0.1, 0.1] (±10% return).
         */
        double return_lo{-0.1};
        double return_hi{ 0.1};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit MutualInformationEstimator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a (sentiment, return) pair.
     *
     * Updates the rolling 2D histogram and recomputes MI.  Thread-safe.
     *
     * @param sentiment  Scalar sentiment (e.g. decayed_bias).
     * @param ret        Concurrent or subsequent price return (normalised).
     */
    void record(double sentiment, double ret);

    /**
     * @brief Reset all state.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current mutual information estimate (in nats).
     *
     * Returns 0 when fewer than min_samples have been recorded.
     * Thread-safe.
     */
    [[nodiscard]] double mi() const noexcept;

    /**
     * @brief Normalised MI ∈ [0, 1].
     *
     * NMI = MI / min(H(X), H(Y)), where H is Shannon entropy.
     * Returns 0 when mi() == 0.  Thread-safe.
     */
    [[nodiscard]] double normalized_mi() const noexcept;

    /**
     * @brief Number of pairs in the rolling window.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total pairs recorded (across all resets). */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears the window.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // 2D histogram (flattened): hist_[i * bins + j] = joint count
    std::vector<int> hist_;

    // Per-observation (i, j) bin indices for the rolling window.
    struct Obs { int i; int j; };
    std::vector<Obs> window_;
    int window_head_{0};
    int window_count_{0};

    double mi_{0.0};
    double nmi_{0.0};

    std::atomic<uint64_t> total_records_{0};

    void recompute();  // called under lock
    int sentiment_bin(double v) const noexcept;
    int return_bin(double v) const noexcept;
};

} // namespace llmquant
