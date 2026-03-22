#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Cross-correlation lead-lag detector for two co-evolving time series.
 *
 * Maintains a rolling window of samples for signal X and signal Y, then
 * computes the Pearson cross-correlation at every integer lag in
 * [-max_lag, +max_lag].  The lag with the highest |correlation| is the
 * *optimal lag*:
 *
 *   lag > 0  →  X leads Y   (X is a leading indicator for Y)
 *   lag < 0  →  Y leads X   (Y is a leading indicator for X)
 *   lag = 0  →  contemporaneous relationship
 *
 * ## Practical uses in this engine
 * | X                  | Y              | Interpretation                         |
 * |--------------------|----------------|----------------------------------------|
 * | TokenEntropy EMA   | delta_bias     | Does entropy spike *before* bias moves? |
 * | Signal velocity    | P99 latency    | Does fast signal cause latency spikes?  |
 * | Block rate EMA     | |bias| magnitude | Does risk gate dampen bias swings?    |
 *
 * ## Thread safety
 * `push_x()` and `push_y()` are individually thread-safe.  `compute()` is
 * also thread-safe (acquires the internal mutex for a snapshot).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_LEAD_LAG` (default ON).
 */
class LeadLagDetector {
public:
    /**
     * @brief Configuration for the detector.
     */
    struct Config {
        /**
         * @brief Rolling window size.  Must be > 2 * max_lag.
         *
         * Default 128.
         */
        int window_size{128};

        /**
         * @brief Maximum lag to test (in number of samples).
         *
         * The detector evaluates lags from -max_lag to +max_lag.
         * Default 16.
         */
        int max_lag{16};
    };

    /**
     * @brief Cross-correlation result at a single lag.
     */
    struct LagResult {
        int    lag{0};      ///< Lag value (positive = X leads Y).
        double corr{0.0};   ///< Pearson r at this lag.
    };

    /**
     * @brief Full cross-correlogram: one entry per lag tested.
     */
    struct Correlogram {
        std::vector<LagResult> lags;   ///< Sorted by lag value.
        int    optimal_lag{0};         ///< Lag with highest |r|.
        double max_corr{0.0};          ///< |r| at optimal_lag (sign preserved).
        bool   x_leads{false};         ///< true when optimal_lag > 0.
        bool   y_leads{false};         ///< true when optimal_lag < 0.

        /**
         * @brief Friendly description of the relationship.
         *
         * e.g. "X leads Y by 3 (r=+0.71)" or "contemporaneous (r=+0.42)".
         */
        [[nodiscard]] std::string describe() const;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit LeadLagDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Data ingestion (thread-safe, one sample at a time)
    // -----------------------------------------------------------------------

    /**
     * @brief Push a new sample for signal X.
     *
     * @param value The new X observation.
     */
    void push_x(double value) noexcept;

    /**
     * @brief Push a new sample for signal Y.
     *
     * @param value The new Y observation.
     */
    void push_y(double value) noexcept;

    /**
     * @brief Push both signals simultaneously (same timestamp).
     *
     * Equivalent to calling push_x then push_y in one lock acquisition.
     *
     * @param x New X observation.
     * @param y New Y observation.
     */
    void push(double x, double y) noexcept;

    // -----------------------------------------------------------------------
    // Computation
    // -----------------------------------------------------------------------

    /**
     * @brief Compute the full cross-correlogram over [-max_lag, +max_lag].
     *
     * Returns an empty Correlogram if fewer than 2*max_lag+1 samples exist
     * for either signal.  Thread-safe (snapshots the ring buffers).
     *
     * @return Correlogram with optimal_lag, max_corr, and the full lag table.
     */
    [[nodiscard]] Correlogram compute() const;

    /**
     * @brief Convenience accessor for the optimal lag only.
     *
     * Returns 0 if insufficient data or no signal above noise_floor.
     */
    [[nodiscard]] int optimal_lag() const { return compute().optimal_lag; }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /**
     * @brief Total samples pushed to X.  Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_x() const noexcept {
        return total_x_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total samples pushed to Y.  Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_y() const noexcept {
        return total_y_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise the latest correlogram to a JSON object string.
     *
     * Fields: optimal_lag, max_corr, x_leads, y_leads, total_x, total_y.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Update the configuration.  Clears all window data.  Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.  Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Reset all ring buffer data.  Thread-safe.
     */
    void reset();

private:
    mutable std::mutex mutex_;
    Config config_;

    std::vector<double> ring_x_;
    std::vector<double> ring_y_;
    int head_x_{0};
    int head_y_{0};
    int fill_x_{0};
    int fill_y_{0};

    std::atomic<uint64_t> total_x_{0};
    std::atomic<uint64_t> total_y_{0};

    // Read `n` samples from a ring in chronological order (oldest first).
    [[nodiscard]] static std::vector<double>
    snapshot(const std::vector<double>& ring, int head, int fill);

    [[nodiscard]] static double pearson_at_lag(const std::vector<double>& x,
                                               const std::vector<double>& y,
                                               int lag);
};

} // namespace llmquant
