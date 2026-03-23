#pragma once
/**
 * @file include/cross_asset_sentiment.hpp
 * @brief Cross-Asset Sentiment Correlation — rolling Pearson correlation
 *        between asset sentiment streams, single-linkage clustering, and
 *        regime/contagion detection.
 *
 * ## Components
 *
 * - `AssetSentiment`              : a (asset_id, sentiment_score, timestamp_ms) tuple.
 * - `SentimentCorrelationMatrix`  : rolling Pearson correlation between any two assets;
 *                                   leading-asset detection via lagged cross-correlation.
 * - `SentimentCluster`            : single-linkage agglomerative clustering over the
 *                                   pairwise correlation matrix.
 * - `SentimentRegimeDetector`     : fires an event when average cross-asset correlation
 *                                   spikes above a configurable threshold (contagion).
 *
 * ## Feature flag
 * Enabled by default via `LLMQUANT_ENABLE_CROSS_ASSET_SENTIMENT` in CMakeLists.txt.
 * Controlled at compile time by `LLMQUANT_CROSS_ASSET_SENTIMENT_ENABLED`.
 *
 * ## Thread safety
 * `SentimentCorrelationMatrix` is NOT thread-safe; protect external access with a mutex
 * when calling `update()` and `correlation()` from different threads.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ─── AssetSentiment ───────────────────────────────────────────────────────────

/**
 * @brief One sentiment observation for a single asset.
 */
struct AssetSentiment {
    std::string asset_id;     ///< Unique asset identifier (e.g. "BTC", "SPY")
    double sentiment_score;   ///< Polarity score in [-1, 1]
    int64_t timestamp_ms;     ///< Wall-clock timestamp in milliseconds

    /// Clamp score to [-1, 1] if needed.
    double clamped_score() const noexcept {
        return std::max(-1.0, std::min(1.0, sentiment_score));
    }
};

// ─── SentimentCorrelationMatrix ───────────────────────────────────────────────

/**
 * @brief Rolling Pearson correlation matrix for N asset sentiment streams.
 *
 * Maintains a per-asset rolling window of recent sentiment values.
 * Pearson correlation is computed on-demand from running sums over the
 * aligned portion of each window pair.
 *
 * Lagged correlation (lag 1..max_lag steps) is used by `leading_asset()` to
 * identify which asset's sentiment anticipates another's.
 */
class SentimentCorrelationMatrix {
public:
    struct Config {
        /** @brief Rolling window length (number of observations per asset). Default 60. */
        int window_size{60};

        /** @brief Minimum samples required to report a correlation. Default 10. */
        int min_samples{10};

        /** @brief Maximum lag (steps) tested in leading_asset(). Default 5. */
        int max_lag{5};

        /** @brief Minimum absolute correlation to consider a lagged relationship meaningful. Default 0.3. */
        double min_lag_corr{0.3};
    };

    explicit SentimentCorrelationMatrix(Config config = {}) noexcept;

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /**
     * @brief Register an asset and initialise its rolling window.
     *
     * If the asset is already registered, this is a no-op and the existing
     * window is preserved.
     *
     * @returns true if the asset was newly registered; false if already present.
     */
    bool register_asset(const std::string& asset_id);

    /**
     * @brief Add a new sentiment observation for one asset.
     *
     * @param sentiment  The observation.  The asset must have been registered
     *                   via `register_asset()` first.
     *
     * @returns true if accepted; false if the asset is unknown.
     */
    bool update(const AssetSentiment& sentiment);

    /**
     * @brief Convenience: add observations for multiple assets at once.
     *
     * @param sentiments  Vector of observations (may span multiple assets).
     *                    Unknown assets are silently skipped.
     */
    void update(const std::vector<AssetSentiment>& sentiments);

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /**
     * @brief Pearson correlation between two registered assets.
     *
     * @returns correlation in [-1, 1] if enough data exist; nullopt otherwise.
     */
    std::optional<double> correlation(
        const std::string& asset_a,
        const std::string& asset_b
    ) const noexcept;

    /**
     * @brief Find the asset whose sentiment most leads the target asset.
     *
     * Tests lags 1..config.max_lag and returns the asset with the highest
     * absolute lagged Pearson correlation (if >= min_lag_corr).
     *
     * @returns The leading asset_id, or nullopt if no significant lead is found.
     */
    std::optional<std::string> leading_asset(
        const std::string& target
    ) const noexcept;

    /**
     * @brief Number of registered assets.
     */
    int asset_count() const noexcept {
        return static_cast<int>(windows_.size());
    }

    /**
     * @brief Return all registered asset IDs.
     */
    std::vector<std::string> asset_ids() const;

    /**
     * @brief Current window size (number of observations) for an asset.
     * Returns 0 if the asset is unknown.
     */
    int window_size_for(const std::string& asset_id) const noexcept;

    /**
     * @brief Full correlation matrix for all registered asset pairs.
     * Pairs with insufficient data have correlation = 0.0.
     *
     * @returns vector<{asset_a, asset_b, rho}> for all unique pairs.
     */
    struct CorrelationEntry {
        std::string asset_a;
        std::string asset_b;
        double rho;
    };
    std::vector<CorrelationEntry> full_matrix() const noexcept;

    /**
     * @brief Average pairwise correlation across all registered asset pairs.
     * Returns 0.0 if fewer than 2 assets are registered.
     */
    double average_correlation() const noexcept;

private:
    Config config_;

    // Per-asset rolling window of recent sentiment values
    struct AssetWindow {
        std::deque<double> values;
    };
    std::unordered_map<std::string, AssetWindow> windows_;
    std::vector<std::string> id_order_;  // insertion order

    // Internal: compute Pearson r between two value sequences of equal length
    static double pearson(
        const std::vector<double>& xs,
        const std::vector<double>& ys
    ) noexcept;

    // Internal: extract the last min(n, window) values from a window
    static std::vector<double> tail(
        const std::deque<double>& dq,
        int n
    ) noexcept;

    // Internal: compute lagged correlation (lag > 0 means xs leads ys)
    static double lagged_pearson(
        const std::vector<double>& xs,
        const std::vector<double>& ys,
        int lag
    ) noexcept;
};

// ─── SentimentCluster ─────────────────────────────────────────────────────────

/**
 * @brief Groups assets by correlation above a threshold using single-linkage
 *        agglomerative clustering.
 *
 * Two assets are in the same cluster if their Pearson correlation (from a
 * SentimentCorrelationMatrix) exceeds `threshold`.  Single-linkage means an
 * asset joins a cluster if it is correlated with ANY member of that cluster.
 */
class SentimentCluster {
public:
    /** @brief One cluster: a set of correlated asset IDs. */
    struct Cluster {
        std::vector<std::string> members;

        int size() const noexcept { return static_cast<int>(members.size()); }

        bool contains(const std::string& id) const noexcept {
            return std::find(members.begin(), members.end(), id) != members.end();
        }
    };

    /**
     * @brief Build clusters from a correlation matrix.
     *
     * @param matrix     The source of pairwise correlations.
     * @param threshold  Minimum |correlation| for two assets to be linked.
     *
     * @returns Vector of clusters (non-singleton clusters listed first).
     */
    static std::vector<Cluster> build(
        const SentimentCorrelationMatrix& matrix,
        double threshold
    );
};

// ─── SentimentRegimeDetector ──────────────────────────────────────────────────

/**
 * @brief Detects contagion events when the average cross-asset correlation
 *        spikes above a configurable threshold.
 *
 * On each call to `update()`, the detector queries the matrix for the current
 * average correlation.  If it crosses `spike_threshold` from below (rising
 * edge), `on_contagion_event` is called.  A cooldown prevents repeated firings.
 */
class SentimentRegimeDetector {
public:
    struct Config {
        /** @brief Average correlation level that triggers a contagion event. Default 0.7. */
        double spike_threshold{0.70};

        /** @brief Minimum calls between consecutive event firings. Default 5. */
        int cooldown_steps{5};

        /** @brief Callback fired on contagion detection: (avg_correlation). */
        std::function<void(double avg_corr)> on_contagion_event;
    };

    explicit SentimentRegimeDetector(Config config = {}) noexcept;

    /**
     * @brief Update the detector with the current state of the correlation matrix.
     *
     * Computes average_correlation() and fires on_contagion_event if the spike
     * threshold is exceeded and the cooldown has expired.
     *
     * @returns Current average correlation.
     */
    double update(const SentimentCorrelationMatrix& matrix);

    /** @brief Number of contagion events fired so far. */
    int event_count() const noexcept { return event_count_; }

    /** @brief Most recently observed average correlation. */
    double last_avg_corr() const noexcept { return last_avg_corr_; }

    /** @brief Whether the detector is currently in a contagion regime. */
    bool in_contagion() const noexcept { return in_contagion_; }

    /** @brief Reset internal state (event count, cooldown, contagion flag). */
    void reset() noexcept;

private:
    Config config_;
    int event_count_{0};
    int steps_since_event_{0};
    double last_avg_corr_{0.0};
    bool in_contagion_{false};
};

}  // namespace llmquant
