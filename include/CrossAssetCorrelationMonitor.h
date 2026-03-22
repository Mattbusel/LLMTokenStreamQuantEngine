#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Tracks rolling pairwise Pearson correlation between sentiment signals
 *        across multiple asset symbols.
 *
 * In multi-asset LLM pipelines, cross-asset correlation shifts carry alpha:
 *
 *  - **Correlation spike (→ +1)**: risk-off flight-to-quality, macro event
 *    affecting all assets simultaneously.  Reduce exposure; hedge.
 *
 *  - **Correlation breakdown (→ 0 or negative)**: potential divergence alpha.
 *    One asset's LLM sentiment diverges from the rest — a candidate for a
 *    spread / pairs trade.
 *
 * ## Method
 * Maintains a rolling window of (signal_a, signal_b) pairs per asset pair.
 * Pearson correlation is computed using online mean/variance accumulators:
 *   ρ = cov(a, b) / (σ_a × σ_b)
 *
 * ## Usage
 * @code
 * CrossAssetCorrelationMonitor mon;
 * mon.register_asset("SPY");
 * mon.register_asset("QQQ");
 *
 * // On each signal update:
 * mon.record("SPY", spyConfidence);
 * mon.record("QQQ", qqqConfidence);
 *
 * // Query:
 * double rho = mon.correlation("SPY", "QQQ");
 * if (rho < 0.2) explore_pairs_trade();
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CROSS_ASSET_CORR` (default ON).
 */
class CrossAssetCorrelationMonitor {
public:
    /** @brief Maximum number of asset symbols supported. */
    static constexpr int kMaxAssets = 16;

    struct Config {
        /**
         * @brief Rolling window size per pair (number of joint observations).
         * Default 100.
         */
        int window_size{100};

        /**
         * @brief Minimum joint observations before correlation is reported.
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Correlation threshold above which on_high_correlation fires.
         * Default 0.85.
         */
        double high_corr_threshold{0.85};

        /**
         * @brief Correlation threshold below which on_low_correlation fires.
         * Default 0.15 (near-zero / negative).
         */
        double low_corr_threshold{0.15};

        /**
         * @brief Callback fired when a pair transitions to high correlation.
         * Parameters: (asset_a, asset_b, correlation).
         */
        std::function<void(const std::string& a, const std::string& b, double rho)>
            on_high_correlation;

        /**
         * @brief Callback fired when a pair transitions to low correlation.
         * Parameters: (asset_a, asset_b, correlation).
         */
        std::function<void(const std::string& a, const std::string& b, double rho)>
            on_low_correlation;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit CrossAssetCorrelationMonitor(Config config = Config{});

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /**
     * @brief Register an asset symbol.
     *
     * Must be called before record() for that asset.
     * Returns the integer ID assigned to the asset, or -1 if kMaxAssets
     * would be exceeded.
     * Thread-safe.
     */
    int register_asset(const std::string& name);

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new signal observation for the given asset.
     *
     * Appends to all rolling windows that involve this asset and recomputes
     * correlations for affected pairs.  Fires threshold callbacks outside
     * the internal lock.
     * Thread-safe.
     *
     * @param name   Asset symbol (must already be registered).
     * @param value  Signal value (e.g. confidence or bias).
     * @return false if the asset is unknown.
     */
    bool record(const std::string& name, double value) noexcept;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Pearson correlation between two registered assets.
     *
     * Returns 0.0 if either asset is unknown, the pair has too few samples,
     * or variance is near zero.  Thread-safe.
     */
    [[nodiscard]] double correlation(const std::string& a, const std::string& b) const noexcept;

    /**
     * @brief Number of registered assets.  Thread-safe.
     */
    [[nodiscard]] int asset_count() const noexcept {
        return n_assets_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total observations recorded across all assets.  Thread-safe.
     */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of high-correlation threshold events fired.  Thread-safe.
     */
    [[nodiscard]] uint64_t high_corr_events() const noexcept {
        return high_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of low-correlation threshold events fired.  Thread-safe.
     */
    [[nodiscard]] uint64_t low_corr_events() const noexcept {
        return low_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /** @brief Update configuration; clears all windows. */
    void update_config(Config config);

    /** @brief Reset all windows and correlation estimates. */
    void reset();

    /** @brief JSON snapshot: asset list + all pairwise correlations. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    // Per-pair rolling window.
    struct PairState {
        std::vector<std::pair<double, double>> ring;
        int    head{0};
        int    fill{0};
        double sum_a{0}, sum_b{0}, sum_aa{0}, sum_bb{0}, sum_ab{0};
        double rho{0.0};
        bool   was_high{false};
        bool   was_low{false};
    };

    int  pair_index(int i, int j) const noexcept;
    void recompute_pair(PairState& ps) noexcept;

    mutable std::mutex mutex_;
    Config config_;

    std::unordered_map<std::string, int> asset_ids_;
    std::vector<std::string>             asset_names_;
    std::vector<PairState>               pairs_;
    std::vector<double>                  last_value_;

    std::atomic<int>      n_assets_{0};
    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> high_events_{0};
    std::atomic<uint64_t> low_events_{0};
};

} // namespace llmquant
