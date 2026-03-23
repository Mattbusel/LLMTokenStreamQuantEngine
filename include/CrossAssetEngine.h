#pragma once

/**
 * @file CrossAssetEngine.h
 * @brief Cross-asset signal correlation engine.
 *
 * CrossAssetEngine correlates token-stream signals across multiple assets
 * simultaneously. It maintains a rolling Pearson correlation matrix updated
 * via Welford's online algorithm for numerical stability.
 *
 * When a signal fires for one asset the engine can:
 *  - Amplify conviction if positively correlated peers also show aligned signals.
 *  - Compute a hedge ratio for portfolio construction against a short-side asset.
 *
 * ## Thread safety
 * All public methods acquire an internal mutex. The class is safe to call from
 * multiple threads provided callers do not hold external locks that could
 * deadlock with each other through this object.
 *
 * ## Algorithm
 * Welford's one-pass online algorithm maintains per-pair running
 * sum-of-cross-products. Pearson r is derived on demand as:
 *
 *   r(a,b) = cov(a,b) / (sigma_a * sigma_b)
 *
 * where cov, sigma are computed from the Welford accumulators. A sliding
 * window of the most recent `window_size` samples is kept per symbol; when
 * the window overflows the oldest sample is evicted via the incremental
 * "remove-one" Welford correction.
 */

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One observed signal sample for a single asset.
struct AssetSignal {
    std::string symbol;
    double      weight{0.0};        ///< Accumulated token weight in [0, 1].
    double      confidence{0.0};    ///< Signal confidence in [0, 1].
    int64_t     timestamp_ns{0};    ///< Nanoseconds since Unix epoch.
};

/// Snapshot of the full N×N correlation matrix.
struct CorrelationMatrix {
    std::vector<std::string>         symbols;        ///< Ordered symbol list (row == column index).
    std::vector<std::vector<double>> matrix;         ///< matrix[i][j] = Pearson r(symbols[i], symbols[j]).
    int64_t                          last_updated_ns{0};
    size_t                           sample_count{0};
};

// ---------------------------------------------------------------------------
// CrossAssetEngine
// ---------------------------------------------------------------------------

/**
 * @brief Rolling cross-asset signal correlation engine.
 *
 * Tracks up to an unbounded number of asset symbols. Symbols are registered
 * implicitly on the first call to update_signal() for that symbol.
 *
 * @note The engine uses a per-pair Welford accumulator.  When the window is
 *       full the oldest (evicted) sample is subtracted via the Welford
 *       online downdate formula to keep accumulators bounded.
 */
class CrossAssetEngine {
public:
    /**
     * @param window_size  Number of most-recent samples to keep per symbol.
     *                     Must be >= 2. Default: 100.
     */
    explicit CrossAssetEngine(size_t window_size = 100);

    // Non-copyable; move is fine.
    CrossAssetEngine(const CrossAssetEngine&)            = delete;
    CrossAssetEngine& operator=(const CrossAssetEngine&) = delete;
    CrossAssetEngine(CrossAssetEngine&&)                 = default;
    CrossAssetEngine& operator=(CrossAssetEngine&&)      = default;

    ~CrossAssetEngine() = default;

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new signal observation for the given asset.
     *
     * If the symbol is not yet tracked it is registered automatically.
     * The effective scalar fed into the correlation is:
     *   value = signal.weight * signal.confidence
     */
    void update_signal(const AssetSignal& signal);

    /**
     * @brief Remove all state for a symbol (window + Welford accumulators + pair accumulators).
     */
    void clear_symbol(const std::string& symbol);

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Full N×N snapshot. O(N^2) — allocates.
    CorrelationMatrix get_correlations() const;

    /**
     * @brief Pearson r between two specific symbols in [-1, 1].
     *
     * Returns 0.0 if either symbol is unknown or the window is too small
     * (fewer than 2 joint observations).
     */
    double get_pairwise_correlation(const std::string& a, const std::string& b) const;

    /**
     * @brief Compute a conviction multiplier in [0.5, 2.0] for a primary signal.
     *
     * Iterates over all other tracked symbols.  For each pair:
     *   - r > +threshold: correlated peer — if peer signal aligns, add boost.
     *   - r < -threshold: anti-correlated peer — if peer signal also fires
     *     the same direction, add a small penalty (divergence).
     *
     * @return Multiplier >1.0 when correlated assets confirm the primary signal,
     *         <1.0 when they contradict it, 1.0 when uncorrelated or no peers.
     */
    double compute_conviction_multiplier(const std::string& primary_symbol) const;

    /**
     * @brief Minimum-variance hedge ratio for a long/short pair.
     *
     * Derived from the beta of `long_sym` regressed on `short_sym`:
     *   beta = r(a,b) * (sigma_a / sigma_b)
     *
     * Returns 0.0 if the pair is not sufficiently correlated (|r| < 0.2)
     * or if standard deviations are near-zero.
     */
    double hedge_ratio(const std::string& long_sym, const std::string& short_sym) const;

    /// List of all currently tracked symbols (arbitrary order).
    std::vector<std::string> tracked_symbols() const;

    /// Number of samples currently stored in the window for a symbol.
    size_t window_size_for(const std::string& symbol) const;

private:
    // -----------------------------------------------------------------------
    // Internal state
    // -----------------------------------------------------------------------

    size_t window_size_;

    /**
     * @brief Per-symbol rolling window of effective scalar values.
     *
     * Each entry is weight * confidence at the time of the call.
     */
    std::unordered_map<std::string, std::vector<double>> signal_windows_;

    /**
     * @brief Welford accumulator for single-symbol variance.
     *
     * Used to compute sigma_a, sigma_b for normalisation in Pearson r.
     */
    struct WelfordState {
        double mean{0.0};
        double M2{0.0};    ///< Sum of squared deviations (for variance = M2/n).
        double count{0.0};
    };
    std::unordered_map<std::string, WelfordState> welford_states_;

    /**
     * @brief Welford-style cross-product accumulator for each symbol pair.
     *
     * Key = canonical pair key produced by pair_key(a, b).
     * Stores running sum of (xi - mean_x)(yi - mean_y) analogues needed
     * for online Pearson r.
     */
    struct PairState {
        double sum_xy{0.0};   ///< Running Σ (xi * yi) — used with per-symbol means.
        double count{0.0};    ///< Number of joint (synchronous) observations.
    };
    std::unordered_map<std::string, PairState> pair_states_;

    /**
     * @brief Global sample count across all update_signal calls.
     */
    size_t total_samples_{0};

    /**
     * @brief Latest wall-clock timestamp observed (ns).
     */
    int64_t last_updated_ns_{0};

    mutable std::mutex mutex_;

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Canonical sort-order pair key so pair_key(a,b) == pair_key(b,a).
    static std::string pair_key(const std::string& a, const std::string& b);

    /**
     * @brief Internal Pearson r computation (must be called with mutex held).
     *
     * Returns 0.0 when sample count < 2 or either sigma is near-zero.
     */
    double pearson_correlation_locked(const std::string& a, const std::string& b) const;

    /**
     * @brief Evict the oldest sample from symbol's window when full.
     *
     * Applies the Welford downdate to keep WelfordState consistent.
     */
    void evict_old_sample(const std::string& symbol, double old_value);

    /**
     * @brief Update the pair accumulator for (sym, peer) given a new value for sym.
     *
     * Iterates all other tracked symbols using their most-recent window entry.
     * Must be called with mutex held after updating sym's own WelfordState.
     */
    void update_pair_states(const std::string& sym, double new_value);
};

} // namespace llmquant
