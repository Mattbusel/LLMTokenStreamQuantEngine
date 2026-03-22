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
 * @brief Herfindahl-Hirschman Index (HHI) concentration guard.
 *
 * Tracks how concentrated the signal pipeline is across named thematic
 * sources ("macro", "earnings", "sentiment", etc.).  Each call to
 * `record_signal(theme, bias)` accumulates the absolute bias for that
 * theme in a rolling window.  The HHI is then:
 *
 *   HHI = Σ (share_i)²   where  share_i = |bias_i| / Σ|bias_j|
 *
 * Interpretation:
 * - HHI ≈ 0: equal contribution from many themes → diversified signals.
 * - HHI → 1: one theme dominates → concentrated, fragile signal pipeline.
 *
 * The guard fires a callback when HHI rises above `concentration_threshold`
 * (and clears when it falls back below a hysteresis level).  It also
 * reports the dominant theme name and its share.
 *
 * ## Design
 * - `record_signal()` is O(1) amortised (ring-buffer per theme).
 * - `hhi()` is O(K) in the number of themes — fine for monitoring.
 * - Thread-safe: atomic hot-path outputs, mutex for per-theme state.
 */
class PositionConcentrationGuard {
public:
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    struct Config {
        int    window_size             = 64;   ///< Rolling window per theme.
        double concentration_threshold = 0.50; ///< HHI above which alert fires.
        double clear_hysteresis        = 0.80; ///< Clear when HHI < threshold * hysteresis.

        /// Fired when HHI rises above concentration_threshold.
        std::function<void(double, const std::string&)> on_concentrated;
        /// Fired when HHI falls back below the hysteresis level.
        std::function<void(double)> on_diversified;
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    PositionConcentrationGuard();
    explicit PositionConcentrationGuard(const Config& cfg);

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------

    /**
     * @brief Record a signal contribution from a named theme.
     * @param theme  Theme identifier (e.g., "macro", "earnings", "fed").
     * @param bias   Absolute or signed bias contribution; magnitude is used.
     */
    void record_signal(const std::string& theme, double bias);

    // -------------------------------------------------------------------------
    // Accessors (lock-free hot path)
    // -------------------------------------------------------------------------

    /// Most recently computed HHI [0,1].
    double hhi() const noexcept;
    /// True if HHI currently exceeds concentration_threshold.
    bool is_concentrated() const noexcept;
    /// Total signals recorded across all themes.
    uint64_t total_signals() const noexcept;
    /// Number of concentration alert events.
    uint64_t concentration_events() const noexcept;

    // -------------------------------------------------------------------------
    // Accessors (mutex-guarded)
    // -------------------------------------------------------------------------

    /// Name of the dominant theme (highest share); empty string if no data.
    std::string dominant_theme() const noexcept;
    /// Share [0,1] of the dominant theme.
    double dominant_share() const noexcept;
    /// Number of distinct themes seen.
    int distinct_themes() const noexcept;
    /// Per-theme shares as a vector of (name, share) pairs, sorted descending.
    std::vector<std::pair<std::string, double>> theme_shares() const;

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /** Reset all accumulated state. */
    void reset();
    /** Replace config and reset. */
    void update_config(const Config& cfg);

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /** Compact JSON with HHI and concentration stats. */
    std::string to_stats_json() const;

private:
    struct ThemeStats {
        std::vector<double> window;
        int   head  = 0;
        int   count = 0;
        double sum  = 0.0;  // absolute sum in current window
    };

    Config cfg_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, ThemeStats> themes_;
    bool   concentrated_ = false;
    std::string dominant_;

    // Atomic outputs
    std::atomic<double>   hhi_{0.0};
    std::atomic<bool>     conc_flag_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> conc_events_{0};

    // Helper
    void recompute_locked();
};

}  // namespace llmquant
