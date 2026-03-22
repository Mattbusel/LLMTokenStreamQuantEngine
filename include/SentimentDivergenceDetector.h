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
 * @brief Detects divergence between multiple concurrent sentiment streams.
 *
 * In a production setup, sentiment signals may arrive from several independent
 * sources (Reddit, Twitter/X, financial news, earnings transcripts).  When
 * these sources agree, signal confidence is high.  When they diverge, the
 * conflicting information warrants caution: position sizes should shrink and
 * alert thresholds should tighten.
 *
 * ## Algorithm
 * Each source maintains an EMA of its recent sentiment.  On each update, the
 * detector computes pairwise absolute differences between all source EMAs and
 * takes the maximum:
 *
 *   divergence = max_{i≠j} |ema_i − ema_j|
 *
 * This is O(S²) where S is the number of registered sources (typically S ≤ 5).
 * When `divergence > divergence_threshold`, `on_divergence` fires.
 * Recovery fires when `divergence < divergence_threshold × recovery_hysteresis`.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_DIVERGENCE` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class SentimentDivergenceDetector {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief EMA alpha applied to each source's observations.
         *
         * Smaller = more stable, slower response.  Default 0.1.
         */
        double ema_alpha{0.1};

        /**
         * @brief Maximum pairwise EMA difference above which divergence fires.
         *
         * Typical range [0.1, 0.8].  Default 0.4.
         */
        double divergence_threshold{0.4};

        /**
         * @brief Hysteresis multiplier for recovery.
         *
         * Recovery fires when divergence < threshold × recovery_hysteresis.
         * Default 0.7.
         */
        double recovery_hysteresis{0.7};

        /**
         * @brief Minimum observations per source before divergence is reported.
         *
         * Prevents spurious alerts during warm-up.  Default 5.
         */
        int min_observations{5};

        /**
         * @brief Fired when pairwise divergence exceeds the threshold.
         *
         * Parameters: (divergence, source_a, source_b) — the pair with the
         * largest difference.
         * Called outside the internal lock.
         */
        std::function<void(double divergence,
                           const std::string& source_a,
                           const std::string& source_b)> on_divergence;

        /**
         * @brief Fired when divergence drops back below the recovery threshold.
         *
         * Parameters: (divergence).
         */
        std::function<void(double divergence)> on_recovery;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SentimentDivergenceDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Source registration
    // -----------------------------------------------------------------------

    /**
     * @brief Register a new sentiment source.
     *
     * Must be called before `record()` for that source.
     *
     * @param source_name Unique identifier for the source (e.g. "reddit").
     * @throws std::runtime_error if more than kMaxSources are registered.
     */
    void register_source(const std::string& source_name);

    // -----------------------------------------------------------------------
    // Observation intake
    // -----------------------------------------------------------------------

    /**
     * @brief Feed an observation from the named source.
     *
     * If the source is unknown, the call is silently ignored.
     *
     * @param source_name Name passed to register_source().
     * @param sentiment   Normalised sentiment in [-1, 1].
     */
    void record(const std::string& source_name, double sentiment);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Current maximum pairwise divergence ∈ [0, 2].
     */
    [[nodiscard]] double divergence() const noexcept {
        return divergence_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when divergence > threshold.
     */
    [[nodiscard]] bool is_diverged() const noexcept {
        return diverged_.load(std::memory_order_relaxed);
    }

    /**
     * @brief EMA value for the named source (-2.0 if not registered).
     */
    [[nodiscard]] double source_ema(const std::string& source_name) const;

    /**
     * @brief Number of registered sources.
     */
    [[nodiscard]] std::size_t source_count() const noexcept;

    /**
     * @brief Total divergence events fired.
     */
    [[nodiscard]] uint64_t divergence_events() const noexcept {
        return divergence_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total recovery events fired.
     */
    [[nodiscard]] uint64_t recovery_events() const noexcept {
        return recovery_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Reset all EMA states; retain source registrations.
     */
    void reset();

    /**
     * @brief Update configuration without resetting EMA state.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

    static constexpr std::size_t kMaxSources = 16;

private:
    struct SourceState {
        std::string name;
        double      ema{0.0};
        int         obs_count{0};
    };

    Config cfg_;
    std::vector<SourceState>                   sources_;
    std::unordered_map<std::string, std::size_t> index_;

    std::atomic<double>   divergence_{0.0};
    std::atomic<bool>     diverged_{false};
    std::atomic<uint64_t> divergence_events_{0};
    std::atomic<uint64_t> recovery_events_{0};

    mutable std::mutex mutex_;

    void update_divergence_locked();
};

} // namespace llmquant
