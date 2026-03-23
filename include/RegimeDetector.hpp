#pragma once

/// @file RegimeDetector.hpp
/// @brief Streaming market-regime detector: classifies token-stream sentiment into
///        five discrete regimes using an online 3-state Hidden Markov Model (HMM).
///
/// ## Regime labels
/// | Regime             | Interpretation                                                     |
/// |--------------------|--------------------------------------------------------------------|
/// | BULL_TRENDING      | Sustained positive sentiment momentum; trend-following bias        |
/// | BEAR_TRENDING      | Sustained negative sentiment momentum; short-side bias             |
/// | HIGH_UNCERTAINTY   | High emission variance across all states; conviction unreliable    |
/// | CONSOLIDATION      | Sentiment range-bound; mean-reversion bias                         |
/// | BREAKOUT           | Rapid regime transition probability shift; potential impulse trade |
///
/// ## Algorithm
/// The detector implements a **3-state HMM forward algorithm** executed online:
///   - States 0=Bearish, 1=Neutral, 2=Bullish
///   - Emission: Gaussian P(bias | state s) = N(bias; μ_s, σ_s)
///   - Transition matrix A[3][3] learned online via soft Viterbi count update
///   - Forward variables α_t[s] updated at each `update()` call
///   - Posterior γ[s] = α[s] / Z used to derive `TokenRegime` label
///   - Emission parameters (μ, σ) estimated online via exponential forgetting
///
/// ## RegimeFilter
/// `RegimeFilter` wraps a user-provided signal predicate and gates it by
/// comparing the current regime against the signal's directional alignment.
/// Only signals whose direction is consistent with the active regime fire.
///
/// ## Thread safety
/// All public methods are thread-safe (mutex-protected internal state).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REGIME_DETECTOR` (default ON — shared with RegimeDetector.h).

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

// ---------------------------------------------------------------------------
// TokenRegime enum
// ---------------------------------------------------------------------------

/**
 * @brief Five-class discrete market-regime label derived from the 3-state HMM.
 */
enum class TokenRegime : int {
    BULL_TRENDING    = 0, ///< Clear positive momentum.
    BEAR_TRENDING    = 1, ///< Clear negative momentum.
    HIGH_UNCERTAINTY = 2, ///< All-state posterior uncertainty above threshold.
    CONSOLIDATION    = 3, ///< Sentiment range-bound; neutral state dominant.
    BREAKOUT         = 4, ///< Rapid cross-state probability mass shift.
};

// ---------------------------------------------------------------------------
// RegimeDetector
// ---------------------------------------------------------------------------

/**
 * @brief Online 3-state HMM streaming regime detector.
 *
 * Maintains a forward-variable filter over three latent states (bearish,
 * neutral, bullish) and maps the resulting posterior distribution to one of
 * the five `TokenRegime` labels.  Emission parameters are updated online
 * via exponential forgetting so the model self-adapts to non-stationary
 * sentiment distributions without requiring a separate training phase.
 */
class RegimeDetectorHMM {
public:
    static constexpr int kStates = 3; ///< Bearish=0, Neutral=1, Bullish=2.

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Prior emission means [bearish, neutral, bullish].
         * Default {-0.15, 0.00, +0.15}.
         */
        std::array<double, kStates> state_mean{-0.15, 0.00, 0.15};

        /**
         * @brief Prior emission standard deviations.
         * Default {0.10, 0.08, 0.10}.
         */
        std::array<double, kStates> state_std{0.10, 0.08, 0.10};

        /**
         * @brief Transition matrix prior A[from][to].
         * High self-transition probability (sticky regimes).
         */
        std::array<std::array<double, kStates>, kStates> transition{{
            {0.90, 0.08, 0.02},   // bearish
            {0.04, 0.92, 0.04},   // neutral
            {0.02, 0.08, 0.90}    // bullish
        }};

        /**
         * @brief Initial state prior [bearish, neutral, bullish].
         * Default uniform.
         */
        std::array<double, kStates> init_prob{0.333, 0.334, 0.333};

        /**
         * @brief EMA forgetting factor for online emission parameter update.
         * 0 = never update (fixed), 1 = fully online.  Default 0.05.
         */
        double emission_learning_rate{0.05};

        /**
         * @brief EMA forgetting factor for online transition matrix update.
         * Default 0.02 (slower than emission to stabilise topology).
         */
        double transition_learning_rate{0.02};

        /**
         * @brief Minimum posterior probability for a single state to be
         * classified as BULL_TRENDING or BEAR_TRENDING.  Default 0.55.
         */
        double trend_threshold{0.55};

        /**
         * @brief Maximum probability in any state below which regime is
         * classified HIGH_UNCERTAINTY.  Default 0.40.
         */
        double uncertainty_threshold{0.40};

        /**
         * @brief KL-divergence threshold between consecutive posteriors above
         * which regime is classified BREAKOUT.  Default 0.12.
         */
        double breakout_kl_threshold{0.12};

        /**
         * @brief Minimum samples before outputting non-CONSOLIDATION regime.
         * Default 10.
         */
        int min_samples{10};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit RegimeDetectorHMM(Config cfg = Config{});

    // Non-copyable (owns mutable EMA state).
    RegimeDetectorHMM(const RegimeDetectorHMM&)            = delete;
    RegimeDetectorHMM& operator=(const RegimeDetectorHMM&) = delete;

    // -----------------------------------------------------------------------
    // Hot-path update
    // -----------------------------------------------------------------------

    /**
     * @brief Ingest a new token-bias observation and advance the HMM filter.
     *
     * Updates the forward variables, optionally updates emission and
     * transition parameters online, reclassifies the regime, and fires the
     * change callback if the regime changed.
     *
     * @param token_bias  Directional bias for the current token ([-1, 1]).
     * @param timestamp   Nanosecond wall-clock timestamp (unused internally
     *                    beyond being forwarded to the change callback).
     */
    void update(double token_bias, int64_t timestamp = 0);

    // -----------------------------------------------------------------------
    // Accessors (lock-free atomic reads)
    // -----------------------------------------------------------------------

    /// Current classified regime.
    [[nodiscard]] TokenRegime current_regime() const noexcept {
        return static_cast<TokenRegime>(regime_.load(std::memory_order_acquire));
    }

    /**
     * @brief Regime confidence in [0, 1].
     *
     * Defined as the maximum posterior probability across all three states,
     * clipped to [0, 1].  High confidence (near 1.0) means the HMM posterior
     * is concentrated on a single state.
     */
    [[nodiscard]] double regime_confidence() const noexcept {
        return confidence_.load(std::memory_order_relaxed);
    }

    /// Posterior probability of the bullish state.
    [[nodiscard]] double p_bullish() const noexcept {
        return p_bull_.load(std::memory_order_relaxed);
    }

    /// Posterior probability of the bearish state.
    [[nodiscard]] double p_bearish() const noexcept {
        return p_bear_.load(std::memory_order_relaxed);
    }

    /// Posterior probability of the neutral state.
    [[nodiscard]] double p_neutral() const noexcept {
        return p_neutral_.load(std::memory_order_relaxed);
    }

    /// Total `update()` calls since construction or last `reset()`.
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Total regime-transition events since construction or last `reset()`.
    [[nodiscard]] uint64_t total_transitions() const noexcept {
        return transitions_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Human-readable name for a `TokenRegime` value.
    [[nodiscard]] static const char* regime_name(TokenRegime r) noexcept;

    /// Human-readable name for the current regime.
    [[nodiscard]] const char* current_regime_name() const noexcept {
        return regime_name(current_regime());
    }

    /**
     * @brief Register a callback fired on every regime change.
     *
     * Called outside the internal mutex.
     * @param cb Callable(new_regime, old_regime, timestamp_ns).
     */
    void set_regime_change_callback(
        std::function<void(TokenRegime, TokenRegime, int64_t)> cb);

    /// Serialise current HMM state and posteriors as a JSON object string.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset HMM state to the configured prior; preserve configuration.
    void reset();

    /// Hot-update configuration; HMM forward variables are preserved.
    void update_config(const Config& cfg);

    /// Return a copy of the current configuration.
    [[nodiscard]] Config get_config() const;

private:
    // Classify from current posteriors (called under mutex).
    [[nodiscard]] TokenRegime classify(
        const std::array<double, kStates>& gamma,
        const std::array<double, kStates>& prev_gamma) const;

    // Online emission update (under mutex).
    void update_emission(int state, double bias);

    // Online transition update (under mutex): soft increment A[prev][new].
    void update_transition(const std::array<double, kStates>& prev_gamma,
                           const std::array<double, kStates>& gamma);

    // Normalise a length-3 array in place; returns the sum.
    static double normalise(std::array<double, kStates>& v) noexcept;

    // Gaussian PDF (no exceptions, no allocations).
    static double gauss_pdf(double x, double mu, double sigma) noexcept;

    // KL divergence D_KL(p || q) for two normalised 3-vectors.
    static double kl_div(const std::array<double, kStates>& p,
                         const std::array<double, kStates>& q) noexcept;

    mutable std::mutex mutex_;
    Config             cfg_;

    // HMM forward variables (normalised).
    std::array<double, kStates> alpha_{0.333, 0.334, 0.333};

    // Previous posterior (for KL-based breakout detection).
    std::array<double, kStates> prev_gamma_{0.333, 0.334, 0.333};

    // Online-learned emission parameters.
    std::array<double, kStates> emu_mean_{};  // initialised from cfg_.state_mean
    std::array<double, kStates> emu_std_{};   // initialised from cfg_.state_std

    // Online-learned transition matrix (row-normalised).
    std::array<std::array<double, kStates>, kStates> trans_{};  // from cfg_

    // Timestamp of the last update (for potential time-decay extensions).
    int64_t last_ts_{0};

    // Atomics for lock-free reads.
    std::atomic<int>    regime_{static_cast<int>(TokenRegime::CONSOLIDATION)};
    std::atomic<double> confidence_{0.0};
    std::atomic<double> p_bull_{0.333};
    std::atomic<double> p_bear_{0.333};
    std::atomic<double> p_neutral_{0.334};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> transitions_{0};

    // Regime-change callback (protected by mutex for set, copied out for call).
    std::function<void(TokenRegime, TokenRegime, int64_t)> regime_cb_;
};

// ---------------------------------------------------------------------------
// RegimeFilter
// ---------------------------------------------------------------------------

/**
 * @brief Signal gate that only fires when the active regime is aligned with
 *        the signal direction.
 *
 * Wraps a `RegimeDetectorHMM` reference (or an owned instance) and a
 * user-supplied signal predicate.  On each `evaluate()` call the filter:
 *   1. Checks whether the signal direction (+1 buy / -1 sell) matches the
 *      current regime.
 *   2. Optionally applies a minimum `regime_confidence()` threshold.
 *   3. Fires the downstream callback if both conditions are satisfied.
 *
 * ### Regime alignment rules
 * | Signal direction | Allowed regimes                                  |
 * |-----------------|--------------------------------------------------|
 * | +1 (buy)        | BULL_TRENDING, BREAKOUT (bullish p > bearish p)  |
 * | -1 (sell)       | BEAR_TRENDING, BREAKOUT (bearish p > bullish p)  |
 * | 0 (neutral)     | CONSOLIDATION, HIGH_UNCERTAINTY                  |
 */
class RegimeFilter {
public:
    struct Config {
        /// Minimum regime_confidence() to pass a signal.  Default 0.45.
        double min_confidence{0.45};

        /// Allow BREAKOUT regime signals regardless of direction.  Default true.
        bool   allow_breakout_bypass{true};

        /// If true, HIGH_UNCERTAINTY signals are silently dropped.  Default true.
        bool   block_high_uncertainty{true};
    };

    /**
     * @brief Construct a RegimeFilter that owns its internal detector.
     *
     * @param detector_cfg Config forwarded to the internal `RegimeDetectorHMM`.
     * @param filter_cfg   Filter gate thresholds.
     */
    explicit RegimeFilter(
        RegimeDetectorHMM::Config detector_cfg = RegimeDetectorHMM::Config{},
        Config                    filter_cfg   = Config{});

    /**
     * @brief Construct a RegimeFilter referencing an *external* detector.
     *
     * The caller is responsible for keeping the detector alive.
     *
     * @param detector  Externally owned detector (must outlive this filter).
     * @param filter_cfg Filter gate thresholds.
     */
    explicit RegimeFilter(RegimeDetectorHMM& detector,
                          Config             filter_cfg = Config{});

    // Non-copyable.
    RegimeFilter(const RegimeFilter&)            = delete;
    RegimeFilter& operator=(const RegimeFilter&) = delete;

    // -----------------------------------------------------------------------
    // Regime ingestion (proxy to underlying detector)
    // -----------------------------------------------------------------------

    /**
     * @brief Forward a token-bias observation to the underlying detector.
     *
     * Must be called on each incoming token, even when no signal is pending.
     */
    void feed(double token_bias, int64_t timestamp = 0);

    // -----------------------------------------------------------------------
    // Signal gate
    // -----------------------------------------------------------------------

    /**
     * @brief Evaluate a directional signal against the current regime.
     *
     * @param signal_direction +1 (buy), -1 (sell), or 0 (neutral).
     * @param signal_bias      Raw bias magnitude (informational only).
     * @return true  if the signal passes the regime gate.
     * @return false if blocked.
     */
    [[nodiscard]] bool evaluate(int signal_direction, double signal_bias = 0.0) const noexcept;

    /// Access the underlying detector (e.g. to read posteriors).
    [[nodiscard]] const RegimeDetectorHMM& detector() const noexcept {
        return *detector_ptr_;
    }
    [[nodiscard]] RegimeDetectorHMM& detector() noexcept {
        return *detector_ptr_;
    }

    /// Total signals evaluated since construction.
    [[nodiscard]] uint64_t total_evaluated() const noexcept {
        return evaluated_.load(std::memory_order_relaxed);
    }

    /// Total signals passed by the regime gate.
    [[nodiscard]] uint64_t total_passed() const noexcept {
        return passed_.load(std::memory_order_relaxed);
    }

    /// Total signals blocked by the regime gate.
    [[nodiscard]] uint64_t total_blocked() const noexcept {
        return blocked_.load(std::memory_order_relaxed);
    }

    /// Pass rate (passed / evaluated).  Returns 0.0 if no signals evaluated.
    [[nodiscard]] double pass_rate() const noexcept;

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Update filter configuration.
    void update_config(const Config& cfg);

private:
    // Owner vs. reference storage: one of these is non-null.
    RegimeDetectorHMM  owned_detector_;      ///< Used when owning.
    RegimeDetectorHMM* detector_ptr_;        ///< Points to owned or external.

    Config cfg_;

    std::atomic<uint64_t> evaluated_{0};
    std::atomic<uint64_t> passed_{0};
    std::atomic<uint64_t> blocked_{0};

    mutable std::mutex cfg_mutex_;
};

} // namespace llmquant
