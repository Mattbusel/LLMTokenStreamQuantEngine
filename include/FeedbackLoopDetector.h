#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Detects reflexive market-impact feedback in the LLM token stream.
 *
 * When an algorithmic trading system emits large orders, it moves the market.
 * If the LLM is reading real-time news or price feeds, it may then narrate
 * those very price moves — which were caused by the system itself — as new
 * "bullish" or "bearish" signals.  Acting on this feedback is a classic
 * reflexivity trap that amplifies losses.
 *
 * This detector tracks two correlated signals:
 *  1. **Self-impact signal** — a measure of the system's recent trading
 *     activity (e.g. position delta, order volume).  Caller feeds this via
 *     record_own_activity().
 *  2. **LLM sentiment signal** — the bias stream from the token engine.
 *     Caller feeds this via record_sentiment().
 *
 * If the cross-correlation between own-activity and subsequent LLM sentiment
 * exceeds a threshold (with a configurable lag range), the detector fires a
 * warning: "the engine's own trades may be driving the signal it is reading."
 *
 * ## Intended use
 * - Feed `abs(filled_quantity)` or `abs(delta_bias_shift)` as own-activity.
 * - Feed `signal.delta_bias_shift` as sentiment.
 * - Monitor `feedback_score()`: if > threshold, reduce position size or pause.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_FEEDBACK_LOOP` (default ON).
 */
class FeedbackLoopDetector {
public:
    /**
     * @brief Callback fired when the feedback score exceeds the threshold.
     *
     * Parameters: (feedback_score, lag_with_peak_correlation).
     */
    using FeedbackCallback = std::function<void(double score, int lag)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Rolling window size for cross-correlation analysis.
         *
         * Default 50 samples.
         */
        int window_size{50};

        /**
         * @brief Minimum samples in each window before analysis starts.
         *
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Maximum lag (in samples) to test for feedback.
         *
         * Tests lags 1..max_lag.  Feedback is most suspicious at lags 1–5
         * (the system trades → market moves → LLM reports → next tokens).
         * Default 10.
         */
        int max_lag{10};

        /**
         * @brief Cross-correlation threshold above which feedback is suspected.
         *
         * Pearson r ≥ this value → feedback_detected() returns true.
         * Default 0.6.
         */
        double threshold{0.6};

        /**
         * @brief Callback fired each time feedback is first detected.
         *
         * Fires once per "episode" (resets when score drops below threshold).
         */
        FeedbackCallback on_feedback;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit FeedbackLoopDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a measure of the system's own trading activity.
     *
     * Use absolute filled quantity, order notional, or |delta_bias| as input.
     * Thread-safe.
     *
     * @param activity  Non-negative activity scalar.
     */
    void record_own_activity(double activity);

    /**
     * @brief Record an LLM sentiment observation.
     *
     * Should be called on every token (or per-signal).  Thread-safe.
     *
     * @param sentiment  delta_bias_shift or similar sentiment scalar.
     */
    void record_sentiment(double sentiment);

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
     * @brief Current maximum cross-correlation across all tested lags.
     *
     * Range [−1, 1].  Values near 1 indicate strong delayed correlation
     * (own activity predicts future sentiment → feedback suspected).
     * Thread-safe.
     */
    [[nodiscard]] double feedback_score() const noexcept;

    /**
     * @brief Lag (in samples) at which the peak cross-correlation occurs.
     *
     * Thread-safe.
     */
    [[nodiscard]] int peak_lag() const noexcept;

    /**
     * @brief True if feedback_score() ≥ threshold and we have enough data.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool feedback_detected() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total own-activity samples recorded. */
    [[nodiscard]] uint64_t total_activity_records() const noexcept {
        return total_activity_.load(std::memory_order_relaxed);
    }

    /** @brief Total sentiment samples recorded. */
    [[nodiscard]] uint64_t total_sentiment_records() const noexcept {
        return total_sentiment_.load(std::memory_order_relaxed);
    }

    /** @brief Times feedback was detected. */
    [[nodiscard]] uint64_t feedback_events() const noexcept {
        return feedback_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; resets state.
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

    std::deque<double> activity_win_;
    std::deque<double> sentiment_win_;

    double feedback_score_{0.0};
    int    peak_lag_{0};
    bool   in_feedback_episode_{false};

    std::atomic<uint64_t> total_activity_{0};
    std::atomic<uint64_t> total_sentiment_{0};
    std::atomic<uint64_t> feedback_events_{0};

    // Recompute cross-correlation for all lags. Called under lock.
    // Returns true if a callback should fire.
    bool recompute();

    // Pearson cross-correlation between x[0..n-1-lag] and y[lag..n-1].
    static double cross_correlation(const std::deque<double>& x,
                                    const std::deque<double>& y,
                                    int lag) noexcept;
};

} // namespace llmquant
