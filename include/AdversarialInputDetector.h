#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llmquant {

/**
 * @brief Detects adversarial or anomalous token inputs that may indicate
 *        prompt injection, adversarial attacks, or a stuck / looping LLM.
 *
 * ## Detection approach
 * The detector maintains an online estimate of "expected" token weight
 * statistics (mean and variance of the absolute bias magnitude) and flags
 * tokens that deviate beyond a z-score threshold.  It also tracks:
 *
 *  1. **Weight anomalies** — tokens whose |bias| is statistically extreme
 *     relative to the rolling baseline (z-score > anomaly_threshold).
 *
 *  2. **Repetition attacks** — a single token appearing more than
 *     `max_repeat_fraction` of the last `repeat_window` tokens.  Repetition
 *     is a classic adversarial-prompt fingerprint.
 *
 *  3. **Vocabulary inflation** — the rate at which unseen tokens arrive
 *     (novel_token_rate) spikes when an attacker injects a large vocabulary.
 *
 * When any of these triggers fires, `on_anomaly` is called and `is_armed()`
 * returns true.  The detector self-resets after `reset_after_n_clean` clean
 * tokens, or can be reset manually.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ADVERSARIAL_DETECT` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class AdversarialInputDetector {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    enum class AnomalyKind {
        WeightAnomaly,    ///< Statistically extreme token weight.
        RepetitionAttack, ///< Single token dominates recent window.
        VocabularyInflation, ///< Novel token rate spike.
    };

    struct Config {
        /**
         * @brief Z-score threshold for weight anomaly detection.
         *
         * Higher = less sensitive.  Default 4.0.
         */
        double anomaly_threshold{4.0};

        /**
         * @brief Minimum tokens before weight anomaly detection is active.
         *
         * Prevents false positives during warm-up.  Default 30.
         */
        int min_warmup_tokens{30};

        /**
         * @brief Window size for repetition detection (token count).
         *
         * Default 20.
         */
        int repeat_window{20};

        /**
         * @brief Maximum fraction of `repeat_window` that a single token
         * may occupy before a repetition attack is declared.
         *
         * Default 0.6 (more than 60% same token = suspicious).
         */
        double max_repeat_fraction{0.6};

        /**
         * @brief Rolling window for novel token rate estimation.
         *
         * Default 50.
         */
        int vocab_window{50};

        /**
         * @brief Novel token fraction above which vocabulary inflation fires.
         *
         * Default 0.8 (80% unseen tokens in the last vocab_window).
         */
        double max_novel_fraction{0.8};

        /**
         * @brief Number of clean tokens needed to auto-reset armed state.
         *
         * Default 100.  Set to 0 to disable auto-reset.
         */
        int reset_after_n_clean{100};

        /**
         * @brief Fired when an anomaly is detected.
         *
         * Parameters: (kind, token, score).
         * Called outside the internal lock; do not recurse into the detector.
         */
        std::function<void(AnomalyKind kind,
                           const std::string& token,
                           double score)> on_anomaly;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit AdversarialInputDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Token intake
    // -----------------------------------------------------------------------

    /**
     * @brief Inspect a token and its signed weight.
     *
     * @param token  Raw token string.
     * @param weight Signed weight applied by the LLM adapter.
     * @return true if an anomaly was detected for this token.
     */
    bool inspect(const std::string& token, double weight);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief True when at least one anomaly has been detected since last reset.
     */
    [[nodiscard]] bool is_armed() const noexcept {
        return armed_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total anomaly events fired since last reset.
     */
    [[nodiscard]] uint64_t anomaly_count() const noexcept {
        return anomaly_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total tokens inspected.
     */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Current online mean of |weight|.
     */
    [[nodiscard]] double weight_mean() const noexcept;

    /**
     * @brief Current online std of |weight|.
     */
    [[nodiscard]] double weight_std() const noexcept;

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Manually disarm and reset state.
     */
    void reset();

    /**
     * @brief Update configuration without resetting state.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    // Welford online statistics for |weight|.
    double w_mean_{0.0};
    double w_M2_{0.0};
    int    w_n_{0};

    // Repetition window.
    std::deque<std::string> repeat_buf_;
    std::unordered_map<std::string, int> repeat_counts_;

    // Vocabulary inflation.
    std::unordered_map<std::string, bool> seen_vocab_;
    std::deque<bool>                       novel_buf_;  // true = novel token
    int                                    novel_in_window_{0};

    // State.
    std::atomic<bool>     armed_{false};
    std::atomic<uint64_t> anomaly_count_{0};
    std::atomic<uint64_t> total_tokens_{0};
    int                   clean_streak_{0};
    double                weight_mean_snap_{0.0};
    double                weight_std_snap_{0.0};

    mutable std::mutex mutex_;
};

} // namespace llmquant
