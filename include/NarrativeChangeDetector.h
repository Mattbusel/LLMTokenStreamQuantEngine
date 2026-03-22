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
 * @brief Detects structural breaks in the LLM token stream via cosine similarity.
 *
 * Maintains two consecutive rolling windows of token frequency vectors and
 * measures the cosine similarity between them:
 *
 *   sim = Σ(a_i * b_i) / (|a| * |b|)
 *
 * A sudden drop in similarity (below `break_threshold`) signals a *narrative
 * break* — the LLM switched from discussing one topic to another.  In a
 * quant context, narrative breaks often precede regime transitions (e.g.,
 * sentiment pivoting from "Fed hike path" to "credit event fear").
 *
 * ## Typical similarity ranges
 * | Similarity | Interpretation                                      |
 * |------------|-----------------------------------------------------|
 * | ≥ 0.9      | Consistent narrative — same themes repeating        |
 * | 0.6 – 0.9  | Gradual evolution — related but shifting topics     |
 * | < 0.4      | Narrative break — abrupt topic change               |
 *
 * ## Usage
 * @code
 * NarrativeChangeDetector ncd;
 * ncd.set_break_callback([](double sim) {
 *     spdlog::warn("[narrative] break detected — similarity={:.3f}", sim);
 * });
 * // In token callback:
 * ncd.record(std::hash<std::string>{}(token));
 * // In stats loop:
 * double sim = ncd.get_similarity();
 * @endcode
 *
 * ## Thread safety
 * `record()` is thread-safe.  All read accessors are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_NARRATIVE_CHANGE` (default ON).
 */
class NarrativeChangeDetector {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Size of each rolling window (number of tokens).
         *
         * Smaller = more reactive; larger = more stable.
         * Default 64.
         */
        int window_size{64};

        /**
         * @brief Cosine similarity below this triggers the break callback.
         *
         * Default 0.40.
         */
        double break_threshold{0.40};

        /**
         * @brief EMA alpha for similarity trend smoothing.
         *
         * Default 0.1.
         */
        double ema_alpha{0.10};

        /**
         * @brief Minimum tokens in both windows before similarity is computed.
         *
         * Default equals window_size (wait for a full window before judging).
         */
        int min_fill{0};  ///< 0 = use window_size.
    };

    /**
     * @brief Callback invoked when a narrative break is detected.
     *
     * Receives the cosine similarity at the moment of the break.
     */
    using BreakCallback = std::function<void(double similarity)>;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit NarrativeChangeDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Hot path
    // -----------------------------------------------------------------------

    /**
     * @brief Record a token observation.
     *
     * Advances the rolling windows and recomputes similarity when the current
     * window is full.  Thread-safe.
     *
     * @param token_hash Hash of the token (e.g. std::hash<std::string>{}(t)).
     */
    void record(uint64_t token_hash) noexcept;

    // -----------------------------------------------------------------------
    // Read accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Current cosine similarity between the two consecutive windows.
     *
     * Returns 1.0 if either window is empty.  Thread-safe (atomic read).
     */
    [[nodiscard]] double get_similarity() const noexcept {
        return similarity_.load(std::memory_order_relaxed);
    }

    /**
     * @brief EMA-smoothed similarity.  Thread-safe (atomic read).
     */
    [[nodiscard]] double get_similarity_ema() const noexcept {
        return similarity_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief true if the current similarity is below the break threshold.
     */
    [[nodiscard]] bool is_narrative_break() const noexcept {
        return similarity_.load(std::memory_order_relaxed)
               < config_break_threshold_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total narrative breaks detected since construction.
     */
    [[nodiscard]] uint64_t total_breaks() const noexcept {
        return breaks_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total tokens recorded.
     */
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Configuration and lifecycle
    // -----------------------------------------------------------------------

    /**
     * @brief Register a callback invoked on every narrative break.
     *
     * Thread-safe.
     */
    void set_break_callback(BreakCallback cb);

    /**
     * @brief Update configuration.  Clears all window data.  Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.  Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Reset all state.  Thread-safe.
     */
    void reset();

    /**
     * @brief JSON stats string.
     *
     * Fields: similarity, similarity_ema, is_break, total_breaks, total_recorded.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // Two consecutive token-frequency windows.
    std::unordered_map<uint64_t, int> window_a_;
    std::unordered_map<uint64_t, int> window_b_;

    // Ring queue for window_b_ (slide out oldest token when full).
    std::vector<uint64_t> ring_b_;
    int ring_head_{0};
    int ring_fill_{0};

    int tokens_in_window_a_{0};

    std::atomic<double>   similarity_{1.0};
    std::atomic<double>   similarity_ema_{1.0};
    std::atomic<double>   config_break_threshold_{0.40};
    std::atomic<uint64_t> breaks_{0};
    std::atomic<uint64_t> total_{0};

    BreakCallback break_cb_;

    [[nodiscard]] double compute_cosine_locked() const;
    void                 update_similarity_locked(double sim);
};

} // namespace llmquant
