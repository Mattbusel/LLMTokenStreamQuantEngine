#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Real-time Shannon entropy monitor for the LLM token stream.
 *
 * Maintains a rolling window of the last N token hashes and computes the
 * normalised Shannon entropy H ∈ [0, 1]:
 *
 *   H = -Σ p_i * log2(p_i) / log2(W)
 *
 * where W is the window size and p_i is the relative frequency of each
 * distinct token type in the window.
 *
 * ## Interpretation
 * | Entropy | Meaning                                      |
 * |---------|----------------------------------------------|
 * | ≈ 0     | Highly repetitive — one token dominates      |
 * | ≈ 0.35  | Focused — a few tokens driving the signal    |
 * | ≈ 1     | Uniform noise — tokens arrive at random      |
 *
 * Low entropy indicates high LLM conviction (the model keeps repeating the
 * same semantically loaded tokens).  High entropy suggests the stream is
 * scattered and the resulting bias estimates are unreliable.
 *
 * ## Typical usage
 * @code
 * TokenEntropyMonitor mon;
 * // in token callback:
 * mon.record(std::hash<std::string>{}(token));
 * // in stats loop:
 * double h = mon.entropy();   // 0 = focused, 1 = noisy
 * bool focused = mon.is_focused();
 * @endcode
 *
 * ## Thread safety
 * `record()` is safe to call from any thread concurrently.
 * `entropy()` and all read accessors acquire the internal mutex.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ENTROPY_MONITOR` (default ON).
 */
class TokenEntropyMonitor {
public:
    /**
     * @brief Configuration for the entropy monitor.
     */
    struct Config {
        /**
         * @brief Rolling window size (number of recent tokens considered).
         *
         * Larger windows give smoother estimates; smaller windows react faster.
         * Default 256.  Capped internally at 4096.
         */
        int window_size{256};

        /**
         * @brief Entropy EMA below which `is_focused()` returns true.
         *
         * Value is in normalised [0, 1] space.  Default 0.40.
         */
        double focused_threshold{0.40};

        /**
         * @brief EMA smoothing factor for the entropy trend.
         *
         * 0.05 = slow / stable; 0.2 = fast / reactive.  Default 0.05.
         */
        double ema_alpha{0.05};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenEntropyMonitor(Config config = Config{});

    // -----------------------------------------------------------------------
    // Hot path
    // -----------------------------------------------------------------------

    /**
     * @brief Record a token observation.
     *
     * Thread-safe.  Passes the token hash (e.g. std::hash<std::string>{}(t))
     * so no heap allocation occurs on the hot path.
     *
     * @param token_hash Hash of the token string.
     */
    void record(uint64_t token_hash) noexcept;

    // -----------------------------------------------------------------------
    // Read accessors (acquire internal mutex)
    // -----------------------------------------------------------------------

    /**
     * @brief Compute the current normalised Shannon entropy of the window.
     *
     * Also updates the internal EMA.
     *
     * @return Value in [0, 1].  Returns 0 if fewer than 2 tokens recorded.
     */
    [[nodiscard]] double entropy();

    /**
     * @brief Return the EMA-smoothed entropy (updated by each `entropy()` call).
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double entropy_ema() const noexcept {
        return entropy_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return true if `entropy_ema()` is below the focused threshold.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] bool is_focused() const noexcept {
        return entropy_ema_.load(std::memory_order_relaxed)
               < config_snapshot_focused_threshold_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of distinct token types in the current window.
     *
     * Acquires the internal mutex.
     */
    [[nodiscard]] int unique_tokens();

    /**
     * @brief Total number of tokens recorded since construction (or last reset).
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Configuration and lifecycle
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration at runtime.
     *
     * The rolling window is resized; existing data is cleared on resize.
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.  Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Reset all window data and counters.  Thread-safe.
     */
    void reset();

    /**
     * @brief Serialise current state to a JSON object string.
     *
     * Fields: entropy_ema, is_focused, unique_tokens (in window),
     * window_fill (fraction of window filled), total_recorded.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json();

private:
    mutable std::mutex mutex_;
    Config config_;

    // Ring buffer of token hashes (capacity = config_.window_size).
    std::vector<uint64_t> ring_;
    int head_{0};   // next write position (wraps modulo ring_.size())
    int fill_{0};   // number of valid entries (saturates at ring_.size())

    std::atomic<uint64_t> total_{0};
    std::atomic<double>   entropy_ema_{0.5};  // start mid-range
    std::atomic<double>   config_snapshot_focused_threshold_{0.40};

    // Compute entropy from current ring contents.  Caller holds mutex_.
    [[nodiscard]] double compute_entropy_locked() const;
};

} // namespace llmquant
