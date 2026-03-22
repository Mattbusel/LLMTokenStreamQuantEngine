#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Pre-seeder that primes pipeline EMA accumulators before live streaming.
 *
 * LLM token-stream pipelines use Exponential Moving Averages (EMAs) to track
 * sentiment momentum, volatility, and win/loss ratios.  On cold start all
 * EMAs are at their zero-value defaults, which can produce misleading signals
 * until enough live tokens have been consumed.  WarmupSequencer replays a
 * curated sequence of synthetic tokens through the same callback used by the
 * live path, accelerating EMA convergence to a meaningful baseline before the
 * first real signal is emitted.
 *
 * ## Typical usage
 * @code
 * WarmupSequencer::Config wcfg;
 * wcfg.synthetic_tokens = {{"bullish", 0.7}, {"rally", 0.5}, {"crash", -0.8}};
 * wcfg.repeat_count = 10;
 * wcfg.on_complete = []{ spdlog::info("warmup done"); };
 *
 * WarmupSequencer warmup(wcfg);
 * warmup.run([&](const std::string& tok, double sent) {
 *     engine.process(tok, sent);   // same hot-path callback
 * });
 * // Pipeline EMAs are now pre-seeded — start live streaming.
 * @endcode
 *
 * ## File loading
 * Alternatively, load a token/sentiment TSV file ("token\tscore\n" per line)
 * with `load_from_file()` before calling `run()`.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_WARMUP_SEQUENCER` (default ON).
 */
class WarmupSequencer {
public:
    /**
     * @brief A single synthetic token with its associated sentiment score.
     */
    struct WarmupToken {
        std::string token;           ///< Token string (same format as live path).
        double      sentiment{0.0};  ///< Sentiment score in [−1.0, 1.0].
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Ordered sequence of synthetic tokens to inject.
         *
         * These are replayed `repeat_count` times.  May be left empty if
         * tokens will be loaded from a file via load_from_file().
         */
        std::vector<WarmupToken> synthetic_tokens;

        /**
         * @brief Number of times the synthetic_tokens sequence is replayed.
         *
         * Higher values cause stronger EMA priming at the cost of slightly
         * more startup time.  Default 1.
         */
        int repeat_count{1};

        /**
         * @brief Optional callback invoked once all tokens have been injected.
         *
         * Called from the same thread as run().
         */
        std::function<void()> on_complete;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit WarmupSequencer(Config config = Config{});

    // -----------------------------------------------------------------------
    // Token loading
    // -----------------------------------------------------------------------

    /**
     * @brief Load synthetic tokens from a TSV file ("token\tscore" per line).
     *
     * Lines beginning with '#' and blank lines are ignored.  Replaces any
     * tokens already in config_.synthetic_tokens.
     *
     * @param filepath Path to the input file.
     * @return true on success; false if the file cannot be opened.
     */
    [[nodiscard]] bool load_from_file(const std::string& filepath);

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Replay all synthetic tokens through the given callback.
     *
     * The callback signature matches the LLMAdapter / TradeSignalEngine
     * hot-path: `(token_string, sentiment_score)`.  Tokens are delivered
     * synchronously in order, `repeat_count` times.  Blocks until done.
     *
     * If synthetic_tokens is empty this is a no-op (on_complete still fires).
     *
     * @param token_cb Callback to invoke for each token.
     */
    void run(std::function<void(const std::string&, double)> token_cb);

    /**
     * @brief Reset completion state so run() can be called again.
     *
     * Does not modify config or the token list.
     */
    void reset();

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /** @brief True after a successful run() has completed. */
    [[nodiscard]] bool is_complete() const noexcept {
        return complete_.load(std::memory_order_acquire);
    }

    /** @brief Total tokens injected across all run() calls since last reset(). */
    [[nodiscard]] uint64_t tokens_injected() const noexcept {
        return tokens_injected_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config config_;
    std::atomic<bool>     complete_{false};
    std::atomic<uint64_t> tokens_injected_{0};
};

} // namespace llmquant
