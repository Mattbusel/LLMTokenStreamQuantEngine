#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Tracks LLM context-window token consumption with tiered overflow guards.
 *
 * Modern LLMs have hard context limits (e.g., 128k tokens for Claude 3, 200k for
 * Claude 3.5).  As the context fills up:
 *  - Performance degrades (lost-in-the-middle effect).
 *  - API calls fail with a context-overflow error.
 *  - Long running streaming sessions silently lose early context without warning.
 *
 * This class tracks tokens consumed in the current "context session" and fires
 * tiered callbacks before overflow, enabling the engine to:
 *  - Log a warning (soft tier).
 *  - Pause token ingestion and flush (hard tier).
 *  - Force a context reset / new conversation start (overflow tier).
 *
 * ## Tiers
 * | Tier     | Condition                          | Default threshold |
 * |----------|------------------------------------|-------------------|
 * | Normal   | tokens_used < soft_fraction * cap  | < 70%             |
 * | Warn     | tokens_used ≥ soft_fraction * cap  | ≥ 70%             |
 * | Critical | tokens_used ≥ hard_fraction * cap  | ≥ 90%             |
 * | Overflow | tokens_used ≥ capacity             | 100%              |
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CONTEXT_BUDGET` (default ON).
 */
class ContextWindowBudget {
public:
    /**
     * @brief Tier returned by consume().
     */
    enum class Tier { Normal, Warn, Critical, Overflow };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Total context capacity in tokens.
         *
         * Default 128000 (Claude 3 / GPT-4 Turbo).
         */
        uint64_t capacity{128'000};

        /**
         * @brief Fraction of capacity at which Warn fires.
         *
         * Default 0.70.
         */
        double soft_fraction{0.70};

        /**
         * @brief Fraction of capacity at which Critical fires.
         *
         * Default 0.90.
         */
        double hard_fraction{0.90};

        /**
         * @brief Callback fired once when the Warn tier is entered.
         *
         * Parameters: (tokens_used, capacity).
         * Use this to log a warning or reduce token input rate.
         */
        std::function<void(uint64_t, uint64_t)> on_warn;

        /**
         * @brief Callback fired once when the Critical tier is entered.
         *
         * Parameters: (tokens_used, capacity).
         * Use this to pause streaming or schedule a context reset.
         */
        std::function<void(uint64_t, uint64_t)> on_critical;

        /**
         * @brief Callback fired once when tokens_used reaches capacity (Overflow).
         *
         * Parameters: (tokens_used, capacity).
         * Use this to force a new conversation / context reset.
         */
        std::function<void(uint64_t, uint64_t)> on_overflow;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit ContextWindowBudget(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record consumption of `tokens` context tokens.
     *
     * Updates the running total, fires tier callbacks on first crossing,
     * and returns the current tier.  Thread-safe.
     *
     * @param tokens  Number of tokens consumed this call (default 1).
     * @return Current Tier after this consumption.
     */
    Tier consume(uint64_t tokens = 1);

    /**
     * @brief Reset the context usage counter to zero.
     *
     * Clears tier state so callbacks will fire again after next crossing.
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /** @brief Tokens consumed in the current context session. Thread-safe. */
    [[nodiscard]] uint64_t tokens_used() const noexcept {
        return tokens_used_.load(std::memory_order_relaxed);
    }

    /** @brief Remaining tokens before overflow. Thread-safe. */
    [[nodiscard]] uint64_t tokens_remaining() const noexcept;

    /** @brief Fill fraction ∈ [0, 1]. Thread-safe. */
    [[nodiscard]] double fill_fraction() const noexcept;

    /** @brief Current tier. Thread-safe. */
    [[nodiscard]] Tier current_tier() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total context resets (calls to reset()). */
    [[nodiscard]] uint64_t total_resets() const noexcept {
        return total_resets_.load(std::memory_order_relaxed);
    }

    /** @brief Times the Warn tier was entered (across all sessions). */
    [[nodiscard]] uint64_t warn_events() const noexcept {
        return warn_events_.load(std::memory_order_relaxed);
    }

    /** @brief Times the Critical tier was entered (across all sessions). */
    [[nodiscard]] uint64_t critical_events() const noexcept {
        return critical_events_.load(std::memory_order_relaxed);
    }

    /** @brief Times overflow was hit (across all sessions). */
    [[nodiscard]] uint64_t overflow_events() const noexcept {
        return overflow_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Human-readable name for the Tier enum value.
     */
    [[nodiscard]] static const char* tier_name(Tier t) noexcept;

    /**
     * @brief Update configuration; implicitly calls reset().
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    std::atomic<uint64_t> tokens_used_{0};

    // Per-session tier flags (reset by reset()).
    bool warned_{false};
    bool critiqued_{false};
    bool overflowed_{false};

    std::atomic<uint64_t> total_resets_{1};  // starts at 1: construction is the initial reset
    std::atomic<uint64_t> warn_events_{0};
    std::atomic<uint64_t> critical_events_{0};
    std::atomic<uint64_t> overflow_events_{0};
};

} // namespace llmquant
