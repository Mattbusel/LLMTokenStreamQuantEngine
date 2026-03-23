#pragma once

/**
 * @file include/ContextWindowTracker.h
 * @brief Tracks token position within an LLM context window and provides a
 *        signal-strength decay factor that reflects model quality degradation
 *        near the context limit.
 *
 * See src/ContextWindowTracker.cpp for the full algorithm description.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CONTEXT_WINDOW_TRACKER` (default ON).
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Decay curve applied to compute the signal scale factor.
 */
enum class DecayCurve {
    Linear,    ///< d(f) = 1 - f                          (linear decay)
    Quadratic, ///< d(f) = (1 - f)^2                      (steeper near end)
    Cosine,    ///< d(f) = 0.5 * (1 + cos(pi * f))        (smooth S-curve, default)
};

/**
 * @brief Tracks LLM context-window token consumption and derives a signal
 *        strength scale factor that decays as the window fills up.
 *
 * ## Summary
 *
 * As token position increases towards `max_tokens`, the decay factor returned
 * by `signal_scale()` decreases from 1.0 towards `min_scale`.  Callers
 * should multiply their raw signal strength by this factor before publishing
 * to downstream consumers.
 *
 * Three tiered callbacks (warn / critical / saturated) are fired once per
 * context session when fill-fraction crosses the configured thresholds.
 */
class ContextWindowTracker {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Configuration for ContextWindowTracker.
     */
    struct Config {
        /**
         * @brief Maximum context window in tokens.
         * Default 128,000 (Claude 3 / GPT-4 Turbo).
         */
        uint64_t max_tokens{128'000};

        /**
         * @brief Decay curve to apply.
         * Default: Cosine.
         */
        DecayCurve decay_curve{DecayCurve::Cosine};

        /**
         * @brief Minimum signal scale factor, even at 100% fill.
         * Prevents the signal from being completely silenced.
         * Default 0.05.
         */
        double min_scale{0.05};

        /**
         * @brief Fill fraction at which the warn callback fires.
         * Default 0.70.
         */
        double warn_fraction{0.70};

        /**
         * @brief Fill fraction at which the critical callback fires.
         * Default 0.90.
         */
        double critical_fraction{0.90};

        /**
         * @brief Callback fired once when fill_fraction >= warn_fraction.
         * Parameters: (tokens_used, max_tokens).
         */
        std::function<void(uint64_t, uint64_t)> on_warn;

        /**
         * @brief Callback fired once when fill_fraction >= critical_fraction.
         * Parameters: (tokens_used, max_tokens).
         */
        std::function<void(uint64_t, uint64_t)> on_critical;

        /**
         * @brief Callback fired once when tokens_used >= max_tokens.
         * Parameters: (tokens_used, max_tokens).
         */
        std::function<void(uint64_t, uint64_t)> on_saturated;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit ContextWindowTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record consumption of `tokens` context tokens.
     *
     * Updates the running total and fires tier callbacks on first crossing.
     * Thread-safe.
     *
     * @param tokens Number of tokens consumed (default 1).
     */
    void consume(uint64_t tokens = 1);

    /**
     * @brief Reset context position to zero and clear tier flags.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Signal strength scale factor in [min_scale, 1.0].
     *
     * Callers multiply their raw signal by this value.  Returns 1.0 at
     * zero fill and declines towards `min_scale` as the window fills.
     *
     * Thread-safe.
     */
    [[nodiscard]] double signal_scale() const;

    /**
     * @brief Current fill fraction in [0.0, 1.0].
     *
     * Thread-safe.
     */
    [[nodiscard]] double fill_fraction() const noexcept;

    /**
     * @brief Tokens consumed so far in this context session.
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t tokens_used() const noexcept {
        return tokens_used_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Remaining tokens before the context is saturated.
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t tokens_remaining() const noexcept;

    /**
     * @brief Human-readable name for a DecayCurve value.
     */
    [[nodiscard]] static const char* decay_curve_name(DecayCurve c) noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Times warn tier was entered across all sessions. */
    [[nodiscard]] uint64_t warn_events() const noexcept {
        return warn_events_.load(std::memory_order_relaxed);
    }

    /** @brief Times critical tier was entered across all sessions. */
    [[nodiscard]] uint64_t critical_events() const noexcept {
        return critical_events_.load(std::memory_order_relaxed);
    }

    /** @brief Times saturation was reached across all sessions. */
    [[nodiscard]] uint64_t saturation_events() const noexcept {
        return saturation_events_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times reset() was called. */
    [[nodiscard]] uint64_t reset_count() const noexcept {
        return reset_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise state to a compact JSON string.
     *
     * Thread-safe.
     */
    [[nodiscard]] std::string to_json() const;

private:
    double fill_fraction_locked(uint64_t used) const noexcept;

    mutable std::mutex mutex_;
    Config config_;

    std::atomic<uint64_t> tokens_used_{0};

    // Per-session tier flags (reset by reset()).
    bool warned_{false};
    bool critical_{false};
    bool saturated_{false};

    std::atomic<uint64_t> warn_events_{0};
    std::atomic<uint64_t> critical_events_{0};
    std::atomic<uint64_t> saturation_events_{0};
    std::atomic<uint64_t> reset_count_{0};
};

} // namespace llmquant
