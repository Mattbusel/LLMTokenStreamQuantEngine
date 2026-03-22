#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief MACD-style momentum oscillator for the trade-signal bias stream.
 *
 * Computes three values in real-time from the incoming bias stream:
 *  - **MACD line**   = fast EMA − slow EMA  (default α≈12-period vs 26-period)
 *  - **Signal line** = EMA of the MACD line (default α≈9-period)
 *  - **Histogram**   = MACD − signal line
 *
 * A positive histogram means the fast EMA is accelerating above the signal
 * (bullish momentum); negative means bearish momentum.  Zero-crossings of the
 * histogram are classic entry/exit triggers in traditional TA; here they flag
 * directional momentum reversals in the LLM sentiment stream.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_MOMENTUM_OSC` (default ON).
 */
class SignalMomentumOscillator {
public:
    /**
     * @brief Direction of the most recent histogram zero-crossing.
     */
    enum class CrossDirection { None, Bullish, Bearish };

    /**
     * @brief Configuration for the oscillator.
     */
    struct Config {
        /** @brief EMA smoothing factor for the fast line (default ≈ 12-period EMA). */
        double fast_alpha{0.15};

        /** @brief EMA smoothing factor for the slow line (default ≈ 26-period EMA). */
        double slow_alpha{0.075};

        /** @brief EMA smoothing factor for the signal line (default ≈ 9-period EMA). */
        double signal_alpha{0.20};

        /**
         * @brief |histogram| threshold to fire on_divergence.
         *
         * When |histogram| >= this value the oscillator is considered in a
         * strong momentum regime and fires the callback.
         */
        double divergence_threshold{0.05};

        /**
         * @brief Callback fired when |histogram| >= divergence_threshold.
         *
         * Parameters: (macd, signal_line, histogram).
         */
        std::function<void(double macd, double signal_line, double histogram)> on_divergence;

        /**
         * @brief Callback fired on histogram zero-cross (bullish or bearish).
         *
         * Parameters: (direction, histogram_after_cross).
         */
        std::function<void(CrossDirection dir, double histogram)> on_cross;
    };

    /** @brief Construct with default configuration. */
    SignalMomentumOscillator() = default;

    /** @brief Construct with the given configuration. */
    explicit SignalMomentumOscillator(Config cfg) : config_(std::move(cfg)) {}

    /**
     * @brief Feed a new bias observation into the oscillator.
     *
     * Updates fast EMA, slow EMA, MACD, and signal line.  Fires
     * on_divergence / on_cross callbacks outside the internal lock.
     */
    void record(double bias) noexcept;

    /** @brief Update configuration; resets all EMA state. */
    void update_config(Config cfg);

    /** @brief Return a copy of the current configuration. */
    [[nodiscard]] Config get_config() const;

    /** @brief Current MACD line (fast_ema − slow_ema). Thread-safe. */
    [[nodiscard]] double macd() const noexcept {
        return macd_.load(std::memory_order_relaxed);
    }

    /** @brief Current signal line (EMA of MACD). Thread-safe. */
    [[nodiscard]] double signal_line() const noexcept {
        return signal_line_.load(std::memory_order_relaxed);
    }

    /** @brief Current histogram (macd − signal_line). Thread-safe. */
    [[nodiscard]] double histogram() const noexcept {
        return histogram_.load(std::memory_order_relaxed);
    }

    /** @brief True if the last histogram zero-cross was bullish. Thread-safe. */
    [[nodiscard]] bool is_bullish() const noexcept {
        return last_cross_.load(std::memory_order_relaxed) ==
               static_cast<int>(CrossDirection::Bullish);
    }

    /** @brief True if the last histogram zero-cross was bearish. Thread-safe. */
    [[nodiscard]] bool is_bearish() const noexcept {
        return last_cross_.load(std::memory_order_relaxed) ==
               static_cast<int>(CrossDirection::Bearish);
    }

    /** @brief Total bias observations recorded. Thread-safe. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /** @brief Total histogram zero-crossings detected. Thread-safe. */
    [[nodiscard]] uint64_t total_crosses() const noexcept {
        return crosses_.load(std::memory_order_relaxed);
    }

    /** @brief Reset all EMA state and counters. */
    void reset();

    /** @brief Serialise current state to a compact JSON string. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config             config_;

    // EMA state (under mutex_)
    double fast_ema_{0.0};
    double slow_ema_{0.0};
    double signal_ema_{0.0};
    double prev_histogram_{0.0};
    bool   seeded_{false};

    // Lock-free readable outputs
    std::atomic<double>   macd_{0.0};
    std::atomic<double>   signal_line_{0.0};
    std::atomic<double>   histogram_{0.0};
    std::atomic<int>      last_cross_{static_cast<int>(CrossDirection::None)};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> crosses_{0};
};

} // namespace llmquant
