#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Latency-aware adaptive signal emission cooldown.
 *
 * Replaces the fixed `signal_cooldown` in TradeSignalEngine::Config with a
 * dynamic cooldown that widens automatically when the pipeline is under
 * latency pressure and narrows when it is healthy.
 *
 * ## Control law
 * The cooldown is a clamped linear function of the current P99 latency:
 *
 *   cooldown_us = clamp(
 *       base_cooldown_us + pressure_gain * max(0, p99_us - target_p99_us),
 *       min_cooldown_us,
 *       max_cooldown_us
 *   )
 *
 * where:
 *   - `target_p99_us` is the desired P99 latency budget (default: 10 µs)
 *   - `pressure_gain`  scales how aggressively cooldown grows per excess µs
 *   - `base_cooldown_us` is the cooldown to use when P99 == target (i.e. nominal)
 *
 * ## Hysteresis
 * To prevent rapid oscillation, the cooldown can only decrease when the P99
 * has been at or below target for at least `recovery_window_s` consecutive
 * seconds.
 *
 * ## Thread safety
 * `update_p99(double)` and all accessors are safe to call from any thread.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ADAPTIVE_COOLDOWN` (default ON).
 */
class AdaptiveCooldownController {
public:
    /**
     * @brief Adaptive cooldown tuning parameters.
     */
    struct Config {
        /**
         * @brief Nominal cooldown (µs) when P99 latency == target_p99_us.
         * Default 500 µs (0.5 ms).
         */
        double base_cooldown_us{500.0};

        /**
         * @brief Minimum allowed cooldown in µs.  Cannot go below this even
         *        when latency is excellent.  Default 100 µs.
         */
        double min_cooldown_us{100.0};

        /**
         * @brief Maximum allowed cooldown in µs.  Caps the backpressure.
         * Default 10 000 µs (10 ms).
         */
        double max_cooldown_us{10000.0};

        /**
         * @brief Target P99 latency budget in µs.
         *
         * When actual P99 exceeds this, the cooldown begins to grow.
         * Default 10 µs (engine SLA).
         */
        double target_p99_us{10.0};

        /**
         * @brief Gain applied to excess P99 latency when computing the
         *        adjusted cooldown.
         *
         * Units: µs of cooldown per µs of excess P99.
         * E.g. gain=5 and P99 20 µs over budget → +100 µs cooldown.
         * Default 5.0.
         */
        double pressure_gain{5.0};

        /**
         * @brief Seconds of consecutive below-target P99 required before the
         *        cooldown is allowed to decrease.
         *
         * Prevents rapid oscillation.  Default 5 s.
         */
        int recovery_window_s{5};

        /**
         * @brief EMA alpha for smoothing the reported P99.
         *
         * Prevents a single spike from doubling the cooldown.
         * Range (0, 1]: 1.0 = no smoothing; 0.1 = heavy smoothing.
         * Default 0.3.
         */
        double ema_alpha{0.3};
    };

    /**
     * @brief Construct with the given configuration.
     *
     * @param config Gain, budget, and clamp parameters.
     */
    explicit AdaptiveCooldownController(Config config = Config{});

    /**
     * @brief Feed the latest P99 latency measurement (in µs).
     *
     * Updates the EMA, recomputes the current cooldown, and manages the
     * hysteresis timer.  Call this once per LatencyController stats sample.
     *
     * Thread-safe.
     *
     * @param p99_us Observed P99 latency in microseconds.
     */
    void update_p99(double p99_us);

    /**
     * @brief Return the current adaptive cooldown as a chrono duration.
     *
     * Suitable for passing directly to TradeSignalEngine::update_config().
     * Thread-safe (atomic read).
     */
    [[nodiscard]] std::chrono::microseconds get_cooldown() const noexcept {
        return std::chrono::microseconds{
            static_cast<long long>(current_cooldown_us_.load(std::memory_order_relaxed))
        };
    }

    /**
     * @brief Return the current cooldown in microseconds (floating-point).
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_cooldown_us() const noexcept {
        return current_cooldown_us_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the EMA-smoothed P99 latency last fed via update_p99().
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_ema_p99() const noexcept {
        return ema_p99_us_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when the cooldown is currently expanded beyond base_cooldown_us
     *        due to latency pressure.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_under_pressure() const noexcept {
        double cd = current_cooldown_us_.load(std::memory_order_relaxed);
        double base = config_snapshot_base_.load(std::memory_order_relaxed);
        return cd > base;
    }

    /**
     * @brief Total number of times the cooldown was increased due to pressure.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t pressure_expansions() const noexcept {
        return pressure_expansions_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of times the cooldown was decreased during recovery.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t recoveries() const noexcept {
        return recoveries_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration at runtime.
     *
     * The EMA state is preserved; only the config parameters change.
     * Thread-safe.
     *
     * @param config New configuration.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.
     * Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Serialise current state to a JSON object string.
     *
     * Fields: cooldown_us, ema_p99_us, under_pressure, pressure_expansions,
     * recoveries, base_cooldown_us, target_p99_us.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset the EMA and hysteresis timer.
     *
     * Restores the cooldown to base_cooldown_us.
     * Thread-safe.
     */
    void reset();

private:
    mutable std::mutex mutex_;
    Config config_;

    std::atomic<double> ema_p99_us_;
    std::atomic<double> current_cooldown_us_;
    std::atomic<double> config_snapshot_base_;  // cached for is_under_pressure

    // Recovery hysteresis timer (protected by mutex_).
    std::chrono::steady_clock::time_point below_target_since_{};
    bool   below_target_timer_running_{false};

    std::atomic<uint64_t> pressure_expansions_{0};
    std::atomic<uint64_t> recoveries_{0};
};

} // namespace llmquant
