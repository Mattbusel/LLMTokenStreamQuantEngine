#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Narrative pressure gauge: spread between fast and slow EMAs of bias.
 *
 * Pressure = fast_ema - slow_ema.  Positive pressure indicates bullish
 * narrative momentum building; negative pressure indicates bearish pressure.
 * Fires callbacks when pressure crosses signed thresholds.
 */
class NarrativePressureGauge {
public:
    struct Config {
        double alpha_fast{0.3};           ///< Fast EMA smoothing factor
        double alpha_slow{0.05};          ///< Slow EMA smoothing factor
        double pressure_threshold{0.01};  ///< |pressure| above this triggers callback
        int    min_samples{5};            ///< Warm-up samples

        /// Fired when pressure rises above +pressure_threshold.
        std::function<void(double /*pressure*/)> on_positive_pressure;
        /// Fired when pressure falls below -pressure_threshold.
        std::function<void(double /*pressure*/)> on_negative_pressure;
    };

    explicit NarrativePressureGauge(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double fast_ema()  const noexcept;
    double slow_ema()  const noexcept;
    double pressure()  const noexcept;  ///< fast_ema - slow_ema

    std::size_t positive_pressure_events() const noexcept;
    std::size_t negative_pressure_events() const noexcept;
    std::size_t total_records()            const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    bool seeded_{false};
    double ema_fast_{0.0};
    double ema_slow_{0.0};
    bool prev_pos_pressure_{false};
    bool prev_neg_pressure_{false};

    std::atomic<double>   fast_{0.0};
    std::atomic<double>   slow_{0.0};
    std::atomic<double>   pressure_{0.0};
    std::atomic<uint64_t> pos_press_ev_{0};
    std::atomic<uint64_t> neg_press_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
