#pragma once
/// SignalAccelerationMeter — second-derivative acceleration of the bias stream.
/// Computes velocity (first diff EMA) and acceleration (second diff EMA).
/// Fires on_inflection when acceleration crosses zero (momentum reversal),
/// and on_surge when |acceleration| exceeds surge_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_ACCELERATION_METER / LLMQUANT_ACCELERATION_METER_ENABLED

#ifdef LLMQUANT_ACCELERATION_METER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalAccelerationMeter {
public:
    struct Config {
        double alpha_vel   = 0.2;   ///< EMA smoothing for velocity
        double alpha_acc   = 0.2;   ///< EMA smoothing for acceleration
        double surge_threshold = 0.01; ///< |accel| above this fires on_surge
        int    min_samples = 5;     ///< minimum records before computing

        /// Fires when acceleration crosses zero (velocity inflection point).
        std::function<void(double vel, double acc)> on_inflection;
        /// Fires when |acceleration| exceeds surge_threshold.
        std::function<void(double acc)> on_surge;
    };

    explicit SignalAccelerationMeter(Config cfg = {});

    /// Ingest a new bias sample and update velocity/acceleration estimates.
    void record(double bias);

    double velocity()     const noexcept;
    double acceleration() const noexcept;
    double max_accel()    const noexcept;

    std::size_t inflection_events() const noexcept;
    std::size_t surge_events()      const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double prev_bias_  = 0.0;
    double ema_vel_    = 0.0;
    double ema_acc_    = 0.0;
    double prev_vel_   = 0.0;
    double max_accel_  = 0.0;
    bool   seeded_     = false;
    bool   prev_acc_pos_ = false;

    std::atomic<double>      vel_{0.0};
    std::atomic<double>      acc_{0.0};
    std::atomic<std::size_t> inflection_events_{0};
    std::atomic<std::size_t> surge_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_ACCELERATION_METER_ENABLED
