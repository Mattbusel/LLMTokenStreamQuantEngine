#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Rate-of-change (velocity) of sentiment EMA.
 *
 * Tracks a fast EMA of bias and computes the one-step velocity as the
 * difference between consecutive EMA values.  An acceleration term
 * (change in velocity) is also tracked.  Callbacks fire when velocity
 * exceeds positive or negative surge thresholds.
 */
class NarrativeSentimentVelocity {
public:
    struct Config {
        double alpha{0.2};               ///< EMA smoothing factor
        double surge_threshold{0.01};    ///< |velocity| above this triggers callback
        int    min_samples{3};           ///< Warm-up samples

        /// Fired when velocity > +surge_threshold (positive momentum surge).
        std::function<void(double /*velocity*/)> on_positive_surge;
        /// Fired when velocity < -surge_threshold (negative momentum surge).
        std::function<void(double /*velocity*/)> on_negative_surge;
    };

    explicit NarrativeSentimentVelocity(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double ema()          const noexcept;
    double velocity()     const noexcept;  ///< First derivative of EMA
    double acceleration() const noexcept;  ///< Second derivative of EMA

    std::size_t positive_surges() const noexcept;
    std::size_t negative_surges() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    bool seeded_{false};
    double ema_raw_{0.0};
    double prev_ema_{0.0};
    double prev_vel_{0.0};

    std::atomic<double>   ema_{0.0};
    std::atomic<double>   vel_{0.0};
    std::atomic<double>   accel_{0.0};
    std::atomic<uint64_t> pos_surges_{0};
    std::atomic<uint64_t> neg_surges_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
