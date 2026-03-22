#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling z-score normalisation of bias values.
 *
 * z = (bias - rolling_mean) / rolling_stddev over a configurable window.
 * Fires callbacks when |z| exceeds an extreme threshold, flagging outlier
 * bias events.
 */
class BiasZScoreNormaliser {
public:
    struct Config {
        int    window{30};              ///< Rolling window for mean/stddev
        int    min_samples{5};          ///< Minimum samples before z-score is valid
        double extreme_threshold{2.0};  ///< |z| above this → extreme event

        /// Fired when z > +extreme_threshold.
        std::function<void(double /*z*/)> on_positive_extreme;
        /// Fired when z < -extreme_threshold.
        std::function<void(double /*z*/)> on_negative_extreme;
    };

    explicit BiasZScoreNormaliser(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double z_score()     const noexcept;
    double rolling_mean() const noexcept;
    double rolling_std()  const noexcept;

    std::size_t positive_extremes() const noexcept;
    std::size_t negative_extremes() const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> buf_;
    int head_{0}, fill_{0};

    std::atomic<double>   z_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<double>   std_{0.0};
    std::atomic<uint64_t> pos_ext_{0};
    std::atomic<uint64_t> neg_ext_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
