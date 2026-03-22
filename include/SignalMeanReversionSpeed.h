#pragma once
/// SignalMeanReversionSpeed — Ornstein-Uhlenbeck speed-of-mean-reversion estimator.
/// Fits an AR(1) model to the bias stream: x[t] = θ·x[t-1] + ε.
/// Mean-reversion speed κ = -log(θ)/dt. Half-life = log(2)/κ.
/// κ > 0 → mean-reverting; θ > 1 → explosive (trend-following).
/// Fires on_fast_reversion / on_explosive on regime transitions.
///
/// Feature flag: LLMQUANT_ENABLE_MEAN_REVERSION_SPEED / LLMQUANT_MEAN_REVERSION_SPEED_ENABLED

#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalMeanReversionSpeed {
public:
    struct Config {
        int    window             = 30;   ///< rolling OLS window
        int    min_samples        = 10;   ///< minimum before emitting estimate
        double fast_kappa_threshold = 1.0; ///< κ > this → on_fast_reversion
        double explosive_theta    = 1.0;  ///< |θ| > this → on_explosive

        /// Fires when κ exceeds fast_kappa_threshold (rapid mean reversion).
        std::function<void(double kappa)> on_fast_reversion;
        /// Fires when |θ| exceeds explosive_theta (explosive / trending).
        std::function<void(double theta)> on_explosive;
    };

    explicit SignalMeanReversionSpeed(Config cfg = {});

    void record(double bias);

    double theta()      const noexcept; ///< AR(1) coefficient
    double kappa()      const noexcept; ///< mean-reversion speed (−log(θ))
    double half_life()  const noexcept; ///< log(2)/κ (positive → mean-reverting)

    std::size_t fast_reversion_events() const noexcept;
    std::size_t explosive_events()      const noexcept;
    std::size_t total_records()         const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_      = 0;
    int    fill_      = 0;
    bool   prev_fast_ = false;
    bool   prev_expl_ = false;

    std::atomic<double>      theta_{0.0};
    std::atomic<double>      kappa_{0.0};
    std::atomic<double>      half_life_{0.0};
    std::atomic<std::size_t> fast_events_{0};
    std::atomic<std::size_t> explosive_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
