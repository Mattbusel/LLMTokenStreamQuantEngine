#pragma once
/// TokenCadenceAnalyser — inter-token timing analyser for cadence and burst detection.
/// Records wall-clock arrival times and computes mean inter-arrival interval (IAI),
/// coefficient of variation (CV = std/mean), and burst score.
/// High CV → bursty arrival pattern; low CV → regular cadence.
/// Fires on_burst when IAI drops below burst_threshold (rapid token flood)
/// and on_gap when IAI exceeds gap_threshold (token drought).
///
/// Feature flag: LLMQUANT_ENABLE_CADENCE_ANALYSER / LLMQUANT_CADENCE_ANALYSER_ENABLED

#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenCadenceAnalyser {
public:
    using Clock    = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct Config {
        int    window          = 20;    ///< rolling window for IAI statistics
        int    min_samples     = 5;     ///< minimum arrivals before computing
        double burst_threshold_ms = 5.0;  ///< IAI below this (ms) fires on_burst
        double gap_threshold_ms   = 500.0; ///< IAI above this (ms) fires on_gap

        std::function<void(double iai_ms)> on_burst; ///< fires on rapid token flood
        std::function<void(double iai_ms)> on_gap;   ///< fires on token drought
    };

    explicit TokenCadenceAnalyser(Config cfg = {});

    /// Record a new token arrival (uses current wall-clock time).
    void record(double bias); // bias ignored — tracks timing only

    double mean_iai_ms() const noexcept; ///< mean inter-arrival interval (ms)
    double cv_iai()      const noexcept; ///< coefficient of variation of IAI
    double last_iai_ms() const noexcept; ///< most recent inter-arrival interval

    std::size_t burst_events() const noexcept;
    std::size_t gap_events()   const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> iai_buf_; ///< rolling IAI values (ms)
    int     iai_head_    = 0;
    int     iai_fill_    = 0;
    bool    has_prev_    = false;
    TimePoint prev_time_;

    std::atomic<double>      mean_iai_{0.0};
    std::atomic<double>      cv_iai_{0.0};
    std::atomic<double>      last_iai_{0.0};
    std::atomic<std::size_t> burst_events_{0};
    std::atomic<std::size_t> gap_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_CADENCE_ANALYSER_ENABLED
