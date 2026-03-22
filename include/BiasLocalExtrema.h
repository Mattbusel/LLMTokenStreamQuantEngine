#pragma once
/// BiasLocalExtrema — zigzag local peak/trough detector for the bias stream.
/// Tracks alternating local maxima and minima. A new extremum is registered
/// when the bias retraces more than reversal_threshold from the current extremum.
/// Fires on_peak(val) / on_trough(val) when a new local extremum is confirmed.
///
/// Feature flag: LLMQUANT_ENABLE_LOCAL_EXTREMA / LLMQUANT_LOCAL_EXTREMA_ENABLED

#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasLocalExtrema {
public:
    struct Config {
        double reversal_threshold = 0.01; ///< minimum reversal to confirm new extremum
        int    min_samples        = 3;    ///< warm-up before detecting extrema

        std::function<void(double val)> on_peak;   ///< fires when local maximum confirmed
        std::function<void(double val)> on_trough; ///< fires when local minimum confirmed
    };

    explicit BiasLocalExtrema(Config cfg = {});

    void record(double bias);

    double last_peak()   const noexcept; ///< most recent confirmed peak value
    double last_trough() const noexcept; ///< most recent confirmed trough value
    bool   seeking_peak() const noexcept; ///< true if currently searching for a peak

    std::size_t peak_events()   const noexcept;
    std::size_t trough_events() const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    bool   seeded_       = false;
    bool   seek_peak_    = true;  ///< start by looking for a peak
    double extremum_     = 0.0;   ///< current candidate extremum

    std::atomic<double>      last_peak_{0.0};
    std::atomic<double>      last_trough_{0.0};
    std::atomic<bool>        seeking_peak_{true};
    std::atomic<std::size_t> peak_ev_{0};
    std::atomic<std::size_t> trough_ev_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_LOCAL_EXTREMA_ENABLED
