#pragma once
/// NarrativePersistenceIndex — simplified Hurst-like persistence measure via run analysis.
/// Counts runs of consecutive values above/below the rolling mean. A run-length above 1
/// indicates auto-correlation (persistence). persistence_index ∈ [0, 1]:
/// > 0.5 → trending (long runs); < 0.5 → mean-reverting (short runs).
/// Fires on_persistent / on_mean_reverting on regime transitions.
///
/// Feature flag: LLMQUANT_ENABLE_PERSISTENCE_INDEX / LLMQUANT_PERSISTENCE_INDEX_ENABLED

#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativePersistenceIndex {
public:
    struct Config {
        int    window             = 30;   ///< rolling window for run analysis
        int    min_samples        = 10;   ///< minimum before emitting index
        double persistent_threshold  = 0.6; ///< PI above this → on_persistent
        double reverting_threshold   = 0.4; ///< PI below this → on_mean_reverting

        std::function<void(double pi)> on_persistent;    ///< fires on persistence regime entry
        std::function<void(double pi)> on_mean_reverting; ///< fires on mean-reversion regime entry
    };

    explicit NarrativePersistenceIndex(Config cfg = {});

    void record(double bias);

    double persistence_index() const noexcept; ///< avg run length / window, ∈ [0, 1]
    bool   is_persistent()     const noexcept;
    bool   is_reverting()      const noexcept;

    std::size_t persistent_events() const noexcept;
    std::size_t reverting_events()  const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_           = 0;
    int    fill_           = 0;
    bool   prev_persistent_ = false;
    bool   prev_reverting_  = false;

    std::atomic<double>      pi_{0.5};
    std::atomic<bool>        is_persistent_{false};
    std::atomic<bool>        is_reverting_{false};
    std::atomic<std::size_t> persistent_events_{0};
    std::atomic<std::size_t> reverting_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_PERSISTENCE_INDEX_ENABLED
