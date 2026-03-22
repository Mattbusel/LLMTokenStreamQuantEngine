#pragma once
/// NarrativeCoherenceTracker — measures directional coherence of the bias stream.
/// Coherence = fraction of rolling window tokens whose sign matches the window median sign.
/// High coherence (→1) = uniform narrative; low coherence (→0) = conflicted/mixed signal.
/// Fires on_high_coherence / on_low_coherence on threshold crossings.
///
/// Feature flag: LLMQUANT_ENABLE_COHERENCE_TRACKER / LLMQUANT_COHERENCE_TRACKER_ENABLED

#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeCoherenceTracker {
public:
    struct Config {
        int    window              = 20;   ///< rolling window for coherence calculation
        int    min_samples         = 5;    ///< minimum before emitting coherence
        double high_threshold      = 0.75; ///< coherence above this → on_high_coherence
        double low_threshold       = 0.40; ///< coherence below this → on_low_coherence

        std::function<void(double coherence)> on_high_coherence; ///< unified narrative
        std::function<void(double coherence)> on_low_coherence;  ///< conflicted narrative
    };

    explicit NarrativeCoherenceTracker(Config cfg = {});

    void record(double bias);

    double coherence() const noexcept; ///< ∈ [0, 1]

    std::size_t high_coherence_events() const noexcept;
    std::size_t low_coherence_events()  const noexcept;
    std::size_t total_records()         const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;

    std::vector<double> buf_;
    int    head_       = 0;
    int    fill_       = 0;
    bool   prev_high_  = false;
    bool   prev_low_   = false;

    std::atomic<double>      coherence_{0.0};
    std::atomic<std::size_t> high_events_{0};
    std::atomic<std::size_t> low_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_COHERENCE_TRACKER_ENABLED
