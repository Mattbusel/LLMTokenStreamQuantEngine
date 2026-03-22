#pragma once
/// BiasEntropyRate — Shannon entropy rate of the discretised bias stream.
/// Bias values are bucketed into n_buckets bins over a rolling window.
/// Entropy ∈ [0, log2(n_buckets)]; normalised to [0, 1].
/// Low entropy → structured/predictable; high entropy → noisy/random.
/// Fires on_high_entropy / on_low_entropy on threshold crossings.
///
/// Feature flag: LLMQUANT_ENABLE_BIAS_ENTROPY_RATE / LLMQUANT_BIAS_ENTROPY_RATE_ENABLED

#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasEntropyRate {
public:
    struct Config {
        int    window          = 32;    ///< rolling window for entropy computation
        int    n_buckets       = 8;     ///< number of discrete histogram bins
        int    min_samples     = 10;    ///< minimum before emitting entropy
        double high_threshold  = 0.75;  ///< normalised entropy above this → on_high_entropy
        double low_threshold   = 0.3;   ///< normalised entropy below this → on_low_entropy

        std::function<void(double entropy)> on_high_entropy; ///< fires on noisy/random regime
        std::function<void(double entropy)> on_low_entropy;  ///< fires on structured regime
    };

    explicit BiasEntropyRate(Config cfg = {});

    void record(double bias);

    double entropy() const noexcept; ///< normalised Shannon entropy ∈ [0, 1]

    std::size_t high_entropy_events() const noexcept;
    std::size_t low_entropy_events()  const noexcept;
    std::size_t total_records()       const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_       = 0;
    int    fill_       = 0;
    bool   prev_high_  = false;
    bool   prev_low_   = false;

    std::atomic<double>      entropy_{0.0};
    std::atomic<std::size_t> high_events_{0};
    std::atomic<std::size_t> low_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
