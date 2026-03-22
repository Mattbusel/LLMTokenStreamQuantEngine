#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Token bias gap/jump detector.
 *
 * Tracks the absolute change between consecutive bias values.  When the
 * change (gap) exceeds a configurable multiple of the rolling ATR, a gap
 * event is fired.  Useful for detecting sudden discrete shifts in narrative
 * sentiment (e.g., a new topic or extreme token).
 */
class TokenBiasGapDetector {
public:
    struct Config {
        double alpha_atr{0.1};         ///< EMA factor for ATR baseline
        double gap_multiplier{3.0};    ///< |Δbias| > gap_multiplier × ATR → gap
        int    min_samples{5};         ///< Warm-up samples

        /// Fired on positive gap (upward jump exceeding threshold).
        std::function<void(double /*gap*/, double /*threshold*/)> on_gap_up;
        /// Fired on negative gap (downward jump exceeding threshold).
        std::function<void(double /*gap*/, double /*threshold*/)> on_gap_down;
    };

    explicit TokenBiasGapDetector(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double atr()       const noexcept;
    double threshold() const noexcept;  ///< gap_multiplier × atr
    double last_gap()  const noexcept;  ///< most recent |Δbias|

    std::size_t gap_up_events()   const noexcept;
    std::size_t gap_down_events() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    bool seeded_{false};
    double prev_bias_{0.0};
    double atr_raw_{0.0};

    std::atomic<double>   atr_{0.0};
    std::atomic<double>   threshold_{0.0};
    std::atomic<double>   last_gap_{0.0};
    std::atomic<uint64_t> gap_up_ev_{0};
    std::atomic<uint64_t> gap_dn_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
