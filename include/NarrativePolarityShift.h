#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Tracks polarity (sign) distribution of bias over a rolling window.
 *
 * Computes the fraction of positive bias values in the window.  A callback
 * fires when the positive fraction crosses above or below configurable
 * thresholds, signalling a polarity shift in the narrative.
 */
class NarrativePolarityShift {
public:
    struct Config {
        int    window{20};              ///< Rolling window size
        int    min_samples{5};          ///< Minimum samples before polarity detection
        double bull_threshold{0.65};    ///< Positive fraction above → bullish shift
        double bear_threshold{0.35};    ///< Positive fraction below → bearish shift

        /// Fired when market shifts to bullish polarity.
        std::function<void(double /*pos_fraction*/)> on_bullish;
        /// Fired when market shifts to bearish polarity.
        std::function<void(double /*pos_fraction*/)> on_bearish;
    };

    explicit NarrativePolarityShift(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double positive_fraction() const noexcept; ///< Fraction of positive values in window
    double negative_fraction() const noexcept; ///< 1 - positive_fraction
    bool   is_bullish()        const noexcept;
    bool   is_bearish()        const noexcept;

    std::size_t bullish_events() const noexcept;
    std::size_t bearish_events() const noexcept;
    std::size_t total_records()  const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<int>    win_buf_;   ///< 1 = positive, 0 = non-positive
    int  win_head_{0};
    int  win_fill_{0};
    bool prev_bullish_{false};
    bool prev_bearish_{false};

    std::atomic<double>   pos_frac_{0.5};
    std::atomic<bool>     bullish_{false};
    std::atomic<bool>     bearish_{false};
    std::atomic<uint64_t> bull_events_{0};
    std::atomic<uint64_t> bear_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
