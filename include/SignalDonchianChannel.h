#pragma once
#include <atomic>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Donchian Channel: rolling high/low bands over a bias window.
 *
 * Tracks the highest (upper band) and lowest (lower band) bias values over a
 * rolling window.  Fires callbacks when a new record-high or record-low is set
 * within the current window.
 */
class SignalDonchianChannel {
public:
    struct Config {
        int window{20};      ///< Rolling window size
        int min_samples{5};  ///< Minimum samples before band computation

        /// Fired when the upper band rises to a new high.
        std::function<void(double /*new_high*/, double /*prev_high*/)> on_upper_breakout;
        /// Fired when the lower band drops to a new low.
        std::function<void(double /*new_low*/, double /*prev_low*/)> on_lower_breakout;
    };

    explicit SignalDonchianChannel(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double upper_band()  const noexcept;
    double lower_band()  const noexcept;
    double midline()     const noexcept;
    bool   is_at_high()  const noexcept;
    bool   is_at_low()   const noexcept;

    std::size_t upper_breakouts() const noexcept;
    std::size_t lower_breakouts() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> win_buf_;
    int  win_head_{0};
    int  win_fill_{0};
    bool seeded_{false};

    std::atomic<double>   upper_{0.0};
    std::atomic<double>   lower_{0.0};
    std::atomic<double>   mid_{0.0};
    std::atomic<bool>     at_high_{false};
    std::atomic<bool>     at_low_{false};
    std::atomic<uint64_t> upper_events_{0};
    std::atomic<uint64_t> lower_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
