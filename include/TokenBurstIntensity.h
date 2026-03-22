#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Token burst intensity detector via local variance spikes.
 *
 * Tracks a short-window variance and compares to a long-window baseline.
 * When short/long variance ratio exceeds a threshold, a burst is detected.
 * Useful for detecting sudden surges in LLM token sentiment volatility.
 */
class TokenBurstIntensity {
public:
    struct Config {
        int    short_window{5};           ///< Short variance window
        int    long_window{30};           ///< Long baseline variance window
        double burst_ratio{3.0};          ///< Short/long variance ratio for burst
        int    min_samples{10};           ///< Minimum samples before burst detection

        /// Fired when burst begins (ratio exceeds threshold).
        std::function<void(double /*ratio*/)> on_burst_start;
        /// Fired when burst ends (ratio drops below threshold).
        std::function<void(double /*ratio*/)> on_burst_end;
    };

    explicit TokenBurstIntensity(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double short_variance() const noexcept;
    double long_variance()  const noexcept;
    double burst_ratio()    const noexcept;
    bool   is_bursting()    const noexcept;

    std::size_t burst_events() const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> short_buf_;
    std::vector<double> long_buf_;
    int short_head_{0}, short_fill_{0};
    int long_head_{0},  long_fill_{0};
    bool prev_bursting_{false};

    static double window_variance(const std::vector<double>& buf, int fill, int sz) noexcept;

    std::atomic<double>   short_var_{0.0};
    std::atomic<double>   long_var_{0.0};
    std::atomic<double>   ratio_{0.0};
    std::atomic<bool>     bursting_{false};
    std::atomic<uint64_t> burst_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
