#pragma once

/// @file OnlineGrangerCausality.h
/// @brief Online simplified Granger causality test between two signal streams.
///
/// ## Motivation
/// Granger causality answers: "does knowing stream X's past help predict
/// stream Y better than Y's own past alone?"  In an LLM sentiment context:
///   - Does the LLM bias *Granger-cause* the options-implied volatility (IV)?
///   - Does IV lead the bias, or do they move together?
/// This guides whether to use sentiment as a leading indicator or as a
/// confirming indicator.
///
/// ## Algorithm
/// Maintains two sliding windows of length `window` for streams X and Y.
/// On each new pair (x, y) the estimator fits two VAR(1) regression models:
///
///   Restricted:   y_t = a + b*y_{t-1} + ε_r      (SS_r)
///   Unrestricted: y_t = a + b*y_{t-1} + c*x_{t-1} + ε_u  (SS_u)
///
/// Granger F-statistic (1 extra regressor, approximated as chi-squared / 1):
///   F = (SS_r − SS_u) / SS_u × (n − 3)
///
/// Positive F with SS_u < SS_r  → X Granger-causes Y.
/// Fires `on_x_causes_y` when F exceeds `f_threshold` (default 3.84 ≈ chi²(1, 0.05)).
/// Fires `on_y_causes_x` symmetrically from the reversed regression.
/// Fires `on_bidirectional` when both directions are significant simultaneously.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ONLINE_GRANGER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class OnlineGrangerCausality {
public:
    struct Config {
        /// Sliding window length (samples). Default 64.
        int window{64};

        /// Minimum samples before F-statistics are valid. Default == window.
        int min_samples{0};

        /// F-statistic threshold for significance. Default 3.84 (χ² α=0.05 df=1).
        double f_threshold{3.84};

        /// Hysteresis factor to clear a causality flag (clear when F < threshold * hysteresis).
        double hysteresis{0.7};

        /// Fired when X significantly Granger-causes Y. Args: F-statistic.
        std::function<void(double f_stat)> on_x_causes_y;

        /// Fired when Y significantly Granger-causes X.
        std::function<void(double f_stat)> on_y_causes_x;

        /// Fired when both directions are simultaneously significant.
        std::function<void()> on_bidirectional;
    };

    explicit OnlineGrangerCausality(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake — provide paired observations (x_t, y_t)
    // -----------------------------------------------------------------------

    void record(double x, double y);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// F-statistic for X→Y causality, or NaN if not enough data.
    [[nodiscard]] double f_xy() const noexcept {
        return f_xy_.load(std::memory_order_relaxed);
    }

    /// F-statistic for Y→X causality, or NaN if not enough data.
    [[nodiscard]] double f_yx() const noexcept {
        return f_yx_.load(std::memory_order_relaxed);
    }

    /// True if X currently Granger-causes Y.
    [[nodiscard]] bool x_causes_y() const noexcept {
        return xcay_.load(std::memory_order_relaxed);
    }

    /// True if Y currently Granger-causes X.
    [[nodiscard]] bool y_causes_x() const noexcept {
        return ycax_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t xy_causality_events() const noexcept {
        return xy_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t yx_causality_events() const noexcept {
        return yx_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    struct Sample { double x, y; };

    /// Fit VAR(1) restricted model: dep_t = a + b*dep_{t-1}.
    /// Returns SS_residual.  Writes fitted slope to b_out.
    static double fit_restricted(const std::vector<double>& dep, int n,
                                  double& a_out, double& b_out);

    /// Fit VAR(1) unrestricted: dep_t = a + b*dep_{t-1} + c*lag_{t-1}.
    /// Returns SS_residual.
    static double fit_unrestricted(const std::vector<double>& dep,
                                    const std::vector<double>& lag, int n);

    void recompute_locked();

    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<Sample> buf_;
    int head_{0};
    int fill_{0};

    bool prev_xcay_{false};
    bool prev_ycax_{false};

    std::atomic<double>   f_xy_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double>   f_yx_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<bool>     xcay_{false};
    std::atomic<bool>     ycax_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> xy_events_{0};
    std::atomic<uint64_t> yx_events_{0};
};

} // namespace llmquant
