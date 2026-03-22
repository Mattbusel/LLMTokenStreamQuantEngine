#pragma once

/// @file SignalTailRiskMeter.h
/// @brief Measures tail risk in the bias stream using rolling Expected Shortfall
///        (ES, also known as Conditional Value-at-Risk) beyond a given quantile.
///
/// ## Motivation
/// VaR at 95% tells you the threshold below which 5% of losses fall; ES tells
/// you the *expected* loss given that you are already in the tail.  For a bias
/// stream where large negative events precede adverse trade fills, ES at 95%
/// gives a more conservative and coherent risk measure than simple drawdown.
///
/// ES_α = E[X | X ≤ Q_α(X)]  (for loss X = −bias)
///
/// ## Algorithm
/// - Maintains a rolling window of `window` bias observations.
/// - Computes the α-quantile of the negative-bias distribution via partial sort.
/// - ES = mean of all values below the quantile.
/// - Fires `on_tail_event(es)` when ES_α drops below `es_threshold` (risk spike).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TAIL_RISK_METER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalTailRiskMeter {
public:
    struct Config {
        /// Rolling window. Default 100.
        int window{100};

        /// Confidence level α for ES (0.95 = 95th percentile). Default 0.95.
        double alpha{0.95};

        /// Minimum fill before computing. Default 20.
        int min_samples{20};

        /// ES below this threshold fires on_tail_event (negative = loss). Default -0.3.
        double es_threshold{-0.3};

        /// Fired when ES_α < es_threshold (outside lock).
        std::function<void(double es)> on_tail_event;
    };

    explicit SignalTailRiskMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Rolling VaR at α (α-quantile of bias distribution).
    [[nodiscard]] double var_alpha() const noexcept {
        return var_.load(std::memory_order_relaxed);
    }

    /// Rolling Expected Shortfall ES_α = E[bias | bias ≤ VaR_α].
    [[nodiscard]] double expected_shortfall() const noexcept {
        return es_.load(std::memory_order_relaxed);
    }

    /// Total tail events fired.
    [[nodiscard]] uint64_t tail_events() const noexcept {
        return tail_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    bool prev_ok_{true};

    std::atomic<double>   var_{0.0};
    std::atomic<double>   es_{0.0};
    std::atomic<uint64_t> tail_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
