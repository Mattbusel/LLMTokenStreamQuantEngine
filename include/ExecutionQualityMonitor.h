#pragma once

/// @file ExecutionQualityMonitor.h
/// @brief Tracks fill latency and slippage distribution between signal emission and OMS ack.
///
/// In a live trading system there are two critical quality metrics between
/// a TradeSignal and its corresponding OMS fill:
///
///   1. **Fill latency** — wall-clock time from signal emission to OMS acknowledgment.
///      High latency → adverse price movement before the order reaches the exchange.
///
///   2. **Slippage** — difference between the signal's implied price and the actual
///      fill price.  Positive slippage = filled at worse price than expected.
///
/// This monitor accepts (signal_id, ack_latency_us, slippage_bps) tuples from
/// the OMS callback and maintains a rolling window with percentile statistics.
///
/// Thread-safe: all public methods are thread-safe.
///
/// Feature flag: LLMQUANT_ENABLE_EXECUTION_QUALITY (default ON).

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class ExecutionQualityMonitor {
public:
    struct FillRecord {
        uint64_t signal_id{0};
        double   latency_us{0.0};  ///< Signal → OMS ack latency in microseconds.
        double   slippage_bps{0.0}; ///< Slippage in basis points (positive = worse).
    };

    struct Stats {
        double mean_latency_us{0.0};
        double p50_latency_us{0.0};
        double p95_latency_us{0.0};
        double p99_latency_us{0.0};
        double mean_slippage_bps{0.0};
        double p95_slippage_bps{0.0};
        uint64_t sample_count{0};
    };

    struct Config {
        /// Rolling window size (number of fill records).  Default 512.
        uint32_t window_size{512};

        /// Latency SLA in microseconds.  on_sla_breach fires when exceeded.
        double latency_sla_us{5000.0};

        /// Slippage SLA in basis points.  on_sla_breach fires when exceeded.
        double slippage_sla_bps{5.0};

        /// Called when either SLA is breached.
        /// Parameters: (fill_record, latency_breach, slippage_breach).
        std::function<void(const FillRecord&, bool, bool)> on_sla_breach;
    };

    explicit ExecutionQualityMonitor(Config cfg = Config{});

    // Non-copyable.
    ExecutionQualityMonitor(const ExecutionQualityMonitor&)            = delete;
    ExecutionQualityMonitor& operator=(const ExecutionQualityMonitor&) = delete;

    /// Record a fill observation.
    void record(const FillRecord& fill);

    /// Convenience overload.
    void record(uint64_t signal_id, double latency_us, double slippage_bps);

    /// Compute statistics over the current window (O(N log N) sort).
    [[nodiscard]] Stats compute_stats() const;

    /// Total fills recorded (including those evicted from window).
    [[nodiscard]] uint64_t total_fills() const noexcept {
        return total_fills_.load(std::memory_order_relaxed);
    }

    /// Total SLA breach events.
    [[nodiscard]] uint64_t sla_breach_count() const noexcept {
        return breach_count_.load(std::memory_order_relaxed);
    }

    /// Reset the window and counters.
    void reset();

    /// Update config; resets the window.
    void update_config(const Config& cfg);

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    struct Entry { double latency_us; double slippage_bps; };
    std::vector<Entry> window_;
    int  ring_head_{0};
    int  ring_fill_{0};

    std::atomic<uint64_t> total_fills_{0};
    std::atomic<uint64_t> breach_count_{0};

    mutable std::mutex mutex_;

    static double percentile(std::vector<double>& sorted_asc, double pct) noexcept;
};

} // namespace llmquant
