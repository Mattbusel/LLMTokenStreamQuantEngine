#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "LLMAdapter.h"   // SemanticWeight
#include "OutputSink.h"

namespace llmquant {

/// A quantitative trade signal derived from one or more semantic weights.
///
/// Both a high-resolution chrono timestamp and a nanosecond integer timestamp
/// are provided: the chrono field is used for latency arithmetic inside the
/// engine; the integer field is used for serialisation and cross-process IPC.
struct TradeSignal {
    /// Nanoseconds since the Unix epoch at signal emission time.
    uint64_t timestamp_ns{0};

    /// High-resolution timestamp at signal emission time (for latency arithmetic).
    std::chrono::high_resolution_clock::time_point timestamp;

    /// Accumulated directional bias shift (negative = sell, positive = buy).
    double delta_bias_shift{0.0};

    /// Volatility adjustment to apply to spread / options pricing models.
    double volatility_adjustment{0.0};

    /// Spread modifier in basis points.
    double spread_modifier{0.0};

    /// Confidence in this signal in [0.0, 1.0].
    double confidence{0.0};

    /// Measured token-to-signal latency in microseconds.
    double latency_us{0.0};

    /// Strategy toggle: 0 = neutral, 1 = bullish strategy, -1 = bearish strategy.
    int strategy_toggle{0};

    /// Weighting applied to the selected strategy (0.0 = ignore, 1.0 = full weight).
    double strategy_weight{0.0};
};

/// Callback invoked once per emitted TradeSignal on the engine's calling thread.
using TradeSignalCallback = std::function<void(const TradeSignal&)>;

/// Converts a stream of SemanticWeights into TradeSignals.
///
/// Incoming weights are accumulated with an exponential decay, then a signal
/// is emitted when the cooldown period has elapsed (realtime mode) or on every
/// token (backtest mode).
///
/// Thread safety: process_semantic_weight() is NOT thread-safe; all calls
/// must arrive from the same thread.  get_stats() is always safe (atomic
/// reads).  set_* configuration methods must not be called concurrently with
/// process_semantic_weight().
class TradeSignalEngine {
public:
    /// Construction-time parameters for the engine.
    struct Config {
        /// Scale factor applied to the directional_bias component.
        double bias_sensitivity{1.0};
        /// Scale factor applied to the volatility_score component.
        double volatility_sensitivity{1.0};
        /// Exponential decay multiplier applied to accumulators on every token.
        double signal_decay_rate{0.95};
        /// Minimum time between consecutive signal emissions in realtime mode.
        std::chrono::microseconds signal_cooldown{std::chrono::microseconds{1000}};
    };

    /// Live statistics updated by the engine.
    struct Stats {
        std::atomic<uint64_t> signals_generated{0};
        std::atomic<uint64_t> signals_suppressed{0};
        std::atomic<double>   avg_signal_strength{0.0};
    };

    /// Construct the engine with the given configuration.
    explicit TradeSignalEngine(const Config& config);

    /// Process a SemanticWeight and potentially emit a TradeSignal.
    ///
    /// The weight is scaled by the configured sensitivities, added to the
    /// decayed accumulators, and — if the cooldown has elapsed — a signal
    /// is emitted via the registered callback.
    ///
    /// # Arguments
    /// * `weight` — Normalised SemanticWeight from LLMAdapter.
    void process_semantic_weight(const SemanticWeight& weight);

    /// Register the callback invoked when a signal is emitted.
    ///
    /// # Arguments
    /// * `callback` — Callable matching TradeSignalCallback; stored by value.
    void set_signal_callback(TradeSignalCallback callback);

    /// Enable or disable realtime mode.
    ///
    /// In realtime mode signals are rate-limited by signal_cooldown.
    /// In backtest mode every token produces a signal.
    ///
    /// # Arguments
    /// * `enabled` — true to enable realtime mode.
    void set_realtime_mode(bool enabled);

    /// Convenience wrapper: set_backtest_mode(true) == set_realtime_mode(false).
    ///
    /// # Arguments
    /// * `enabled` — true to enable backtest (every-token) mode.
    void set_backtest_mode(bool enabled);

    /// Return a const reference to the live statistics struct.
    const Stats& get_stats() const { return stats_; }

    /// Register an OutputSink to receive all emitted signals.
    ///
    /// The sink is called synchronously inside emit_signal() after the
    /// user callback.  Multiple sinks can be added; all receive every signal.
    ///
    /// # Arguments
    /// * `sink` — Shared pointer to an OutputSink implementation.
    void add_output_sink(std::shared_ptr<OutputSink> sink);

    /// Remove all registered output sinks.
    void clear_output_sinks();

private:
    bool should_emit_signal() const;
    void emit_signal(const TradeSignal& signal);

    Config config_;
    TradeSignalCallback callback_;
    std::atomic<double> accumulated_bias_{0.0};
    std::atomic<double> accumulated_volatility_{0.0};
    std::atomic<bool>   realtime_mode_{true};
    /// Last confidence score observed from process_semantic_weight(); used to
    /// populate TradeSignal::confidence on emission.
    std::atomic<double> last_confidence_{0.5};
    std::chrono::high_resolution_clock::time_point last_signal_time_;
    Stats stats_;
    std::vector<std::shared_ptr<OutputSink>> output_sinks_;
};

} // namespace llmquant
