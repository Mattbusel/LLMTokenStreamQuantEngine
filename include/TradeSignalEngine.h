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

/**
 * @brief A quantitative trade signal derived from one or more semantic weights.
 *
 * Both a high-resolution chrono timestamp and a nanosecond integer timestamp
 * are provided: the chrono field is used for latency arithmetic inside the
 * engine; the integer field is used for serialisation and cross-process IPC.
 */
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

/**
 * @brief Callback invoked once per emitted TradeSignal on the engine's calling thread.
 */
using TradeSignalCallback = std::function<void(const TradeSignal&)>;

/**
 * @brief Converts a stream of SemanticWeights into TradeSignals.
 *
 * Incoming weights are accumulated with an exponential decay, then a signal
 * is emitted when the cooldown period has elapsed (realtime mode) or on every
 * token (backtest mode).
 *
 * Thread safety: process_semantic_weight() is NOT thread-safe; all calls
 * must arrive from the same thread.  last_signal_time_ is not protected by
 * any lock and must be read and written only from the same thread.
 * get_stats() is always safe (atomic reads).  set_* configuration methods
 * must not be called concurrently with process_semantic_weight().
 */
class TradeSignalEngine {
public:
    /**
     * @brief Construction-time parameters for the engine.
     */
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

    /**
     * @brief Live statistics updated atomically by the engine.
     */
    struct Stats {
        std::atomic<uint64_t> signals_generated{0};
        std::atomic<uint64_t> signals_suppressed{0};
        std::atomic<double>   avg_signal_strength{0.0};

        Stats() = default;

        /// @brief Explicit copy constructor: loads each atomic value individually.
        Stats(const Stats& other)
            : signals_generated{other.signals_generated.load()}
            , signals_suppressed{other.signals_suppressed.load()}
            , avg_signal_strength{other.avg_signal_strength.load()} {}

        /// @brief Explicit copy assignment: stores each atomic value individually.
        Stats& operator=(const Stats& other) {
            if (this != &other) {
                signals_generated.store(other.signals_generated.load());
                signals_suppressed.store(other.signals_suppressed.load());
                avg_signal_strength.store(other.avg_signal_strength.load());
            }
            return *this;
        }
    };

    /**
     * @brief Construct the engine with the given configuration.
     *
     * @param config Scale factors, decay rate, and signal cooldown parameters.
     */
    explicit TradeSignalEngine(const Config& config);

    /**
     * @brief Process a SemanticWeight and potentially emit a TradeSignal.
     *
     * The weight is scaled by the configured sensitivities, added to the
     * decayed accumulators, and — if the cooldown has elapsed — a signal
     * is emitted via the registered callback.
     *
     * @param weight Normalised SemanticWeight produced by LLMAdapter.
     */
    void process_semantic_weight(const SemanticWeight& weight);

    /**
     * @brief Register the callback invoked when a signal is emitted.
     *
     * @param callback Callable matching TradeSignalCallback; stored by value.
     */
    void set_signal_callback(TradeSignalCallback callback);

    /**
     * @brief Enable or disable realtime mode.
     *
     * In realtime mode signals are rate-limited by signal_cooldown.
     * In backtest mode every token produces a signal.
     *
     * @param enabled true to enable realtime mode.
     */
    void set_realtime_mode(bool enabled);

    /**
     * @brief Convenience wrapper: set_backtest_mode(true) == set_realtime_mode(false).
     *
     * @param enabled true to enable backtest (every-token) mode.
     */
    void set_backtest_mode(bool enabled);

    /**
     * @brief Replace the engine configuration at runtime (e.g. on hot-reload).
     *
     * Sensitivity changes take effect on the next call to process_semantic_weight().
     * Must be called from the same thread as process_semantic_weight().
     *
     * @param config New configuration; all fields are replaced atomically from
     *               the caller's perspective (single-threaded use only).
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current engine configuration.
     *
     * @return Copy of the active Config struct.
     */
    Config get_config() const noexcept { return config_; }

    /**
     * @brief Return a copy of the current signal statistics.
     *
     * Thread-safe atomic snapshot.
     *
     * @return Copy of the current Stats struct.
     */
    Stats get_stats() const noexcept { return stats_; }

    /**
     * @brief Returns the current accumulated directional bias (atomic read, instantaneous snapshot).
     *
     * @return Current accumulated bias value.
     */
    double get_accumulated_bias() const noexcept {
        return accumulated_bias_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current accumulated volatility (atomic read, instantaneous snapshot).
     *
     * @return Current accumulated volatility value.
     */
    double get_accumulated_volatility() const noexcept {
        return accumulated_volatility_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Register an OutputSink to receive all emitted signals.
     *
     * The sink is called synchronously inside emit_signal() after the
     * user callback.  Multiple sinks can be added; all receive every signal.
     *
     * @param sink Shared pointer to an OutputSink implementation.
     */
    void add_output_sink(std::shared_ptr<OutputSink> sink);

    /**
     * @brief Remove all registered output sinks.
     */
    void clear_output_sinks();

    /**
     * @brief Reset accumulated bias and volatility to zero and clear statistics.
     *
     * Intended for use between trading sessions. Not thread-safe — must be
     * called from the same thread as process_semantic_weight().
     */
    void reset() noexcept;

    /**
     * @brief Flush all registered output sinks.
     *
     * Call this at shutdown to ensure any buffered file-sink output is written
     * to disk.  Not thread-safe — call from the same thread as emit_signal().
     */
    void flush_sinks();

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
    /// Updated by emit_signal() and read by should_emit_signal().  Both are
    /// called exclusively from the single-threaded process_semantic_weight()
    /// path (see class-level thread-safety note); no synchronisation is needed.
    /// Do NOT access this field from any other thread.
    std::chrono::high_resolution_clock::time_point last_signal_time_;
    /// Set at the start of process_semantic_weight() to allow emit_signal() to
    /// compute the token-to-signal latency for the TradeSignal::latency_us field.
    std::chrono::high_resolution_clock::time_point processing_start_;
    Stats stats_;
    std::vector<std::shared_ptr<OutputSink>> output_sinks_;
};

} // namespace llmquant
