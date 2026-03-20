#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
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

    /// Composite quality score in [0.0, 1.0].
    /// Computed as: confidence * clamp((|delta_bias_shift| + |volatility_adjustment|) / 2, 0, 1).
    /// Higher = more confident and larger magnitude signal.
    double signal_quality{0.0};
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
        /// Maximum allowed token-to-signal processing time in microseconds.
        /// Signals with latency_us > this value are suppressed as stale.
        /// Set to 0.0 to disable the guard (default).
        double max_signal_age_us{0.0};
        /// Minimum absolute bias required to emit a signal.
        /// Signals with |delta_bias_shift| < this value are treated as noise
        /// and suppressed (counted in signals_suppressed).
        /// Set to 0.0 to disable (default).
        double min_bias_threshold{0.0};
        /// Maximum absolute value of the accumulated bias.
        /// The accumulator is clamped to [-max_accumulated_bias, +max_accumulated_bias]
        /// after every token update to prevent runaway compounding.
        /// Set to 0.0 to disable (default).
        double max_accumulated_bias{0.0};
    };

    /**
     * @brief Live statistics updated atomically by the engine.
     */
    struct Stats {
        std::atomic<uint64_t> signals_generated{0};
        std::atomic<uint64_t> signals_suppressed{0};
        std::atomic<uint64_t> signals_aged_out{0};    ///< Signals suppressed by staleness guard.
        std::atomic<uint64_t> accumulator_clamped{0}; ///< Times max_accumulated_bias cap was applied.
        std::atomic<uint64_t> tokens_processed{0};    ///< Total process_semantic_weight() calls since last reset.
        /// Tokens rejected by the noise gate (|accumulated_bias| < min_bias_threshold).
        /// Unlike signals_suppressed, this counter is NOT incremented for signals
        /// that were emitted but had no registered callback or sink.
        std::atomic<uint64_t> noise_filtered{0};
        std::atomic<double>   avg_signal_strength{0.0};
        std::atomic<double>   peak_bias{0.0}; ///< Maximum |accumulated_bias| observed since last reset.

        Stats() = default;

        /// @brief Explicit copy constructor: loads each atomic value individually.
        Stats(const Stats& other)
            : signals_generated{other.signals_generated.load()}
            , signals_suppressed{other.signals_suppressed.load()}
            , signals_aged_out{other.signals_aged_out.load()}
            , accumulator_clamped{other.accumulator_clamped.load()}
            , tokens_processed{other.tokens_processed.load()}
            , noise_filtered{other.noise_filtered.load()}
            , avg_signal_strength{other.avg_signal_strength.load()}
            , peak_bias{other.peak_bias.load()} {}

        /// @brief Explicit copy assignment: stores each atomic value individually.
        Stats& operator=(const Stats& other) {
            if (this != &other) {
                signals_generated.store(other.signals_generated.load());
                signals_suppressed.store(other.signals_suppressed.load());
                signals_aged_out.store(other.signals_aged_out.load());
                accumulator_clamped.store(other.accumulator_clamped.load());
                tokens_processed.store(other.tokens_processed.load());
                noise_filtered.store(other.noise_filtered.load());
                avg_signal_strength.store(other.avg_signal_strength.load());
                peak_bias.store(other.peak_bias.load());
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
     * @brief Process a batch of SemanticWeights in sequence.
     *
     * Equivalent to calling process_semantic_weight() for each element in
     * order.  Useful in backtest mode where an entire token sequence is
     * available ahead of time.
     *
     * @param weights Ordered sequence of SemanticWeights to process.
     */
    void process_batch(const std::vector<SemanticWeight>& weights);

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
     * @brief Update only the signal cooldown period without replacing the whole config.
     *
     * Equivalent to reading get_config(), setting signal_cooldown, and calling
     * update_config(), but avoids re-validating unrelated fields.
     * Must be called from the same thread as process_semantic_weight().
     *
     * @param cooldown New minimum time between consecutive signal emissions.
     */
    void set_signal_cooldown(std::chrono::microseconds cooldown);

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
     * @brief Return the time elapsed since the last emitted signal, in microseconds.
     *
     * Returns 0.0 if no signal has been emitted yet.
     * Thread-safe (atomic read of nanosecond timestamp).
     *
     * @return Elapsed microseconds since the last emit_signal() call.
     */
    double get_signal_age_us() const noexcept {
        uint64_t ts = last_signal_timestamp_ns_.load(std::memory_order_relaxed);
        if (ts == 0) return 0.0;
        auto now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        return (now_ns > ts) ? static_cast<double>(now_ns - ts) / 1000.0 : 0.0;
    }

    /**
     * @brief Returns the current accumulated directional bias (atomic read, instantaneous snapshot).
     *
     * @return Current accumulated bias value.
     */
    double get_accumulated_bias() const noexcept {
        return accumulated_bias_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Compute the fraction of weight-processing calls that were suppressed.
     *
     * Returns signals_suppressed / (signals_generated + signals_suppressed), or
     * 0.0 if no tokens have been processed yet.  A high suppression rate
     * (approaching 1.0) indicates the min_bias_threshold is filtering most of
     * the incoming signal energy.
     *
     * Thread-safe (atomic reads).
     *
     * @return Suppression rate in [0.0, 1.0].
     */
    double suppression_rate() const noexcept {
        uint64_t gen  = stats_.signals_generated.load(std::memory_order_relaxed);
        uint64_t supp = stats_.signals_suppressed.load(std::memory_order_relaxed);
        uint64_t total = gen + supp;
        return (total == 0) ? 0.0 : static_cast<double>(supp) / static_cast<double>(total);
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
     * @brief Return the signal_quality field of the most recently emitted signal.
     *
     * Returns 0.0 if no signal has been emitted yet.
     * Thread-safe (atomic read).
     *
     * @return signal_quality of the last emitted signal in [0.0, 1.0].
     */
    double get_last_signal_quality() const noexcept {
        return last_signal_quality_.load(std::memory_order_relaxed);
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
     * @brief Predicate type used to filter signals before forwarding to a sink.
     *
     * Return true to forward the signal to the associated sink; false to skip it.
     */
    using SinkPredicate = std::function<bool(const TradeSignal&)>;

    /**
     * @brief Register an OutputSink that only receives signals matching the predicate.
     *
     * Useful for routing high-confidence signals to a separate sink, or for
     * separating bullish and bearish signals into different output files.
     *
     * @param sink      Shared pointer to an OutputSink implementation.
     * @param predicate Callable that returns true for signals to forward.
     */
    void add_sink_with_filter(std::shared_ptr<OutputSink> sink, SinkPredicate predicate);

    /**
     * @brief Remove all registered output sinks (both unfiltered and filtered).
     */
    void clear_output_sinks();

    /**
     * @brief Emit a final signal from the current accumulators then reset.
     *
     * Useful at graceful shutdown or session boundary: if the absolute
     * accumulated bias exceeds min_bias_threshold (or is non-zero when the
     * threshold is disabled), emits one last signal before clearing state.
     * If the engine is in realtime mode, the cooldown is bypassed so the
     * drain signal is always emitted regardless of the last emission time.
     *
     * Not thread-safe — must be called from the same thread as
     * process_semantic_weight().
     */
    void drain_pending();

    /**
     * @brief Reset accumulated bias and volatility to zero and clear statistics.
     *
     * Intended for use between trading sessions. Not thread-safe — must be
     * called from the same thread as process_semantic_weight().
     */
    void reset() noexcept;

    /**
     * @brief Return the average token processing throughput since construction or last reset().
     *
     * Computed as tokens_processed / elapsed_seconds.  Returns 0.0 if no
     * tokens have been processed or if less than 1 microsecond has elapsed
     * (to avoid division by near-zero).
     *
     * Not thread-safe for the elapsed-time computation; call from the same
     * thread as process_semantic_weight().
     *
     * @return Tokens processed per second.
     */
    double get_tokens_per_second() const noexcept;

    /**
     * @brief Flush all registered output sinks.
     *
     * Call this at shutdown to ensure any buffered file-sink output is written
     * to disk.  Not thread-safe — call from the same thread as emit_signal().
     */
    void flush_sinks();

    /**
     * @brief Immutable state snapshot for dashboards and health checks.
     */
    struct Snapshot {
        Config   config;
        Stats    stats;
        double   accumulated_bias{0.0};
        double   accumulated_volatility{0.0};
        double   last_signal_quality{0.0};
        double   signal_age_us{0.0};
        double   suppression_rate_val{0.0};
        double   tokens_per_second{0.0};
        bool     realtime_mode{true};
    };

    /**
     * @brief Capture all current engine state in a single consistent call.
     *
     * Reads each field via their existing thread-safe accessors.  Suitable
     * for dashboards and health-check endpoints.
     *
     * Thread-safe (atomic reads only).
     *
     * @return Snapshot of all current TradeSignalEngine state.
     */
    Snapshot snapshot() const noexcept;

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
    /// Set at construction and reset() to support get_tokens_per_second().
    std::chrono::high_resolution_clock::time_point reset_time_{
        std::chrono::high_resolution_clock::now()};
    Stats stats_;
    /// signal_quality of the last emitted signal; updated by emit_signal().
    std::atomic<double>   last_signal_quality_{0.0};
    /// Nanosecond timestamp (since epoch) of the most recent emit_signal() call; 0 = never.
    std::atomic<uint64_t> last_signal_timestamp_ns_{0};
    std::vector<std::shared_ptr<OutputSink>> output_sinks_;
    std::vector<std::pair<std::shared_ptr<OutputSink>, SinkPredicate>> filtered_sinks_;
};

} // namespace llmquant
