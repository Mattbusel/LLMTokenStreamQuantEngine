#pragma once

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
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

    /**
     * @brief Return a compact human-readable one-liner for logging and debugging.
     *
     * Format: "bias=<val> vol=<val> conf=<val> quality=<val> lat=<val>us strategy=<+1|0|-1>"
     *
     * @return Single-line summary string.
     */
    std::string to_string() const {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "bias=%.4f vol=%.4f spread=%.4f conf=%.4f quality=%.4f"
            " lat=%.2fus strategy=%+d weight=%.4f",
            delta_bias_shift, volatility_adjustment, spread_modifier,
            confidence, signal_quality, latency_us,
            strategy_toggle, strategy_weight);
        return buf;
    }

    /**
     * @brief Return the signal as a compact JSON object string.
     *
     * All floating-point fields use 6 decimal places.  The timestamp is
     * serialised as a 64-bit integer (nanoseconds since Unix epoch).
     *
     * @return Single-line JSON string, no trailing newline.
     */
    std::string to_json() const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"timestamp_ns\":%" PRIu64
            ",\"delta_bias_shift\":%.6f"
            ",\"volatility_adjustment\":%.6f"
            ",\"spread_modifier\":%.6f"
            ",\"confidence\":%.6f"
            ",\"latency_us\":%.6f"
            ",\"strategy_toggle\":%d"
            ",\"strategy_weight\":%.6f"
            ",\"signal_quality\":%.6f}",
            timestamp_ns,
            delta_bias_shift, volatility_adjustment, spread_modifier,
            confidence, latency_us, strategy_toggle,
            strategy_weight, signal_quality);
        return buf;
    }
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
     * @brief One bucket of the signal quality histogram.
     *
     * Each bucket holds a per-bucket (not cumulative) count of signals whose
     * signal_quality falls within (prev_upper_bound, upper_bound].
     * The first bucket covers [0.0, 0.2).
     */
    struct QualityHistogramBucket {
        double   upper_bound{0.0}; ///< Inclusive upper bound of this bucket.
        uint64_t count{0};         ///< Number of signals in this bucket.
    };

    /**
     * @brief Breakdown of suppressed signals by suppression type.
     *
     * Provides fine-grained visibility into why TradeSignalEngine suppressed
     * tokens before they were emitted.  The three named categories are mutually
     * exclusive; @c total is the sum of all three plus any unrouted signals.
     */
    struct SuppressionBreakdown {
        uint64_t noise_filtered{0};         ///< Suppressed by the min_bias_threshold noise gate.
        uint64_t aged_out{0};               ///< Suppressed by the max_signal_age_us staleness guard.
        uint64_t cooldown_suppressed{0};    ///< Skipped because signal_cooldown had not elapsed.
        uint64_t total{0};                  ///< Total suppressed (noise + aged_out + cooldown + unrouted).
    };

    /**
     * @brief Return a breakdown of suppressed signals by suppression type.
     *
     * Reads the relevant atomic counters from Stats and packages them into a
     * SuppressionBreakdown for callers that need per-category visibility.
     * Thread-safe (atomic loads with relaxed ordering).
     *
     * @return SuppressionBreakdown snapshot.
     */
    [[nodiscard]] SuppressionBreakdown get_suppression_breakdown() const noexcept {
        SuppressionBreakdown bd;
        bd.noise_filtered       = stats_.noise_filtered.load(std::memory_order_relaxed);
        bd.aged_out             = stats_.signals_aged_out.load(std::memory_order_relaxed);
        bd.cooldown_suppressed  = stats_.signals_suppressed_cooldown.load(std::memory_order_relaxed);
        bd.total                = stats_.signals_suppressed.load(std::memory_order_relaxed)
                                + bd.aged_out
                                + bd.cooldown_suppressed;
        return bd;
    }

    /**
     * @brief Return the distribution of emitted signal qualities in 5 fixed buckets.
     *
     * Bucket ranges: [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0].
     * Counts are per-bucket (not cumulative).
     *
     * Thread-safe (atomic loads).
     *
     * @return Vector of 5 QualityHistogramBucket entries in ascending bound order.
     */
    [[nodiscard]] std::vector<QualityHistogramBucket> get_quality_histogram() const;

    /**
     * @brief Smoothing factor for the signal quality EMA.
     *
     * 0.1 gives slow-reacting average: recent signals weighted 10%.
     * Increase for faster response to recent quality changes.
     */
    static constexpr double SIGNAL_QUALITY_EMA_ALPHA = 0.1;

    /**
     * @brief Return the exponential moving average of signal_quality.
     *
     * Updated after each emitted signal:
     *   EMA = alpha * quality + (1 - alpha) * prev_EMA
     * where alpha = SIGNAL_QUALITY_EMA_ALPHA (0.1).
     *
     * Returns -1.0 when no signals have been emitted yet.
     * Thread-safe (atomic load).
     *
     * @return EMA in [0.0, 1.0], or -1.0 if no signals emitted.
     */
    [[nodiscard]] double get_signal_quality_ema() const noexcept {
        return stats_.signal_quality_ema.load(std::memory_order_relaxed);
    }

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
        /// Tokens skipped because the signal cooldown had not yet elapsed.
        std::atomic<uint64_t> signals_suppressed_cooldown{0};
        std::atomic<double>   avg_signal_strength{0.0};
        std::atomic<double>   peak_bias{0.0}; ///< Maximum |accumulated_bias| observed since last reset.
        /// Welford running mean of signal_quality across emitted signals.
        std::atomic<double>   avg_signal_quality{0.0};
        /// Per-bucket signal quality histogram counters (5 fixed buckets).
        std::atomic<uint64_t> quality_bucket_0_20{0};   ///< [0.0, 0.2)
        std::atomic<uint64_t> quality_bucket_20_40{0};  ///< [0.2, 0.4)
        std::atomic<uint64_t> quality_bucket_40_60{0};  ///< [0.4, 0.6)
        std::atomic<uint64_t> quality_bucket_60_80{0};  ///< [0.6, 0.8)
        std::atomic<uint64_t> quality_bucket_80_100{0}; ///< [0.8, 1.0]
        /// Exponential moving average of signal_quality (alpha = SIGNAL_QUALITY_EMA_ALPHA).
        /// Initialised to -1.0 so callers can detect "no signals yet".
        std::atomic<double>   signal_quality_ema{-1.0};

        Stats() = default;

        /// @brief Explicit copy constructor: loads each atomic value individually.
        Stats(const Stats& other)
            : signals_generated{other.signals_generated.load()}
            , signals_suppressed{other.signals_suppressed.load()}
            , signals_aged_out{other.signals_aged_out.load()}
            , accumulator_clamped{other.accumulator_clamped.load()}
            , tokens_processed{other.tokens_processed.load()}
            , noise_filtered{other.noise_filtered.load()}
            , signals_suppressed_cooldown{other.signals_suppressed_cooldown.load()}
            , avg_signal_strength{other.avg_signal_strength.load()}
            , peak_bias{other.peak_bias.load()}
            , avg_signal_quality{other.avg_signal_quality.load()}
            , quality_bucket_0_20{other.quality_bucket_0_20.load()}
            , quality_bucket_20_40{other.quality_bucket_20_40.load()}
            , quality_bucket_40_60{other.quality_bucket_40_60.load()}
            , quality_bucket_60_80{other.quality_bucket_60_80.load()}
            , quality_bucket_80_100{other.quality_bucket_80_100.load()}
            , signal_quality_ema{other.signal_quality_ema.load()} {}

        /// @brief Explicit copy assignment: stores each atomic value individually.
        Stats& operator=(const Stats& other) {
            if (this != &other) {
                signals_generated.store(other.signals_generated.load());
                signals_suppressed.store(other.signals_suppressed.load());
                signals_aged_out.store(other.signals_aged_out.load());
                accumulator_clamped.store(other.accumulator_clamped.load());
                tokens_processed.store(other.tokens_processed.load());
                noise_filtered.store(other.noise_filtered.load());
                signals_suppressed_cooldown.store(other.signals_suppressed_cooldown.load());
                avg_signal_strength.store(other.avg_signal_strength.load());
                peak_bias.store(other.peak_bias.load());
                avg_signal_quality.store(other.avg_signal_quality.load());
                quality_bucket_0_20.store(other.quality_bucket_0_20.load());
                quality_bucket_20_40.store(other.quality_bucket_20_40.load());
                quality_bucket_40_60.store(other.quality_bucket_40_60.load());
                quality_bucket_60_80.store(other.quality_bucket_60_80.load());
                quality_bucket_80_100.store(other.quality_bucket_80_100.load());
                signal_quality_ema.store(other.signal_quality_ema.load());
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
     * @brief Update only the noise-gate threshold without replacing the whole config.
     *
     * Signals whose |accumulated_bias| falls below this threshold are suppressed
     * and counted in stats_.noise_filtered.  Set to 0.0 to disable the noise gate.
     * Must be called from the same thread as process_semantic_weight().
     *
     * @param threshold New minimum absolute bias required to emit a signal.
     *                  Negative values are clamped to 0.0.
     */
    void set_min_bias_threshold(double threshold);

    /**
     * @brief Return a copy of the current engine configuration.
     *
     * @return Copy of the active Config struct.
     */
    [[nodiscard]] Config get_config() const noexcept { return config_; }

    /**
     * @brief Return a copy of the current signal statistics.
     *
     * Thread-safe atomic snapshot.
     *
     * @return Copy of the current Stats struct.
     */
    [[nodiscard]] Stats get_stats() const noexcept { return stats_; }

    /**
     * @brief Return the time elapsed since the last emitted signal, in microseconds.
     *
     * Returns 0.0 if no signal has been emitted yet.
     * Thread-safe (atomic read of nanosecond timestamp).
     *
     * @return Elapsed microseconds since the last emit_signal() call.
     */
    [[nodiscard]] double get_signal_age_us() const noexcept {
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
    [[nodiscard]] double get_accumulated_bias() const noexcept {
        return accumulated_bias_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return true if the accumulated bias exceeds the noise gate threshold.
     *
     * Equivalent to checking |accumulated_bias| >= config.min_bias_threshold.
     * If min_bias_threshold is 0 (disabled), returns true whenever bias != 0.
     * Thread-safe (atomic reads).
     */
    [[nodiscard]] bool has_pending_bias() const noexcept {
        double bias = accumulated_bias_.load(std::memory_order_relaxed);
        if (config_.min_bias_threshold > 0.0)
            return std::fabs(bias) >= config_.min_bias_threshold;
        return bias != 0.0;
    }

    /**
     * @brief Return the directional sign of the current accumulated bias.
     *
     * @return +1 if accumulated bias > 0 (bullish), -1 if < 0 (bearish), 0 if zero.
     */
    [[nodiscard]] int get_bias_direction() const noexcept {
        double b = accumulated_bias_.load(std::memory_order_relaxed);
        if (b > 0.0) return  1;
        if (b < 0.0) return -1;
        return 0;
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
    [[nodiscard]] double suppression_rate() const noexcept {
        uint64_t gen  = stats_.signals_generated.load(std::memory_order_relaxed);
        uint64_t supp = stats_.signals_suppressed.load(std::memory_order_relaxed);
        uint64_t total = gen + supp;
        return (total == 0) ? 0.0 : static_cast<double>(supp) / static_cast<double>(total);
    }

    /**
     * @brief Return the total number of signals emitted since construction or last reset().
     *
     * Thread-safe (atomic read with relaxed ordering).
     *
     * @return Total emitted signal count.
     */
    [[nodiscard]] uint64_t get_signals_generated() const noexcept {
        return stats_.signals_generated.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the total number of signals suppressed since construction or last reset().
     *
     * Thread-safe (atomic read with relaxed ordering).
     *
     * @return Total suppressed signal count.
     */
    [[nodiscard]] uint64_t get_signals_suppressed() const noexcept {
        return stats_.signals_suppressed.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the total number of process_semantic_weight() calls since construction or last reset().
     *
     * Thread-safe (atomic read with relaxed ordering).
     *
     * @return Total tokens processed count.
     */
    [[nodiscard]] uint64_t get_tokens_processed() const noexcept {
        return stats_.tokens_processed.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the total number of signals discarded by the staleness guard since
     *        construction or last reset().
     *
     * Thread-safe (atomic read with relaxed ordering).
     *
     * @return Total aged-out signal count.
     */
    [[nodiscard]] uint64_t get_signals_aged_out() const noexcept {
        return stats_.signals_aged_out.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the fraction of processed tokens where the signal was aged out.
     *
     * Computed as signals_aged_out / tokens_processed.
     * Returns 0.0 if no tokens have been processed yet.
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Aged-out rate in [0.0, 1.0].
     */
    [[nodiscard]] double get_aged_out_rate() const noexcept {
        uint64_t tokens = stats_.tokens_processed.load(std::memory_order_relaxed);
        if (tokens == 0) return 0.0;
        return static_cast<double>(stats_.signals_aged_out.load(std::memory_order_relaxed))
             / static_cast<double>(tokens);
    }

    /**
     * @brief Return the running average signal_quality across all emitted signals.
     *
     * Computed as a Welford running mean updated in emit_signal().
     * Returns 0.0 if no signals have been emitted yet.
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_avg_signal_quality() const noexcept {
        return stats_.avg_signal_quality.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current accumulated volatility (atomic read, instantaneous snapshot).
     *
     * @return Current accumulated volatility value.
     */
    [[nodiscard]] double get_accumulated_volatility() const noexcept {
        return accumulated_volatility_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Compute the ratio of generated signals to tokens processed.
     *
     * Returns signals_generated / tokens_processed, or 0.0 if no tokens
     * have been processed. Useful for measuring how often the noise gate
     * allows signal emission.
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Signal efficiency in [0.0, 1.0].
     */
    [[nodiscard]] double get_signal_efficiency() const noexcept {
        uint64_t tokens = stats_.tokens_processed.load(std::memory_order_relaxed);
        if (tokens == 0) return 0.0;
        return static_cast<double>(stats_.signals_generated.load(std::memory_order_relaxed))
             / static_cast<double>(tokens);
    }

    /**
     * @brief Return the average accumulated bias per token processed.
     *
     * Computed as |accumulated_bias| / tokens_processed.
     * Returns 0.0 if no tokens have been processed yet.
     * Thread-safe (reads atomics with relaxed ordering).
     *
     * @return Average absolute bias contribution per token.
     */
    [[nodiscard]] double get_avg_bias_per_token() const noexcept {
        uint64_t tokens = stats_.tokens_processed.load(std::memory_order_relaxed);
        if (tokens == 0) return 0.0;
        return std::fabs(accumulated_bias_.load(std::memory_order_relaxed))
             / static_cast<double>(tokens);
    }

    /**
     * @brief Return the signal_quality field of the most recently emitted signal.
     *
     * Returns 0.0 if no signal has been emitted yet.
     * Thread-safe (atomic read).
     *
     * @return signal_quality of the last emitted signal in [0.0, 1.0].
     */
    [[nodiscard]] double get_last_signal_quality() const noexcept {
        return last_signal_quality_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return signals emitted per second since last reset() or construction.
     *
     * Computed as signals_generated / elapsed_seconds since reset_time_.
     * Returns 0.0 if fewer than 1 ns has elapsed (avoids divide-by-near-zero).
     *
     * Thread-safe (reads atomic counter and construction time).
     *
     * @return Signal velocity in signals/second.
     */
    [[nodiscard]] double get_signal_velocity() const noexcept {
        uint64_t generated = stats_.signals_generated.load(std::memory_order_relaxed);
        if (generated == 0) return 0.0;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - reset_time_).count());
        if (elapsed_ns < 1.0) return 0.0;
        return static_cast<double>(generated) / (elapsed_ns / 1e9);
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
    [[nodiscard]] double get_tokens_per_second() const noexcept;

    /**
     * @brief Return the fraction of processed tokens rejected by the noise gate.
     *
     * Computed as noise_filtered / tokens_processed.  Returns 0.0 when no
     * tokens have been processed.  A high rate indicates the min_bias_threshold
     * is filtering most incoming signal energy.
     *
     * Thread-safe (atomic reads).
     *
     * @return Noise filter rate in [0.0, 1.0].
     */
    [[nodiscard]] double get_noise_filter_rate() const noexcept;

    /**
     * @brief Return the peak absolute accumulated bias observed since construction or last reset().
     *
     * Thread-safe (atomic read).
     *
     * @return Maximum |accumulated_bias| observed since last reset, in signal units.
     */
    [[nodiscard]] double get_peak_bias() const noexcept {
        return stats_.peak_bias.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the fraction of processed tokens where the accumulator was clamped.
     *
     * Computed as accumulator_clamped / tokens_processed.  Returns 0.0 when no
     * tokens have been processed or max_accumulated_bias is disabled (0.0).
     * A high rate indicates the accumulator cap is frequently hit.
     *
     * Thread-safe (atomic reads).
     *
     * @return Accumulator clamp rate in [0.0, 1.0].
     */
    [[nodiscard]] double get_accumulator_clamp_rate() const noexcept {
        uint64_t total   = stats_.tokens_processed.load(std::memory_order_relaxed);
        uint64_t clamped = stats_.accumulator_clamped.load(std::memory_order_relaxed);
        return (total == 0) ? 0.0 : static_cast<double>(clamped) / static_cast<double>(total);
    }

    /**
     * @brief Return elapsed milliseconds since construction or the last reset() call.
     *
     * Useful for monitoring session uptime and computing throughput metrics.
     * Thread-safe (reads reset_time_ which is only written on construction and reset()).
     *
     * @return Elapsed session duration in milliseconds.
     */
    [[nodiscard]] double get_session_duration_ms() const noexcept;

    /**
     * @brief Return the time elapsed since the last signal emission in microseconds.
     *
     * Returns the full session duration if no signal has been emitted yet
     * (i.e., last_signal_time_ is at the epoch).
     *
     * Not thread-safe: call from the same thread as process_semantic_weight().
     *
     * @return Microseconds since last signal emission.
     */
    [[nodiscard]] double get_time_since_last_signal_us() const noexcept;

    /**
     * @brief Return true if the engine is currently within the cooldown window.
     *
     * Equivalent to get_time_since_last_signal_us() < signal_cooldown.count().
     *
     * Not thread-safe: call from the same thread as process_semantic_weight().
     *
     * @return true if a new signal would be rate-limited by the cooldown.
     */
    [[nodiscard]] bool is_in_cooldown() const noexcept;

    /**
     * @brief Flush all registered output sinks.
     *
     * Call this at shutdown to ensure any buffered file-sink output is written
     * to disk.  Not thread-safe — call from the same thread as emit_signal().
     */
    void flush_sinks();

    /**
     * @brief Return a single-line human-readable summary of engine statistics.
     *
     * Format: "tokens=<n> generated=<n> suppressed=<n> aged_out=<n>
     *          efficiency=<rate> suppression_rate=<rate> bias=<val> vol=<val>"
     * If no tokens have been processed, returns "tokens=0 (no data)".
     *
     * Thread-safe (reads atomic counters and accumulator).
     *
     * @return Single-line stats summary string.
     */
    [[nodiscard]] std::string format_stats() const;

    /**
     * @brief Serialise the current engine statistics to a JSON string.
     *
     * Produces a JSON object with counters for signals generated/suppressed/
     * aged-out/noise-filtered, quality metrics (avg, peak_bias, EMA), and the
     * five signal-quality histogram buckets.
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return JSON object as std::string.
     */
    [[nodiscard]] std::string to_stats_json() const noexcept {
        const Stats s = get_stats();
        char buf[768];
        std::snprintf(buf, sizeof(buf),
            "{\"signals_generated\":%" PRIu64
            ",\"signals_suppressed\":%" PRIu64
            ",\"signals_aged_out\":%" PRIu64
            ",\"noise_filtered\":%" PRIu64
            ",\"tokens_processed\":%" PRIu64
            ",\"accumulator_clamped\":%" PRIu64
            ",\"avg_signal_strength\":%.6f"
            ",\"avg_signal_quality\":%.6f"
            ",\"peak_bias\":%.6f"
            ",\"signal_quality_ema\":%.6f"
            ",\"quality_hist\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]}",
            s.signals_generated.load(std::memory_order_relaxed),
            s.signals_suppressed.load(std::memory_order_relaxed),
            s.signals_aged_out.load(std::memory_order_relaxed),
            s.noise_filtered.load(std::memory_order_relaxed),
            s.tokens_processed.load(std::memory_order_relaxed),
            s.accumulator_clamped.load(std::memory_order_relaxed),
            s.avg_signal_strength.load(std::memory_order_relaxed),
            s.avg_signal_quality.load(std::memory_order_relaxed),
            s.peak_bias.load(std::memory_order_relaxed),
            s.signal_quality_ema.load(std::memory_order_relaxed),
            s.quality_bucket_0_20.load(std::memory_order_relaxed),
            s.quality_bucket_20_40.load(std::memory_order_relaxed),
            s.quality_bucket_40_60.load(std::memory_order_relaxed),
            s.quality_bucket_60_80.load(std::memory_order_relaxed),
            s.quality_bucket_80_100.load(std::memory_order_relaxed));
        return buf;
    }

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
    [[nodiscard]] Snapshot snapshot() const noexcept;

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
