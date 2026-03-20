#include "TradeSignalEngine.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace llmquant {

TradeSignalEngine::TradeSignalEngine(const Config& config)
    : config_(config)
    // Initialise last_signal_time_ to the epoch so the very first token always
    // passes the cooldown check regardless of the configured cooldown duration.
    , last_signal_time_(std::chrono::high_resolution_clock::time_point{}) {
    if (config_.bias_sensitivity <= 0.0)
        throw std::invalid_argument("TradeSignalEngine: bias_sensitivity must be > 0");
    if (config_.volatility_sensitivity <= 0.0)
        throw std::invalid_argument("TradeSignalEngine: volatility_sensitivity must be > 0");
    if (config_.signal_decay_rate <= 0.0 || config_.signal_decay_rate > 1.0)
        throw std::invalid_argument("TradeSignalEngine: signal_decay_rate must be in (0, 1]");
}

void TradeSignalEngine::process_semantic_weight(const SemanticWeight& weight) {
    // Record start time for latency_us population in emit_signal().
    processing_start_ = std::chrono::high_resolution_clock::now();

    // Apply sensitivity scaling
    double bias_contribution = weight.directional_bias * weight.confidence_score * config_.bias_sensitivity;
    double vol_contribution = weight.volatility_score * weight.confidence_score * config_.volatility_sensitivity;
    
    // Accumulate signals with decay using CAS loops for linearisable RMW.
    // Yield after 8 failed retries to avoid livelock under high contention.
    double expected_bias = accumulated_bias_.load(std::memory_order_relaxed);
    double desired_bias;
    {
        int retries = 0;
        do {
            desired_bias = expected_bias * config_.signal_decay_rate + bias_contribution;
            if (config_.max_accumulated_bias > 0.0)
                desired_bias = std::clamp(desired_bias,
                                          -config_.max_accumulated_bias,
                                           config_.max_accumulated_bias);
            if (++retries > 8) { std::this_thread::yield(); retries = 0; }
        } while (!accumulated_bias_.compare_exchange_weak(
                     expected_bias, desired_bias,
                     std::memory_order_release,
                     std::memory_order_relaxed));
    }

    double expected_vol = accumulated_volatility_.load(std::memory_order_relaxed);
    double desired_vol;
    {
        int retries = 0;
        do {
            desired_vol = expected_vol * config_.signal_decay_rate + vol_contribution;
            if (++retries > 8) { std::this_thread::yield(); retries = 0; }
        } while (!accumulated_volatility_.compare_exchange_weak(
                     expected_vol, desired_vol,
                     std::memory_order_release,
                     std::memory_order_relaxed));
    }

    double current_bias = desired_bias;
    double current_vol  = desired_vol;
    
    // Record latest confidence for use in emitted signals.
    last_confidence_ = weight.confidence_score;

    // Noise filter: suppress signals too weak to act on.
    if (config_.min_bias_threshold > 0.0 &&
        std::fabs(current_bias) < config_.min_bias_threshold) {
        if (stats_.signals_suppressed.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max() - 1)
            stats_.signals_suppressed.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Check if we should emit a signal
    if (should_emit_signal()) {
        TradeSignal signal;
        signal.delta_bias_shift = current_bias;
        signal.volatility_adjustment = current_vol;
        
        // Strategy selection logic
        if (std::fabs(current_bias) > 0.5) {
            signal.strategy_toggle = (current_bias > 0) ? 1 : -1;
        }
        
        signal.strategy_weight = std::min(1.0, weight.confidence_score * 2.0);
        // Set confidence on the signal before emit_signal() so the field is
        // correctly populated without relying on an overwrite inside emit_signal().
        signal.confidence = weight.confidence_score;

        emit_signal(signal);
        
        // Reset accumulators after significant signal using CAS loops so the
        // halving is a linearisable RMW even if another thread is accumulating.
        if (std::fabs(current_bias) > 0.8 || std::fabs(current_vol) > 0.8) {
            double b = accumulated_bias_.load(std::memory_order_relaxed);
            {
                int retries = 0;
                while (!accumulated_bias_.compare_exchange_weak(
                           b, b * 0.5,
                           std::memory_order_release, std::memory_order_relaxed)) {
                    if (++retries > 8) { std::this_thread::yield(); retries = 0; }
                }
            }
            double v = accumulated_volatility_.load(std::memory_order_relaxed);
            {
                int retries = 0;
                while (!accumulated_volatility_.compare_exchange_weak(
                           v, v * 0.5,
                           std::memory_order_release, std::memory_order_relaxed)) {
                    if (++retries > 8) { std::this_thread::yield(); retries = 0; }
                }
            }
        }
    }
}

void TradeSignalEngine::update_config(const Config& config) {
    if (config.bias_sensitivity <= 0.0)
        throw std::invalid_argument("TradeSignalEngine: bias_sensitivity must be > 0");
    if (config.volatility_sensitivity <= 0.0)
        throw std::invalid_argument("TradeSignalEngine: volatility_sensitivity must be > 0");
    if (config.signal_decay_rate <= 0.0 || config.signal_decay_rate > 1.0)
        throw std::invalid_argument("TradeSignalEngine: signal_decay_rate must be in (0, 1]");
    config_ = config;
}

void TradeSignalEngine::set_signal_callback(TradeSignalCallback callback) {
    callback_ = std::move(callback);
}

void TradeSignalEngine::set_realtime_mode(bool enabled) {
    realtime_mode_ = enabled;
}

void TradeSignalEngine::set_backtest_mode(bool enabled) {
    realtime_mode_ = !enabled;
}

bool TradeSignalEngine::should_emit_signal() const {
    if (!realtime_mode_.load()) return true; // Always emit in backtest mode
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_signal_time_);
    
    return elapsed >= config_.signal_cooldown;
}

void TradeSignalEngine::emit_signal(const TradeSignal& signal_in) {
    TradeSignal signal = signal_in;
    auto now = std::chrono::high_resolution_clock::now();
    signal.timestamp    = now;
    signal.timestamp_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
    // Populate latency_us: time from start of process_semantic_weight() to now.
    signal.latency_us = std::chrono::duration<double, std::micro>(
        now - processing_start_).count();

    // Staleness guard: suppress signals that took too long to process.
    // When max_signal_age_us > 0 and latency_us exceeds the threshold the
    // signal is discarded and counted in signals_aged_out rather than emitted.
    if (config_.max_signal_age_us > 0.0 && signal.latency_us > config_.max_signal_age_us) {
        if (stats_.signals_aged_out.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max() - 1)
            stats_.signals_aged_out.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    signal.spread_modifier = (std::fabs(signal.delta_bias_shift) > 0.5)
                                 ? -0.1 * signal.delta_bias_shift
                                 : 0.0;
    // Do NOT overwrite signal.confidence here — it must be set by the caller
    // (process_semantic_weight) before calling emit_signal().  Overwriting it
    // would silently discard any per-signal confidence already populated.
    // signal.confidence = last_confidence_.load();  // removed (improvement #14)

    if (callback_) {
        try { callback_(signal); } catch (...) {}
    } else if (output_sinks_.empty()) {
        // Signal has no callback and no sinks — it is fully suppressed.
        // Saturating increment — wraps at UINT64_MAX then holds
        if (stats_.signals_suppressed.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max() - 1)
            stats_.signals_suppressed.fetch_add(1, std::memory_order_relaxed);
    }
    // signals_generated counts every emission regardless of routing;
    // signals_suppressed counts only those with no destination at all.

    // Emit to all registered output sinks.
    for (const auto& sink : output_sinks_) {
        try { sink->emit(signal); } catch (...) {}
    }

    // Update stats unconditionally — count every emitted signal regardless of
    // whether a callback or only sinks are registered.
    // signals_generated is incremented BEFORE loading n so that n >= 1 always
    // holds when the Welford mean update runs below.  The guard against n==0
    // defends the rare case where the saturating check skips the increment.
    // Saturating increment — wraps at UINT64_MAX then holds
    if (stats_.signals_generated.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max() - 1)
        stats_.signals_generated.fetch_add(1, std::memory_order_relaxed);
    uint64_t n = stats_.signals_generated.load();
    if (n == 0) n = 1; // guard: n must be >= 1 for the Welford denominator
    double old_avg = stats_.avg_signal_strength.load();
    // Welford running mean: mean_n = mean_{n-1} + (x - mean_{n-1}) / n
    stats_.avg_signal_strength = old_avg + (std::fabs(signal.delta_bias_shift) - old_avg) / static_cast<double>(n);
    last_signal_time_ = now;
}

void TradeSignalEngine::reset() noexcept {
    accumulated_bias_.store(0.0, std::memory_order_relaxed);
    accumulated_volatility_.store(0.0, std::memory_order_relaxed);
    last_confidence_.store(0.5, std::memory_order_relaxed);
    last_signal_time_ = std::chrono::high_resolution_clock::time_point{};
    stats_.signals_generated.store(0, std::memory_order_relaxed);
    stats_.signals_suppressed.store(0, std::memory_order_relaxed);
    stats_.signals_aged_out.store(0, std::memory_order_relaxed);
    stats_.avg_signal_strength.store(0.0, std::memory_order_relaxed);
}

void TradeSignalEngine::add_output_sink(std::shared_ptr<OutputSink> sink) {
    output_sinks_.push_back(std::move(sink));
}

void TradeSignalEngine::clear_output_sinks() {
    output_sinks_.clear();
}

void TradeSignalEngine::flush_sinks() {
    for (const auto& sink : output_sinks_) {
        try { sink->flush(); } catch (...) {}
    }
}

} // namespace llmquant
