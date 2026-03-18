#include "TradeSignalEngine.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

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
    // Apply sensitivity scaling
    double bias_contribution = weight.directional_bias * weight.confidence_score * config_.bias_sensitivity;
    double vol_contribution = weight.volatility_score * weight.confidence_score * config_.volatility_sensitivity;
    
    // Accumulate signals with decay using CAS loops for linearisable RMW.
    double expected_bias = accumulated_bias_.load(std::memory_order_relaxed);
    double desired_bias;
    do {
        desired_bias = expected_bias * config_.signal_decay_rate + bias_contribution;
    } while (!accumulated_bias_.compare_exchange_weak(
                 expected_bias, desired_bias,
                 std::memory_order_release,
                 std::memory_order_relaxed));

    double expected_vol = accumulated_volatility_.load(std::memory_order_relaxed);
    double desired_vol;
    do {
        desired_vol = expected_vol * config_.signal_decay_rate + vol_contribution;
    } while (!accumulated_volatility_.compare_exchange_weak(
                 expected_vol, desired_vol,
                 std::memory_order_release,
                 std::memory_order_relaxed));

    double current_bias = desired_bias;
    double current_vol  = desired_vol;
    
    // Record latest confidence for use in emitted signals.
    last_confidence_ = weight.confidence_score;

    // Check if we should emit a signal
    if (should_emit_signal()) {
        TradeSignal signal;
        signal.delta_bias_shift = current_bias;
        signal.volatility_adjustment = current_vol;
        
        // Strategy selection logic
        if (std::abs(current_bias) > 0.5) {
            signal.strategy_toggle = (current_bias > 0) ? 1 : -1;
        }
        
        signal.strategy_weight = std::min(1.0, weight.confidence_score * 2.0);
        
        emit_signal(signal);
        
        // Reset accumulators after significant signal using CAS loops so the
        // halving is a linearisable RMW even if another thread is accumulating.
        if (std::abs(current_bias) > 0.8 || std::abs(current_vol) > 0.8) {
            double b = accumulated_bias_.load(std::memory_order_relaxed);
            while (!accumulated_bias_.compare_exchange_weak(
                       b, b * 0.5,
                       std::memory_order_release, std::memory_order_relaxed)) {}
            double v = accumulated_volatility_.load(std::memory_order_relaxed);
            while (!accumulated_volatility_.compare_exchange_weak(
                       v, v * 0.5,
                       std::memory_order_release, std::memory_order_relaxed)) {}
        }
    }
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
    signal.spread_modifier = (std::abs(signal.delta_bias_shift) > 0.5)
                                 ? -0.1 * signal.delta_bias_shift
                                 : 0.0;
    signal.confidence = last_confidence_.load();

    if (callback_) {
        try { callback_(signal); } catch (...) {}
    } else if (output_sinks_.empty()) {
        // Signal has no callback and no sinks — it is fully suppressed.
        stats_.signals_suppressed++;
    }
    // signals_generated counts every emission regardless of routing;
    // signals_suppressed counts only those with no destination at all.

    // Emit to all registered output sinks.
    for (const auto& sink : output_sinks_) {
        try { sink->emit(signal); } catch (...) {}
    }

    // Update stats unconditionally — count every emitted signal regardless of
    // whether a callback or only sinks are registered.
    stats_.signals_generated++;
    uint64_t n = stats_.signals_generated.load();
    double old_avg = stats_.avg_signal_strength.load();
    // Welford running mean: mean_n = mean_{n-1} + (x - mean_{n-1}) / n
    stats_.avg_signal_strength = old_avg + (std::abs(signal.delta_bias_shift) - old_avg) / static_cast<double>(n);
    last_signal_time_ = now;
}

void TradeSignalEngine::add_output_sink(std::shared_ptr<OutputSink> sink) {
    output_sinks_.push_back(std::move(sink));
}

void TradeSignalEngine::clear_output_sinks() {
    output_sinks_.clear();
}

} // namespace llmquant
