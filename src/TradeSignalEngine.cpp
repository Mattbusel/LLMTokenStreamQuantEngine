#include "TradeSignalEngine.h"
#include <algorithm>
#include <cmath>

namespace llmquant {

TradeSignalEngine::TradeSignalEngine(const Config& config) 
    : config_(config), last_signal_time_(std::chrono::high_resolution_clock::now()) {}

void TradeSignalEngine::process_semantic_weight(const SemanticWeight& weight) {
    // Apply sensitivity scaling
    double bias_contribution = weight.directional_bias * weight.confidence_score * config_.bias_sensitivity;
    double vol_contribution = weight.volatility_score * weight.confidence_score * config_.volatility_sensitivity;
    
    // Accumulate signals with decay
    double current_bias = accumulated_bias_.load();
    double current_vol = accumulated_volatility_.load();
    
    // Apply decay
    current_bias *= config_.signal_decay_rate;
    current_vol *= config_.signal_decay_rate;
    
    // Add new contribution
    current_bias += bias_contribution;
    current_vol += vol_contribution;
    
    accumulated_bias_ = current_bias;
    accumulated_volatility_ = current_vol;
    
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
        
        // Reset accumulators after significant signal
        if (std::abs(current_bias) > 0.8 || std::abs(current_vol) > 0.8) {
            accumulated_bias_ = current_bias * 0.5;
            accumulated_volatility_ = current_vol * 0.5;
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

void TradeSignalEngine::emit_signal(const TradeSignal& signal) {
    if (callback_) {
        callback_(signal);
        stats_.signals_generated++;
        stats_.avg_signal_strength = (stats_.avg_signal_strength.load() + 
                                    std::abs(signal.delta_bias_shift)) / 2.0;
        last_signal_time_ = std::chrono::high_resolution_clock::now();
    } else {
        stats_.signals_suppressed++;
    }
}

} // namespace llmquant
