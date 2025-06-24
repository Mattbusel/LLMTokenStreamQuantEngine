#pragma once
#include "LLMAdapter.h"
#include <atomic>
#include <chrono>
#include <functional>

namespace llmquant {

struct TradeSignal {
    double delta_bias_shift;        // Position bias adjustment
    double volatility_adjustment;   // Vol index change
    double strategy_weight;         // Strategy selection weight
    int strategy_toggle;           // Strategy switch signal
    std::chrono::high_resolution_clock::time_point timestamp;
    
    TradeSignal() : delta_bias_shift(0.0), volatility_adjustment(0.0), 
                   strategy_weight(1.0), strategy_toggle(0),
                   timestamp(std::chrono::high_resolution_clock::now()) {}
};

using TradeSignalCallback = std::function<void(const TradeSignal&)>;

class TradeSignalEngine {
public:
    struct Config {
        double bias_sensitivity{1.0};
        double volatility_sensitivity{1.0};
        double signal_decay_rate{0.95};
        std::chrono::microseconds signal_cooldown{1000}; // 1ms
    };

    explicit TradeSignalEngine(const Config& config);
    ~TradeSignalEngine() = default;

    // Core signal processing
    void process_semantic_weight(const SemanticWeight& weight);
    void set_signal_callback(TradeSignalCallback callback);
    
    // Real-time vs backtest modes
    void set_realtime_mode(bool enabled);
    void set_backtest_mode(bool enabled);
    
    // Performance metrics
    struct Stats {
        std::atomic<uint64_t> signals_generated{0};
        std::atomic<uint64_t> signals_suppressed{0};
        std::atomic<double> avg_signal_strength{0.0};
    };
    
    const Stats& get_stats() const { return stats_; }

private:
    void emit_signal(const TradeSignal& signal);
    bool should_emit_signal() const;
    
    Config config_;
    TradeSignalCallback callback_;
    std::atomic<bool> realtime_mode_{true};
    std::chrono::high_resolution_clock::time_point last_signal_time_;
    mutable Stats stats_;
    
    // Signal state
    std::atomic<double> accumulated_bias_{0.0};
    std::atomic<double> accumulated_volatility_{0.0};
};

} // namespace llmquant
