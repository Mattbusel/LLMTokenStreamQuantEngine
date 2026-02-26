#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include "OutputSink.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

using namespace llmquant;

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nShutting down gracefully..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    
    // Load configuration
    Config config;
    std::string config_file = (argc > 1) ? argv[1] : "config.yaml";
    if (!config.load_from_file(config_file)) {
        std::cout << "Using default configuration" << std::endl;
    }

    config.start_watching(config_file, [](const llmquant::SystemConfig& updated) {
        std::cout << "\n[config] Hot-reloaded: bias_sensitivity="
                  << updated.trading.bias_sensitivity << std::endl;
    });

    const auto& sys_config = config.get_config();
    
    // Initialize components
    MetricsLogger logger({
        .log_file_path = sys_config.logging.log_file_path,
        .format = sys_config.logging.format == "CSV" ? 
                 MetricsLogger::OutputFormat::CSV : MetricsLogger::OutputFormat::JSON,
        .enable_console_output = sys_config.logging.enable_console,
        .flush_interval = std::chrono::milliseconds(sys_config.logging.flush_interval_ms)
    });
    
    LatencyController latency_ctrl({
        .target_latency = std::chrono::microseconds(sys_config.latency.target_latency_us),
        .sample_window = sys_config.latency.sample_window,
        .enable_profiling = sys_config.latency.enable_profiling
    });
    
    // Arrival rate tracking for pressure system.
    std::atomic<uint64_t> token_count_window{0};
    std::atomic<double>   sentiment_variance_accum{0.0};
    std::atomic<double>   sentiment_mean_accum{0.0};
    std::atomic<uint64_t> variance_n{0};

    LLMAdapter llm_adapter;
    
    TradeSignalEngine trade_engine({
        .bias_sensitivity = sys_config.trading.bias_sensitivity,
        .volatility_sensitivity = sys_config.trading.volatility_sensitivity,
        .signal_decay_rate = sys_config.trading.signal_decay_rate,
        .signal_cooldown = std::chrono::microseconds(sys_config.trading.signal_cooldown_us)
    });
    
    // Wire an in-memory sink for telemetry (signals accessible for inspection/export).
    auto memory_sink = std::make_shared<llmquant::MemoryOutputSink>();
    trade_engine.add_output_sink(memory_sink);

    TokenStreamSimulator token_sim({
        .token_interval = std::chrono::microseconds(sys_config.token_stream.token_interval_ms * 1000),
        .buffer_size = sys_config.token_stream.buffer_size,
        .use_memory_stream = sys_config.token_stream.use_memory_stream,
        .data_file_path = sys_config.token_stream.data_file_path
    });
    
    // Set up callbacks
    token_sim.set_token_callback([&](const Token& token) {
        latency_ctrl.start_measurement();
        
        // Log token received
        logger.log_token_received(token.text, token.sequence_id);
        
        // Map token to semantic weight
        auto weight = llm_adapter.map_token_to_weight(token.text);
        
        // Process through trade signal engine
        trade_engine.process_semantic_weight(weight);

        latency_ctrl.end_measurement();

        // Track token arrival for ingestion pressure.
        token_count_window++;

        // Welford online variance for semantic pressure.
        double s = weight.sentiment_score;
        uint64_t n = variance_n.fetch_add(1) + 1;
        double mean = sentiment_mean_accum.load();
        double delta = s - mean;
        sentiment_mean_accum.store(mean + delta / static_cast<double>(n));
        double delta2 = s - sentiment_mean_accum.load();
        double var = sentiment_variance_accum.load();
        sentiment_variance_accum.store(var + delta * delta2);

        // Update pressure (semantic only here; ingestion + queue updated in monitoring loop).
        double current_variance = (n > 1) ? (sentiment_variance_accum.load() / static_cast<double>(n - 1)) : 0.0;
        latency_ctrl.update_semantic_pressure(current_variance);
    });
    
    trade_engine.set_signal_callback([&](const TradeSignal& signal) {
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - signal.timestamp
        );
        
        logger.log_signal_generated(
            signal.delta_bias_shift, 
            signal.volatility_adjustment, 
            latency.count()
        );
        
        // Example output matching your specification
        std::cout << "[" << std::chrono::duration_cast<std::chrono::milliseconds>(
            signal.timestamp.time_since_epoch()).count() << "] "
                  << "Trading engine updated. Δ Skew: " << signal.delta_bias_shift 
                  << " | Δ Volatility: " << signal.volatility_adjustment
                  << " | Latency: " << latency.count() << "μs" << std::endl;
    });
    
    // Load test tokens
    if (sys_config.token_stream.use_memory_stream) {
        token_sim.load_tokens_from_memory({
            "crash", "panic", "inevitable", "guarantee", "bullish", "collapse",
            "volatile", "surge", "confident", "uncertain", "rally", "plunge",
            "breakout", "support", "resistance", "momentum"
        });
    } else {
        token_sim.load_tokens_from_file(sys_config.token_stream.data_file_path);
    }
    
    // Start the engine
    std::cout << "🚀 Starting LLMTokenStreamQuantEngine..." << std::endl;
    std::cout << "Target latency: " << sys_config.latency.target_latency_us << "μs" << std::endl;
    std::cout << "Token interval: " << sys_config.token_stream.token_interval_ms << "ms" << std::endl;
    
    token_sim.start();
    
    // Main loop
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto stats    = latency_ctrl.get_stats();
        auto pressure = latency_ctrl.get_pressure();

        // Update ingestion pressure (tokens emitted in the last second).
        uint64_t tps = token_count_window.exchange(0);
        double max_tps = static_cast<double>(1000000 / std::max(1, sys_config.token_stream.token_interval_ms));
        latency_ctrl.update_ingestion_pressure(static_cast<double>(tps), max_tps);

        // Queue pressure: use signal-suppressed count as proxy for backlog depth.
        auto& engine_stats = trade_engine.get_stats();
        latency_ctrl.update_queue_pressure(engine_stats.signals_suppressed.load(), 1024);

        double backoff = latency_ctrl.get_backoff_multiplier();

        std::cout << "\r"
                  << "Tokens: "     << token_sim.get_stats().tokens_emitted
                  << " | Avg: "     << stats.avg_latency.count()     << "us"
                  << " | Max: "     << stats.max_latency.count()     << "us"
                  << " | P99: "     << stats.p99_latency.count()     << "us"
                  << " | Press: "   << std::fixed << std::setprecision(2) << pressure.composite
                  << " | Backoff: " << backoff << "x"
                  << std::flush;
    }
    
    token_sim.stop();
    std::cout << "Signals captured by memory sink: " << memory_sink->get_signals().size() << std::endl;
    logger.log_performance_summary();
    config.stop_watching();

    std::cout << "\nEngine stopped successfully" << std::endl;
    return 0;
}
