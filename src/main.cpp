#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>

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
    
    LLMAdapter llm_adapter;
    
    TradeSignalEngine trade_engine({
        .bias_sensitivity = sys_config.trading.bias_sensitivity,
        .volatility_sensitivity = sys_config.trading.volatility_sensitivity,
        .signal_decay_rate = sys_config.trading.signal_decay_rate,
        .signal_cooldown = std::chrono::microseconds(sys_config.trading.signal_cooldown_us)
    });
    
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
        
        // Print performance stats
        auto stats = latency_ctrl.get_stats();
        std::cout << "\r📊 Tokens: " << token_sim.get_stats().tokens_emitted 
                  << " | Avg Latency: " << stats.avg_latency.count() << "μs"
                  << " | Max: " << stats.max_latency.count() << "μs" << std::flush;
    }
    
    token_sim.stop();
    logger.log_performance_summary();
    
    std::cout << "\n✅ Engine stopped successfully" << std::endl;
    return 0;
}
