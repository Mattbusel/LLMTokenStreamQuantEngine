#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include "OutputSink.h"
#include "Deduplicator.h"
#include "LLMStreamClient.h"
#include "OmsAdapter.h"
#include "RestOmsAdapter.h"
#include "MockOmsAdapter.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

using namespace llmquant;

std::atomic<bool> g_running{true};

void signal_handler(int /*signal*/) {
    std::cout << "\nShutting down gracefully..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    // Load configuration.
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

    // Deduplication layer: skip repeated tokens within a sliding TTL window.
    auto dedup_backend = std::make_shared<llmquant::InProcessDeduplicator>();
    llmquant::Deduplicator deduplicator(dedup_backend,
        std::chrono::milliseconds(sys_config.token_stream.token_interval_ms * 10));

    // Initialize subsystem components.
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

    // Risk manager.
    llmquant::RiskManager::Config risk_cfg;
    risk_cfg.max_bias_magnitude      = 2.0;
    risk_cfg.max_volatility_magnitude = 2.0;
    risk_cfg.max_signals_per_second  = 500;
    risk_cfg.max_drawdown            = 10.0;
    llmquant::RiskManager risk_mgr(risk_cfg);

    // OMS adapter: use MockOmsAdapter by default; REST if --oms <host:port> is passed.
    std::unique_ptr<llmquant::OmsAdapter> oms_adapter;
    if (argc > 2 && std::string(argv[2]) == "--oms" && argc > 3) {
        std::string endpoint(argv[3]);
        llmquant::RestOmsAdapter::Config oms_cfg;
        size_t colon = endpoint.find(':');
        if (colon != std::string::npos) {
            oms_cfg.host = endpoint.substr(0, colon);
            oms_cfg.port = static_cast<uint16_t>(std::stoi(endpoint.substr(colon + 1)));
        } else {
            oms_cfg.host = endpoint;
        }
        oms_adapter = std::make_unique<llmquant::RestOmsAdapter>(oms_cfg);
    } else {
        auto mock = std::make_unique<llmquant::MockOmsAdapter>();
        mock->load_states({
            {0.1,  1.0,  0.5, -10.0},
            {0.25, 1.0,  0.3, -10.0},
            {-0.1, 1.0, -0.2, -10.0},
        });
        oms_adapter = std::move(mock);
    }

    oms_adapter->set_position_callback([&](const llmquant::RiskManager::PositionState& state) {
        risk_mgr.update_position(state);
    });
    risk_mgr.set_oms_callback([](const std::string& event,
                                  const llmquant::RiskManager::PositionState&,
                                  const llmquant::TradeSignal&) {
        std::cout << "\n[risk] " << event << std::endl;
    });
    oms_adapter->start();

    TokenStreamSimulator token_sim({
        .token_interval = std::chrono::microseconds(sys_config.token_stream.token_interval_ms * 1000),
        .buffer_size = sys_config.token_stream.buffer_size,
        .use_memory_stream = sys_config.token_stream.use_memory_stream,
        .data_file_path = sys_config.token_stream.data_file_path
    });

    // Shared token processing lambda used by both the simulator and the
    // LLMStreamClient paths.  Encapsulates dedup, latency, logging, and
    // semantic-weight pipeline so neither call site duplicates logic.
    auto process_token = [&](const std::string& text, uint64_t seq_id) {
        // Skip duplicate tokens within the dedup window.
        if (deduplicator.check(text) == llmquant::DedupResult::Duplicate) {
            return;
        }

        latency_ctrl.start_measurement();

        logger.log_token_received(text, seq_id);

        auto weight = llm_adapter.map_token_to_weight(text);

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

        // Update pressure (semantic only; ingestion + queue updated in monitoring loop).
        double current_variance = (n > 1)
            ? (sentiment_variance_accum.load() / static_cast<double>(n - 1))
            : 0.0;
        latency_ctrl.update_semantic_pressure(current_variance);
    };

    // Set up simulator callback.
    token_sim.set_token_callback([&](const Token& token) {
        process_token(token.text, token.sequence_id);
    });

    trade_engine.set_signal_callback([&](const TradeSignal& signal) {
        if (!risk_mgr.evaluate(signal)) return;   // blocked by risk

        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - signal.timestamp
        );

        logger.log_signal_generated(
            signal.delta_bias_shift,
            signal.volatility_adjustment,
            latency.count()
        );

        std::cout << "[" << std::chrono::duration_cast<std::chrono::milliseconds>(
            signal.timestamp.time_since_epoch()).count() << "] "
                  << "Trading engine updated. Δ Skew: " << signal.delta_bias_shift
                  << " | Δ Volatility: " << signal.volatility_adjustment
                  << " | Latency: " << latency.count() << "us" << std::endl;
    });

    // Load test tokens for simulator path.
    if (sys_config.token_stream.use_memory_stream) {
        token_sim.load_tokens_from_memory({
            "crash", "panic", "inevitable", "guarantee", "bullish", "collapse",
            "volatile", "surge", "confident", "uncertain", "rally", "plunge",
            "breakout", "support", "resistance", "momentum"
        });
    } else {
        token_sim.load_tokens_from_file(sys_config.token_stream.data_file_path);
    }

    // Start the engine — either via LLMStreamClient (--stream flag) or the
    // local TokenStreamSimulator.
    std::cout << "Starting LLMTokenStreamQuantEngine..." << std::endl;
    std::cout << "Target latency: " << sys_config.latency.target_latency_us << "us" << std::endl;
    std::cout << "Token interval: " << sys_config.token_stream.token_interval_ms << "ms" << std::endl;

    std::unique_ptr<llmquant::LLMStreamClient> stream_client;
    if (argc > 2 && std::string(argv[2]) == "--stream") {
        llmquant::LLMStreamClient::Config stream_cfg;
        stream_cfg.host    = "api.openai.com";
        stream_cfg.port    = 80;      // plain HTTP for now; TLS via proxy
        stream_cfg.api_key = (argc > 3) ? argv[3] : "";
        stream_cfg.use_tls = false;

        stream_client = std::make_unique<llmquant::LLMStreamClient>(stream_cfg);
        stream_client->set_token_callback([&](const std::string& text) {
            process_token(text, 0);
        });
        stream_client->set_done_callback([](const std::string& err) {
            if (!err.empty()) std::cerr << "[stream] Error: " << err << std::endl;
        });
        stream_client->connect();
        std::cout << "Streaming from LLM API..." << std::endl;
    } else {
        token_sim.start();
    }

    // Main monitoring loop.
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto stats    = latency_ctrl.get_stats();
        auto pressure = latency_ctrl.get_pressure();

        // Update ingestion pressure (tokens processed in the last second).
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
                  << " | Dedup hits: " << dedup_backend->total_duplicates()
                  << std::flush;
    }

    token_sim.stop();
    if (stream_client) stream_client->stop();
    oms_adapter->stop();

    std::cout << "Signals captured by memory sink: " << memory_sink->get_signals().size() << std::endl;
    logger.log_performance_summary();
    config.stop_watching();

    std::cout << "\nEngine stopped successfully" << std::endl;
    return 0;
}
