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
    // OMS alert callback wired after signal callback is registered (see below).
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

    // Shared risk-block reason for display on the same line.
    std::string last_block_reason;
    std::mutex  block_reason_mutex;

    risk_mgr.set_oms_callback([&](const std::string& event,
                                   const llmquant::RiskManager::PositionState&,
                                   const llmquant::TradeSignal&) {
        std::lock_guard<std::mutex> lk(block_reason_mutex);
        last_block_reason = event;
    });

    trade_engine.set_signal_callback([&](const TradeSignal& signal) {
        auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         signal.timestamp.time_since_epoch()).count();
        auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now() - signal.timestamp
                          ).count();

        bool passed = risk_mgr.evaluate(signal);

        std::string gate_str;
        if (passed) {
            gate_str = " \033[32mPASS\033[0m";
        } else {
            std::lock_guard<std::mutex> lk(block_reason_mutex);
            std::string reason = last_block_reason.empty() ? "risk" : last_block_reason;
            // Truncate to 16 chars for column alignment.
            if (reason.size() > 16) reason = reason.substr(0, 16);
            gate_str = " \033[31mBLOCK\033[0m(" + reason + ")";
            last_block_reason.clear();
        }

        // Aligned columns: TIME(ms)  BIAS     VOL      LATENCY  GATE
        std::cout << "\n  "
                  << std::setw(12) << ts_ms          << "  "
                  << std::setw(8)  << std::fixed << std::setprecision(4)
                                   << signal.delta_bias_shift  << "  "
                  << std::setw(8)  << signal.volatility_adjustment << "  "
                  << std::setw(6)  << latency_us << "μs"
                  << gate_str
                  << std::flush;

        if (passed) {
            logger.log_signal_generated(
                signal.delta_bias_shift,
                signal.volatility_adjustment,
                static_cast<uint64_t>(latency_us));
        }
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

    // Detect --stream flag in any position after argv[0].
    // Accepted forms:
    //   ./engine --stream <api_key>
    //   ./engine config.yaml --stream <api_key>
    std::string stream_api_key;
    bool stream_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--stream" && i + 1 < argc) {
            stream_mode    = true;
            stream_api_key = argv[i + 1];
            break;
        }
    }

    // Print banner.
    std::cout << "\n";
    std::cout << "  LLMTokenStreamQuantEngine\n";
    std::cout << "  ─────────────────────────────────────────────────────────\n";
    if (stream_mode) {
        std::cout << "  MODE    : LIVE STREAM  (gpt-4o → api.openai.com:443)\n";
        std::cout << "  PROMPT  : market sentiment / tickers / directional\n";
        std::cout << "  INTERVAL: 5s per request\n";
    } else {
        std::cout << "  MODE    : SIMULATOR  (in-memory token loop)\n";
        std::cout << "  INTERVAL: " << sys_config.token_stream.token_interval_ms << "ms/token\n";
    }
    std::cout << "  LATENCY : target p99 < " << sys_config.latency.target_latency_us << "μs\n";
    std::cout << "  ─────────────────────────────────────────────────────────\n\n";
    std::cout << "  TIME(ms)     TOKEN              BIAS      VOL       LATENCY   GATE\n";
    std::cout << "  ─────────────────────────────────────────────────────────────────\n";

    std::unique_ptr<llmquant::LLMStreamClient> stream_client;
    if (stream_mode) {
        llmquant::LLMStreamClient::Config stream_cfg;
        stream_cfg.host         = "api.openai.com";
        stream_cfg.port         = 443;
        stream_cfg.api_key      = stream_api_key;
        stream_cfg.model        = "gpt-4o";
        stream_cfg.use_tls      = true;
        stream_cfg.max_tokens   = 300;
        stream_cfg.loop_interval = std::chrono::seconds(5);
        stream_cfg.system_prompt =
            "You are a financial markets analyst providing real-time commentary "
            "on market conditions, options flow, and sentiment. Be specific, "
            "use tickers, use directional language.";
        stream_cfg.user_prompt =
            "Give a fresh real-time market sentiment update with specific "
            "tickers and directional signals.";

        stream_client = std::make_unique<llmquant::LLMStreamClient>(stream_cfg);
        stream_client->set_token_callback([&](const std::string& text) {
            process_token(text, 0);
        });
        stream_client->set_done_callback([](const std::string& err) {
            if (!err.empty())
                std::cerr << "\n  [stream] " << err << std::endl;
        });
        stream_client->connect();
    } else {
        token_sim.start();
    }

    // Main monitoring loop — prints a rolling stats bar every second.
    uint64_t last_tick = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto stats    = latency_ctrl.get_stats();
        auto pressure = latency_ctrl.get_pressure();

        // Update ingestion pressure.
        uint64_t tps = token_count_window.exchange(0);
        double   max_tps = stream_mode
                               ? 50.0   // gpt-4o emits ~10-30 tokens/s
                               : static_cast<double>(1000000 / std::max(1, sys_config.token_stream.token_interval_ms));
        latency_ctrl.update_ingestion_pressure(static_cast<double>(tps), max_tps);

        // Queue pressure via suppressed-signal count.
        auto& eng_stats = trade_engine.get_stats();
        latency_ctrl.update_queue_pressure(eng_stats.signals_suppressed.load(), 1024);

        double backoff = latency_ctrl.get_backoff_multiplier();

        // Colour the P99 value: green < 10μs, yellow < 50μs, red otherwise.
        auto p99 = stats.p99_latency.count();
        const char* p99_colour =
            (p99 < 10)  ? "\033[32m" :
            (p99 < 50)  ? "\033[33m" : "\033[31m";

        // Colour the pressure bar.
        const char* press_colour =
            (pressure.composite < 0.5) ? "\033[32m" :
            (pressure.composite < 0.8) ? "\033[33m" : "\033[31m";

        uint64_t tokens_total = stream_mode
                                    ? static_cast<uint64_t>(variance_n.load())
                                    : token_sim.get_stats().tokens_emitted.load();

        // Overwrite the stats line in-place.
        std::cout << "\n  ─ STATS  "
                  << " TPS:"   << std::setw(4) << tps
                  << "  TOK:"  << std::setw(7) << tokens_total
                  << "  AVG:"  << std::setw(5) << stats.avg_latency.count() << "μs"
                  << "  P99:"  << p99_colour
                               << std::setw(5) << p99 << "μs\033[0m"
                  << "  PRESS:" << press_colour
                               << std::fixed << std::setprecision(2)
                               << pressure.composite << "\033[0m"
                  << "  BKOF:" << std::setprecision(1) << backoff << "x"
                  << "  DEDUP:" << dedup_backend->total_duplicates()
                  << "  SIG-PASS:" << eng_stats.signals_generated.load()
                  << "  BLOCK:"   << (eng_stats.signals_suppressed.load()
                                      + risk_mgr.get_stats().signals_blocked_magnitude.load()
                                      + risk_mgr.get_stats().signals_blocked_confidence.load()
                                      + risk_mgr.get_stats().signals_blocked_rate.load()
                                      + risk_mgr.get_stats().signals_blocked_drawdown.load()
                                      + risk_mgr.get_stats().signals_blocked_position.load())
                  << std::flush;

        // Alert if P99 exceeds budget.
        if (p99 > sys_config.latency.target_latency_us && last_tick != stats.measurements) {
            std::cout << "  \033[31m[!] P99 > target\033[0m" << std::flush;
        }
        last_tick = stats.measurements;
    }

    token_sim.stop();
    if (stream_client) stream_client->stop();
    oms_adapter->stop();
    config.stop_watching();

    auto final_stats = latency_ctrl.get_stats();
    std::cout << "\n\n  ═══════════════════════════════════════════════════════\n";
    std::cout << "  SESSION SUMMARY\n";
    std::cout << "  ───────────────────────────────────────────────────────\n";
    std::cout << "  Tokens processed : " << variance_n.load() << "\n";
    std::cout << "  Signals emitted  : " << trade_engine.get_stats().signals_generated.load() << "\n";
    std::cout << "  Signals blocked  : "
              << (risk_mgr.get_stats().signals_blocked_magnitude.load()
                  + risk_mgr.get_stats().signals_blocked_confidence.load()
                  + risk_mgr.get_stats().signals_blocked_rate.load()
                  + risk_mgr.get_stats().signals_blocked_drawdown.load()
                  + risk_mgr.get_stats().signals_blocked_position.load()) << "\n";
    std::cout << "  Memory sink size : " << memory_sink->get_signals().size() << "\n";
    std::cout << "  Avg latency      : " << final_stats.avg_latency.count() << "μs\n";
    std::cout << "  P99 latency      : " << final_stats.p99_latency.count() << "μs\n";
    std::cout << "  Max latency      : " << final_stats.max_latency.count() << "μs\n";
    std::cout << "  ═══════════════════════════════════════════════════════\n\n";

    logger.log_performance_summary();
    return 0;
}
