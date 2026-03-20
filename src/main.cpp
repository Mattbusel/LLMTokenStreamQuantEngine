#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include "OutputSinkImpl.h"
#include "Deduplicator.h"
#include "LLMStreamClient.h"
#include "OmsAdapter.h"
#include "RestOmsAdapter.h"
#include "FixOmsAdapter.h"
#include "MockOmsAdapter.h"
#include "PrometheusExporter.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <sstream>

using namespace llmquant;

std::atomic<bool> g_running{true};

void signal_handler(int /*signal*/) {
    // std::cout is not async-signal-safe; only set the atomic flag here.
    // The main loop detects g_running==false and prints the shutdown summary.
    g_running = false;
}

int main(int argc, char* argv[]) {
  try {
    std::signal(SIGINT, signal_handler);

    // Record engine start time for uptime metrics.
    const auto engine_start_time = std::chrono::steady_clock::now();

    // Parse flags before anything else.
    bool        stream_mode    = false;
    std::string stream_api_key;
    bool        no_color       = false;
    bool        debug_raw      = false;
    bool        dry_run        = false;
    bool        backtest_mode  = false;
    std::string oms_address;
    std::string fix_address;
    std::string config_file    = "config.yaml"; // may be overridden by --config
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: LLMTokenStreamQuantEngine [config.yaml] [options]\n"
                "\n"
                "Options:\n"
                "  --stream [key]    Enable live LLM stream mode (optional API key)\n"
                "  --oms host:port   Connect to REST OMS adapter\n"
                "  --fix host:port   Connect to FIX 4.2 OMS adapter\n"
                "  --config path     Path to config YAML (default: config.yaml)\n"
                "  --dry-run         Process tokens through LLMAdapter only; skip signal emission\n"
                "  --backtest        Enable backtest mode (emit signal on every token, no cooldown)\n"
                "  --no-color        Disable ANSI colour output\n"
                "  --debug-raw       Print raw LLM stream bytes\n"
                "  --version         Print version and exit\n"
                "  --help            Print this help and exit\n"
                "\n"
                "Environment:\n"
                "  LLMQUANT_API_KEY  LLM API key (fallback when --stream has no key)\n"
                "\n"
                "Config file (YAML) keys: token_stream, trading, latency, logging,\n"
                "  pressure, risk_thresholds, risk (override flags).\n";
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            // Version string injected by CMake via llmquant_version.h.in
            std::cout << "LLMTokenStreamQuantEngine 1.1.0\n";
            return 0;
        } else if (arg == "--stream") {
            stream_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                stream_api_key = argv[++i];  // explicit key provided on CLI
        } else if (arg == "--no-color") {
            no_color = true;
        } else if (arg == "--debug-raw") {
            debug_raw = true;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--backtest") {
            backtest_mode = true;
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--oms" && i + 1 < argc) {
            oms_address = argv[++i];
        } else if (arg == "--fix" && i + 1 < argc) {
            fix_address = argv[++i];
        }
    }

    // API key security: fall back to LLMQUANT_API_KEY env var if not given on CLI.
    // Never echo the key value into logs or the banner.
    if (stream_api_key.empty()) {
        // getenv is safe here: the environment is not modified concurrently
        // during this single-threaded init phase.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        const char* env_key_raw = std::getenv("LLMQUANT_API_KEY");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        if (const char* env_key = env_key_raw) {
            stream_api_key = env_key;
            spdlog::warn("API key loaded from environment variable LLMQUANT_API_KEY"
                         " — consider using a key file (mode 0600) for better security");
        }
    } else {
        spdlog::debug("API key loaded from command-line argument");
    }

    // Colour helpers — emit empty string when --no-color is active.
    auto C = [&](const char* code) -> const char* { return no_color ? "" : code; };
    // Line helpers — ASCII dividers when --no-color, Unicode otherwise.
    const char* DIV1 = no_color
        ? "  ---------------------------------------------------------\n"
        : "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n";
    const char* DIV2 = no_color
        ? "  -----------------------------------------------------------------\n"
        : "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n";
    const char* ARROW = no_color ? "->" : "\xe2\x86\x92";

    // Load configuration.
    // config_file may already have been set by --config; otherwise treat the
    // first positional argument (no leading dash) as the config path so that
    // the legacy invocation `engine config.yaml` still works.
    Config config;
    if (config_file == "config.yaml") {
        for (int i = 1; i < argc; ++i) {
            if (argv[i][0] != '-') { config_file = argv[i]; break; }
        }
    }
    bool config_loaded = config.load_from_file(config_file);
    if (!config_loaded) {
        std::cout << "Using default configuration" << std::endl;
        // No config file: fall back to in-memory token stream so no file I/O required.
        config.set_use_memory_stream(true);
    }

    const auto& sys_config = config.get_config();

    // Deduplication layer: skip repeated tokens within a sliding TTL window.
    auto dedup_backend = std::make_shared<llmquant::InProcessDeduplicator>();
    llmquant::Deduplicator deduplicator(dedup_backend,
        std::chrono::milliseconds(sys_config.token_stream.token_interval_ms * 10));
    // Prevent unbounded memory growth: purge expired entries every 60 s.
    dedup_backend->start_background_purge(60);

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
    // Welford variance accumulators: plain (non-atomic) variables protected exclusively
    // by variance_mutex so readers can never see an inconsistent state between reset
    // and update (Improvement 2 — fix Welford variance race).
    double   sentiment_variance_accum{0.0};
    double   sentiment_mean_accum{0.0};
    uint64_t variance_n{0};
    // Wall-clock timestamp of the last Welford accumulator reset.
    // Reset every 60 seconds to prevent catastrophic cancellation.
    auto variance_last_reset = std::chrono::steady_clock::now();
    // Protects the three Welford variables as a unit: both the token-callback
    // update and the monitoring-loop reset must hold this mutex so they are
    // never interleaved, which would silently corrupt the variance estimate.
    std::mutex            variance_mutex;

    LLMAdapter llm_adapter;

    TradeSignalEngine trade_engine({
        .bias_sensitivity = sys_config.trading.bias_sensitivity,
        .volatility_sensitivity = sys_config.trading.volatility_sensitivity,
        .signal_decay_rate = sys_config.trading.signal_decay_rate,
        .signal_cooldown = std::chrono::microseconds(sys_config.trading.signal_cooldown_us)
    });

    // Backtest mode: emit on every token, ignoring the cooldown timer.
    if (backtest_mode) {
        trade_engine.set_backtest_mode(true);
    }

    // Wire an in-memory sink for telemetry (signals accessible for inspection/export).
    auto memory_sink = std::make_shared<llmquant::MemoryOutputSink>();
    trade_engine.add_output_sink(memory_sink);

    // Risk manager — thresholds driven from config (hot-reloadable via YAML).
    const auto& rt = sys_config.risk_thresholds;
    llmquant::RiskManager::Config risk_cfg;
    risk_cfg.max_bias_magnitude       = rt.max_bias_magnitude;
    risk_cfg.max_volatility_magnitude = rt.max_volatility_magnitude;
    risk_cfg.max_spread_magnitude     = rt.max_spread_magnitude;
    risk_cfg.min_confidence           = rt.min_confidence;
    risk_cfg.max_signals_per_second   = rt.max_signals_per_second;
    risk_cfg.max_drawdown             = rt.max_drawdown;
    risk_cfg.drawdown_window          = std::chrono::seconds(rt.drawdown_window_s);
    risk_cfg.position_warn_fraction   = rt.position_warn_fraction;
    risk_cfg.disable_magnitude_gate   = sys_config.risk_overrides.disable_magnitude_gate;
    risk_cfg.disable_confidence_gate  = sys_config.risk_overrides.disable_confidence_gate;
    risk_cfg.disable_rate_gate        = sys_config.risk_overrides.disable_rate_gate;
    risk_cfg.disable_drawdown_gate    = sys_config.risk_overrides.disable_drawdown_gate;
    risk_cfg.disable_position_gate    = sys_config.risk_overrides.disable_position_gate;
    llmquant::RiskManager risk_mgr(risk_cfg);
    risk_mgr.set_metrics_logger(&logger);

    // Start config hot-reload watcher now that risk_mgr exists so the callback
    // can update risk thresholds live without requiring a restart.
    if (!config.start_watching(config_file, [&risk_mgr, &trade_engine](const llmquant::SystemConfig& updated) {
        const auto& u = updated.risk_thresholds;
        llmquant::RiskManager::Config new_risk_cfg;
        new_risk_cfg.max_bias_magnitude       = u.max_bias_magnitude;
        new_risk_cfg.max_volatility_magnitude = u.max_volatility_magnitude;
        new_risk_cfg.max_spread_magnitude     = u.max_spread_magnitude;
        new_risk_cfg.min_confidence           = u.min_confidence;
        new_risk_cfg.max_signals_per_second   = u.max_signals_per_second;
        new_risk_cfg.max_drawdown             = u.max_drawdown;
        new_risk_cfg.drawdown_window          = std::chrono::seconds(u.drawdown_window_s);
        new_risk_cfg.position_warn_fraction   = u.position_warn_fraction;
        new_risk_cfg.disable_magnitude_gate   = updated.risk_overrides.disable_magnitude_gate;
        new_risk_cfg.disable_confidence_gate  = updated.risk_overrides.disable_confidence_gate;
        new_risk_cfg.disable_rate_gate        = updated.risk_overrides.disable_rate_gate;
        new_risk_cfg.disable_drawdown_gate    = updated.risk_overrides.disable_drawdown_gate;
        new_risk_cfg.disable_position_gate    = updated.risk_overrides.disable_position_gate;
        risk_mgr.update_config(new_risk_cfg);
        llmquant::TradeSignalEngine::Config new_eng_cfg;
        new_eng_cfg.bias_sensitivity       = updated.trading.bias_sensitivity;
        new_eng_cfg.volatility_sensitivity = updated.trading.volatility_sensitivity;
        new_eng_cfg.signal_decay_rate      = updated.trading.signal_decay_rate;
        new_eng_cfg.signal_cooldown        = std::chrono::microseconds(updated.trading.signal_cooldown_us);
        trade_engine.update_config(new_eng_cfg);
        std::cout << "\n[config] Hot-reloaded: bias_sensitivity="
                  << updated.trading.bias_sensitivity
                  << "  max_bias=" << u.max_bias_magnitude
                  << "  max_signals/s=" << u.max_signals_per_second << std::endl;
    })) {
        std::cerr << "[warn] Config hot-reload watcher failed to start\n";
    }

    // OMS adapter: MockOmsAdapter by default; REST via --oms, FIX 4.2 via --fix.
    std::unique_ptr<llmquant::OmsAdapter> oms_adapter;
    if (!fix_address.empty()) {
        llmquant::FixOmsAdapter::Config fix_cfg;
        size_t colon = fix_address.find(':');
        if (colon != std::string::npos) {
            fix_cfg.host = fix_address.substr(0, colon);
            fix_cfg.port = static_cast<uint16_t>(std::stoi(fix_address.substr(colon + 1)));
        } else {
            fix_cfg.host = fix_address;
        }
        oms_adapter = std::make_unique<llmquant::FixOmsAdapter>(fix_cfg);
    } else if (!oms_address.empty()) {
        std::string endpoint = oms_address;
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

        // In dry-run mode, tokens are mapped through LLMAdapter for
        // dictionary coverage analysis but no signals are emitted.
        if (!dry_run) {
            trade_engine.process_semantic_weight(weight);
        }

        latency_ctrl.end_measurement();

        // Track token arrival for ingestion pressure.
        token_count_window++;

        // Welford online variance for semantic pressure.
        // The mutex ensures this three-variable update is never interleaved
        // with the periodic reset in the monitoring loop.
        double current_variance = 0.0;
        {
            std::lock_guard<std::mutex> lk(variance_mutex);
            double s = weight.sentiment_score;
            ++variance_n;
            uint64_t n = variance_n;
            double delta = s - sentiment_mean_accum;
            sentiment_mean_accum += delta / static_cast<double>(n);
            double delta2 = s - sentiment_mean_accum;
            sentiment_variance_accum += delta * delta2;
            current_variance = (n > 1)
                ? (sentiment_variance_accum / static_cast<double>(n - 1))
                : 0.0;
        }

        // Update pressure (semantic only; ingestion + queue updated in monitoring loop).
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
            gate_str = std::string(" ") + C("\033[32m") + "PASS" + C("\033[0m");
        } else {
            std::lock_guard<std::mutex> lk(block_reason_mutex);
            std::string reason = last_block_reason.empty() ? "risk" : last_block_reason;
            if (reason.size() > 16) reason = reason.substr(0, 16);
            gate_str = std::string(" ") + C("\033[31m") + "BLOCK" + C("\033[0m") + "(" + reason + ")";
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

    // Print banner.
    std::cout << "\n";
    std::cout << "  LLMTokenStreamQuantEngine\n";
    std::cout << DIV1;
    if (stream_mode) {
        std::cout << "  MODE    : LIVE STREAM  (gpt-4o " << ARROW << " api.openai.com:443)\n";
        std::cout << "  PROMPT  : market sentiment / tickers / directional\n";
        std::cout << "  INTERVAL: 5s per request\n";
    } else {
        std::cout << "  MODE    : SIMULATOR  (in-memory token loop)\n";
        std::cout << "  INTERVAL: " << sys_config.token_stream.token_interval_ms << "ms/token\n";
    }
    std::cout << "  LATENCY : target p99 < " << sys_config.latency.target_latency_us << "us\n";
    if (dry_run)
        std::cout << "  DRY-RUN : signals suppressed — dictionary coverage mode\n";
    std::cout << DIV1 << "\n";
    std::cout << "  TIME(ms)     BIAS      VOL       LATENCY   GATE\n";
    std::cout << DIV2;

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
        stream_cfg.debug_raw    = debug_raw;
        stream_cfg.system_prompt =
            "You are a high-frequency financial markets analyst. Every response "
            "must include specific tickers and explicit directional words: "
            "bullish, bearish, crash, surge, panic, breakout, collapse, volatile, "
            "guarantee, inevitable. Be terse and signal-dense.";
        stream_cfg.user_prompt =
            "Give a terse real-time market signal update. Use tickers. "
            "Use words: bullish, bearish, surge, crash, breakout, collapse, volatile.";

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

    // Prometheus metrics endpoint on port 9100.
    // The snapshot is built once per second in the monitoring loop so the
    // scrape thread never contends with the hot path for latency stats.
    std::string prom_snapshot;
    std::mutex  prom_snapshot_mutex;

    llmquant::PrometheusExporter prom_exporter({.port = 9100});
    prom_exporter.set_metrics_callback([&]() -> std::string {
        std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
        return prom_snapshot;
    });
    if (!prom_exporter.start()) {
        std::cerr << "[warn] PrometheusExporter failed to bind on port 9100\n";
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
                               ? sys_config.pressure.max_ingestion_rate_tps   // gpt-4o emits ~10-30 tokens/s
                               : static_cast<double>(1000000 / std::max(1, sys_config.token_stream.token_interval_ms));
        latency_ctrl.update_ingestion_pressure(static_cast<double>(tps), max_tps);

        // Queue pressure via suppressed-signal count.
        auto eng_stats = trade_engine.get_stats();  // snapshot, not reference
        latency_ctrl.update_queue_pressure(eng_stats.signals_suppressed.load(), 1024);

        double backoff = latency_ctrl.get_backoff_multiplier();

        // Colour the P99 value: green < 10μs, yellow < 50μs, red otherwise.
        auto p99 = stats.p99_latency.count();
        const char* p99_colour =
            (p99 < 10)  ? C("\033[32m") :
            (p99 < 50)  ? C("\033[33m") : C("\033[31m");

        // Colour the pressure bar.
        const char* press_colour =
            (pressure.composite < 0.5) ? C("\033[32m") :
            (pressure.composite < 0.8) ? C("\033[33m") : C("\033[31m");

        uint64_t tokens_total_local;
        {
            std::lock_guard<std::mutex> lk(variance_mutex);
            tokens_total_local = variance_n;
        }
        uint64_t tokens_total = stream_mode
                                    ? tokens_total_local
                                    : token_sim.get_stats().tokens_emitted.load();

        // Saturating addition for the BLOCK counter — prevent silent wrap-around.
        const auto& rs = risk_mgr.get_stats();
        uint64_t blocked = eng_stats.signals_suppressed.load();
        auto sat_add = [](uint64_t a, uint64_t b) -> uint64_t {
            return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
        };
        blocked = sat_add(blocked, rs.signals_blocked_magnitude.load());
        blocked = sat_add(blocked, rs.signals_blocked_confidence.load());
        blocked = sat_add(blocked, rs.signals_blocked_rate.load());
        blocked = sat_add(blocked, rs.signals_blocked_drawdown.load());
        blocked = sat_add(blocked, rs.signals_blocked_position.load());
        blocked = sat_add(blocked, rs.signals_blocked_pnl.load());

        // Welford periodic reset: avoid precision loss over time.
        // Reset the accumulators every 60 seconds (wall-clock) instead of
        // by sample count, preventing catastrophic cancellation while
        // bounding the reset interval to a predictable time window.
        // The mutex ensures the three stores are never interleaved with the
        // Welford update running in the token callback.
        {
            std::lock_guard<std::mutex> lk(variance_mutex);
            auto now_steady = std::chrono::steady_clock::now();
            if (now_steady - variance_last_reset > std::chrono::seconds{60}) {
                variance_n = 0;
                sentiment_mean_accum = 0.0;
                sentiment_variance_accum = 0.0;
                variance_last_reset = now_steady;
            }
        }

        // Update Prometheus snapshot (read once per second from the monitoring
        // thread; the scrape callback returns this cached string without
        // acquiring any latency-path locks).
        {
            std::ostringstream snap;
            snap << "# HELP llmquant_signals_generated_total Total trade signals generated\n"
                 << "# TYPE llmquant_signals_generated_total counter\n"
                 << "llmquant_signals_generated_total " << eng_stats.signals_generated.load() << "\n"
                 << "# HELP llmquant_signals_suppressed_total Signals with no callback or sink (fully suppressed)\n"
                 << "# TYPE llmquant_signals_suppressed_total counter\n"
                 << "llmquant_signals_suppressed_total " << eng_stats.signals_suppressed.load() << "\n"
                 << "# HELP llmquant_signals_blocked_total Total trade signals blocked by risk\n"
                 << "# TYPE llmquant_signals_blocked_total counter\n"
                 << "llmquant_signals_blocked_total " << blocked << "\n"
                 << "# HELP llmquant_signals_blocked_magnitude_total Signals blocked: magnitude exceeded\n"
                 << "# TYPE llmquant_signals_blocked_magnitude_total counter\n"
                 << "llmquant_signals_blocked_magnitude_total " << rs.signals_blocked_magnitude.load() << "\n"
                 << "# HELP llmquant_signals_blocked_confidence_total Signals blocked: confidence below minimum\n"
                 << "# TYPE llmquant_signals_blocked_confidence_total counter\n"
                 << "llmquant_signals_blocked_confidence_total " << rs.signals_blocked_confidence.load() << "\n"
                 << "# HELP llmquant_signals_blocked_rate_total Signals blocked: rate limit exceeded\n"
                 << "# TYPE llmquant_signals_blocked_rate_total counter\n"
                 << "llmquant_signals_blocked_rate_total " << rs.signals_blocked_rate.load() << "\n"
                 << "# HELP llmquant_signals_blocked_drawdown_total Signals blocked: drawdown limit exceeded\n"
                 << "# TYPE llmquant_signals_blocked_drawdown_total counter\n"
                 << "llmquant_signals_blocked_drawdown_total " << rs.signals_blocked_drawdown.load() << "\n"
                 << "# HELP llmquant_signals_blocked_position_total Signals blocked: position limit breached\n"
                 << "# TYPE llmquant_signals_blocked_position_total counter\n"
                 << "llmquant_signals_blocked_position_total " << rs.signals_blocked_position.load() << "\n"
                 << "# HELP llmquant_signals_blocked_pnl_total Signals blocked: PnL limit breached\n"
                 << "# TYPE llmquant_signals_blocked_pnl_total counter\n"
                 << "llmquant_signals_blocked_pnl_total " << rs.signals_blocked_pnl.load() << "\n"
                 << "# HELP llmquant_latency_p99_us p99 token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p99_us gauge\n"
                 << "llmquant_latency_p99_us " << p99 << "\n"
                 << "# HELP llmquant_latency_avg_us Average token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_avg_us gauge\n"
                 << "llmquant_latency_avg_us " << stats.avg_latency.count() << "\n"
                 << "# HELP llmquant_latency_p50_us p50 (median) token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p50_us gauge\n"
                 << "llmquant_latency_p50_us " << stats.p50_latency.count() << "\n"
                 << "# HELP llmquant_latency_p95_us p95 token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p95_us gauge\n"
                 << "llmquant_latency_p95_us " << stats.p95_latency.count() << "\n"
                 << "# HELP llmquant_tokens_emitted_total Tokens emitted by simulator (0 in stream mode)\n"
                 << "# TYPE llmquant_tokens_emitted_total counter\n"
                 << "llmquant_tokens_emitted_total " << (!stream_mode ? token_sim.get_stats().tokens_emitted.load() : 0) << "\n"
                 << "# HELP llmquant_oms_update_count_total Total successful OMS position updates\n"
                 << "# TYPE llmquant_oms_update_count_total counter\n"
                 << "llmquant_oms_update_count_total " << [&]() -> uint64_t {
                        if (auto* rest = dynamic_cast<llmquant::RestOmsAdapter*>(oms_adapter.get()))
                            return rest->update_count();
                        return 0;
                    }() << "\n"
                 << "# HELP llmquant_oms_error_count_total Total OMS connection errors\n"
                 << "# TYPE llmquant_oms_error_count_total counter\n"
                 << "llmquant_oms_error_count_total " << [&]() -> uint64_t {
                        if (auto* rest = dynamic_cast<llmquant::RestOmsAdapter*>(oms_adapter.get()))
                            return rest->error_count();
                        return 0;
                    }() << "\n"
                 << "# HELP llmquant_dedup_novel_total Tokens processed as novel (not seen in TTL window)\n"
                 << "# TYPE llmquant_dedup_novel_total counter\n"
                 << "llmquant_dedup_novel_total " << dedup_backend->total_novel() << "\n"
                 << "# HELP llmquant_dedup_duplicates_total Tokens suppressed as duplicates within the TTL window\n"
                 << "# TYPE llmquant_dedup_duplicates_total counter\n"
                 << "llmquant_dedup_duplicates_total " << dedup_backend->total_duplicates() << "\n"
                 << "# HELP llmquant_dedup_redis_connected Whether a Redis dedup connection is active\n"
                 << "# TYPE llmquant_dedup_redis_connected gauge\n"
                 << "llmquant_dedup_redis_connected 0\n"
                 << "# HELP llmquant_dedup_redis_reconnect_attempts_total Total Redis reconnect attempts\n"
                 << "# TYPE llmquant_dedup_redis_reconnect_attempts_total counter\n"
                 << "llmquant_dedup_redis_reconnect_attempts_total 0\n"
                 << "# HELP llmquant_signals_passed_total Total trade signals that passed all risk gates\n"
                 << "# TYPE llmquant_signals_passed_total counter\n"
                 << "llmquant_signals_passed_total " << rs.signals_passed.load() << "\n"
                 << "# HELP llmquant_tokens_processed_total Total tokens processed since startup\n"
                 << "# TYPE llmquant_tokens_processed_total counter\n"
                 << "llmquant_tokens_processed_total " << llm_adapter.get_stats().tokens_processed << "\n"
                 << "# HELP llmquant_cache_hits_total Tokens resolved from the in-memory dictionary cache\n"
                 << "# TYPE llmquant_cache_hits_total counter\n"
                 << "llmquant_cache_hits_total " << llm_adapter.get_stats().cache_hits << "\n"
                 << "# HELP llmquant_cache_misses_total Tokens not found in the dictionary (neutral fallback)\n"
                 << "# TYPE llmquant_cache_misses_total counter\n"
                 << "llmquant_cache_misses_total " << llm_adapter.get_stats().cache_misses << "\n"
                 << "# HELP llmquant_dictionary_size Number of entries in the LLMAdapter token dictionary\n"
                 << "# TYPE llmquant_dictionary_size gauge\n"
                 << "llmquant_dictionary_size " << llm_adapter.get_dictionary_size() << "\n"
                 << "# HELP llmquant_pressure_composite Current composite back-pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_composite gauge\n"
                 << "llmquant_pressure_composite " << std::fixed << std::setprecision(4) << pressure.composite << "\n"
                 << "# HELP llmquant_pressure_ingestion Current ingestion pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_ingestion gauge\n"
                 << "llmquant_pressure_ingestion " << pressure.ingestion_pressure << "\n"
                 << "# HELP llmquant_pressure_semantic Current semantic variance pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_semantic gauge\n"
                 << "llmquant_pressure_semantic " << pressure.semantic_pressure << "\n"
                 << "# HELP llmquant_pressure_queue Current signal queue pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_queue gauge\n"
                 << "llmquant_pressure_queue " << pressure.queue_pressure << "\n"
                 << "# HELP llmquant_backoff_multiplier Current exponential backoff multiplier [1,5]\n"
                 << "# TYPE llmquant_backoff_multiplier gauge\n"
                 << "llmquant_backoff_multiplier " << std::setprecision(2) << backoff << "\n"
                 << "# HELP llmquant_latency_min_us Minimum observed token-to-signal latency\n"
                 << "# TYPE llmquant_latency_min_us gauge\n"
                 << "llmquant_latency_min_us " << stats.min_latency.count() << "\n"
                 << "# HELP llmquant_latency_max_us Maximum observed token-to-signal latency\n"
                 << "# TYPE llmquant_latency_max_us gauge\n"
                 << "llmquant_latency_max_us " << stats.max_latency.count() << "\n"
                 << "# HELP llmquant_latency_jitter_ms Latency standard deviation in milliseconds\n"
                 << "# TYPE llmquant_latency_jitter_ms gauge\n"
                 << "llmquant_latency_jitter_ms " << std::setprecision(4) << stats.jitter_ms << "\n"
                 << "# HELP llmquant_ring_buffer_drops_total Tokens dropped due to full simulator ring buffer\n"
                 << "# TYPE llmquant_ring_buffer_drops_total counter\n"
                 << "llmquant_ring_buffer_drops_total " << (!stream_mode ? token_sim.get_stats().ring_buffer_drops.load() : 0) << "\n"
                 << "# HELP llmquant_uptime_seconds Engine uptime since startup\n"
                 << "# TYPE llmquant_uptime_seconds counter\n"
                 << "llmquant_uptime_seconds " << std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - engine_start_time).count() << "\n"
                 << "# HELP llmquant_dry_run Whether the engine is running in dry-run mode (1=yes)\n"
                 << "# TYPE llmquant_dry_run gauge\n"
                 << "llmquant_dry_run " << (dry_run ? 1 : 0) << "\n"
                 << "# HELP llmquant_avg_signal_strength Running Welford mean of |delta_bias_shift|\n"
                 << "# TYPE llmquant_avg_signal_strength gauge\n"
                 << "llmquant_avg_signal_strength " << std::setprecision(6)
                     << eng_stats.avg_signal_strength.load() << "\n"
                 << "# HELP llmquant_latency_measurements_total Total latency samples recorded\n"
                 << "# TYPE llmquant_latency_measurements_total counter\n"
                 << "llmquant_latency_measurements_total " << stats.measurements << "\n";
            std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
            prom_snapshot = snap.str();
        }

        // Cache hit rate for LLMAdapter dictionary efficiency.
        auto adapter_stats = llm_adapter.get_stats();
        uint64_t hit_pct = (adapter_stats.tokens_processed > 0)
            ? (adapter_stats.cache_hits * 100 / adapter_stats.tokens_processed) : 0;

        // Overwrite the stats line in-place.
        std::cout << "\n  -- STATS "
                  << " TPS:"   << std::setw(4) << tps
                  << "  TOK:"  << std::setw(7) << tokens_total
                  << "  AVG:"  << std::setw(5) << stats.avg_latency.count() << "us"
                  << "  P99:"  << p99_colour
                               << std::setw(5) << p99 << "us" << C("\033[0m")
                  << "  PRESS:" << press_colour
                               << std::fixed << std::setprecision(2)
                               << pressure.composite << C("\033[0m")
                  << "  BKOF:" << std::setprecision(1) << backoff << "x"
                  << "  HIT%:" << hit_pct
                  << "  DEDUP:" << dedup_backend->total_duplicates()
                  << "  PASS:" << risk_mgr.get_stats().signals_passed.load()
                  << "  BLOCK:" << blocked
                  << "  RATE%:" << [&]() -> uint64_t {
                        uint64_t passed = risk_mgr.get_stats().signals_passed.load();
                        uint64_t total  = (passed > UINT64_MAX - blocked) ? UINT64_MAX : passed + blocked;
                        return (total > 0) ? (passed * 100 / total) : 100;
                     }()
                  << (!stream_mode ? (std::string("  DROPS:") + std::to_string(token_sim.get_stats().ring_buffer_drops.load())) : "")
                  << std::flush;

        // Alert if P99 exceeds budget.
        if (p99 > sys_config.latency.target_latency_us && last_tick != stats.measurements) {
            std::cout << "  " << C("\033[31m") << "[!] P99 > target" << C("\033[0m") << std::flush;
        }
        last_tick = stats.measurements;
    }

    token_sim.stop();
    if (stream_client) stream_client->stop();
    oms_adapter->stop();
    prom_exporter.stop();
    // Flush all output sinks (CSV/JSON) before printing the session summary
    // so any buffered writes are visible if a crash follows.
    trade_engine.flush_sinks();
    config.stop_watching();

    auto final_stats = latency_ctrl.get_stats();
    std::cout << "\n\n  =========================================================\n";
    std::cout << "  SESSION SUMMARY\n";
    std::cout << "  ---------------------------------------------------------\n";
    uint64_t final_variance_n;
    {
        std::lock_guard<std::mutex> lk(variance_mutex);
        final_variance_n = variance_n;
    }
    std::cout << "  Tokens processed : " << final_variance_n << "\n";
    std::cout << "  Signals emitted  : " << trade_engine.get_stats().signals_generated.load() << "\n";
    {
        const auto& frs = risk_mgr.get_stats();
        auto fsat = [](uint64_t a, uint64_t b) -> uint64_t {
            return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
        };
        uint64_t fblocked = frs.signals_blocked_magnitude.load();
        fblocked = fsat(fblocked, frs.signals_blocked_confidence.load());
        fblocked = fsat(fblocked, frs.signals_blocked_rate.load());
        fblocked = fsat(fblocked, frs.signals_blocked_drawdown.load());
        fblocked = fsat(fblocked, frs.signals_blocked_position.load());
        fblocked = fsat(fblocked, frs.signals_blocked_pnl.load());
        std::cout << "  Signals blocked  : " << fblocked << "\n";
    }
    std::cout << "  Memory sink size : " << memory_sink->get_signals().size() << "\n";
    std::cout << "  Avg latency      : " << final_stats.avg_latency.count() << "us\n";
    std::cout << "  Min latency      : " << final_stats.min_latency.count() << "us\n";
    std::cout << "  P50 latency      : " << final_stats.p50_latency.count() << "us\n";
    std::cout << "  P95 latency      : " << final_stats.p95_latency.count() << "us\n";
    std::cout << "  P99 latency      : " << final_stats.p99_latency.count() << "us\n";
    std::cout << "  Max latency      : " << final_stats.max_latency.count() << "us\n";
    std::cout << "  Avg sig strength : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().avg_signal_strength.load() << "\n";
    std::cout << "  Jitter           : " << std::fixed << std::setprecision(3)
              << final_stats.jitter_ms << "ms\n";
    {
        auto ads = llm_adapter.get_stats();
        uint64_t hit_pct2 = (ads.tokens_processed > 0)
            ? (ads.cache_hits * 100 / ads.tokens_processed) : 0;
        std::cout << "  Cache hit rate   : " << hit_pct2 << "% ("
                  << ads.cache_hits << "/" << ads.tokens_processed << ")\n";
    }
    std::cout << "  Signals passed   : " << risk_mgr.get_stats().signals_passed.load() << "\n";
    std::cout << "  ---------------------------------------------------------\n\n";

    trade_engine.flush_sinks();
    logger.log_performance_summary();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "\n[FATAL] Unhandled exception: " << ex.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\n[FATAL] Unknown exception" << std::endl;
    return 1;
  }
}
