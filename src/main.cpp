#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include "OutputSinkImpl.h"
#ifdef LLMQUANT_DEDUP_ENABLED
#  include "Deduplicator.h"
#endif
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
#  include "LLMStreamClient.h"
#endif
#include "OmsAdapter.h"
#ifdef LLMQUANT_REST_OMS_ENABLED
#  include "RestOmsAdapter.h"
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
#  include "FixOmsAdapter.h"
#endif
#ifdef LLMQUANT_MOCK_OMS_ENABLED
#  include "MockOmsAdapter.h"
#endif
#ifdef LLMQUANT_PROMETHEUS_ENABLED
#  include "PrometheusExporter.h"
#endif
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
#  include "SignalAuditLog.h"
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
#  include "PipelineCircuitBreaker.h"
#endif
#ifdef LLMQUANT_KELLY_SIZER_ENABLED
#  include "KellyPositionSizer.h"
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
#  include "HealthServer.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
#  include "AdaptiveCooldownController.h"
#endif
#ifdef LLMQUANT_SIGNAL_BLEND_ENABLED
#  include "SignalBlendLayer.h"
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
#  include "StaleTokenDetector.h"
#endif
#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
#  include "RegimeDetector.h"
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED
#  include "RegimeTransitionModel.h"
#endif
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
#  include "TokenEntropyMonitor.h"
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
#  include "NarrativeChangeDetector.h"
#endif
#ifdef LLMQUANT_TRADING_HOURS_ENABLED
#  include "TradingHoursGuard.h"
#endif
#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
#  include "SignalCorrelationTracker.h"
#endif
#ifdef LLMQUANT_WARMUP_SEQUENCER_ENABLED
#  include "WarmupSequencer.h"
#endif
#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
#  include "DrawdownProtector.h"
#endif
#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
#  include "MultiTimeframeAggregator.h"
#endif
#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
#  include "VolatilityForecaster.h"
#endif
#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
#  include "BayesianSignalFilter.h"
#endif
#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
#  include "AnomalyDetector.h"
#endif
#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
#  include "TokenBurstDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
#  include "SignalPersistenceTracker.h"
#endif
#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
#  include "RollingSharpeBiasTracker.h"
#endif
#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
#  include "OrderBookSimulator.h"
#endif
#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
#  include "TokenSentimentHeatmap.h"
#endif
#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
#  include "SentimentMomentumFilter.h"
#endif
#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
#  include "PositionTracker.h"
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
#  include "SignalDecayEnvelope.h"
#endif
#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
#  include "LatencyBudgetEnforcer.h"
#endif
#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
#  include "PnLAttributionEngine.h"
#endif
#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
#  include "PortfolioHeatMonitor.h"
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
#  include "ContextWindowBudget.h"
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
#  include "FractalDimensionEstimator.h"
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
#  include "MarketMicrostructureFilter.h"
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
#  include "SignalEnsembleLayer.h"
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
#  include "SignalMomentumOscillator.h"
#endif
#ifdef LLMQUANT_CVAR_ENABLED
#  include "CVaRCalculator.h"
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
#  include "TemporalPatternLibrary.h"
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
#  include "FeedbackLoopDetector.h"
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
#  include "SentimentCycleDetector.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
#  include "AdaptiveSamplingController.h"
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
#  include "MutualInformationEstimator.h"
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
#  include "SignalBlindSpotDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
#  include "SignalSurpriseIndex.h"
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
#  include "TokenStreamHealthMonitor.h"
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
#  include "RegimeAwareSizer.h"
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
#  include "ConfidenceDecayTracker.h"
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
#  include "CrossAssetCorrelationMonitor.h"
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
#  include "TokenVelocityTracker.h"
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
#  include "NarrativeMomentumClock.h"
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
#  include "AdaptiveVelocityBreaker.h"
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
#  include "SignalCalibrationEngine.h"
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
#  include "TokenBiasHeatmap.h"
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
#  include "OrderFlowImbalanceDetector.h"
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
#  include "CrossSessionMemory.h"
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
#  include "MarketRegimeProbabilityEstimator.h"
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
#  include "SignalReplayBuffer.h"
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
#  include "TokenNgramProfiler.h"
#endif
#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
#  include "ExecutionQualityMonitor.h"
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
#  include "SentimentDispersionIndex.h"
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
#  include "SentimentDivergenceDetector.h"
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
#  include "TokenInfluenceAttributor.h"
#endif
#ifdef LLMQUANT_WALK_FORWARD_ENABLED
#  include "WalkForwardValidator.h"
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
#  include "AdversarialInputDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
#  include "SignalConfidenceInterval.h"
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
#  include "SentimentPersistenceMatrix.h"
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
#  include "CausalImpactEstimator.h"
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
#  include "OptionsFlowSentimentBridge.h"
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
#  include "SentimentPhasePortrait.h"
#endif
#include "llmquant_version.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#else
#  include <unistd.h>   // sysconf(_SC_CLK_TCK) for CPU fraction on Linux
#endif

using namespace llmquant;

/// @brief Returns the process RSS (resident set size) in bytes, or 0 if unavailable.
static uint64_t get_process_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<uint64_t>(pmc.WorkingSetSize);
    return 0;
#else
    // Linux/macOS: parse VmRSS from /proc/self/status (kB units).
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            uint64_t kb = 0;
            std::sscanf(line.c_str(), "VmRSS: %llu kB", &kb);
            return kb * 1024ULL;
        }
    }
    return 0;
#endif
}

/// @brief Returns process CPU usage as a fraction [0.0, N_cores] since last call.
///        Returns 0.0 if unavailable.  Not async-signal-safe; call from monitoring thread only.
static double get_process_cpu_fraction() {
#ifdef _WIN32
    static bool  win_initialized = false;
    static FILETIME prev_kernel{}, prev_user{}, prev_wall{};
    FILETIME creation, exit_ft, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit_ft, &kernel, &user))
        return 0.0;
    FILETIME now_ft;
    GetSystemTimeAsFileTime(&now_ft);
    // On first call: seed previous values and return 0 to avoid cumulative-uptime spike.
    if (!win_initialized) {
        win_initialized = true;
        prev_kernel = kernel; prev_user = user; prev_wall = now_ft;
        return 0.0;
    }
    auto to_u64 = [](FILETIME ft) -> uint64_t {
        return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    uint64_t k = to_u64(kernel), u = to_u64(user), w = to_u64(now_ft);
    uint64_t dk = k - to_u64(prev_kernel);
    uint64_t du = u - to_u64(prev_user);
    uint64_t dw = w - to_u64(prev_wall);
    prev_kernel = kernel; prev_user = user; prev_wall = now_ft;
    if (dw == 0) return 0.0;
    return static_cast<double>(dk + du) / static_cast<double>(dw);
#else
    // Linux: read /proc/self/stat fields utime+stime (jiffies), compare with wall clock.
    static uint64_t prev_cpu_jiffies = UINT64_MAX;  // sentinel: UINT64_MAX = uninitialized
    static std::chrono::steady_clock::time_point prev_tp = std::chrono::steady_clock::now();
    std::ifstream f("/proc/self/stat");
    if (!f.is_open()) return 0.0;
    std::string stat_line;
    std::getline(f, stat_line);
    // Fields 14 and 15 (1-indexed) are utime and stime; skip past the comm field.
    auto rp = stat_line.rfind(')');
    if (rp == std::string::npos) return 0.0;
    std::istringstream iss(stat_line.substr(rp + 2));
    uint64_t utime = 0, stime = 0;
    std::string tok;
    try {
        for (int i = 3; i <= 15; ++i) {
            if (!(iss >> tok)) break;
            if (i == 14) utime = std::stoull(tok);
            if (i == 15) stime = std::stoull(tok);
        }
    } catch (...) { return 0.0; }
    uint64_t cpu_jiffies = utime + stime;
    auto now = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(now - prev_tp).count();
    // On first call prev_cpu_jiffies == UINT64_MAX (sentinel): seed and return 0.
    if (prev_cpu_jiffies == UINT64_MAX) {
        prev_cpu_jiffies = cpu_jiffies;
        prev_tp = now;
        return 0.0;
    }
    double delta_jiffies = static_cast<double>(cpu_jiffies - prev_cpu_jiffies);
    prev_cpu_jiffies = cpu_jiffies;
    prev_tp = now;
    if (elapsed_s <= 0.0) return 0.0;
    long hz = sysconf(_SC_CLK_TCK);
    return delta_jiffies / (elapsed_s * static_cast<double>(hz > 0 ? hz : 100));
#endif
}

std::atomic<bool> g_running{true};

void signal_handler(int /*signal*/) {
    // std::cout is not async-signal-safe; only set the atomic flag here.
    // The main loop detects g_running==false and prints the shutdown summary.
    g_running = false;
}

int main(int argc, char* argv[]) {
  try {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Record engine start time for uptime metrics.
    const auto engine_start_time = std::chrono::steady_clock::now();
    // Also capture a system_clock snapshot so Prometheus can expose the absolute
    // start timestamp (system_clock and steady_clock have different epochs).
    const int64_t engine_start_unix_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Parse flags before anything else.
    bool        stream_mode    = false;
    std::string stream_api_key;
    bool        no_color       = false;
    bool        debug_raw      = false;
    bool        dry_run         = false;
    bool        backtest_mode   = false;
    bool        list_tokens     = false;
    bool        dump_config     = false;
    bool        validate_config = false;
    bool        quiet           = false;
    std::string export_dict_path;   // non-empty = write TSV to file and exit
    std::string oms_address;
    std::string fix_address;
    std::string config_file    = "config.yaml"; // may be overridden by --config
    uint16_t    stats_port_override = 0;        // 0 = use config value
    int         token_interval_override = 0;    // 0 = use config value
    std::string log_level_str  = "info";        // spdlog level name
    int         stats_interval_ms  = 1000;      // monitoring loop tick period
    bool        no_prometheus  = false;         // skip Prometheus exporter
#ifdef LLMQUANT_DEDUP_ENABLED
    bool        no_dedup       = false;         // disable deduplication at runtime
#endif
    bool        no_hot_reload  = false;         // skip config file watcher
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    std::string audit_log_path;                  // non-empty = enable audit log at this path
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    uint16_t    health_port_override = 0;        // 0 = use default (8080)
    bool        no_health_server     = false;    // skip health server
#endif
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: LLMTokenStreamQuantEngine [config.yaml] [options]\n"
                "\n"
                "Options:\n"
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
                "  --stream [key]    Enable live LLM stream mode (optional API key)\n"
#endif
                "  --oms host:port   Connect to REST OMS adapter\n"
                "  --fix host:port   Connect to FIX 4.2 OMS adapter\n"
                "  --config path     Path to config YAML (default: config.yaml)\n"
                "  --stats-port N    Override Prometheus metrics port (default: from config, 9100)\n"
                "  --token-interval N  Override token_stream.token_interval_ms (ms between tokens, min 1)\n"
                "  --log-level LEVEL Set spdlog log level: trace|debug|info|warn|error|critical (default: info)\n"
                "  --dry-run         Process tokens through LLMAdapter only; skip signal emission\n"
                "  --backtest        Enable backtest mode (emit signal on every token, no cooldown)\n"
                "  --no-color        Disable ANSI colour output\n"
                "  --debug-raw       Print raw LLM stream bytes\n"
                "  --list-tokens     Print the full semantic dictionary and exit\n"
                "  --export-dict FILE  Export semantic dictionary to a TSV file and exit\n"
                "  --dump-config     Print effective configuration and exit\n"
                "  --validate-config Validate configuration, print any errors, exit 0=OK 1=invalid\n"
                "  --quiet           Suppress console signal/stats output (log-file only)\n"
                "  --stats-interval N  Monitoring loop tick period in ms (default: 1000)\n"
                "  --no-prometheus   Disable the Prometheus /metrics scrape endpoint\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                "  --no-dedup        Disable token deduplication (all tokens treated as novel)\n"
#endif
                "  --no-hot-reload   Disable config file hot-reload watcher\n"
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                "  --audit-log FILE  Write per-signal NDJSON audit log to FILE\n"
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
                "  --health-port N   Override HTTP /health endpoint port (default: 8080)\n"
                "  --no-health       Disable the HTTP /health endpoint\n"
#endif
                "  --version         Print version and exit\n"
                "  --show-flags      Print compile-time feature flags and exit\n"
                "  --help            Print this help and exit\n"
                "\n"
                "Environment:\n"
                "  LLMQUANT_API_KEY        LLM API key (fallback when --stream has no key)\n"
                "  LLMQUANT_NO_PROMETHEUS  Set to 1/true/yes to disable Prometheus endpoint\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                "  LLMQUANT_NO_DEDUP       Set to 1/true/yes to disable token deduplication\n"
#endif
                "  LLMQUANT_NO_HOT_RELOAD  Set to 1/true/yes to disable config hot-reload\n"
                "  LLMQUANT_DRY_RUN        Set to 1/true/yes for dry-run (signal only, no OMS)\n"
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                "  LLMQUANT_AUDIT_LOG_PATH Path to NDJSON audit log file\n"
#endif
                "  LLMQUANT_QUIET          Set to 1/true/yes to suppress console output\n"
                "  LLMQUANT_BACKTEST       Set to 1/true/yes to enable backtest mode\n"
                "\n"
                "Config file (YAML) keys: token_stream, trading, latency, logging,\n"
                "  pressure, risk_thresholds, risk (override flags).\n";
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "LLMTokenStreamQuantEngine " << LLMQUANT_VERSION
                      << " (" << LLMQUANT_GIT_COMMIT
                      << ", " << LLMQUANT_BUILD_TIMESTAMP << ")\n";
            return 0;
        } else if (arg == "--show-flags") {
            // Print all compile-time feature flag states and exit.
            // Useful for verifying embedded/minimal builds have the expected features.
            std::cout << "Compile-time feature flags:\n"
#ifdef LLMQUANT_PROMETHEUS_ENABLED
                      << "  LLMQUANT_ENABLE_PROMETHEUS    ON\n"
#else
                      << "  LLMQUANT_ENABLE_PROMETHEUS    OFF\n"
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
                      << "  LLMQUANT_ENABLE_FIX_OMS       ON\n"
#else
                      << "  LLMQUANT_ENABLE_FIX_OMS       OFF\n"
#endif
#ifdef LLMQUANT_REST_OMS_ENABLED
                      << "  LLMQUANT_ENABLE_REST_OMS      ON\n"
#else
                      << "  LLMQUANT_ENABLE_REST_OMS      OFF\n"
#endif
#ifdef LLMQUANT_DEDUP_ENABLED
                      << "  LLMQUANT_ENABLE_DEDUP         ON\n"
#else
                      << "  LLMQUANT_ENABLE_DEDUP         OFF\n"
#endif
#ifdef LLMQUANT_PROFILING_ENABLED
                      << "  LLMQUANT_ENABLE_PROFILING     ON\n"
#else
                      << "  LLMQUANT_ENABLE_PROFILING     OFF\n"
#endif
#ifdef LLMQUANT_JSON_STATS_SUMMARY
                      << "  LLMQUANT_ENABLE_JSON_STATS    ON\n"
#else
                      << "  LLMQUANT_ENABLE_JSON_STATS    OFF\n"
#endif
#ifdef LLMQUANT_TLS_ENABLED
                      << "  LLMQUANT_ENABLE_TLS           ON\n"
#else
                      << "  LLMQUANT_ENABLE_TLS           OFF\n"
#endif
#ifdef LLMQUANT_REDIS_ENABLED
                      << "  LLMQUANT_ENABLE_REDIS         ON\n"
#else
                      << "  LLMQUANT_ENABLE_REDIS         OFF\n"
#endif
#ifdef LLMQUANT_HOT_RELOAD_ENABLED
                      << "  LLMQUANT_ENABLE_HOT_RELOAD    ON\n"
#else
                      << "  LLMQUANT_ENABLE_HOT_RELOAD    OFF\n"
#endif
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
                      << "  LLMQUANT_ENABLE_STREAM_CLIENT ON\n"
#else
                      << "  LLMQUANT_ENABLE_STREAM_CLIENT OFF\n"
#endif
#ifdef LLMQUANT_SIMD_DISABLED
                      << "  LLMQUANT_ENABLE_SIMD          OFF\n"
#else
                      << "  LLMQUANT_ENABLE_SIMD          ON (SSE2"
#  ifdef LLMQUANT_HAVE_SSE2
                      << " active)\n"
#  else
                      << " not detected — scalar fallback)\n"
#  endif
#endif
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
                      << "  LLMQUANT_ENABLE_SIGNAL_TRACE  ON\n"
#else
                      << "  LLMQUANT_ENABLE_SIGNAL_TRACE  OFF\n"
#endif
                      ;
            return 0;
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
        } else if (arg == "--stream") {
            stream_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                stream_api_key = argv[++i];  // explicit key provided on CLI
#endif
        } else if (arg == "--no-color") {
            no_color = true;
        } else if (arg == "--debug-raw") {
            debug_raw = true;
        } else if (arg == "--list-tokens") {
            list_tokens = true;
        } else if (arg == "--export-dict" && i + 1 < argc) {
            export_dict_path = argv[++i];
        } else if (arg == "--dump-config") {
            dump_config = true;
        } else if (arg == "--validate-config") {
            validate_config = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--backtest") {
            backtest_mode = true;
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--stats-port" && i + 1 < argc) {
            try {
                int p = std::stoi(argv[++i]);
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                stats_port_override = static_cast<uint16_t>(p);
            }
            catch (...) { std::cerr << "error: --stats-port requires an integer in range 1-65535\n"; return 1; }
        } else if (arg == "--token-interval" && i + 1 < argc) {
            try { token_interval_override = std::max(1, std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --token-interval requires an integer\n"; return 1; }
        } else if (arg == "--log-level" && i + 1 < argc) {
            log_level_str = argv[++i];
        } else if (arg == "--stats-interval" && i + 1 < argc) {
            try {
                stats_interval_ms = std::clamp(std::stoi(argv[++i]), 100, 60000);
            }
            catch (...) { std::cerr << "error: --stats-interval requires an integer\n"; return 1; }
        } else if (arg == "--no-prometheus") {
            no_prometheus = true;
#ifdef LLMQUANT_DEDUP_ENABLED
        } else if (arg == "--no-dedup") {
            no_dedup = true;
#endif
        } else if (arg == "--no-hot-reload") {
            no_hot_reload = true;
        } else if (arg == "--oms" && i + 1 < argc) {
            oms_address = argv[++i];
        } else if (arg == "--fix" && i + 1 < argc) {
            fix_address = argv[++i];
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        } else if (arg == "--audit-log" && i + 1 < argc) {
            audit_log_path = argv[++i];
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
        } else if (arg == "--health-port" && i + 1 < argc) {
            try {
                int p = std::stoi(argv[++i]);
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                health_port_override = static_cast<uint16_t>(p);
            }
            catch (...) { std::cerr << "error: --health-port requires an integer in range 1-65535\n"; return 1; }
        } else if (arg == "--no-health") {
            no_health_server = true;
#endif
        }
    }

    // Environment variable overrides for runtime feature flags.
    // Useful in containerised / Kubernetes deployments where editing the command
    // line is inconvenient.  CLI flags take precedence; env vars only set the flag
    // when the CLI has NOT already set it.
    //
    // LLMQUANT_NO_PROMETHEUS=1   equivalent to --no-prometheus
    // LLMQUANT_NO_DEDUP=1        equivalent to --no-dedup
    // LLMQUANT_NO_HOT_RELOAD=1   equivalent to --no-hot-reload
    // LLMQUANT_DRY_RUN=1         equivalent to --dry-run
    // LLMQUANT_QUIET=1           equivalent to --quiet
    // LLMQUANT_BACKTEST=1        equivalent to --backtest
    {
        auto env_flag = [](const char* name) -> bool {
#ifdef _WIN32
            char buf[8] = {};
            size_t sz = 0;
            if (getenv_s(&sz, buf, sizeof(buf), name) != 0 || sz == 0) return false;
            const char* v = buf;
#else
            const char* v = std::getenv(name);
#endif
            return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
        };
        if (!no_prometheus  && env_flag("LLMQUANT_NO_PROMETHEUS"))  no_prometheus  = true;
#ifdef LLMQUANT_DEDUP_ENABLED
        if (!no_dedup       && env_flag("LLMQUANT_NO_DEDUP"))       no_dedup       = true;
#endif
        if (!no_hot_reload  && env_flag("LLMQUANT_NO_HOT_RELOAD"))  no_hot_reload  = true;
        if (!dry_run        && env_flag("LLMQUANT_DRY_RUN"))        dry_run        = true;
        if (!quiet          && env_flag("LLMQUANT_QUIET"))          quiet          = true;
        if (!backtest_mode  && env_flag("LLMQUANT_BACKTEST"))       backtest_mode  = true;
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        // LLMQUANT_AUDIT_LOG_PATH=<file>  set audit log path when --audit-log was not passed
        if (audit_log_path.empty()) {
#ifdef _WIN32
            char env_buf[512] = {};
            size_t env_sz = 0;
            if (getenv_s(&env_sz, env_buf, sizeof(env_buf), "LLMQUANT_AUDIT_LOG_PATH") == 0
                    && env_sz > 0)
                audit_log_path = env_buf;
#else
            const char* env_p = std::getenv("LLMQUANT_AUDIT_LOG_PATH");
            if (env_p && env_p[0] != '\0') audit_log_path = env_p;
#endif
        }
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
        if (!no_health_server && env_flag("LLMQUANT_NO_HEALTH")) no_health_server = true;
#endif
    }

    // Apply log level before any spdlog calls so early warnings are visible.
    {
        auto level = spdlog::level::from_str(log_level_str);
        // from_str returns off for unknown names — warn and fall back to info.
        if (level == spdlog::level::off && log_level_str != "off") {
            spdlog::warn("Unknown --log-level '{}'; defaulting to info", log_level_str);
            level = spdlog::level::info;
        }
        spdlog::set_level(level);
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
    // Apply CLI overrides before capturing sys_config so all subsystems see the
    // effective values (token_sim is constructed later but reads from sys_config).
    if (token_interval_override > 0) {
        config.set_token_interval_ms(token_interval_override);
        spdlog::info("--token-interval: overriding token_interval_ms to {}ms", token_interval_override);
    }

    const auto& sys_config = config.get_config();

    // --dump-config: print effective configuration and exit.
    if (dump_config) {
        const auto& ts  = sys_config.token_stream;
        const auto& tr  = sys_config.trading;
        const auto& lat = sys_config.latency;
        const auto& log = sys_config.logging;
        const auto& met = sys_config.metrics;
        const auto& rt  = sys_config.risk_thresholds;
        std::cout << "# Effective configuration loaded from: " << config_file << "\n"
                  << "token_stream.use_memory_stream:       " << (ts.use_memory_stream ? "true" : "false") << "\n"
                  << "token_stream.token_interval_ms:       " << ts.token_interval_ms << "\n"
                  << "token_stream.buffer_size:             " << ts.buffer_size << "\n"
                  << "token_stream.data_file_path:          " << ts.data_file_path << "\n"
                  << "token_stream.dedup_ttl_ms:            " << ts.dedup_ttl_ms
                      << (ts.dedup_ttl_ms == 0 ? " (auto: 10x token_interval_ms)" : "") << "\n"
                  << "trading.bias_sensitivity:             " << tr.bias_sensitivity << "\n"
                  << "trading.volatility_sensitivity:       " << tr.volatility_sensitivity << "\n"
                  << "trading.signal_decay_rate:            " << tr.signal_decay_rate << "\n"
                  << "trading.signal_cooldown_us:           " << tr.signal_cooldown_us << "\n"
                  << "trading.max_signal_age_us:            " << tr.max_signal_age_us << "\n"
                  << "trading.min_bias_threshold:           " << tr.min_bias_threshold << "\n"
                  << "trading.max_accumulated_bias:         " << tr.max_accumulated_bias << "\n"
                  << "latency.target_latency_us:            " << lat.target_latency_us << "\n"
                  << "latency.sample_window:                " << lat.sample_window << "\n"
                  << "latency.enable_profiling:             " << (lat.enable_profiling ? "true" : "false") << "\n"
                  << "logging.log_file_path:                " << log.log_file_path << "\n"
                  << "logging.format:                       " << log.format << "\n"
                  << "logging.enable_console:               " << (log.enable_console ? "true" : "false") << "\n"
                  << "logging.flush_interval_ms:            " << log.flush_interval_ms << "\n"
                  << "metrics.stats_port:                   " << met.stats_port << "\n"
                  << "metrics.bind_address:                 " << met.bind_address << "\n"
                  << "risk_thresholds.max_bias_magnitude:   " << rt.max_bias_magnitude << "\n"
                  << "risk_thresholds.max_volatility_magnitude: " << rt.max_volatility_magnitude << "\n"
                  << "risk_thresholds.min_confidence:       " << rt.min_confidence << "\n"
                  << "risk_thresholds.max_signals_per_second: " << rt.max_signals_per_second << "\n"
                  << "risk_thresholds.max_drawdown:         " << rt.max_drawdown << "\n"
                  << "risk_thresholds.drawdown_window_s:    " << rt.drawdown_window_s << "\n";
        const auto& pr = sys_config.pressure;
        const auto& sw = sys_config.semantic_weights;
        std::cout << "pressure.max_ingestion_rate_tps:      " << pr.max_ingestion_rate_tps << "\n"
                  << "pressure.backoff_scale_factor:        " << pr.backoff_scale_factor << "\n"
                  << "semantic_weights.sentiment_multiplier: " << sw.sentiment_multiplier << "\n"
                  << "semantic_weights.confidence_multiplier: " << sw.confidence_multiplier << "\n"
                  << "semantic_weights.volatility_multiplier: " << sw.volatility_multiplier << "\n"
                  << "semantic_weights.bias_multiplier:     " << sw.bias_multiplier << "\n";
        // Print non-default fields as a diff section for quick operator review.
        auto diffs = config.diff_from_defaults();
        if (!diffs.empty()) {
            std::cout << "\n# Non-default fields (changed from compiled defaults):\n";
            for (const auto& d : diffs)
                std::cout << "  " << d << "\n";
        } else {
            std::cout << "\n# All fields are at compiled defaults.\n";
        }
        return 0;
    }

    // --validate-config: run the validation suite and report errors.
    if (validate_config) {
        auto errors = config.validate();
        if (errors.empty()) {
            std::cout << "Config OK: " << config_file << " is valid.\n";
            return 0;
        }
        std::cerr << "Config INVALID: " << errors.size() << " error(s) in " << config_file << ":\n";
        for (const auto& e : errors)
            std::cerr << "  - " << e << "\n";
        return 1;
    }

#ifdef LLMQUANT_DEDUP_ENABLED
    // Deduplication layer: skip repeated tokens within a sliding TTL window.
    // Dedup TTL: use config value when set (> 0), else default to 10× the token interval.
    const int dedup_ttl_ms = (sys_config.token_stream.dedup_ttl_ms > 0)
        ? sys_config.token_stream.dedup_ttl_ms
        : sys_config.token_stream.token_interval_ms * 10;
    // Backend selection: Redis (distributed) when redis_url is configured and
    // LLMQUANT_ENABLE_REDIS=ON; otherwise fall back to in-process.
    std::shared_ptr<llmquant::DeduplicatorBackend> dedup_backend;
#if defined(LLMQUANT_REDIS_ENABLED)
    if (!sys_config.token_stream.redis_url.empty()) {
        auto redis_backend = std::make_shared<llmquant::RedisDeduplicator>(
            sys_config.token_stream.redis_url);
        if (redis_backend->is_connected()) {
            spdlog::info("[dedup] using Redis backend: {}", sys_config.token_stream.redis_url);
            redis_backend->set_disconnect_callback([](const std::string& err) {
                spdlog::warn("[dedup] Redis disconnected: {}; falling back to in-process", err);
            });
            dedup_backend = std::move(redis_backend);
        } else {
            spdlog::warn("[dedup] Redis connection failed ({}); falling back to in-process",
                         sys_config.token_stream.redis_url);
        }
    }
#endif
    if (!dedup_backend) {
        auto ip = std::make_shared<llmquant::InProcessDeduplicator>();
        // Prevent unbounded memory growth: purge expired entries every 60 s.
        ip->start_background_purge(60);
        dedup_backend = std::move(ip);
    }
    llmquant::Deduplicator deduplicator(dedup_backend,
        std::chrono::milliseconds(dedup_ttl_ms));
#endif

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
        // When compiled with LLMQUANT_ENABLE_PROFILING=OFF the sample ring buffer
        // is always disabled regardless of the YAML/env runtime setting.
#ifdef LLMQUANT_PROFILING_ENABLED
        .enable_profiling = sys_config.latency.enable_profiling
#else
        .enable_profiling = false
#endif
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

    // --list-tokens: dump the full semantic dictionary and exit immediately.
    if (list_tokens) {
        auto keys = llm_adapter.get_all_token_keys();
        std::cout << "token\tsentiment\tconfidence\tvolatility\tbias\n";
        for (const auto& k : keys) {
            SemanticWeight w;
            (void)llm_adapter.get_token_mapping(k, w);
            std::cout << k
                      << "\t" << std::fixed << std::setprecision(3) << w.sentiment_score
                      << "\t" << w.confidence_score
                      << "\t" << w.volatility_score
                      << "\t" << w.directional_bias
                      << "\n";
        }
        std::cout << "-- " << keys.size() << " entries --\n";
        return 0;
    }

    // --export-dict FILE: write the semantic dictionary to a TSV file and exit.
    if (!export_dict_path.empty()) {
        std::string tsv = llm_adapter.export_dictionary();
        std::ofstream out(export_dict_path);
        if (!out) {
            spdlog::error("--export-dict: cannot open '{}' for writing", export_dict_path);
            return 1;
        }
        out << tsv;
        std::cout << "Exported " << llm_adapter.get_dictionary_size()
                  << " entries to " << export_dict_path << "\n";
        return 0;
    }

    TradeSignalEngine trade_engine({
        .bias_sensitivity     = sys_config.trading.bias_sensitivity,
        .volatility_sensitivity = sys_config.trading.volatility_sensitivity,
        .signal_decay_rate    = sys_config.trading.signal_decay_rate,
        .signal_cooldown      = std::chrono::microseconds(sys_config.trading.signal_cooldown_us),
        .max_signal_age_us    = sys_config.trading.max_signal_age_us,
        .min_bias_threshold   = sys_config.trading.min_bias_threshold,
        .min_vol_threshold    = sys_config.trading.min_vol_threshold,
        .max_accumulated_bias = sys_config.trading.max_accumulated_bias
    });

    // Backtest mode: emit on every token, ignoring the cooldown timer.
    if (backtest_mode) {
        trade_engine.set_backtest_mode(true);
    }

    // Wire an in-memory sink for telemetry (signals accessible for inspection/export).
    auto memory_sink = std::make_shared<llmquant::MemoryOutputSink>();
    trade_engine.add_output_sink(memory_sink);

    // Semantic weight multipliers as atomics so the hot-reload callback can update
    // them without a mutex on the process_token hot path.
    std::atomic<double> sem_mult_sentiment{sys_config.semantic_weights.sentiment_multiplier};
    std::atomic<double> sem_mult_confidence{sys_config.semantic_weights.confidence_multiplier};
    std::atomic<double> sem_mult_volatility{sys_config.semantic_weights.volatility_multiplier};
    std::atomic<double> sem_mult_bias{sys_config.semantic_weights.bias_multiplier};

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
    risk_cfg.dry_run_mode             = sys_config.risk_overrides.dry_run_mode;
    llmquant::RiskManager risk_mgr(risk_cfg);
    risk_mgr.set_metrics_logger(&logger);
    // Register gate trip-wire callbacks for real-time alerting on first block.
    // Each callback fires once per pass→block edge; subsequent consecutive blocks
    // on the same gate are silent until the gate passes and trips again.
    for (const char* gate : {"magnitude", "confidence", "rate", "drawdown", "position"}) {
        risk_mgr.set_gate_trip_callback(gate, [gate](const std::string& /*g*/,
                                                      const llmquant::TradeSignal& sig) {
            spdlog::warn("[risk] Gate '{}' tripped: bias={:+.3f} vol={:.3f} conf={:.3f}",
                         gate, sig.delta_bias_shift, sig.volatility_adjustment, sig.confidence);
        });
    }

    // Hot-reload watcher is started after token_sim is constructed (below) so
    // the callback can also reload the token file when data_file_path changes.

    // OMS adapter: MockOmsAdapter by default; REST via --oms, FIX 4.2 via --fix.
    std::unique_ptr<llmquant::OmsAdapter> oms_adapter;
    if (!fix_address.empty()) {
#ifdef LLMQUANT_FIX_OMS_ENABLED
        llmquant::FixOmsAdapter::Config fix_cfg;
        size_t colon = fix_address.find(':');
        if (colon != std::string::npos) {
            fix_cfg.host = fix_address.substr(0, colon);
            try {
                int p = std::stoi(fix_address.substr(colon + 1));
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                fix_cfg.port = static_cast<uint16_t>(p);
            }
            catch (...) { spdlog::error("--fix: invalid port in '{}' (must be 1-65535)", fix_address); return 1; }
        } else {
            fix_cfg.host = fix_address;
        }
        oms_adapter = std::make_unique<llmquant::FixOmsAdapter>(fix_cfg);
#else
        spdlog::error("--fix requested but FIX OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_FIX_OMS=OFF).");
        return 1;
#endif
    } else if (!oms_address.empty()) {
#ifdef LLMQUANT_REST_OMS_ENABLED
        std::string endpoint = oms_address;
        llmquant::RestOmsAdapter::Config oms_cfg;
        size_t colon = endpoint.find(':');
        if (colon != std::string::npos) {
            oms_cfg.host = endpoint.substr(0, colon);
            try {
                int p = std::stoi(endpoint.substr(colon + 1));
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                oms_cfg.port = static_cast<uint16_t>(p);
            }
            catch (...) { spdlog::error("--oms: invalid port in '{}' (must be 1-65535)", endpoint); return 1; }
        } else {
            oms_cfg.host = endpoint;
        }
        oms_adapter = std::make_unique<llmquant::RestOmsAdapter>(oms_cfg);
#else
        spdlog::error("--oms requested but REST OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_REST_OMS=OFF).");
        return 1;
#endif
    } else {
#ifdef LLMQUANT_MOCK_OMS_ENABLED
        auto mock = std::make_unique<llmquant::MockOmsAdapter>();
        mock->load_states({
            {0.1,  1.0,  0.5, -10.0},
            {0.25, 1.0,  0.3, -10.0},
            {-0.1, 1.0, -0.2, -10.0},
        });
        oms_adapter = std::move(mock);
#else
        spdlog::error("No OMS adapter specified and MockOmsAdapter is disabled "
                      "(LLMQUANT_ENABLE_MOCK_OMS=OFF). Pass --oms or --fix, or enable "
                      "LLMQUANT_ENABLE_MOCK_OMS=ON.");
        return 1;
#endif
    }

    oms_adapter->set_position_callback([&](const llmquant::RiskManager::PositionState& state) {
        risk_mgr.update_position(state);
    });
    // OMS alert callback wired after signal callback is registered (see below).
    if (!oms_adapter->start())
        spdlog::warn("[oms] start() returned false — adapter may not be polling");

    TokenStreamSimulator token_sim({
        .token_interval = std::chrono::microseconds(sys_config.token_stream.token_interval_ms * 1000),
        .buffer_size = sys_config.token_stream.buffer_size,
        .use_memory_stream = sys_config.token_stream.use_memory_stream,
        .data_file_path = sys_config.token_stream.data_file_path
    });

    // Start config hot-reload watcher now that all pipeline objects exist.
    // The callback can update every subsystem live, including reloading the
    // token file when token_stream.data_file_path changes at runtime.
    // Disabled when --no-hot-reload is passed OR when hot-reload was compiled
    // out with -DLLMQUANT_ENABLE_HOT_RELOAD=OFF (useful in CI/embedded contexts).
#ifndef LLMQUANT_HOT_RELOAD_ENABLED
    (void)no_hot_reload;  // suppress unused-variable warning
    spdlog::info("Config hot-reload watcher compiled out (LLMQUANT_ENABLE_HOT_RELOAD=OFF)");
    if (false) {
#else
    // prev_hot_config: snapshot of the last-seen config used to diff-log what changed on reload.
    llmquant::SystemConfig prev_hot_config = sys_config;
    if (no_hot_reload) {
        spdlog::info("--no-hot-reload: config file watcher disabled");
    } else if (!config.start_watching(config_file, [&risk_mgr, &trade_engine, &token_sim,
                                              &logger, &config_file,
                                              &sem_mult_sentiment, &sem_mult_confidence,
                                              &sem_mult_volatility, &sem_mult_bias,
                                              &prev_hot_config](const llmquant::SystemConfig& updated) {
        // Config diff: log only fields that changed so operators can see exactly what hot-reload applied.
        {
            const auto& o = prev_hot_config;
            const auto& n = updated;
            auto log_ch = [](const char* key, auto ov, auto nv) {
                if (ov != nv) spdlog::info("[config_diff] {}: {} → {}", key, ov, nv);
            };
            log_ch("trading.bias_sensitivity",       o.trading.bias_sensitivity,       n.trading.bias_sensitivity);
            log_ch("trading.volatility_sensitivity", o.trading.volatility_sensitivity, n.trading.volatility_sensitivity);
            log_ch("trading.signal_decay_rate",      o.trading.signal_decay_rate,      n.trading.signal_decay_rate);
            log_ch("trading.signal_cooldown_us",     o.trading.signal_cooldown_us,     n.trading.signal_cooldown_us);
            log_ch("trading.min_bias_threshold",     o.trading.min_bias_threshold,     n.trading.min_bias_threshold);
            log_ch("trading.min_vol_threshold",      o.trading.min_vol_threshold,      n.trading.min_vol_threshold);
            log_ch("trading.max_accumulated_bias",   o.trading.max_accumulated_bias,   n.trading.max_accumulated_bias);
            log_ch("risk.max_bias_magnitude",        o.risk_thresholds.max_bias_magnitude,       n.risk_thresholds.max_bias_magnitude);
            log_ch("risk.max_volatility_magnitude",  o.risk_thresholds.max_volatility_magnitude, n.risk_thresholds.max_volatility_magnitude);
            log_ch("risk.min_confidence",            o.risk_thresholds.min_confidence,           n.risk_thresholds.min_confidence);
            log_ch("risk.max_signals_per_second",    o.risk_thresholds.max_signals_per_second,   n.risk_thresholds.max_signals_per_second);
            log_ch("risk.max_drawdown",              o.risk_thresholds.max_drawdown,             n.risk_thresholds.max_drawdown);
            log_ch("semantic.sentiment_multiplier",  o.semantic_weights.sentiment_multiplier,    n.semantic_weights.sentiment_multiplier);
            log_ch("semantic.confidence_multiplier", o.semantic_weights.confidence_multiplier,   n.semantic_weights.confidence_multiplier);
            log_ch("semantic.volatility_multiplier", o.semantic_weights.volatility_multiplier,   n.semantic_weights.volatility_multiplier);
            log_ch("semantic.bias_multiplier",       o.semantic_weights.bias_multiplier,         n.semantic_weights.bias_multiplier);
            log_ch("latency.target_latency_us",      o.latency.target_latency_us,                n.latency.target_latency_us);
            prev_hot_config = updated;
        }
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
        new_risk_cfg.dry_run_mode             = updated.risk_overrides.dry_run_mode;
        risk_mgr.update_config(new_risk_cfg);
        llmquant::TradeSignalEngine::Config new_eng_cfg;
        new_eng_cfg.bias_sensitivity       = updated.trading.bias_sensitivity;
        new_eng_cfg.volatility_sensitivity = updated.trading.volatility_sensitivity;
        new_eng_cfg.signal_decay_rate      = updated.trading.signal_decay_rate;
        new_eng_cfg.signal_cooldown        = std::chrono::microseconds(updated.trading.signal_cooldown_us);
        new_eng_cfg.max_signal_age_us      = updated.trading.max_signal_age_us;
        new_eng_cfg.min_bias_threshold     = updated.trading.min_bias_threshold;
        new_eng_cfg.min_vol_threshold      = updated.trading.min_vol_threshold;
        new_eng_cfg.max_accumulated_bias   = updated.trading.max_accumulated_bias;
        trade_engine.update_config(new_eng_cfg);
        // Update semantic weight multipliers atomically — visible to process_token
        // on the very next token without requiring a process restart.
        const auto& sw = updated.semantic_weights;
        sem_mult_sentiment.store(sw.sentiment_multiplier,  std::memory_order_relaxed);
        sem_mult_confidence.store(sw.confidence_multiplier, std::memory_order_relaxed);
        sem_mult_volatility.store(sw.volatility_multiplier, std::memory_order_relaxed);
        sem_mult_bias.store(sw.bias_multiplier,            std::memory_order_relaxed);
        // Reload token file when data_file_path changes and not in memory-stream mode.
        if (!updated.token_stream.use_memory_stream) {
            token_sim.load_tokens_from_file(updated.token_stream.data_file_path);
            spdlog::info("[config] Token file reloaded: {}", updated.token_stream.data_file_path);
        }
        // Apply token pacing changes immediately so interval tuning takes
        // effect without a restart.
        if (updated.token_stream.token_interval_ms > 0) {
            token_sim.set_token_interval(
                std::chrono::microseconds(updated.token_stream.token_interval_ms * 1000));
        }
        logger.log_config_reload(config_file, true);
        std::cout << "\n[config] Hot-reloaded: bias_sensitivity="
                  << updated.trading.bias_sensitivity
                  << "  max_bias=" << u.max_bias_magnitude
                  << "  max_signals/s=" << u.max_signals_per_second
                  << "  sem_wts=[" << sw.sentiment_multiplier << ","
                  << sw.confidence_multiplier << "," << sw.volatility_multiplier
                  << "," << sw.bias_multiplier << "]" << std::endl;
    })) {
        spdlog::warn("Config hot-reload watcher failed to start");
    }
#endif // LLMQUANT_HOT_RELOAD_ENABLED

#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
    // Rolling Shannon entropy of token type diversity.
    // Declared before process_token lambda so the lambda can capture it by ref.
    llmquant::TokenEntropyMonitor entropy_monitor;
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
    // Narrative change detector: cosine similarity break for topic-switch events.
    llmquant::NarrativeChangeDetector narrative_detector;
    narrative_detector.set_break_callback([](double sim) {
        spdlog::info("[narrative] topic break detected — cosine_sim={:.3f}; "
                     "LLM may have switched themes", sim);
    });
#endif

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
    // Stale-token watchdog: fires if no token arrives for >30 s (configurable).
    // Declared before process_token lambda so the lambda can capture it by ref.
    llmquant::StaleTokenDetector stale_detector;
    stale_detector.reset();
    stale_detector.set_stale_callback([](int64_t gap_ms) {
        spdlog::error("[stale_detector] LLM token stream SILENT for {}ms — "
                      "no tokens received; check upstream API / network",
                      gap_ms);
    });
    stale_detector.set_recovery_callback([]() {
        spdlog::info("[stale_detector] LLM token stream RECOVERED — tokens flowing again");
    });
#endif

    // These must be declared BEFORE the process_token lambda so the lambda can
    // capture them by reference even though they appear under #ifdef guards.
#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    llmquant::SentimentMomentumFilter sentiment_momentum_filter;
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
    llmquant::SignalDecayEnvelope signal_decay;
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    llmquant::ContextWindowBudget context_budget;
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    llmquant::TemporalPatternLibrary tpl;
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
    llmquant::TokenStreamHealthMonitor stream_health;
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
    llmquant::TokenBiasHeatmap token_bias_heatmap;
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    llmquant::OrderFlowImbalanceDetector order_flow_detector;
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    llmquant::TokenInfluenceAttributor token_influence;
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    llmquant::TokenNgramProfiler ngram_profiler;
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    llmquant::AdversarialInputDetector adversarial_detector;
#endif

    // Shared token processing lambda used by both the simulator and the
    // LLMStreamClient paths.  Encapsulates dedup, latency, logging, and
    // semantic-weight pipeline so neither call site duplicates logic.
    auto process_token = [&](const std::string& text, uint64_t seq_id) {
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
        entropy_monitor.record(std::hash<std::string>{}(text));
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
        narrative_detector.record(std::hash<std::string>{}(text));
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
        // Feed raw token text into the phrase matcher; fires on complete patterns.
        tpl.push_token(text);
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
        context_budget.consume(1);
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        stream_health.ping();
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.record_token();
#endif
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("[trace] token seq={} text={}", seq_id, text);
#endif
#ifdef LLMQUANT_DEDUP_ENABLED
        // Skip duplicate tokens within the dedup window (unless --no-dedup).
        if (!no_dedup) {
            auto dedup_result = deduplicator.check(text);
            logger.log_dedup_event(text, dedup_result == llmquant::DedupResult::Duplicate);
            if (dedup_result == llmquant::DedupResult::Duplicate) {
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
                spdlog::trace("[trace] token seq={} DEDUP_SKIP", seq_id);
#endif
                return;
            }
        }
#endif

        latency_ctrl.start_measurement();

        logger.log_token_received(text, seq_id);

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.record_token();
#endif

        auto weight = llm_adapter.map_token_to_weight(text);

#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("token seq={} text='{}' sent={:.4f} conf={:.4f} vol={:.4f} bias={:.4f}",
                      seq_id, text,
                      weight.sentiment_score, weight.confidence_score,
                      weight.volatility_score, weight.directional_bias);
#endif

        // Apply per-category semantic weight multipliers (hot-reloadable).
        // Read atomically so hot-reload updates are visible without a mutex.
        weight.sentiment_score  *= sem_mult_sentiment.load(std::memory_order_relaxed);
        weight.confidence_score *= sem_mult_confidence.load(std::memory_order_relaxed);
        weight.volatility_score *= sem_mult_volatility.load(std::memory_order_relaxed);
        weight.directional_bias *= sem_mult_bias.load(std::memory_order_relaxed);

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
        // Feed raw sentiment score into the momentum filter's trajectory analyzer.
        sentiment_momentum_filter.record_sample(weight.sentiment_score);
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
        // Reinforce the decay envelope with each token's directional bias.
        signal_decay.reinforce(weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
        // Accumulate per-token signed bias so operators can identify dominant tokens.
        token_bias_heatmap.record(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        // Also feed raw token text into the keyword dictionary matcher.
        order_flow_detector.record(text);
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
        // Track n-gram frequencies; fires on repeated patterns.
        ngram_profiler.push(text);
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
        // Screen each token for weight anomalies, repetition, and vocab inflation.
        adversarial_detector.inspect(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
        // Attribute per-token marginal contribution to the current bias.
        token_influence.record(text, weight.directional_bias);
#endif

        // In dry-run mode, tokens are mapped through LLMAdapter for
        // dictionary coverage analysis but no signals are emitted.
        if (!dry_run) {
            trade_engine.process_semantic_weight(weight);
        }
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("signal bias={:.4f} vol={:.4f} latency={}us",
                      trade_engine.get_accumulated_bias(),
                      trade_engine.get_accumulated_volatility(),
                      latency_ctrl.get_stats().avg_latency.count());
#endif

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

#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    std::unique_ptr<llmquant::SignalAuditLog> audit_log;
    if (!audit_log_path.empty()) {
        llmquant::SignalAuditLog::Config audit_cfg;
        audit_cfg.filepath = audit_log_path;
        audit_log = std::make_unique<llmquant::SignalAuditLog>(audit_cfg);
        spdlog::info("[audit_log] started — writing to '{}'", audit_log_path);
    }
#endif

#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    // Circuit-breaker: auto-pause signal emission when block rate stays
    // above the threshold for the configured sustained window.
    llmquant::PipelineCircuitBreaker circuit_breaker;
    circuit_breaker.set_state_change_callback(
        [](llmquant::PipelineCircuitBreaker::State s, double rate) {
            if (s == llmquant::PipelineCircuitBreaker::State::Open) {
                spdlog::error("[circuit_breaker] pipeline OPEN — {:.0f}% of signals blocked; "
                              "signal emission suppressed until block rate drops",
                              rate * 100.0);
            } else if (s == llmquant::PipelineCircuitBreaker::State::Closed) {
                spdlog::info("[circuit_breaker] pipeline CLOSED — recovered");
            } else {
                spdlog::info("[circuit_breaker] pipeline HALF-OPEN — probing recovery");
            }
        });
#endif

#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
    // Latency budget enforcer: tiered SLA escalation (Normal→Warn→Throttle→Drop→Breaker).
    // When the Breaker tier is reached the pipeline circuit breaker is tripped immediately.
    llmquant::LatencyBudgetEnforcer latency_budget_enforcer;
    latency_budget_enforcer.set_warn_callback([](int64_t p99) {
        spdlog::warn("[lbe] p99={}µs — warn budget exceeded", p99);
    });
    latency_budget_enforcer.set_throttle_callback([](int64_t p99) {
        spdlog::warn("[lbe] p99={}µs — throttle tier: slowing token intake", p99);
    });
    latency_budget_enforcer.set_drop_callback([](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — drop tier: signal emission suspended", p99);
    });
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    latency_budget_enforcer.set_breaker_callback([&](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — breaker tier: tripping circuit breaker", p99);
        circuit_breaker.force_open();
    });
#else
    latency_budget_enforcer.set_breaker_callback([](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — breaker tier: critical latency", p99);
    });
#endif
    latency_budget_enforcer.set_recovery_callback([](int64_t p99) {
        spdlog::info("[lbe] p99={}µs — recovered to Normal tier", p99);
    });
#endif

#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
    // P&L attribution: attributes realized trade outcomes to sentiment driver categories.
    llmquant::PnLAttributionEngine pnl_attribution;
#endif

#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
    // Portfolio heat monitor: aggregates cross-instrument risk heat.
    llmquant::PortfolioHeatMonitor portfolio_heat;
    portfolio_heat.set_warn_callback([](double heat) {
        spdlog::warn("[portfolio_heat] heat={:.2f} — approaching risk budget", heat);
    });
    portfolio_heat.set_critical_callback([](double heat) {
        spdlog::error("[portfolio_heat] heat={:.2f} — critical; shedding risk", heat);
    });
    portfolio_heat.set_recovery_callback([](double heat) {
        spdlog::info("[portfolio_heat] heat={:.2f} — recovered to Cool", heat);
    });
#endif

#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    // ContextWindowBudget: configure the instance declared before process_token.
    {
        llmquant::ContextWindowBudget::Config cb_cfg;
        cb_cfg.capacity = 128000;  // Claude 3 / GPT-4 128k context
        cb_cfg.on_warn = [](uint64_t used, uint64_t cap) {
            spdlog::warn("[ctx_budget] context warn  used={} / {} ({:.0f}%)",
                         used, cap, 100.0 * static_cast<double>(used) / cap);
        };
        cb_cfg.on_critical = [](uint64_t used, uint64_t cap) {
            spdlog::error("[ctx_budget] context CRITICAL  used={} / {} ({:.0f}%)",
                          used, cap, 100.0 * static_cast<double>(used) / cap);
        };
        cb_cfg.on_overflow = [](uint64_t used, uint64_t cap) {
            spdlog::critical("[ctx_budget] context OVERFLOW used={} cap={} — reset required",
                             used, cap);
        };
        context_budget.update_config(cb_cfg);
    }
#endif

#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
    // FractalDimensionEstimator: tracks Hurst exponent of bias stream (H>0.5=trending).
    llmquant::FractalDimensionEstimator fractal_dim;
    {
        llmquant::FractalDimensionEstimator::Config fd_cfg;
        fd_cfg.on_regime_change = [](double prev_h, double new_h) {
            const char* prev_r = (prev_h > 0.55) ? "trending" : (prev_h < 0.45 ? "mean-rev" : "random");
            const char* new_r  = (new_h  > 0.55) ? "trending" : (new_h  < 0.45 ? "mean-rev" : "random");
            spdlog::info("[fractal] Hurst {:.3f}→{:.3f}  {} → {}", prev_h, new_h, prev_r, new_r);
        };
        fractal_dim.update_config(fd_cfg);
    }
#endif

#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
    // MarketMicrostructureFilter: gates signals whose predicted edge < bid-ask + impact cost.
    llmquant::MarketMicrostructureFilter microstructure_filter;
#endif

#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
    // SignalEnsembleLayer: combines bias/vol/confidence sub-signals with online weight learning.
    llmquant::SignalEnsembleLayer signal_ensemble;
    int ens_bias_id = -1, ens_vol_id = -1, ens_conf_id = -1;
    {
        ens_bias_id = signal_ensemble.register_source("bias");
        ens_vol_id  = signal_ensemble.register_source("vol_adj");
        ens_conf_id = signal_ensemble.register_source("confidence");
        llmquant::SignalEnsembleLayer::Config ec;
        ec.on_weight_update = [](const std::vector<double>& w) {
            if (w.size() >= 3)
                spdlog::debug("[ensemble] weights  bias={:.3f}  vol={:.3f}  conf={:.3f}",
                              w[0], w[1], w[2]);
        };
        signal_ensemble.update_config(ec);
    }
#endif

#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
    // SignalMomentumOscillator: MACD-style oscillator on the bias stream.
    // Fires on_cross when the histogram crosses zero (directional momentum shift).
    llmquant::SignalMomentumOscillator signal_momentum_osc;
    {
        llmquant::SignalMomentumOscillator::Config smo_cfg;
        smo_cfg.divergence_threshold = 0.02;
        smo_cfg.on_cross = [](llmquant::SignalMomentumOscillator::CrossDirection dir, double hist) {
            spdlog::info("[smo] histogram zero-cross  dir={}  hist={:.5f}",
                         (dir == llmquant::SignalMomentumOscillator::CrossDirection::Bullish)
                             ? "BULLISH" : "BEARISH",
                         hist);
        };
        smo_cfg.on_divergence = [](double macd, double sig, double hist) {
            spdlog::debug("[smo] divergence  macd={:.5f}  signal={:.5f}  hist={:.5f}",
                          macd, sig, hist);
        };
        signal_momentum_osc.update_config(smo_cfg);
    }
#endif

#ifdef LLMQUANT_CVAR_ENABLED
    // CVaRCalculator: rolling Expected Shortfall at α=0.95 for tail-risk gating.
    // Records delta_bias_shift as a proxy PnL; fires on_breach when CVaR < -5%.
    llmquant::CVaRCalculator cvar_calc;
    {
        llmquant::CVaRCalculator::Config cv_cfg;
        cv_cfg.breach_threshold = -0.05;
        cv_cfg.on_breach = [](double cvar, double var, double alpha) {
            spdlog::warn("[cvar] tail-risk breach  cvar={:.4f}  var={:.4f}  alpha={:.2f}",
                         cvar, var, alpha);
        };
        cvar_calc.update_config(cv_cfg);
    }
#endif

#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    // TemporalPatternLibrary: configure the instance declared before process_token.
    {
        tpl.register_pattern("earnings_beat",    {"earnings", "beat"},      0.6);
        tpl.register_pattern("earnings_miss",    {"earnings", "miss"},     -0.6);
        tpl.register_pattern("rate_hike",        {"rate", "hike"},         -0.4);
        tpl.register_pattern("rate_cut",         {"rate", "cut"},           0.4);
        tpl.register_pattern("guidance_raised",  {"guidance", "raised"},    0.5);
        tpl.register_pattern("guidance_lowered", {"guidance", "lowered"},  -0.5);
        tpl.register_pattern("short_squeeze",    {"short", "squeeze"},      0.7);
        tpl.register_pattern("margin_call",      {"margin", "call"},       -0.8);
    }
#endif

#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
    // FeedbackLoopDetector: cross-correlation reflexivity trap detector.
    // Warns when the system's own trades appear to drive the LLM sentiment signal.
    llmquant::FeedbackLoopDetector feedback_detector;
    {
        llmquant::FeedbackLoopDetector::Config fb_cfg;
        fb_cfg.threshold  = 0.65;
        fb_cfg.on_feedback = [](double score, int lag) {
            spdlog::warn("[feedback] reflexivity suspected  score={:.3f}  peak_lag={}", score, lag);
        };
        feedback_detector.update_config(fb_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
    // SentimentCycleDetector: detects periodic news-cycle patterns via ACF analysis.
    llmquant::SentimentCycleDetector sentiment_cycle;
    {
        llmquant::SentimentCycleDetector::Config sc_cfg;
        sc_cfg.window_size       = 256;
        sc_cfg.max_lag           = 64;
        sc_cfg.cyclic_threshold  = 0.35;
        sc_cfg.on_period_change  = [](int new_p, int old_p, double strength) {
            spdlog::info("[cycle] dominant period {}→{}  acf={:.3f}", old_p, new_p, strength);
        };
        sentiment_cycle.update_config(sc_cfg);
    }
#endif

#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
    // AdaptiveSamplingController: shrinks poll interval on high activity,
    // grows it on quiet periods to reduce wasteful LLM API calls.
    llmquant::AdaptiveSamplingController adaptive_sampler;
    {
        llmquant::AdaptiveSamplingController::Config as_cfg;
        as_cfg.min_interval_ms  = 10;
        as_cfg.max_interval_ms  = 2000;
        as_cfg.initial_interval_ms = 100;
        as_cfg.on_interval_change = [](int64_t new_ms) {
            spdlog::debug("[sampler] poll interval → {} ms", new_ms);
        };
        adaptive_sampler.update_config(as_cfg);
    }
#endif

#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
    // MutualInformationEstimator: captures non-linear sentiment→return dependency.
    llmquant::MutualInformationEstimator mi_estimator;
    // record(sentiment, return) is called from the OMS PnL callback below.
#endif

#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
    // SignalBlindSpotDetector: flags calendar slots with poor historical win rate.
    llmquant::SignalBlindSpotDetector blind_spot;
    {
        llmquant::SignalBlindSpotDetector::Config bs_cfg;
        bs_cfg.min_samples         = 10;
        bs_cfg.blind_spot_threshold = 0.4;
        bs_cfg.on_blind_spot_found = [](int slot, double wr) {
            spdlog::warn("[blind_spot] hour {} flagged — win_rate={:.2f}", slot, wr);
        };
        blind_spot.update_config(bs_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
    // SignalSurpriseIndex: flags signals that are statistically anomalous
    // relative to the engine's own learned distribution (self-information).
    llmquant::SignalSurpriseIndex signal_surprise;
    {
        llmquant::SignalSurpriseIndex::Config ss_cfg;
        ss_cfg.min_samples             = 50;
        ss_cfg.high_surprise_threshold = 0.80;
        ss_cfg.on_high_surprise = [](double bias, double score) {
            spdlog::warn("[surprise] HIGH I(x)={:.3f}  bias={:.5f} — anomalous signal",
                         score, bias);
        };
        signal_surprise.update_config(ss_cfg);
    }
#endif

    // TokenStreamHealthMonitor: watchdog for feed stalls and token floods.
    {
        llmquant::TokenStreamHealthMonitor::Config sh_cfg;
        sh_cfg.stall_timeout_ms   = 3000;
        sh_cfg.max_tokens_per_sec = 1000.0;
        sh_cfg.on_stall = [](int64_t elapsed_ms) {
            spdlog::warn("[stream_health] STALL detected — {}ms since last token", elapsed_ms);
        };
        sh_cfg.on_flood = [](double rate) {
            spdlog::warn("[stream_health] FLOOD {:.0f} tok/s — exceeds threshold", rate);
        };
        sh_cfg.on_recovery = [] {
            spdlog::info("[stream_health] stream recovered to Healthy state");
        };
        stream_health.update_config(sh_cfg);
    }

#ifdef LLMQUANT_REGIME_SIZER_ENABLED
    // RegimeAwareSizer: scales notional by Hurst exponent × vol-targeting factor.
    // update_hurst() and update_vol() should be called after each signal.
    llmquant::RegimeAwareSizer regime_sizer;
    {
        llmquant::RegimeAwareSizer::Config rs_cfg;
        rs_cfg.target_vol = 0.20;
        rs_cfg.on_size_change = [](double nw, double old) {
            spdlog::info("[regime_sizer] multiplier {:.3f}→{:.3f}", old, nw);
        };
        regime_sizer.update_config(rs_cfg);
    }
#endif

#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
    // ConfidenceDecayTracker: fits exponential decay to signal.confidence over time.
    // Slow decay → news-driven; fast decay → noise spike.
    llmquant::ConfidenceDecayTracker conf_decay;
    {
        llmquant::ConfidenceDecayTracker::Config cd_cfg;
        cd_cfg.fast_decay_threshold_ms = 500.0;
        cd_cfg.on_decay_change = [](double nw, double old) {
            spdlog::info("[conf_decay] half-life {:.0f}ms→{:.0f}ms", old, nw);
        };
        conf_decay.update_config(std::move(cd_cfg));
    }
#endif

#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
    // CrossAssetCorrelationMonitor: rolling Pearson correlation between bias and vol.
    // Register "bias" and "vol" as two virtual "assets" to track their co-movement.
    llmquant::CrossAssetCorrelationMonitor cross_asset_corr;
    {
        llmquant::CrossAssetCorrelationMonitor::Config ca_cfg;
        ca_cfg.on_high_correlation = [](const std::string& a, const std::string& b, double rho) {
            spdlog::info("[cross_asset] high correlation  {}<>{} rho={:.3f}", a, b, rho);
        };
        ca_cfg.on_low_correlation  = [](const std::string& a, const std::string& b, double rho) {
            spdlog::info("[cross_asset] low correlation   {}<>{} rho={:.3f}", a, b, rho);
        };
        cross_asset_corr.update_config(std::move(ca_cfg));
        cross_asset_corr.register_asset("bias");
        cross_asset_corr.register_asset("vol");
        cross_asset_corr.register_asset("confidence");
    }
#endif

#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
    // TokenVelocityTracker: measures first and second time-derivatives of bias.
    // High velocity = rapid sentiment shift; high acceleration = regime change.
    llmquant::TokenVelocityTracker velocity_tracker;
    {
        llmquant::TokenVelocityTracker::Config vt_cfg;
        vt_cfg.window_size         = 16;
        vt_cfg.fast_move_threshold = 0.5;
        vt_cfg.on_fast_move = [](double vel, double accel) {
            spdlog::warn("[velocity] fast move  vel={:.4f}  accel={:.4f}", vel, accel);
        };
        velocity_tracker.update_config(vt_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
    // NarrativeMomentumClock: four-quadrant investment-clock on smoothed bias.
    // Rising(Q1)/Fading(Q2)/Falling(Q3)/Recovering(Q4) — fires on rotation.
    llmquant::NarrativeMomentumClock narrative_clock;
    {
        llmquant::NarrativeMomentumClock::Config nc_cfg;
        nc_cfg.bias_alpha     = 0.10;
        nc_cfg.velocity_alpha = 0.20;
        nc_cfg.on_quadrant_change = [](llmquant::NarrativeMomentumClock::Quadrant from,
                                       llmquant::NarrativeMomentumClock::Quadrant to,
                                       double b, double v) {
            static const char* names[] = {"Rising", "Fading", "Falling", "Recovering"};
            spdlog::info("[clock] {} → {}  bias={:.4f}  vel={:.5f}",
                         names[static_cast<int>(from)], names[static_cast<int>(to)], b, v);
        };
        narrative_clock.update_config(nc_cfg);
    }
#endif

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    // AdaptiveVelocityBreaker: EMA-smoothed circuit-breaker that trips when
    // bias changes too rapidly, guarding against reflexive LLM feedback loops.
    llmquant::AdaptiveVelocityBreaker velocity_breaker;
    {
        llmquant::AdaptiveVelocityBreaker::Config vb_cfg;
        vb_cfg.trip_threshold   = 10.0;
        vb_cfg.recovery_factor  = 0.5;
        vb_cfg.velocity_alpha   = 0.3;
        vb_cfg.on_trip = [](double vel) {
            spdlog::warn("[vel_breaker] TRIPPED  smoothed_vel={:.4f}", vel);
        };
        vb_cfg.on_recovery = [](double vel) {
            spdlog::info("[vel_breaker] RECOVERED  smoothed_vel={:.4f}", vel);
        };
        velocity_breaker.update_config(vb_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
    // SignalCalibrationEngine: online Platt-scaling logistic calibration that
    // maps raw confidence scores to well-calibrated P(win) probabilities.
    llmquant::SignalCalibrationEngine signal_calibration;
#endif

#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    {
        llmquant::OrderFlowImbalanceDetector::Config of_cfg;
        of_cfg.ema_alpha            = 0.1;
        of_cfg.imbalance_threshold  = 0.6;
        of_cfg.on_imbalance = [](double imb, bool buy) {
            spdlog::info("[order_flow] imbalance={:.3f}  side={}", imb, buy ? "BUY" : "SELL");
        };
        order_flow_detector.update_config(of_cfg);
    }
#endif

#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
    // CrossSessionMemory: persist Kelly / drawdown state across restarts
    // so the engine warms up instantly rather than starting cold.
    llmquant::CrossSessionMemory cross_session_mem;
    {
        llmquant::CrossSessionMemory::Config csm_cfg;
        csm_cfg.ignore_missing = true;
        csm_cfg.on_load = [](const std::string& path, uint64_t sess) {
            spdlog::info("[cross_session] loaded session={} from {}", sess, path);
        };
        csm_cfg.on_save = [](const std::string& path, uint64_t sess) {
            spdlog::info("[cross_session] saved  session={} to {}", sess, path);
        };
        cross_session_mem.update_config(csm_cfg);
        cross_session_mem.load();  // warm start
    }
#endif

#ifdef LLMQUANT_REGIME_PROB_ENABLED
    // MarketRegimeProbabilityEstimator: online 2-state HMM Bayesian filter
    // that produces a soft probability distribution over risk-on/risk-off regimes.
    llmquant::MarketRegimeProbabilityEstimator regime_prob_est;
    {
        llmquant::MarketRegimeProbabilityEstimator::Config rp_cfg;
        rp_cfg.min_observations = 20;
        rp_cfg.transition_threshold = 0.70;
        rp_cfg.on_regime_change = [](double prob_on, bool is_on) {
            spdlog::info("[regime_hmm] {} → p_risk_on={:.4f}",
                         is_on ? "RISK-ON" : "RISK-OFF", prob_on);
        };
        regime_prob_est.update_config(rp_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
    // SignalReplayBuffer: retains the last 1024 signals for replay and
    // post-hoc analysis without needing to re-run the full token stream.
    llmquant::SignalReplayBuffer signal_replay;
#endif

#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    // TokenNgramProfiler: config — tracks 2-gram and 3-gram frequencies;
    // fires on hot n-grams that may indicate adversarial injection or stuck LLM output.
    {
        llmquant::TokenNgramProfiler::Config ng_cfg;
        ng_cfg.hot_threshold = 10;
        ng_cfg.on_hot_ngram = [](const std::string& ng, uint64_t cnt, int n) {
            spdlog::warn("[ngram] hot {}-gram \"{}\" count={}", n, ng, cnt);
        };
        ngram_profiler.update_config(ng_cfg);
    }
#endif

#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
    // ExecutionQualityMonitor: tracks fill latency and slippage between signal
    // emission and OMS acknowledgment; fires on SLA breaches.
    llmquant::ExecutionQualityMonitor exec_quality;
    {
        llmquant::ExecutionQualityMonitor::Config eq_cfg;
        eq_cfg.latency_sla_us   = 5000.0;  // 5 ms SLA
        eq_cfg.slippage_sla_bps = 5.0;
        eq_cfg.on_sla_breach = [](const llmquant::ExecutionQualityMonitor::FillRecord& f,
                                  bool lat_breach, bool slip_breach) {
            spdlog::warn("[exec_quality] SLA breach sig={} lat={:.0f}us slip={:.2f}bps "
                         "lat_breach={} slip_breach={}",
                         f.signal_id, f.latency_us, f.slippage_bps,
                         lat_breach, slip_breach);
        };
        exec_quality.update_config(eq_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
    // SentimentDispersionIndex: measures incoherence across bias / vol / confidence
    // streams via coefficient of variation; fires on high/low dispersion events.
    llmquant::SentimentDispersionIndex sentiment_dispersion;
    {
        llmquant::SentimentDispersionIndex::Config sd_cfg;
        sd_cfg.high_threshold = 0.8;
        sd_cfg.low_threshold  = 0.2;
        sd_cfg.on_high_dispersion = [](double sdi) {
            spdlog::warn("[dispersion] HIGH sdi={:.4f} — signals incoherent", sdi);
        };
        sd_cfg.on_low_dispersion = [](double sdi) {
            spdlog::info("[dispersion] COHERENT sdi={:.4f}", sdi);
        };
        sentiment_dispersion.update_config(sd_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
    // SentimentDivergenceDetector: pairwise EMA divergence across bias/vol/conf.
    // Fires when any pair's |ema_a - ema_b| exceeds the threshold.
    llmquant::SentimentDivergenceDetector sentiment_divergence;
    {
        llmquant::SentimentDivergenceDetector::Config svd_cfg;
        svd_cfg.divergence_threshold  = 0.4;
        svd_cfg.recovery_hysteresis   = 0.7;
        svd_cfg.on_divergence = [](double d, const std::string& a, const std::string& b) {
            spdlog::warn("[divergence] {} <> {} diverge={:.4f}", a, b, d);
        };
        svd_cfg.on_recovery = [](double d) {
            spdlog::info("[divergence] recovered  diverge={:.4f}", d);
        };
        sentiment_divergence.update_config(svd_cfg);
    }
#endif

#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    // TokenInfluenceAttributor: Shapley-inspired per-token marginal attribution.
    // Declared in the pre-lambda block; configure here.
    {
        llmquant::TokenInfluenceAttributor::Config ti_cfg;
        ti_cfg.window_size = 64;
        ti_cfg.top_k       = 5;
        token_influence.update_config(ti_cfg);
    }
#endif

#ifdef LLMQUANT_WALK_FORWARD_ENABLED
    // WalkForwardValidator: rolling OOS validation — offline/diagnostic tool.
    // Tokens must be loaded via load_tokens() before run() can be called.
    llmquant::WalkForwardValidator::Config wf_cfg;
    wf_cfg.train_size = 200;
    wf_cfg.test_size  = 50;
    wf_cfg.step_size  = 50;
    wf_cfg.optimize   = false;  // skip parameter sweep in live mode
    llmquant::WalkForwardValidator walk_forward(wf_cfg);
#endif

#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    // AdversarialInputDetector: monitors token stream for weight anomalies,
    // repetition attacks, and vocabulary inflation in real time.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::AdversarialInputDetector::Config ad_cfg;
        ad_cfg.anomaly_threshold    = 4.0;
        ad_cfg.min_warmup_tokens    = 30;
        ad_cfg.max_repeat_fraction  = 0.6;
        ad_cfg.max_novel_fraction   = 0.8;
        ad_cfg.on_anomaly = [](llmquant::AdversarialInputDetector::AnomalyKind kind,
                               const std::string& token, double score) {
            const char* k =
                kind == llmquant::AdversarialInputDetector::AnomalyKind::WeightAnomaly
                ? "weight"
                : kind == llmquant::AdversarialInputDetector::AnomalyKind::RepetitionAttack
                  ? "repetition" : "vocab_inflation";
            spdlog::warn("[adversarial] {} token=\"{}\" score={:.3f}", k, token, score);
        };
        adversarial_detector.update_config(ad_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CI_ENABLED
    // SignalConfidenceInterval: jackknife CI on rolling signal window; narrow
    // CI = high-confidence environment, wide CI = noisy / uncertain signals.
    llmquant::SignalConfidenceInterval signal_ci;
    {
        llmquant::SignalConfidenceInterval::Config ci_cfg;
        ci_cfg.window_size      = 64;
        ci_cfg.z                = 1.96;
        ci_cfg.narrow_threshold = 0.05;
        ci_cfg.wide_threshold   = 0.30;
        ci_cfg.on_narrow_interval = [](double mean, double hw) {
            spdlog::info("[signal_ci] NARROW mean={:.4f} hw={:.4f}", mean, hw);
        };
        ci_cfg.on_wide_interval = [](double mean, double hw) {
            spdlog::warn("[signal_ci] WIDE   mean={:.4f} hw={:.4f}", mean, hw);
        };
        signal_ci.update_config(ci_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
    // SentimentPersistenceMatrix: Markov chain over N discretized bias states.
    // Tracks N×N transition counts, row-normalised probabilities, stickiness,
    // and a stationary-distribution estimate.  Fires on state transitions.
    llmquant::SentimentPersistenceMatrix sentiment_persistence;
    {
        llmquant::SentimentPersistenceMatrix::Config mp_cfg;
        mp_cfg.n_states      = 5;
        mp_cfg.min_row_count = 4;
        mp_cfg.on_state_change = [](int from, int to, double p) {
            spdlog::info("[markov] state {} → {} p={:.3f}", from, to, p);
        };
        sentiment_persistence.update_config(mp_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
    // SentimentPhasePortrait: treats (bias, velocity) as a 2-D dynamical system.
    // Discretises the phase plane into an N×N grid; tracks dwell time, attractors
    // and period-2 cycles in the trajectory.
    llmquant::SentimentPhasePortrait phase_portrait;
    {
        llmquant::SentimentPhasePortrait::Config pp_cfg;
        pp_cfg.grid_size           = 8;
        pp_cfg.attractor_threshold = 0.15;
        pp_cfg.min_visits          = 10;
        pp_cfg.on_attractor_change = [](int r, int c) {
            spdlog::info("[phase] attractor → ({}, {})", r, c);
        };
        pp_cfg.on_cycle_detected = [](int r1, int c1, int r2, int c2) {
            spdlog::warn("[phase] CYCLE ({},{})↔({},{})", r1, c1, r2, c2);
        };
        phase_portrait.update_config(pp_cfg);
    }
#endif

#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
    // CausalImpactEstimator: CUSUM structural-break detector — attributes
    // return regime shifts to preceding LLM sentiment events.
    // Record signals via record_event() and returns via record_return().
    llmquant::CausalImpactEstimator causal_impact;
    {
        llmquant::CausalImpactEstimator::Config ci_causal_cfg;
        ci_causal_cfg.warmup_window  = 50;
        ci_causal_cfg.sensitivity    = 0.001;
        ci_causal_cfg.threshold      = 0.05;
        ci_causal_cfg.event_lookback = 20;
        ci_causal_cfg.on_break = [](double stat, const std::string& label, double impact) {
            spdlog::warn("[causal] break stat={:.4f} event=\"{}\" impact={:.5f}",
                         stat, label.empty() ? "<none>" : label, impact);
        };
        causal_impact.update_config(ci_causal_cfg);
    }
#endif

#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
    // OptionsFlowSentimentBridge: detects divergence between LLM sentiment
    // velocity and options IV skew.  Smart-money bear: narrative bullish but
    // IV skew widening.  Smart-money bull: narrative bearish but calls bid.
    // record_skew() would be driven by a real options feed; here we seed a
    // neutral skew so the detector is ready to accept live updates.
    llmquant::OptionsFlowSentimentBridge options_flow_bridge;
    {
        llmquant::OptionsFlowSentimentBridge::Config ofb_cfg;
        ofb_cfg.velocity_alpha = 0.15;
        ofb_cfg.skew_alpha     = 0.20;
        ofb_cfg.div_threshold  = 0.03;
        ofb_cfg.hysteresis     = 0.30;
        ofb_cfg.min_warmup     = 15;
        ofb_cfg.on_divergence  = [](llmquant::OptionsFlowSentimentBridge::DivergenceKind kind,
                                    double score, double vel, double skew) {
            const char* label = "NONE";
            if (kind == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBear)
                label = "SMART_MONEY_BEAR";
            else if (kind == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBull)
                label = "SMART_MONEY_BULL";
            spdlog::warn("[options_flow] divergence={} score={:.4f} vel={:.4f} skew={:.4f}",
                         label, score, vel, skew);
        };
        options_flow_bridge.update_config(ofb_cfg);
    }
#endif

#ifdef LLMQUANT_KELLY_SIZER_ENABLED
    // Kelly Criterion position sizer: scales delta_bias_shift by the optimal
    // fraction given the observed win/loss history.  Outcomes should be fed
    // back via kelly_sizer.record_outcome() from the OMS P&L callback.
    llmquant::KellyPositionSizer kelly_sizer;
#endif

#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
    // Adaptive cooldown: widens signal cooldown when P99 latency exceeds budget,
    // narrows it again during recovery.  Updated by the stats ticker.
    llmquant::AdaptiveCooldownController adaptive_cooldown;
#endif

#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
    // Regime detector: classifies pipeline as Bull/Bear/Volatile/RiskOff/Neutral.
    // Updated per signal in the hot-path callback; logged on regime transitions.
    llmquant::RegimeDetector regime_detector;
#endif
#if defined(LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED) && defined(LLMQUANT_REGIME_DETECTOR_ENABLED)
    // Markov transition model: learns P(next|current) from regime change history.
    llmquant::RegimeTransitionModel regime_transition_model;
    regime_detector.set_regime_change_callback(
        [&](llmquant::RegimeDetector::Regime next, llmquant::RegimeDetector::Regime prev) {
            regime_transition_model.record_transition(prev, next);
        });
#endif

#ifdef LLMQUANT_TRADING_HOURS_ENABLED
    // Market-hours guard: blocks signals outside NYSE/NASDAQ regular session.
    llmquant::TradingHoursGuard trading_hours_guard;
    trading_hours_guard.update_config([]{
        llmquant::TradingHoursGuard::Config cfg;
        cfg.on_session_change = [](bool open) {
            spdlog::info("[trading_hours] session {} — signal gate {}",
                         open ? "OPEN" : "CLOSED",
                         open ? "enabled" : "disabled");
        };
        return cfg;
    }());
#endif

#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
    // Cross-source correlation tracker: watches for diverging/converging sources.
    llmquant::SignalCorrelationTracker signal_corr;
    signal_corr.set_divergence_callback([](const std::string& a, const std::string& b, double r) {
        spdlog::warn("[signal_corr] DIVERGE {}<>{} r={:.3f} — sources moving oppositely", a, b, r);
    });
    signal_corr.set_convergence_callback([](const std::string& a, const std::string& b, double r) {
        spdlog::info("[signal_corr] CONVERGE {}<>{} r={:.3f} — sources in agreement", a, b, r);
    });
#endif

#ifdef LLMQUANT_WARMUP_SEQUENCER_ENABLED
    // Pre-seed EMA accumulators with a short synthetic token burst.
    {
        llmquant::WarmupSequencer::Config wcfg;
        wcfg.synthetic_tokens = {
            {"bullish", 0.6}, {"rally",   0.5}, {"growth",  0.4},
            {"neutral", 0.0}, {"concern", -0.3}, {"crash",  -0.7},
            {"bearish", -0.5}, {"recover", 0.3}, {"stable",  0.1},
        };
        wcfg.repeat_count = 3;
        wcfg.on_complete  = [] { spdlog::info("[warmup] EMA pre-seeding complete"); };
        llmquant::WarmupSequencer warmup(wcfg);
        warmup.run([&](const std::string& tok, double /*sent*/) {
            (void)llm_adapter.map_token_to_weight(tok);
        });
    }
#endif

#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
    // DrawdownProtector: tightens risk thresholds as cumulative losses deepen.
    llmquant::DrawdownProtector drawdown_protector;
    drawdown_protector.update_config([]{
        llmquant::DrawdownProtector::Config cfg;
        cfg.on_tier_change = [](int t, double scale, double dd_pct) {
            spdlog::warn("[drawdown] tier {} active  scale={:.2f}  drawdown={:.1f}%",
                         t, scale, dd_pct * 100.0);
        };
        return cfg;
    }());
#endif

#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
    // MultiTimeframeAggregator: fuses bias signals across 1s/5s/30s/5m EMAs.
    llmquant::MultiTimeframeAggregator multi_tf;
    multi_tf.update_config([]{
        llmquant::MultiTimeframeAggregator::Config cfg;
        cfg.on_divergence = [](double spread, double, double) {
            spdlog::debug("[multi_tf] timeframe divergence spread={:.3f}", spread);
        };
        return cfg;
    }());
#endif

// stale_detector already declared above (before the process_token lambda).

#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
    // VolatilityForecaster: tracks GARCH(1,1) conditional variance of sentiment stream.
    llmquant::VolatilityForecaster vol_forecaster;
    {
        llmquant::VolatilityForecaster::Config vf_cfg;
        vf_cfg.on_high_vol = [](double vol, double /*var*/) {
            spdlog::warn("[vol_fcst] HIGH conditional vol={:.4f}", vol);
        };
        vol_forecaster.update_config(vf_cfg);
    }
#endif

#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
    // BayesianSignalFilter: Beta-Bernoulli posterior confidence per direction.
    llmquant::BayesianSignalFilter bayes_filter;
    {
        llmquant::BayesianSignalFilter::Config bf_cfg;
        bf_cfg.on_low_confidence = [](bool bullish, double post) {
            spdlog::warn("[bayes] low posterior={:.3f} dir={}", post,
                         bullish ? "BULL" : "BEAR");
        };
        bayes_filter.update_config(bf_cfg);
    }
#endif

#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
    // AnomalyDetector: online Z-score anomaly detection on the bias stream.
    llmquant::AnomalyDetector anomaly_detector;
    {
        llmquant::AnomalyDetector::Config ad_cfg;
        ad_cfg.name = "bias";
        ad_cfg.soft_cb = [](const llmquant::AnomalyDetector::AnomalyEvent& e) {
            spdlog::info("[anomaly] soft z={:.2f} val={:.6f}", e.z_score, e.value);
        };
        ad_cfg.hard_cb = [](const llmquant::AnomalyDetector::AnomalyEvent& e) {
            spdlog::warn("[anomaly] HARD z={:.2f} val={:.6f}", e.z_score, e.value);
        };
        anomaly_detector.update_config(ad_cfg);
    }
#endif

#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
    // TokenBurstDetector: flags high token arrival rates to detect backlog flushes.
    llmquant::TokenBurstDetector burst_detector;
    {
        llmquant::TokenBurstDetector::Config bd_cfg;
        bd_cfg.on_burst_start = [](double rate) {
            spdlog::warn("[burst_det] BURST {:.1f} tok/s — throttling may apply", rate);
        };
        bd_cfg.on_burst_end = [](double rate) {
            spdlog::info("[burst_det] burst end {:.1f} tok/s", rate);
        };
        burst_detector.update_config(bd_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
    // SignalPersistenceTracker: conviction multiplier from directional streak.
    llmquant::SignalPersistenceTracker persistence_tracker;
    {
        llmquant::SignalPersistenceTracker::Config pt_cfg;
        pt_cfg.on_conviction = [](int streak) {
            spdlog::info("[persistence] conviction streak={}", streak);
        };
        persistence_tracker.update_config(pt_cfg);
    }
#endif

#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
    // RollingSharpeBiasTracker: rolling Sharpe of the bias stream.
    llmquant::RollingSharpeBiasTracker rolling_sharpe;
#endif

#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
    // OrderBookSimulator: sentiment-driven LOB for slippage-aware fill estimation.
    llmquant::OrderBookSimulator order_book_sim;
#endif

#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
    // TokenSentimentHeatmap: per-token sentiment distribution for attribution.
    llmquant::TokenSentimentHeatmap sentiment_heatmap;
#endif

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    // SentimentMomentumFilter: gates trade signals that contradict the macro
    // sentiment trajectory (Improving/Declining/Stable/Volatile).
    {
        llmquant::SentimentMomentumFilter::Config smf_cfg;
        smf_cfg.mode           = llmquant::SentimentMomentumFilter::Mode::Relaxed;
        smf_cfg.scale_by_slope = true;
        smf_cfg.slope_scale    = 0.05;
        sentiment_momentum_filter.update_config(smf_cfg);
    }
#endif

#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
    // PositionTracker: records open/close trades and feeds realised P&L back
    // into the Kelly sizer to keep position sizing adaptive.
    llmquant::PositionTracker position_tracker(kelly_sizer);
    position_tracker.set_trade_close_callback([](uint64_t id, double ret, bool win) {
        spdlog::info("[pos_tracker] trade#{} closed  return={:.4f}  {}",
                     id, ret, win ? "WIN" : "LOSS");
    });
#endif

#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
    // SignalDecayEnvelope: attenuates accumulated bias after token-stream silence.
    {
        llmquant::SignalDecayEnvelope::Config sd_cfg;
        sd_cfg.half_life_ms = 15'000.0;  // bias halves after 15 s of silence
        sd_cfg.clamp        = true;
        sd_cfg.min_bias     = -2.0;
        sd_cfg.max_bias     =  2.0;
        signal_decay.update_config(sd_cfg);
        signal_decay.set_zero_cross_callback([](double old_b, double new_b) {
            spdlog::info("[signal_decay] bias zero-cross: {:.4f} → {:.4f}", old_b, new_b);
        });
    }
#endif

    // Sparkline ring buffer: 24 most-recent delta_bias_shift values → unicode blocks.
    constexpr int kSparkSlots = 24;
    std::array<double, kSparkSlots> spark_ring{};
    std::atomic<int> spark_head{0};

    // Signal velocity: rate of change of delta_bias_shift (units/second).
    // Written by signal callback (single writer), read by stats ticker.
    double sig_vel_prev_bias = 0.0;
    std::chrono::steady_clock::time_point sig_vel_prev_time{};
    std::atomic<double> sig_velocity{0.0};  // current velocity estimate

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

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
        // Block signals that contradict the macro sentiment trajectory.
        const TradeSignal momentum_filtered = sentiment_momentum_filter.filter_signal(signal);
#else
        const TradeSignal& momentum_filtered = signal;
#endif

#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
        // Scale delta_bias_shift by the decay-attenuated envelope factor.
        // Reuse the filtered signal, cloning to allow modification.
        TradeSignal decay_adjusted = momentum_filtered;
        {
            double envelope = signal_decay.decayed_bias();
            if (std::abs(envelope) > 1e-9) {
                // Attenuate: multiply bias by |envelope| / |raw_bias| ratio clamped to [0,1].
                double raw = signal_decay.raw_bias();
                if (std::abs(raw) > 1e-9)
                    decay_adjusted.delta_bias_shift *= std::min(1.0, std::abs(envelope) / std::abs(raw));
            }
        }
        const TradeSignal& pre_kelly_signal = decay_adjusted;
#else
        const TradeSignal& pre_kelly_signal = momentum_filtered;
#endif

#ifdef LLMQUANT_KELLY_SIZER_ENABLED
        // Scale delta_bias_shift by the current Kelly fraction before risk gating.
        const TradeSignal sized_signal = kelly_sizer.size_signal(pre_kelly_signal);
        bool passed = risk_mgr.evaluate(sized_signal);
#else
        const TradeSignal& sized_signal = pre_kelly_signal;
        bool passed = risk_mgr.evaluate(pre_kelly_signal);
#endif

#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
        circuit_breaker.record_signal(!passed);
        // When the circuit is open, treat the signal as blocked regardless
        // of the risk manager decision (unless suppress_when_open is false).
        if (passed && circuit_breaker.is_open()) {
            passed = false;
        }
#endif

#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
        regime_detector.update(signal.delta_bias_shift,
                               signal.volatility_adjustment,
                               !passed);
#endif

#ifdef LLMQUANT_TRADING_HOURS_ENABLED
        // Block signals outside NYSE market hours.
        if (passed && trading_hours_guard.should_block()) {
            passed = false;
        }
#endif

#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
        // Track bias value under the "main" source for correlation monitoring.
        signal_corr.record("main", signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
        // Feed the raw bias into all timeframe EMAs for multi-horizon consensus.
        multi_tf.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
        // Record simulated P&L outcome (passed signals only) into drawdown protector.
        if (passed) {
            // Use a simple proxy: bias * 0.001 as a per-signal PnL estimate.
            drawdown_protector.record_pnl(signal.delta_bias_shift * 0.001);
        }
#endif

#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
        // Update GARCH(1,1) volatility estimate from the raw bias value.
        vol_forecaster.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
        // Record signal direction; outcomes would be fed by a live P&L feed.
        bayes_filter.record_signal(signal.delta_bias_shift > 0.0);
#endif

#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
        // Check for statistical anomalies in the bias stream.
        anomaly_detector.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
        // Record token arrival for burst rate tracking.
        burst_detector.record();
#endif

#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
        // Track directional streak for conviction scoring.
        persistence_tracker.record_bias(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
        // Update rolling Sharpe of the bias stream.
        rolling_sharpe.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        // Update Hurst exponent estimate with each new bias observation.
        fractal_dim.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
        // Update ensemble sub-signals and train from outcome (passed = positive reward).
        if (ens_bias_id >= 0)  signal_ensemble.update_source(ens_bias_id,  signal.delta_bias_shift);
        if (ens_vol_id  >= 0)  signal_ensemble.update_source(ens_vol_id,   signal.volatility_adjustment);
        if (ens_conf_id >= 0)  signal_ensemble.update_source(ens_conf_id,  signal.confidence);
        signal_ensemble.record_outcome(passed ? 1.0 : -1.0);
#endif

#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
        // Feed bias into MACD oscillator; callbacks fire on zero-crosses.
        signal_momentum_osc.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
        // Update synthetic LOB with the latest bias signal.
        order_book_sim.update_bias(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
        // Record direction–sentiment pair for attribution.
        {
            const char* dir_key = (signal.strategy_toggle > 0) ? "bull"
                                : (signal.strategy_toggle < 0) ? "bear" : "neutral";
            sentiment_heatmap.record(dir_key, signal.delta_bias_shift);
        }
#endif

#ifdef LLMQUANT_CVAR_ENABLED
        // Treat delta_bias_shift as proxy PnL for tail-risk tracking.
        cvar_calc.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
        // Own-activity = |bias shift| emitted; sentiment = raw bias shift.
        feedback_detector.record_own_activity(std::abs(signal.delta_bias_shift));
        feedback_detector.record_sentiment(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
        // Feed bias into ACF cycle detector.
        sentiment_cycle.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
        // Adapt poll interval based on recent signal magnitude.
        adaptive_sampler.record_activity(std::abs(signal.delta_bias_shift));
#endif

#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
        // Compute self-information of this signal relative to learned distribution.
        signal_surprise.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_REGIME_SIZER_ENABLED
        // Feed Hurst exponent and volatility into the regime-aware position sizer.
#  ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        regime_sizer.update_hurst(fractal_dim.hurst());
#  endif
#  ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
        regime_sizer.update_vol(vol_forecaster.conditional_vol());
#  endif
#endif

#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
        // Track how quickly signal confidence decays over time.
        conf_decay.record(signal.confidence);
#endif

#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
        // Track rolling correlation between bias, vol, and confidence streams.
        cross_asset_corr.record("bias",       signal.delta_bias_shift);
        cross_asset_corr.record("vol",        signal.volatility_adjustment);
        cross_asset_corr.record("confidence", signal.confidence);
#endif

#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
        velocity_tracker.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
        narrative_clock.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        // Trip the breaker if bias velocity is excessive; block signal when open.
        (void)velocity_breaker.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
        // Record a synthetic outcome: positive confidence = treat as win.
        // In production, wire record_outcome() from the OMS P&L callback.
        signal_calibration.record_outcome(signal.confidence, signal.confidence > 0.5);
#endif

#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        // Push the raw bias as a signed pressure proxy.
        order_flow_detector.record_pressure(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_REGIME_PROB_ENABLED
        // Feed sentiment + volatility into the HMM filter for regime probability.
        regime_prob_est.update(signal.delta_bias_shift,
                               signal.volatility_adjustment);
#endif

#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
        // Capture signal in the replay ring for post-hoc analysis.
        signal_replay.push(signal);
#endif

#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
        // Measure incoherence across bias, vol, and confidence dimensions.
        sentiment_dispersion.record(std::abs(signal.delta_bias_shift),
                                    signal.volatility_adjustment,
                                    signal.confidence);
#endif

#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
        // Track pairwise divergence between bias, vol, and confidence streams.
        sentiment_divergence.record("bias",       signal.delta_bias_shift);
        sentiment_divergence.record("vol",        signal.volatility_adjustment);
        sentiment_divergence.record("confidence", signal.confidence);
#endif

#ifdef LLMQUANT_SIGNAL_CI_ENABLED
        // Track jackknife CI on bias stream; narrow=reliable, wide=uncertain.
        signal_ci.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
        // Feed each signal's bias shift into the Markov state chain.
        sentiment_persistence.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
        // Plot the signal on the (bias, velocity) phase plane.
        phase_portrait.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
        // Record the signal as a sentiment event sentinel for causal attribution.
        // Return observations should be fed from the OMS P&L callback in production.
        causal_impact.record_event("signal:" + std::to_string(signal.delta_bias_shift));
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
        // Feed current aggregated bias and elapsed time into the IV-skew divergence
        // detector.  A live options feed would also call record_skew() separately.
        {
            static auto ofb_last_t = std::chrono::steady_clock::now();
            auto ofb_now = std::chrono::steady_clock::now();
            double ofb_dt = std::chrono::duration<double>(ofb_now - ofb_last_t).count();
            ofb_last_t = ofb_now;
            if (ofb_dt > 0.0)
                options_flow_bridge.record_bias(signal.delta_bias_shift, ofb_dt);
        }
#endif

        // Record bias value in sparkline ring (lock-free: only one writer thread).
        {
            int idx = spark_head.fetch_add(1, std::memory_order_relaxed) % kSparkSlots;
            spark_ring[idx] = signal.delta_bias_shift;
        }

        // Signal velocity: delta_bias_shift per second (first-order finite difference).
        {
            auto now = std::chrono::steady_clock::now();
            if (sig_vel_prev_time.time_since_epoch().count() != 0) {
                double dt_s = std::chrono::duration<double>(now - sig_vel_prev_time).count();
                if (dt_s > 1e-6) {
                    double vel = (signal.delta_bias_shift - sig_vel_prev_bias) / dt_s;
                    sig_velocity.store(vel, std::memory_order_relaxed);
                }
            }
            sig_vel_prev_bias = signal.delta_bias_shift;
            sig_vel_prev_time = now;
        }

        // Capture the rejection reason once, under the lock, so it is available
        // for both the console gate_str and the structured log call below.
        // Previously the reason was cleared after gate_str was built, making
        // the second read always fall back to "risk".
        std::string block_reason_copy;
        if (!passed) {
            std::lock_guard<std::mutex> lk(block_reason_mutex);
            block_reason_copy = last_block_reason.empty() ? "risk" : last_block_reason;
            if (block_reason_copy.size() > 16) block_reason_copy = block_reason_copy.substr(0, 16);
            last_block_reason.clear();
        }

        std::string gate_str;
        if (passed) {
            gate_str = std::string(" ") + C("\033[32m") + "PASS" + C("\033[0m");
        } else {
            gate_str = std::string(" ") + C("\033[31m") + "BLOCK" + C("\033[0m") + "(" + block_reason_copy + ")";
        }

        // Aligned columns: TIME(ms)  BIAS     VOL      LATENCY  GATE
        // Suppressed in --quiet mode; all data still flows to MetricsLogger.
        if (!quiet) {
            std::cout << "\n  "
                      << std::setw(12) << ts_ms          << "  "
                      << std::setw(8)  << std::fixed << std::setprecision(4)
                                       << sized_signal.delta_bias_shift  << "  "
                      << std::setw(8)  << sized_signal.volatility_adjustment << "  "
                      << std::setw(6)  << latency_us << "μs"
                      << gate_str
                      << std::flush;
        }

        if (passed) {
            logger.log_trade_signal(
                sized_signal.delta_bias_shift,
                sized_signal.volatility_adjustment,
                sized_signal.confidence,
                static_cast<double>(latency_us),
                sized_signal.signal_quality);

#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
            // Open a tracked position for this signal.  In a live system the
            // entry_price would come from the OMS fill report; here we use the
            // sized bias as a normalised proxy (1.0 base).
            (void)position_tracker.open_trade(sized_signal, 1.0);
#endif
        } else {
            logger.log_risk_rejection(block_reason_copy,
                                      signal.delta_bias_shift,
                                      signal.confidence);
        }

#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        if (audit_log) {
            audit_log->log_signal(signal, passed, passed ? "" : block_reason_copy);
        }
#endif
    });

    // Load test tokens for simulator path.
    if (sys_config.token_stream.use_memory_stream) {
        token_sim.load_tokens_from_memory({
            // Fear / panic
            "crash", "panic", "collapse", "plunge", "selloff", "rout",
            // Bullish directional
            "bullish", "rally", "surge", "breakout", "rebound", "accumulate",
            // Bearish directional
            "bearish", "short", "downtrend", "distribution",
            // Volatility
            "volatile", "spike", "whipsaw", "choppy", "gamma", "vega",
            // Certainty / confidence
            "inevitable", "guarantee", "confident", "confirmed",
            // Corporate / earnings
            "earnings", "beats", "misses", "guidance", "dividend", "buyback",
            // Macro / regime
            "inflation", "fed", "pivot", "recession", "risk-on", "risk-off",
            // Analyst
            "upgrade", "downgrade", "overweight", "outperform",
            // Options
            "calls", "puts", "squeeze", "hedge",
            // Crypto / retail
            "pump", "fud", "hodl",
            // Neutral filler (tests zero-weight path)
            "the", "and", "is"
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
    std::cout << "  OMS     : " << oms_adapter->description() << "\n";
    std::cout << "  LATENCY : target p99 < " << sys_config.latency.target_latency_us << "us\n";
    if (dry_run)
        std::cout << "  DRY-RUN : signals suppressed — dictionary coverage mode\n";
    if (backtest_mode)
        std::cout << "  BACKTEST: cooldown disabled — signal emitted on every token\n";
    std::cout << DIV1 << "\n";
    std::cout << config.to_summary_string() << "\n";
    std::cout << "  TIME(ms)     BIAS      VOL       LATENCY   GATE\n";
    std::cout << DIV2;

#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
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
        // Each stream token gets a unique monotonically-increasing sequence ID so
        // MetricsLogger and dedup logs can distinguish individual stream tokens.
        std::atomic<uint64_t> stream_seq_id{0};
        stream_client->set_token_callback([&](const std::string& text) {
            process_token(text, stream_seq_id.fetch_add(1, std::memory_order_relaxed));
        });
        stream_client->set_done_callback([](const std::string& err) {
            if (!err.empty())
                spdlog::warn("stream: {}", err);
        });
        if (!stream_client->connect())
            spdlog::warn("[stream] connect() returned false — stream may not start");
    } else {
        token_sim.start();
    }
#else
    // LLMStreamClient compiled out: always use the token simulator.
    if (stream_mode) {
        spdlog::warn("--stream requested but LLMStreamClient was compiled out "
                     "(-DLLMQUANT_ENABLE_STREAM_CLIENT=OFF); falling back to simulator.");
        stream_mode = false;
    }
    token_sim.start();
#endif

    // Prometheus metrics endpoint on port 9100.
    // The snapshot is built once per second in the monitoring loop so the
    // scrape thread never contends with the hot path for latency stats.
    std::string prom_snapshot;
    std::mutex  prom_snapshot_mutex;

#ifdef LLMQUANT_PROMETHEUS_ENABLED
    uint16_t eff_stats_port = (stats_port_override != 0)
                                  ? stats_port_override
                                  : sys_config.metrics.stats_port;
    llmquant::PrometheusExporter prom_exporter({.port = eff_stats_port,
                                                .bind_address = sys_config.metrics.bind_address});
    prom_exporter.set_metrics_callback([&]() -> std::string {
        std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
        return prom_snapshot;
    });
    if (!no_prometheus) {
        if (!prom_exporter.start()) {
            spdlog::warn("PrometheusExporter failed to bind on port {}", eff_stats_port);
        }
    } else {
        spdlog::info("--no-prometheus: Prometheus scrape endpoint disabled");
    }
#else
    (void)stats_port_override;
    if (!no_prometheus)
        spdlog::info("Prometheus scrape endpoint not available "
                     "(built with LLMQUANT_ENABLE_PROMETHEUS=OFF)");
#endif

#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    uint16_t eff_health_port = (health_port_override != 0) ? health_port_override : uint16_t{8080};
    llmquant::HealthServer::Config health_cfg;
    health_cfg.port         = eff_health_port;
    health_cfg.bind_address = sys_config.metrics.bind_address;
    llmquant::HealthServer health_server(health_cfg);
    health_server.set_health_callback([&]() -> std::pair<bool, std::string> {
        auto lc_stats         = latency_ctrl.get_stats();
        auto te_stats         = trade_engine.get_stats();
        const auto& rm_stats  = risk_mgr.get_stats();
        auto uptime_s  = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - engine_start_time).count();
        bool oms_ok    = oms_adapter && oms_adapter->is_running();
#  ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
        bool cb_open         = circuit_breaker.is_open();
        double blk_rate      = circuit_breaker.block_rate();
        std::string cb_name  = circuit_breaker.state_name();
#  else
        bool cb_open         = false;
        double blk_rate      = 0.0;
        std::string cb_name  = "closed";
#  endif
#  if defined(LLMQUANT_DEDUP_ENABLED) && defined(LLMQUANT_REDIS_ENABLED)
        bool redis_ok = deduplicator.is_connected();
#  else
        bool redis_ok = false;
#  endif
#  ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        bool stream_stale = stale_detector.is_stale();
#  else
        bool stream_stale = false;
#  endif
        bool ok = !cb_open && oms_ok && !stream_stale;
        uint64_t blocked = rm_stats.signals_blocked_magnitude.load()
                         + rm_stats.signals_blocked_confidence.load()
                         + rm_stats.signals_blocked_rate.load()
                         + rm_stats.signals_blocked_drawdown.load();
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "{\"ok\":%s"
            ",\"uptime_s\":%lld"
            ",\"circuit_breaker\":\"%s\""
            ",\"block_rate\":%.4f"
            ",\"oms_connected\":%s"
            ",\"stream_stale\":%s"
            ",\"p99_latency_us\":%lld"
            ",\"signals_generated\":%llu"
            ",\"signals_blocked\":%llu"
            ",\"redis_connected\":%s"
            ",\"version\":\"%s\"}",
            ok ? "true" : "false",
            static_cast<long long>(uptime_s),
            cb_name.c_str(),
            blk_rate,
            oms_ok ? "true" : "false",
            stream_stale ? "true" : "false",
            static_cast<long long>(lc_stats.p99_latency.count()),
            static_cast<unsigned long long>(te_stats.signals_generated.load()),
            static_cast<unsigned long long>(blocked),
            redis_ok ? "true" : "false",
            LLMQUANT_VERSION
        );
        return {ok, std::string(buf)};
    });
    if (!no_health_server) {
        if (!health_server.start())
            spdlog::warn("[health_server] failed to bind on port {}", eff_health_port);
    } else {
        spdlog::info("--no-health: HTTP /health endpoint disabled");
    }
#endif  // LLMQUANT_HEALTH_SERVER_ENABLED

    // Main monitoring loop — prints a rolling stats bar every second.
    // Interruptible sleep: wake every 100ms to check g_running so that
    // SIGINT/SIGTERM is handled promptly regardless of --stats-interval.
    uint64_t last_tick = 0;
    std::string last_regime;    // For regime-change transition alerts.
    std::string last_morphology; // Last detected sparkline pattern name.
    while (g_running) {
        {
            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(stats_interval_ms);
            while (g_running && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!g_running) break;
        }

        auto stats    = latency_ctrl.get_stats();
        auto pressure = latency_ctrl.get_pressure();

        // Update ingestion pressure.
        // Normalise the per-tick token count to tokens/second regardless of
        // the configured stats_interval_ms (fixes pressure and TPS display
        // when --stats-interval differs from the default 1000ms).
        uint64_t raw_count = token_count_window.exchange(0);
        double   tps_d     = (stats_interval_ms > 0)
                                 ? (static_cast<double>(raw_count) * 1000.0
                                    / static_cast<double>(stats_interval_ms))
                                 : static_cast<double>(raw_count);
        uint64_t tps = static_cast<uint64_t>(tps_d + 0.5);  // rounded for display
        double   max_tps = stream_mode
                               ? sys_config.pressure.max_ingestion_rate_tps   // gpt-4o emits ~10-30 tokens/s
                               : static_cast<double>(1000000 / std::max(1, sys_config.token_stream.token_interval_ms));
        latency_ctrl.update_ingestion_pressure(tps_d, max_tps);

        // Queue pressure via suppressed-signal count.
        auto eng_stats = trade_engine.get_stats();  // snapshot, not reference
        latency_ctrl.update_queue_pressure(eng_stats.signals_suppressed.load(), 1024);

        double backoff = latency_ctrl.get_backoff_multiplier();
        double cpu_fraction = get_process_cpu_fraction();  // sampled once per loop tick

#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
        // Feed current P99 to the adaptive cooldown controller.
        // If pressure is detected, update the trade engine's cooldown on the fly.
        {
            double p99_us_f = static_cast<double>(stats.p99_latency.count());
            adaptive_cooldown.update_p99(p99_us_f);
            if (adaptive_cooldown.is_under_pressure()) {
                auto new_cd = adaptive_cooldown.get_cooldown();
                auto te_cfg = trade_engine.get_config();
                if (te_cfg.signal_cooldown != new_cd) {
                    te_cfg.signal_cooldown = new_cd;
                    trade_engine.update_config(te_cfg);
                }
            }
        }
#endif

#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        // Poll stall watchdog — fires on_stall if no token has arrived within timeout.
        stream_health.poll();
#endif

#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
        // Feed current P99 to the latency budget enforcer.
        // Callbacks fire on tier transitions; Breaker tier trips the circuit breaker.
        {
            int64_t p99_us_i = static_cast<int64_t>(stats.p99_latency.count());
            (void)latency_budget_enforcer.check(p99_us_i);
        }
#endif

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.check();
#endif

        // Colour the P99 value: green < 10μs, yellow < 50μs, red otherwise.
        auto p99 = stats.p99_latency.count();
        const char* p99_colour =
            (p99 < 10)  ? C("\033[32m") :
            (p99 < 50)  ? C("\033[33m") : C("\033[31m");

        // Colour the pressure bar.
        const char* press_colour =
            (pressure.composite < 0.5) ? C("\033[32m") :
            (pressure.composite < 0.8) ? C("\033[33m") : C("\033[31m");

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.check();
#endif

        // Use the LLMAdapter's own token counter for the stats bar — variance_n
        // is reset every 60 seconds (Welford reset) and would undercount after
        // the first reset interval.
        uint64_t tokens_total = llm_adapter.get_stats().tokens_processed;

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
#ifdef LLMQUANT_PROMETHEUS_ENABLED
        {
            std::ostringstream snap;
            snap << "# HELP llmquant_signals_generated_total Total trade signals generated\n"
                 << "# TYPE llmquant_signals_generated_total counter\n"
                 << "llmquant_signals_generated_total " << eng_stats.signals_generated.load() << "\n"
                 << "# HELP llmquant_signals_suppressed_total Signals with no callback or sink (fully suppressed)\n"
                 << "# TYPE llmquant_signals_suppressed_total counter\n"
                 << "llmquant_signals_suppressed_total " << eng_stats.signals_suppressed.load() << "\n"
                 << "# HELP llmquant_signals_aged_out_total Signals suppressed by the staleness guard\n"
                 << "# TYPE llmquant_signals_aged_out_total counter\n"
                 << "llmquant_signals_aged_out_total " << eng_stats.signals_aged_out.load() << "\n"
                 << "# HELP llmquant_signals_cooldown_suppressed_total Signals skipped because the signal cooldown had not elapsed\n"
                 << "# TYPE llmquant_signals_cooldown_suppressed_total counter\n"
                 << "llmquant_signals_cooldown_suppressed_total " << eng_stats.signals_suppressed_cooldown.load() << "\n"
                 << "# HELP llmquant_accumulator_clamped_total Times the bias accumulator cap was applied\n"
                 << "# TYPE llmquant_accumulator_clamped_total counter\n"
                 << "llmquant_accumulator_clamped_total " << eng_stats.accumulator_clamped.load() << "\n"
                 << "# HELP llmquant_memory_sink_size Current number of signals buffered in the in-memory sink\n"
                 << "# TYPE llmquant_memory_sink_size gauge\n"
                 << "llmquant_memory_sink_size " << memory_sink->size() << "\n"
                 << "# HELP llmquant_memory_sink_dropped_total Signals evicted from memory sink due to capacity cap\n"
                 << "# TYPE llmquant_memory_sink_dropped_total counter\n"
                 << "llmquant_memory_sink_dropped_total " << memory_sink->dropped_count() << "\n"
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
                 << "llmquant_oms_update_count_total "    << oms_adapter->update_count()    << "\n"
                 << "# HELP llmquant_oms_error_count_total Total OMS connection errors\n"
                 << "# TYPE llmquant_oms_error_count_total counter\n"
                 << "llmquant_oms_error_count_total "     << oms_adapter->error_count()     << "\n"
                 << "# HELP llmquant_oms_reconnect_count_total Total FIX session reconnect attempts\n"
                 << "# TYPE llmquant_oms_reconnect_count_total counter\n"
                 << "llmquant_oms_reconnect_count_total " << oms_adapter->reconnect_count() << "\n"
#ifdef LLMQUANT_DEDUP_ENABLED
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
#endif
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
                 << "# HELP llmquant_adapter_cache_hit_rate Fraction of token lookups served from dictionary [0,1]\n"
                 << "# TYPE llmquant_adapter_cache_hit_rate gauge\n"
                 << "llmquant_adapter_cache_hit_rate " << std::fixed << std::setprecision(6) << llm_adapter.get_cache_hit_rate() << "\n"
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
                 << "# TYPE llmquant_uptime_seconds gauge\n"
                 << "llmquant_uptime_seconds " << std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - engine_start_time).count() << "\n"
                 << "# HELP llmquant_dry_run Whether the engine is running in dry-run mode (1=yes)\n"
                 << "# TYPE llmquant_dry_run gauge\n"
                 << "llmquant_dry_run " << (dry_run ? 1 : 0) << "\n"
                 << "# HELP llmquant_shadow_mode_active 1 when RiskManager shadow/dry-run mode is active (gates evaluate but never block)\n"
                 << "# TYPE llmquant_shadow_mode_active gauge\n"
                 << "llmquant_shadow_mode_active " << (risk_mgr.get_config().dry_run_mode ? 1 : 0) << "\n"
                 << "# HELP llmquant_backtest_mode Whether the engine is running in backtest mode (1=yes)\n"
                 << "# TYPE llmquant_backtest_mode gauge\n"
                 << "llmquant_backtest_mode " << (backtest_mode ? 1 : 0) << "\n"
                 << "# HELP llmquant_version_info Engine version info (always 1; use labels for version string)\n"
                 << "# TYPE llmquant_version_info gauge\n"
                 << "llmquant_version_info{version=\"" LLMQUANT_VERSION "\"} 1\n"
                 << "# HELP llmquant_start_time_seconds Unix timestamp (seconds) when the engine process started\n"
                 << "# TYPE llmquant_start_time_seconds gauge\n"
                 << "llmquant_start_time_seconds " << engine_start_unix_s << "\n"
                 << "# HELP llmquant_process_rss_bytes Process resident set size in bytes\n"
                 << "# TYPE llmquant_process_rss_bytes gauge\n"
                 << "llmquant_process_rss_bytes " << get_process_rss_bytes() << "\n"
                 << "# HELP llmquant_process_cpu_fraction Process CPU usage fraction since last scrape (0=idle, 1=one full core)\n"
                 << "# TYPE llmquant_process_cpu_fraction gauge\n"
                 << "llmquant_process_cpu_fraction " << std::setprecision(4) << cpu_fraction << "\n"
                 << "# HELP llmquant_avg_signal_strength Running Welford mean of |delta_bias_shift|\n"
                 << "# TYPE llmquant_avg_signal_strength gauge\n"
                 << "llmquant_avg_signal_strength " << std::setprecision(6)
                     << eng_stats.avg_signal_strength.load() << "\n"
                 << "# HELP llmquant_latency_window_fill_ratio Fraction of sample window populated [0,1]\n"
                 << "# TYPE llmquant_latency_window_fill_ratio gauge\n"
                 << "llmquant_latency_window_fill_ratio " << std::fixed << std::setprecision(4) << latency_ctrl.get_window_fill_ratio() << "\n"
                 << "# HELP llmquant_latency_measurements_total Total latency samples recorded\n"
                 << "# TYPE llmquant_latency_measurements_total counter\n"
                 << "llmquant_latency_measurements_total " << stats.measurements << "\n"
                 << "# HELP llmquant_signal_age_threshold_us Configured staleness guard threshold (0=disabled)\n"
                 << "# TYPE llmquant_signal_age_threshold_us gauge\n"
                 << "llmquant_signal_age_threshold_us " << trade_engine.get_config().max_signal_age_us << "\n"
                 << "# HELP llmquant_min_bias_threshold Configured noise-filter minimum |bias| threshold (0=disabled)\n"
                 << "# TYPE llmquant_min_bias_threshold gauge\n"
                 << "llmquant_min_bias_threshold " << trade_engine.get_config().min_bias_threshold << "\n"
                 << "# HELP llmquant_min_vol_threshold Configured noise-filter minimum |vol| threshold (0=disabled)\n"
                 << "# TYPE llmquant_min_vol_threshold gauge\n"
                 << "llmquant_min_vol_threshold " << trade_engine.get_config().min_vol_threshold << "\n"
                 << "# HELP llmquant_max_accumulated_bias Configured accumulator cap (0=disabled)\n"
                 << "# TYPE llmquant_max_accumulated_bias gauge\n"
                 << "llmquant_max_accumulated_bias " << trade_engine.get_config().max_accumulated_bias << "\n"
                 << "# HELP llmquant_p5_latency_us 5th-percentile latency of the sample window (microseconds)\n"
                 << "# TYPE llmquant_p5_latency_us gauge\n"
                 << "llmquant_p5_latency_us " << stats.p5_latency.count() << "\n"
                 << "# HELP llmquant_p25_latency_us 25th-percentile (Q1) latency (microseconds)\n"
                 << "# TYPE llmquant_p25_latency_us gauge\n"
                 << "llmquant_p25_latency_us " << stats.p25_latency.count() << "\n"
                 << "# HELP llmquant_peak_bias Peak absolute value of the accumulated bias since last reset\n"
                 << "# TYPE llmquant_peak_bias gauge\n"
                 << "llmquant_peak_bias " << std::fixed << std::setprecision(6) << trade_engine.get_stats().peak_bias.load() << "\n"
                 << "# HELP llmquant_signal_efficiency Ratio of signals emitted to tokens processed [0,1]\n"
                 << "# TYPE llmquant_signal_efficiency gauge\n"
                 << "llmquant_signal_efficiency " << std::setprecision(6) << trade_engine.get_signal_efficiency() << "\n"
                 << "# HELP llmquant_tokens_per_second Token throughput (tokens processed per second)\n"
                 << "# TYPE llmquant_tokens_per_second gauge\n"
                 << "llmquant_tokens_per_second " << std::setprecision(2) << trade_engine.get_tokens_per_second() << "\n"
                 << "# HELP llmquant_avg_signal_quality Welford running mean of signal_quality [0,1]\n"
                 << "# TYPE llmquant_avg_signal_quality gauge\n"
                 << "llmquant_avg_signal_quality " << std::setprecision(6) << eng_stats.avg_signal_quality.load() << "\n"
                 << [&]() -> std::string {
                        double q_ema = trade_engine.get_signal_quality_ema();
                        if (q_ema < 0.0) return "";
                        std::ostringstream o;
                        o << "# HELP llmquant_signal_quality_ema EMA(alpha=0.1) of signal_quality [0,1]; omitted until first signal\n"
                          << "# TYPE llmquant_signal_quality_ema gauge\n"
                          << "llmquant_signal_quality_ema " << std::setprecision(6) << q_ema << "\n";
                        return o.str();
                    }()
#ifdef LLMQUANT_DEDUP_ENABLED
                 << "# HELP llmquant_dedup_duplicate_rate Fraction of checked tokens that were duplicates [0,1]\n"
                 << "# TYPE llmquant_dedup_duplicate_rate gauge\n"
                 << "llmquant_dedup_duplicate_rate " << [&]() -> double {
                        uint64_t dupes = dedup_backend->total_duplicates();
                        uint64_t novel = dedup_backend->total_novel();
                        uint64_t total = novel + dupes;
                        return (total > 0) ? (static_cast<double>(dupes) / static_cast<double>(total)) : 0.0;
                    }() << "\n"
#endif
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                 << [&]() -> std::string {
                        if (!audit_log) return "";
                        std::ostringstream o;
                        o << "# HELP llmquant_audit_records_written_total Total signal audit records written to disk\n"
                          << "# TYPE llmquant_audit_records_written_total counter\n"
                          << "llmquant_audit_records_written_total " << audit_log->records_written() << "\n"
                          << "# HELP llmquant_audit_records_dropped_total Signal audit records dropped (queue full)\n"
                          << "# TYPE llmquant_audit_records_dropped_total counter\n"
                          << "llmquant_audit_records_dropped_total " << audit_log->records_dropped() << "\n";
                        return o.str();
                    }()
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
                 << "# HELP llmquant_circuit_breaker_state Circuit breaker state: 0=closed 1=open 2=half_open\n"
                 << "# TYPE llmquant_circuit_breaker_state gauge\n"
                 << "llmquant_circuit_breaker_state " << static_cast<int>(circuit_breaker.state()) << "\n"
                 << "# HELP llmquant_circuit_breaker_block_rate EMA block rate observed by circuit breaker [0,1]\n"
                 << "# TYPE llmquant_circuit_breaker_block_rate gauge\n"
                 << "llmquant_circuit_breaker_block_rate " << std::setprecision(4) << circuit_breaker.block_rate() << "\n"
                 << "# HELP llmquant_circuit_breaker_trips_total Times circuit has tripped to OPEN\n"
                 << "# TYPE llmquant_circuit_breaker_trips_total counter\n"
                 << "llmquant_circuit_breaker_trips_total " << circuit_breaker.trips() << "\n"
                 << "# HELP llmquant_circuit_breaker_recoveries_total Times circuit has recovered to CLOSED\n"
                 << "# TYPE llmquant_circuit_breaker_recoveries_total counter\n"
                 << "llmquant_circuit_breaker_recoveries_total " << circuit_breaker.recoveries() << "\n"
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
                 << "# HELP llmquant_stream_stale Whether the LLM token stream is currently silent (1=stale)\n"
                 << "# TYPE llmquant_stream_stale gauge\n"
                 << "llmquant_stream_stale " << (stale_detector.is_stale() ? 1 : 0) << "\n"
                 << "# HELP llmquant_stream_stale_events_total Times the token stream went silent\n"
                 << "# TYPE llmquant_stream_stale_events_total counter\n"
                 << "llmquant_stream_stale_events_total " << stale_detector.stale_events() << "\n"
                 << "# HELP llmquant_stream_ms_since_last_token Milliseconds since last token arrived\n"
                 << "# TYPE llmquant_stream_ms_since_last_token gauge\n"
                 << "llmquant_stream_ms_since_last_token " << stale_detector.ms_since_last_token() << "\n"
#endif
                ;
            // Signal quality histogram — per-bucket counts emitted as a Prometheus histogram.
            {
                auto qhb = trade_engine.get_quality_histogram();
                snap << "# HELP llmquant_signal_quality_histogram Distribution of emitted signal quality scores\n"
                     << "# TYPE llmquant_signal_quality_histogram histogram\n";
                uint64_t cumulative = 0;
                for (const auto& b : qhb) {
                    cumulative += b.count;
                    snap << "llmquant_signal_quality_histogram_bucket{le=\"" << b.upper_bound << "\"} " << cumulative << "\n";
                }
                snap << "llmquant_signal_quality_histogram_bucket{le=\"+Inf\"} " << cumulative << "\n";
                snap << "llmquant_signal_quality_histogram_count " << cumulative << "\n";
                double avg_q = eng_stats.avg_signal_quality.load();
                snap << "llmquant_signal_quality_histogram_sum " << std::setprecision(6)
                     << (std::isfinite(avg_q) ? avg_q * static_cast<double>(cumulative) : 0.0) << "\n";
            }
            // Prometheus native histogram — cumulative latency buckets.
            {
                auto hb = latency_ctrl.histogram_buckets();
                snap << "# HELP llmquant_token_latency_us Cumulative latency histogram of token-to-signal processing time (µs)\n"
                     << "# TYPE llmquant_token_latency_us histogram\n";
                uint64_t last_count = 0;
                for (const auto& b : hb) {
                    if (std::isinf(b.upper_bound_us)) {
                        snap << "llmquant_token_latency_us_bucket{le=\"+Inf\"} " << b.count << "\n";
                    } else {
                        snap << "llmquant_token_latency_us_bucket{le=\"" << b.upper_bound_us << "\"} " << b.count << "\n";
                    }
                    last_count = b.count;
                }
                snap << "llmquant_token_latency_us_count " << last_count << "\n"
                     << "llmquant_token_latency_us_sum "   << latency_ctrl.get_total_latency_us() << "\n";
            }
            {
                auto hs = latency_ctrl.get_health_state();
                snap << "# HELP llmquant_latency_warmed_up 1 when the sample window is >=50% populated, 0 during warmup\n"
                     << "# TYPE llmquant_latency_warmed_up gauge\n"
                     << "llmquant_latency_warmed_up " << (hs.warmed_up ? 1 : 0) << "\n"
                     << "# HELP llmquant_latency_budget_remaining_us Signed latency budget: target_us minus p99_us (negative = over budget)\n"
                     << "# TYPE llmquant_latency_budget_remaining_us gauge\n"
                     << "llmquant_latency_budget_remaining_us " << std::fixed << std::setprecision(1) << hs.budget_remaining_us << "\n";
            }
            snap << "# HELP llmquant_drawdown_cumulative_bias Current cumulative bias in the drawdown window\n"
                 << "# TYPE llmquant_drawdown_cumulative_bias gauge\n"
                 << "llmquant_drawdown_cumulative_bias " << std::fixed << std::setprecision(4) << risk_mgr.get_cumulative_bias() << "\n"
                 << "# HELP llmquant_risk_pass_rate_pct Percentage of signals that passed all risk gates (0-100)\n"
                 << "# TYPE llmquant_risk_pass_rate_pct gauge\n"
                 << "llmquant_risk_pass_rate_pct " << std::setprecision(4) << ((1.0 - risk_mgr.get_blocked_rate()) * 100.0) << "\n"
                 << "# HELP llmquant_slo_breach_rate Fraction of latency samples that exceeded the p99 target [0,1]\n"
                 << "# TYPE llmquant_slo_breach_rate gauge\n"
                 << "llmquant_slo_breach_rate " << std::setprecision(6) << latency_ctrl.get_slo_breach_rate() << "\n"
                 << "# HELP llmquant_drawdown_utilization Fraction of drawdown budget consumed in current window [0,1]\n"
                 << "# TYPE llmquant_drawdown_utilization gauge\n"
                 << "llmquant_drawdown_utilization " << std::setprecision(4) << risk_mgr.get_drawdown_utilization() << "\n"
                 << "# HELP llmquant_rate_limit_utilization Fraction of per-second rate cap consumed in current window [0,1]\n"
                 << "# TYPE llmquant_rate_limit_utilization gauge\n"
                 << "llmquant_rate_limit_utilization " << std::setprecision(4) << risk_mgr.get_rate_limit_utilization() << "\n"
                 << "# HELP llmquant_noise_filtered_total Tokens suppressed by the min-bias noise gate\n"
                 << "# TYPE llmquant_noise_filtered_total counter\n"
                 << "llmquant_noise_filtered_total " << eng_stats.noise_filtered.load() << "\n"
                 << "# HELP llmquant_bias_reversals_total Number of times accumulated_bias changed direction (sign reversal / momentum crossover)\n"
                 << "# TYPE llmquant_bias_reversals_total counter\n"
                 << "llmquant_bias_reversals_total " << eng_stats.bias_reversals.load() << "\n"
                 << "# HELP llmquant_risk_healthy Whether all risk gates are nominally healthy (1=yes)\n"
                 << "# TYPE llmquant_risk_healthy gauge\n"
                 << "llmquant_risk_healthy " << (risk_mgr.is_healthy() ? 1 : 0) << "\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                 << "# HELP llmquant_dedup_dup_rate_pct Duplicate token rate as percentage [0,100]\n"
                 << "# TYPE llmquant_dedup_dup_rate_pct gauge\n"
                 << "llmquant_dedup_dup_rate_pct " << [&]() -> double {
                        uint64_t nov = dedup_backend->total_novel();
                        uint64_t dup = dedup_backend->total_duplicates();
                        uint64_t tot = nov + dup;
                        return (tot > 0) ? (static_cast<double>(dup) * 100.0 / static_cast<double>(tot)) : 0.0;
                    }() << "\n"
#endif
                ;
            // Top-5 influential tokens as labeled gauges for Grafana dashboards.
            {
                snap << "# HELP llmquant_top_influence_token Composite influence score (freq+bias blend) [0,1]\n"
                     << "# TYPE llmquant_top_influence_token gauge\n";
                for (const auto& [tok, score] : llm_adapter.export_hot_tokens(5)) {
                    // Prometheus text format requires escaping \, ", and \n in label values.
                    std::string safe_tok;
                    safe_tok.reserve(tok.size());
                    for (char c : tok) {
                        if (c == '\\')     { safe_tok += "\\\\"; }
                        else if (c == '"') { safe_tok += "\\\""; }
                        else if (c == '\n') { safe_tok += "\\n"; }
                        else               { safe_tok += c; }
                    }
                    snap << "llmquant_top_influence_token{token=\"" << safe_tok << "\"} "
                         << std::setprecision(4) << score << "\n";
                }
            }
            std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
            prom_snapshot = snap.str();
        }
#endif // LLMQUANT_PROMETHEUS_ENABLED

        // Cache hit rate for LLMAdapter dictionary efficiency.
        auto adapter_stats = llm_adapter.get_stats();
        uint64_t hit_pct = (adapter_stats.tokens_processed > 0)
            ? (adapter_stats.cache_hits * 100 / adapter_stats.tokens_processed) : 0;

        // Log periodic latency snapshot and pipeline health (independent of --quiet).
        logger.log_latency_measurement(static_cast<uint64_t>(p99));
        {
            bool slo_healthy = (p99 <= sys_config.latency.target_latency_us);
            logger.log_pipeline_health(slo_healthy,
                                       latency_ctrl.get_slo_breach_rate(),
                                       backoff);
            if (!slo_healthy && last_tick != stats.measurements) {
                spdlog::warn("P99 latency {}us exceeds target {}us (breach rate {:.1f}%)",
                             p99, sys_config.latency.target_latency_us,
                             latency_ctrl.get_slo_breach_rate() * 100.0);
            }
        }
        // Log system resource usage once per second (memory RSS + CPU).
        // MetricsLogger::log_system_stats expects cpu_usage as percentage (0-100);
        // get_process_cpu_fraction() returns a fraction [0, N_cores], so multiply by 100.
        logger.log_system_stats(get_process_rss_bytes(), cpu_fraction * 100.0);

        // Compute rolling regime classification from sparkline ring (lag-1 AC).
        // mean > +0.05 && AC > 0.10  → BULL  mean < -0.05 && AC > 0.10  → BEAR
        // AC < -0.15 → CHOP  |mean| < 0.02 → FLAT  else → NOIS
        std::string current_regime;
        std::string current_regime_str;
        {
            int rgm_head = spark_head.load(std::memory_order_relaxed);
            int rgm_n    = std::min(rgm_head, kSparkSlots);
            if (rgm_n >= 4) {
                int rgm_start = (rgm_head >= kSparkSlots) ? (rgm_head % kSparkSlots) : 0;
                double rgm_sum = 0.0;
                for (int i = 0; i < rgm_n; ++i)
                    rgm_sum += spark_ring[(rgm_start + i) % kSparkSlots];
                double rgm_mean = rgm_sum / static_cast<double>(rgm_n);
                double rgm_cov = 0.0, rgm_var = 0.0;
                for (int i = 1; i < rgm_n; ++i) {
                    double x0 = spark_ring[(rgm_start + i - 1) % kSparkSlots] - rgm_mean;
                    double x1 = spark_ring[(rgm_start + i)     % kSparkSlots] - rgm_mean;
                    rgm_cov += x0 * x1;
                    rgm_var += x0 * x0;
                }
                double rgm_ac = (rgm_var > 1e-12) ? rgm_cov / rgm_var : 0.0;
                const char* rgm_col;
                if      (rgm_mean >  0.05 && rgm_ac >  0.10) { current_regime = "BULL"; rgm_col = "\033[32m"; }
                else if (rgm_mean < -0.05 && rgm_ac >  0.10) { current_regime = "BEAR"; rgm_col = "\033[31m"; }
                else if (rgm_ac < -0.15)                      { current_regime = "CHOP"; rgm_col = "\033[33m"; }
                else if (std::abs(rgm_mean) < 0.02)           { current_regime = "FLAT"; rgm_col = "\033[90m"; }
                else                                          { current_regime = "NOIS"; rgm_col = "\033[35m"; }
                current_regime_str = std::string("  RGM:") + C(rgm_col) + current_regime + C("\033[0m");
            }
        }

        // Signal morphology detector: scan the last 6 sparkline values for
        // named candlestick-like shapes in the bias stream.
        // Patterns: RALLY (4+ up), SELLOFF (4+ down), V_RVSL (3d+3u),
        //           INV_V (3u+3d), CONSOLIDATE (6 within ±0.02).
        std::string current_morphology;
        {
            int morph_head = spark_head.load(std::memory_order_relaxed);
            int morph_n    = std::min(morph_head, kSparkSlots);
            if (morph_n >= 6) {
                int morph_start = (morph_head >= kSparkSlots) ? (morph_head % kSparkSlots) : 0;
                // Extract the 6 most-recent values (indices morph_n-6 .. morph_n-1).
                double v[6];
                for (int i = 0; i < 6; ++i)
                    v[i] = spark_ring[(morph_start + morph_n - 6 + i) % kSparkSlots];

                // Compute differences
                int up6 = 0, dn6 = 0;
                for (int i = 1; i < 6; ++i) {
                    if (v[i] > v[i-1] + 0.005) ++up6;
                    else if (v[i] < v[i-1] - 0.005) ++dn6;
                }
                // First-half and second-half runs
                int up3a = (v[1]>v[0]+0.005)+(v[2]>v[1]+0.005);
                int dn3a = (v[1]<v[0]-0.005)+(v[2]<v[1]-0.005);
                int up3b = (v[4]>v[3]+0.005)+(v[5]>v[4]+0.005);
                int dn3b = (v[4]<v[3]-0.005)+(v[5]<v[4]-0.005);
                double band = *std::max_element(v, v+6) - *std::min_element(v, v+6);

                if      (up6 >= 4)                  current_morphology = "RALLY";
                else if (dn6 >= 4)                  current_morphology = "SELLOFF";
                else if (dn3a >= 2 && up3b >= 2)    current_morphology = "V_RVSL";
                else if (up3a >= 2 && dn3b >= 2)    current_morphology = "INV_V";
                else if (band < 0.02)               current_morphology = "CONSOL";

                if (!current_morphology.empty() && current_morphology != last_morphology) {
                    spdlog::info("[morphology] pattern detected: {}", current_morphology);
                }
                last_morphology = current_morphology;
            }
        }

        // Overwrite the stats line in-place. Suppressed in --quiet mode.
        if (!quiet) {
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
#ifdef LLMQUANT_DEDUP_ENABLED
                      << "  DEDUP:" << dedup_backend->total_duplicates()
#endif
                      << "  NOISE:" << trade_engine.get_stats().noise_filtered.load()
                      << "  PASS:" << risk_mgr.get_stats().signals_passed.load()
                      << "  BLOCK:" << blocked
                      << "  RATE%:" << [&]() -> uint64_t {
                            uint64_t passed = risk_mgr.get_stats().signals_passed.load();
                            uint64_t total  = (passed > UINT64_MAX - blocked) ? UINT64_MAX : passed + blocked;
                            return (total > 0) ? (passed * 100 / total) : 100;
                         }()
                      << (!stream_mode ? (std::string("  DROPS:") + std::to_string(token_sim.get_stats().ring_buffer_drops.load())) : "")
                      << [&]() -> std::string {
                            double q_ema = trade_engine.get_signal_quality_ema();
                            if (q_ema < 0.0) return "";
                            std::ostringstream o;
                            o << "  Q-EMA:" << std::fixed << std::setprecision(2) << q_ema;
                            return o.str();
                         }()
                      << [&]() -> std::string {
                            // Signal velocity: bias per second. Skip if no signal yet.
                            double vel = sig_velocity.load(std::memory_order_relaxed);
                            if (vel == 0.0) return "";
                            std::ostringstream o;
                            o << "  VEL:";
                            if (vel > 0.01)       o << C("\033[32m");
                            else if (vel < -0.01) o << C("\033[31m");
                            else                  o << C("\033[90m");
                            o << std::showpos << std::fixed << std::setprecision(3) << vel;
                            o << "/s" << C("\033[0m");
                            return o.str();
                         }()
                      << [&]() -> std::string {
                            // Render 24-slot sparkline of recent delta_bias_shift values.
                            // Values are clamped to [-1, 1] and mapped to 8 block levels.
                            // Slots not yet written (head < kSparkSlots) render as '·'.
                            static const char* kBlocks[] = {
                                "▁","▂","▃","▄","▅","▆","▇","█"
                            };
                            int head = spark_head.load(std::memory_order_relaxed);
                            if (head == 0) return "";
                            std::string out = "  BIAS:";
                            int filled = std::min(head, kSparkSlots);
                            int start  = (head >= kSparkSlots) ? (head % kSparkSlots) : 0;
                            for (int i = 0; i < kSparkSlots; ++i) {
                                if (i >= filled) { out += C("\033[90m"); out += "·"; out += C("\033[0m"); continue; }
                                int slot = (start + i) % kSparkSlots;
                                double v = spark_ring[slot];
                                // Map [-1,1] → [0,7]; neutral (0) → level 3 (▄)
                                double clamped = std::max(-1.0, std::min(1.0, v));
                                int level = static_cast<int>((clamped + 1.0) / 2.0 * 7.0 + 0.5);
                                // Colour: positive=green, negative=red, near-zero=yellow
                                if (v > 0.05)       out += C("\033[32m");
                                else if (v < -0.05) out += C("\033[31m");
                                else                out += C("\033[33m");
                                out += kBlocks[level];
                                out += C("\033[0m");
                            }
                            return out;
                         }()
                      << [&]() -> std::string { return current_regime_str; }()
                      << [&]() -> std::string {
                            if (current_morphology.empty()) return "";
                            // Colour by pattern type
                            const char* col;
                            if      (current_morphology == "RALLY")  col = "\033[32m";
                            else if (current_morphology == "SELLOFF") col = "\033[31m";
                            else if (current_morphology == "V_RVSL") col = "\033[36m";
                            else if (current_morphology == "INV_V")  col = "\033[35m";
                            else                                      col = "\033[90m";
                            return std::string("  MORPH:") + C(col) + current_morphology + C("\033[0m");
                         }()
                      << [&]() -> std::string {
                            // Bias Sharpe: mean(bias)/stddev(bias) over the sparkline window.
                            // Signal-to-noise ratio. |SHP|>1.0=directional, <0.5=noise.
                            int sh_n = std::min(spark_head.load(std::memory_order_relaxed), kSparkSlots);
                            if (sh_n < 4) return "";
                            int sh_start = (spark_head.load(std::memory_order_relaxed) >= kSparkSlots)
                                           ? (spark_head.load(std::memory_order_relaxed) % kSparkSlots) : 0;
                            double sh_mean = 0.0, sh_m2 = 0.0;
                            for (int i = 0; i < sh_n; ++i) {
                                double x = spark_ring[(sh_start + i) % kSparkSlots];
                                double d = x - sh_mean;
                                sh_mean += d / (i + 1);
                                sh_m2   += d * (x - sh_mean);
                            }
                            double sh_std = (sh_n > 1 && sh_m2 > 1e-12)
                                            ? std::sqrt(sh_m2 / (sh_n - 1)) : 0.0;
                            if (sh_std < 1e-9) return "";
                            double sharpe = sh_mean / sh_std;
                            std::ostringstream o;
                            o << "  SHP:";
                            if      (sharpe >  1.0) o << C("\033[32m");
                            else if (sharpe < -1.0) o << C("\033[31m");
                            else                    o << C("\033[90m");
                            o << std::showpos << std::fixed << std::setprecision(2) << sharpe << C("\033[0m");
                            return o.str();
                         }()
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
                      << [&]() -> std::string {
                            // Token entropy: 0=focused/repetitive, 1=uniform/noisy.
                            double h     = entropy_monitor.entropy();
                            bool focused = entropy_monitor.is_focused();
                            std::ostringstream o;
                            o << "  ENT:";
                            if      (focused)  o << C("\033[32m");   // green  = focused
                            else if (h > 0.75) o << C("\033[31m");   // red    = noisy
                            else               o << C("\033[33m");   // yellow = mixed
                            o << std::fixed << std::setprecision(2) << h << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
                      << [&]() -> std::string {
                            // Narrative similarity: 1=consistent, 0=topic break.
                            double sim  = narrative_detector.get_similarity();
                            bool brk    = narrative_detector.is_narrative_break();
                            std::ostringstream o;
                            o << "  NRR:";
                            if      (brk)        o << C("\033[31m");   // red    = break
                            else if (sim > 0.75)  o << C("\033[32m");   // green  = stable
                            else                  o << C("\033[33m");   // yellow = shifting
                            o << std::fixed << std::setprecision(2) << sim << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
                      << [&]() -> std::string {
                            // Hurst exponent: >0.55=trending, <0.45=mean-rev, else random.
                            double h = fractal_dim.hurst();
                            std::ostringstream o;
                            o << "  FRC:";
                            if      (h > 0.55) o << C("\033[36m");   // cyan   = trending
                            else if (h < 0.45) o << C("\033[35m");   // magenta= mean-rev
                            else               o << C("\033[90m");   // grey   = random
                            o << std::fixed << std::setprecision(2) << h << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
                      << [&]() -> std::string {
                            // Context fill fraction: green=ok, yellow=warn, red=critical.
                            double f = context_budget.fill_fraction();
                            std::ostringstream o;
                            o << "  CTX:";
                            if      (f >= 0.90) o << C("\033[31m");  // red    = critical
                            else if (f >= 0.70) o << C("\033[33m");  // yellow = warn
                            else                o << C("\033[32m");  // green  = normal
                            o << std::fixed << std::setprecision(0) << (f * 100.0) << "%" << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
                      << [&]() -> std::string {
                            // MACD histogram: positive=bullish momentum, negative=bearish.
                            double hist = signal_momentum_osc.histogram();
                            std::ostringstream o;
                            o << "  OSC:";
                            if      (hist >  0.005) o << C("\033[32m");  // green  = bullish
                            else if (hist < -0.005) o << C("\033[31m");  // red    = bearish
                            else                    o << C("\033[90m");  // grey   = flat
                            o << std::showpos << std::fixed << std::setprecision(3) << hist << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
                      << [&]() -> std::string {
                            // Dominant cycle period; highlighted when cyclic pattern detected.
                            int    p  = sentiment_cycle.dominant_period();
                            bool   cy = sentiment_cycle.is_cyclic();
                            std::ostringstream o;
                            o << "  CYC:";
                            if      (cy && p > 0)  o << C("\033[36m");  // cyan  = active cycle
                            else if (p > 0)         o << C("\033[33m");  // yellow= weak cycle
                            else                    o << C("\033[90m");  // grey  = none
                            if (p > 0) o << p;
                            else       o << "?";
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
                      << [&]() -> std::string {
                            // Normalized self-information: 0=expected, 1=maximally surprising.
                            double s   = signal_surprise.surprise();
                            bool   hi  = signal_surprise.is_high_surprise();
                            std::ostringstream o;
                            o << "  SUR:";
                            if      (hi)       o << C("\033[35m");   // magenta = surprise
                            else if (s > 0.5)  o << C("\033[33m");   // yellow  = moderate
                            else               o << C("\033[32m");   // green   = expected
                            o << std::fixed << std::setprecision(2) << s << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
                      << [&]() -> std::string {
                            // Stream health: green=healthy, red=stalled/flooded.
                            bool healthy = stream_health.is_healthy();
                            auto st = stream_health.status();
                            std::ostringstream o;
                            o << "  HLT:";
                            if (healthy) {
                                o << C("\033[32m") << "OK";
                            } else if (st == llmquant::TokenStreamHealthMonitor::Status::Stalled) {
                                o << C("\033[31m") << "STALL";
                            } else {
                                o << C("\033[31m") << "FLOOD";
                            }
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
                      << [&]() -> std::string {
                            // Bias velocity: green=fast-positive, red=fast-negative, grey=slow.
                            double v = velocity_tracker.velocity();
                            bool   fast = velocity_tracker.is_fast_move();
                            std::ostringstream o;
                            o << "  VEL:";
                            if      (fast && v > 0) o << C("\033[32m");   // green  = fast bullish
                            else if (fast && v < 0) o << C("\033[31m");   // red    = fast bearish
                            else                    o << C("\033[90m");   // grey   = slow
                            o << std::showpos << std::fixed << std::setprecision(3) << v << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
                      << [&]() -> std::string {
                            // Narrative clock quadrant: Q1=Rising/Q2=Fading/Q3=Falling/Q4=Recovering.
                            auto q = narrative_clock.quadrant();
                            using Q = llmquant::NarrativeMomentumClock::Quadrant;
                            std::ostringstream o;
                            o << "  CLK:";
                            switch (q) {
                                case Q::Rising:     o << C("\033[32m") << "Q1"; break;  // green
                                case Q::Fading:     o << C("\033[33m") << "Q2"; break;  // yellow
                                case Q::Falling:    o << C("\033[31m") << "Q3"; break;  // red
                                case Q::Recovering: o << C("\033[36m") << "Q4"; break;  // cyan
                            }
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
                      << [&]() -> std::string {
                            int cur  = sentiment_persistence.current_state();
                            int pred = sentiment_persistence.predicted_state();
                            std::ostringstream o;
                            o << "  MKV:";
                            if (cur < 0) { o << C("\033[90m") << "---" << C("\033[0m"); return o.str(); }
                            // colour: lower states bearish (red), upper states bullish (green)
                            const char* col = (cur >= 3) ? "\033[32m" : (cur <= 1) ? "\033[31m" : "\033[33m";
                            o << C(col) << cur << C("\033[0m");
                            if (pred >= 0) o << C("\033[90m") << "→" << pred << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
                      << [&]() -> std::string {
                            int r = phase_portrait.current_row();
                            int c = phase_portrait.current_col();
                            bool cyc = phase_portrait.cycle_detected();
                            std::ostringstream o;
                            o << "  PHS:" << C("\033[36m") << r << "," << c << C("\033[0m");
                            if (cyc) o << C("\033[33m") << "~" << C("\033[0m");
                            return o.str();
                         }()
#endif
                      << std::flush;

            // Regime-change alert: log to spdlog when classified regime transitions.
            if (!current_regime.empty() && current_regime != last_regime && !last_regime.empty()) {
                spdlog::info("[regime] {} → {}", last_regime, current_regime);
            }
            last_regime = current_regime;

            // Alert if P99 exceeds budget.
            if (p99 > sys_config.latency.target_latency_us && last_tick != stats.measurements) {
                std::cout << "  " << C("\033[31m") << "[!] P99 > target" << C("\033[0m") << std::flush;
            }
        }
        last_tick = stats.measurements;
    }

    token_sim.stop();
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
    if (stream_client) stream_client->stop();
#endif
    oms_adapter->stop();
#ifdef LLMQUANT_PROMETHEUS_ENABLED
    prom_exporter.stop();
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    health_server.stop();
#endif
    // Flush all output sinks (CSV/JSON) before printing the session summary
    // so any buffered writes are visible if a crash follows.
    trade_engine.flush_sinks();
    config.stop_watching();

    auto final_stats = latency_ctrl.get_stats();
    std::cout << "\n\n  =========================================================\n";
    std::cout << "  SESSION SUMMARY\n";
    std::cout << "  ---------------------------------------------------------\n";
    // Use LLMAdapter's cumulative counter — variance_n resets every 60 s.
    std::cout << "  Tokens processed : " << llm_adapter.get_stats().tokens_processed << "\n";
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
        std::cout << "  Blocked by gate  : " << risk_mgr.format_blocked_by_gate() << "\n";
    }
    std::cout << "  Most blocked gate: " << risk_mgr.get_most_blocked_gate() << "\n";
    std::cout << "  Memory sink size : " << memory_sink->get_signals().size() << "\n";
    std::cout << "  Avg latency      : " << final_stats.avg_latency.count() << "us\n";
    std::cout << "  Min latency      : " << final_stats.min_latency.count() << "us\n";
    std::cout << "  P50 latency      : " << final_stats.p50_latency.count() << "us\n";
    std::cout << "  P95 latency      : " << final_stats.p95_latency.count() << "us\n";
    std::cout << "  P99 latency      : " << final_stats.p99_latency.count() << "us\n";
    std::cout << "  Max latency      : " << final_stats.max_latency.count() << "us\n";
    std::cout << "  P5  latency      : " << final_stats.p5_latency.count()  << "us\n";
    std::cout << "  P25 latency      : " << final_stats.p25_latency.count() << "us\n";
    std::cout << "  Avg sig strength : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().avg_signal_strength.load() << "\n";
    std::cout << "  Avg sig quality  : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().avg_signal_quality.load() << "\n";
    {
        double ema = trade_engine.get_signal_quality_ema();
        if (ema >= 0.0) {
            std::cout << "  Quality EMA(0.1) : " << std::fixed << std::setprecision(4) << ema << "\n";
        }
    }
    std::cout << "  Noise filtered   : " << trade_engine.get_stats().noise_filtered.load() << "\n";
    std::cout << "  Cooldown skip    : " << trade_engine.get_stats().signals_suppressed_cooldown.load() << "\n";
    std::cout << "  Peak bias        : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().peak_bias.load() << "\n";
    std::cout << "  SLO breach rate  : " << std::fixed << std::setprecision(2)
              << (latency_ctrl.get_slo_breach_rate() * 100.0) << "%\n";
    std::cout << "  Jitter           : " << std::fixed << std::setprecision(3)
              << final_stats.jitter_ms << "ms\n";
    {
        auto ads = llm_adapter.get_stats();
        uint64_t hit_pct2 = (ads.tokens_processed > 0)
            ? (ads.cache_hits * 100 / ads.tokens_processed) : 0;
        std::cout << "  Cache hit rate   : " << hit_pct2 << "% ("
                  << ads.cache_hits << "/" << ads.tokens_processed << ")\n";
    }
    std::cout << "  Signals aged out : " << trade_engine.get_stats().signals_aged_out.load() << "\n";
    std::cout << "  Accum. clamped   : " << trade_engine.get_stats().accumulator_clamped.load() << "\n";
    std::cout << "  Signals passed   : " << risk_mgr.get_stats().signals_passed.load() << "\n";
    std::cout << "  Latency warmup   : " << std::fixed << std::setprecision(0)
              << (latency_ctrl.get_window_fill_ratio() * 100.0) << "% window filled\n";
#ifdef LLMQUANT_DEDUP_ENABLED
    {
        auto ds = deduplicator.get_stats();
        uint64_t total_dedup = ds.total_novel + ds.total_duplicates;
        double dup_rate = (total_dedup > 0)
            ? (static_cast<double>(ds.total_duplicates) * 100.0 / static_cast<double>(total_dedup))
            : 0.0;
        std::cout << "  Dedup novel      : " << ds.total_novel << "\n";
        std::cout << "  Dedup duplicates : " << ds.total_duplicates
                  << "  (" << std::fixed << std::setprecision(1) << dup_rate << "% dup rate)\n";
    }
#endif
    {
        auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - engine_start_time).count();
        std::cout << "  Uptime           : " << uptime_s << "s\n";
    }
    std::cout << "  Log entries      : " << logger.get_log_entry_count() << "\n";
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    if (audit_log) {
        std::cout << "  Audit written    : " << audit_log->records_written()
                  << "  (dropped=" << audit_log->records_dropped()
                  << "  rot=" << audit_log->rotations() << ")\n";
        std::cout << "  Audit log file   : " << audit_log_path << "\n";
    }
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    std::cout << "  Circuit breaker  : " << circuit_breaker.state_name()
              << "  trips=" << circuit_breaker.trips()
              << "  recoveries=" << circuit_breaker.recoveries()
              << "  block_rate=" << std::fixed << std::setprecision(1)
              << (circuit_breaker.block_rate() * 100.0) << "%\n";
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    std::cout << "  Health requests  : " << health_server.requests_served() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
    std::cout << "  Adaptive cooldown: " << std::fixed << std::setprecision(0)
              << adaptive_cooldown.get_cooldown_us() << "µs"
              << "  expansions=" << adaptive_cooldown.pressure_expansions()
              << "  recoveries=" << adaptive_cooldown.recoveries() << "\n";
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
    std::cout << "  Stream stale evts: " << stale_detector.stale_events()
              << "  recoveries=" << stale_detector.recovery_events()
              << "  ms_since_last=" << stale_detector.ms_since_last_token() << "\n";
#endif
#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
    std::cout << "  Market regime    : " << regime_detector.current_regime_name()
              << "  transitions=" << regime_detector.total_transitions()
              << "  momentum=" << std::fixed << std::setprecision(3)
              << regime_detector.get_momentum() << "\n";
#endif
#ifdef LLMQUANT_TRADING_HOURS_ENABLED
    std::cout << "  Market hrs guard : "
              << (trading_hours_guard.is_market_open() ? "OPEN" : "CLOSED")
              << "  blocked=" << trading_hours_guard.signals_blocked()
              << "  transitions=" << trading_hours_guard.session_transitions()
              << "  et=" << trading_hours_guard.current_et_time_str() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
    std::cout << "  Signal corr      : "
              << "sources=" << signal_corr.source_names().size()
              << "  diverge_evts=" << signal_corr.divergence_events()
              << "  converge_evts=" << signal_corr.convergence_events() << "\n";
#endif
#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
    std::cout << "  Drawdown protect : "
              << "tier=" << drawdown_protector.current_tier()
              << "  drawdown=" << std::fixed << std::setprecision(1)
              << (drawdown_protector.current_drawdown_pct() * 100.0) << "%"
              << "  hwm=" << std::setprecision(6) << drawdown_protector.high_water_mark()
              << "  transitions=" << drawdown_protector.tier_transitions() << "\n";
#endif
#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
    std::cout << "  Multi-timeframe  : "
              << "consensus=" << std::fixed << std::setprecision(4) << multi_tf.consensus()
              << "  spread=" << multi_tf.timeframe_spread()
              << "  diverging=" << (multi_tf.is_diverging() ? "Y" : "N")
              << "  records=" << multi_tf.total_records() << "\n";
#endif
#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
    std::cout << "  Vol forecast     : "
              << "cond_vol=" << std::fixed << std::setprecision(4) << vol_forecaster.conditional_vol()
              << "  high_vol=" << (vol_forecaster.is_high_vol() ? "Y" : "N")
              << "  events=" << vol_forecaster.high_vol_events() << "\n";
#endif
#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
    std::cout << "  Bayes filter     : "
              << "bull=" << std::fixed << std::setprecision(3)
              << bayes_filter.posterior_confidence(true)
              << "  bear=" << bayes_filter.posterior_confidence(false)
              << "  signals=" << bayes_filter.total_signals() << "\n";
#endif
#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
    std::cout << "  Anomaly detect   : "
              << "soft=" << anomaly_detector.soft_anomalies()
              << "  hard=" << anomaly_detector.hard_anomalies()
              << "  last_z=" << std::fixed << std::setprecision(2)
              << anomaly_detector.last_z_score() << "\n";
#endif
#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
    std::cout << "  Burst detector   : "
              << "rate=" << std::fixed << std::setprecision(1) << burst_detector.current_rate()
              << "tok/s  burst=" << (burst_detector.is_burst() ? "Y" : "N")
              << "  events=" << burst_detector.burst_events() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
    std::cout << "  Persistence      : "
              << "streak=" << persistence_tracker.current_streak()
              << "  scale=" << std::fixed << std::setprecision(2)
              << persistence_tracker.conviction_scale()
              << "  reversals=" << persistence_tracker.total_reversals() << "\n";
#endif
#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
    std::cout << "  Rolling Sharpe   : "
              << "sharpe=" << std::fixed << std::setprecision(3) << rolling_sharpe.last_sharpe()
              << "  poor=" << (rolling_sharpe.is_poor_quality() ? "Y" : "N")
              << "  n=" << rolling_sharpe.sample_count() << "\n";
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    std::cout << "  Context budget   : "
              << "used=" << context_budget.tokens_used()
              << "  fill=" << std::fixed << std::setprecision(1)
              << (context_budget.fill_fraction() * 100.0) << "%\n";
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
    std::cout << "  Fractal dim      : "
              << "hurst=" << std::fixed << std::setprecision(3) << fractal_dim.hurst()
              << "  "
              << (fractal_dim.is_trending() ? "TRENDING" :
                  fractal_dim.is_mean_reverting() ? "MEAN-REV" : "RANDOM")
              << "  n=" << fractal_dim.total_records() << "\n";
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
    std::cout << "  Microstructure   : "
              << "half_spread=" << std::fixed << std::setprecision(5)
              << microstructure_filter.estimated_half_spread()
              << "  blocks=" << microstructure_filter.total_blocked()
              << "  passes=" << microstructure_filter.total_passed() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
    std::cout << "  Signal ensemble  : "
              << "output=" << std::fixed << std::setprecision(4) << signal_ensemble.ensemble_output()
              << "  outcomes=" << signal_ensemble.total_outcomes() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
    std::cout << "  Signal momentum  : "
              << "macd=" << std::showpos << std::fixed << std::setprecision(5)
              << signal_momentum_osc.macd()
              << "  hist=" << signal_momentum_osc.histogram()
              << std::noshowpos
              << "  " << (signal_momentum_osc.is_bullish() ? "BULL" :
                           signal_momentum_osc.is_bearish() ? "BEAR" : "FLAT")
              << "  crosses=" << signal_momentum_osc.total_crosses() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
    std::cout << "  Sentiment cycle  : "
              << "period=" << sentiment_cycle.dominant_period()
              << "  strength=" << std::fixed << std::setprecision(3) << sentiment_cycle.cycle_strength()
              << "  " << (sentiment_cycle.is_cyclic() ? "CYCLIC" : "none")
              << "  changes=" << sentiment_cycle.period_changes() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
    std::cout << "  Adaptive sampler : "
              << "interval=" << adaptive_sampler.recommended_interval_ms() << "ms"
              << "  accel=" << adaptive_sampler.accelerations()
              << "  decel=" << adaptive_sampler.decelerations()
              << "  " << (adaptive_sampler.is_at_min() ? "FAST" : adaptive_sampler.is_at_max() ? "SLOW" : "mid") << "\n";
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
    std::cout << "  Mutual info      : "
              << "mi=" << std::fixed << std::setprecision(4) << mi_estimator.mi()
              << "  nmi=" << mi_estimator.normalized_mi()
              << "  n=" << mi_estimator.sample_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
    std::cout << "  Blind spots      : "
              << "flagged_slots=" << blind_spot.blind_spot_count()
              << "  outcomes=" << blind_spot.total_outcomes()
              << "  events=" << blind_spot.detection_events() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
    std::cout << "  Signal surprise  : "
              << "last=" << std::fixed << std::setprecision(3) << signal_surprise.surprise()
              << "  mean=" << signal_surprise.mean_surprise()
              << "  high=" << (signal_surprise.is_high_surprise() ? "YES" : "no")
              << "  events=" << signal_surprise.high_surprise_count() << "\n";
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
    {
        auto st = stream_health.status();
        const char* st_str = (st == llmquant::TokenStreamHealthMonitor::Status::Healthy) ? "HEALTHY"
                           : (st == llmquant::TokenStreamHealthMonitor::Status::Stalled) ? "STALLED"
                           : "FLOODED";
        std::cout << "  Stream health    : "
                  << st_str
                  << "  rate=" << std::fixed << std::setprecision(1) << stream_health.current_rate() << "tok/s"
                  << "  stalls=" << stream_health.stall_count()
                  << "  floods=" << stream_health.flood_count() << "\n";
    }
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
    std::cout << "  Regime sizer     : "
              << "mult=" << std::fixed << std::setprecision(3) << regime_sizer.size_multiplier()
              << "  H=" << regime_sizer.current_hurst()
              << "  vol=" << regime_sizer.current_vol()
              << "  regime_f=" << regime_sizer.regime_factor()
              << "  vol_f=" << regime_sizer.vol_factor()
              << "  changes=" << regime_sizer.change_events() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
    std::cout << "  Conf decay       : "
              << "half_life=" << std::fixed << std::setprecision(0) << conf_decay.half_life_ms() << "ms"
              << "  lambda=" << std::setprecision(5) << conf_decay.lambda()
              << "  fast=" << (conf_decay.is_fast_decay() ? "YES" : "no")
              << "  n=" << conf_decay.total_records() << "\n";
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
    std::cout << "  Cross-asset corr : "
              << "assets=" << cross_asset_corr.asset_count()
              << "  bias~vol=" << std::fixed << std::setprecision(3)
              << cross_asset_corr.correlation("bias", "vol")
              << "  bias~conf=" << cross_asset_corr.correlation("bias", "confidence")
              << "  high_events=" << cross_asset_corr.high_corr_events()
              << "  low_events=" << cross_asset_corr.low_corr_events() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
    std::cout << "  Bias velocity    : "
              << "vel=" << std::showpos << std::fixed << std::setprecision(4)
              << velocity_tracker.velocity()
              << "  accel=" << velocity_tracker.acceleration()
              << "  fast_move=" << (velocity_tracker.is_fast_move() ? "YES" : "no")
              << "  n=" << velocity_tracker.total_records() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
    {
        using Q = llmquant::NarrativeMomentumClock::Quadrant;
        static const char* qnames[] = {"Rising(Q1)", "Fading(Q2)", "Falling(Q3)", "Recovering(Q4)"};
        std::cout << "  Narrative clock  : "
                  << "quadrant=" << qnames[static_cast<int>(narrative_clock.quadrant())]
                  << "  transitions=" << narrative_clock.quadrant_transitions()
                  << "  bias_ema=" << std::noshowpos << std::fixed << std::setprecision(5)
                  << narrative_clock.bias_ema()
                  << "  vel_ema=" << narrative_clock.velocity_ema()
                  << "  n=" << narrative_clock.total_records() << "\n";
    }
#endif
#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
    std::cout << "  Order book sim   : "
              << "mid=" << std::fixed << std::setprecision(4) << order_book_sim.mid_price()
              << "  bias=" << order_book_sim.cumulative_bias()
              << "  updates=" << order_book_sim.total_updates() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
    std::cout << "  Sentiment heatmap: "
              << "tokens=" << sentiment_heatmap.token_count()
              << "  records=" << sentiment_heatmap.total_records() << "\n";
#endif
#ifdef LLMQUANT_CVAR_ENABLED
    std::cout << "  CVaR (ES α=0.95) : "
              << "cvar=" << std::showpos << std::fixed << std::setprecision(5) << cvar_calc.cvar()
              << "  var=" << cvar_calc.var()
              << std::noshowpos
              << "  breach=" << (cvar_calc.is_in_breach() ? "YES" : "no")
              << "  events=" << cvar_calc.breach_events() << "\n";
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    std::cout << "  Phrase patterns  : "
              << "patterns=" << tpl.pattern_count()
              << "  tokens=" << tpl.total_tokens()
              << "  matches=" << tpl.total_matches() << "\n";
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
    std::cout << "  Feedback loop    : "
              << "score=" << std::fixed << std::setprecision(3) << feedback_detector.feedback_score()
              << "  peak_lag=" << feedback_detector.peak_lag()
              << "  detected=" << (feedback_detector.feedback_detected() ? "YES" : "no")
              << "  events=" << feedback_detector.feedback_events() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    std::cout << "  Vel breaker      : "
              << "open=" << (velocity_breaker.is_open() ? "YES" : "no")
              << "  trips=" << velocity_breaker.trip_count()
              << "  vel=" << std::fixed << std::setprecision(4) << velocity_breaker.smoothed_velocity() << "\n";
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    std::cout << "  Order flow imb   : "
              << "imb=" << std::showpos << std::fixed << std::setprecision(3) << order_flow_detector.imbalance()
              << std::noshowpos
              << "  events=" << order_flow_detector.imbalance_events()
              << "  tokens=" << order_flow_detector.total_tokens() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
    {
        auto top = token_bias_heatmap.top_by_abs_contribution(3);
        std::cout << "  Token heatmap    : distinct=" << token_bias_heatmap.distinct_tokens()
                  << "  records=" << token_bias_heatmap.total_records();
        if (!top.empty()) {
            std::cout << "  top=[";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << top[i].token << ":" << std::showpos << std::fixed
                          << std::setprecision(3) << top[i].total_bias << std::noshowpos;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
    std::cout << "  Sig calibration  : "
              << "samples=" << signal_calibration.sample_count()
              << "  ece=" << std::fixed << std::setprecision(4) << signal_calibration.expected_calibration_error()
              << "  A=" << signal_calibration.platt_a()
              << "  B=" << signal_calibration.platt_b() << "\n";
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
    std::cout << "  Cross-session    : session=" << cross_session_mem.session_number()
              << "  loaded=" << (cross_session_mem.has_loaded_state() ? "yes" : "no") << "\n";
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
    std::cout << "  Regime HMM       : "
              << "p_risk_on=" << std::fixed << std::setprecision(4) << regime_prob_est.prob_risk_on()
              << "  p_risk_off=" << regime_prob_est.prob_risk_off()
              << "  transitions=" << regime_prob_est.transition_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
    std::cout << "  Signal replay    : "
              << "retained=" << signal_replay.size()
              << "  total=" << signal_replay.total_pushed() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    {
        auto top = ngram_profiler.top_by_frequency(3);
        std::cout << "  N-gram profiler  : "
                  << "distinct=" << ngram_profiler.distinct_ngrams()
                  << "  hot_events=" << ngram_profiler.hot_events();
        if (!top.empty()) {
            std::cout << "  top=[";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << "\"" << top[i].ngram << "\":" << top[i].count;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
    std::cout << "  Sent dispersion  : "
              << "sdi=" << std::fixed << std::setprecision(4) << sentiment_dispersion.sdi()
              << "  dispersed=" << (sentiment_dispersion.is_dispersed() ? "YES" : "no")
              << "  coherent=" << (sentiment_dispersion.is_coherent() ? "YES" : "no")
              << "  events=" << sentiment_dispersion.high_dispersion_events() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
    std::cout << "  Sent divergence  : "
              << "diverge=" << std::fixed << std::setprecision(4) << sentiment_divergence.divergence()
              << "  active=" << (sentiment_divergence.is_diverged() ? "YES" : "no")
              << "  events=" << sentiment_divergence.divergence_events()
              << "  sources=" << sentiment_divergence.source_count() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    {
        std::cout << "  Token influence  : "
                  << "n=" << token_influence.total_recorded()
                  << "  window=" << token_influence.window_size();
        auto top = token_influence.attribute();
        if (!top.empty()) {
            std::cout << "  top=" << top[0].token
                      << "(inf=" << std::showpos << std::fixed << std::setprecision(4)
                      << top[0].influence << std::noshowpos << ")";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_WALK_FORWARD_ENABLED
    std::cout << "  Walk-forward     : "
              << "folds=" << walk_forward.num_folds()
              << "  (tokens not loaded in live mode — offline use only)\n";
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    std::cout << "  Adversarial det  : "
              << "armed=" << (adversarial_detector.is_armed() ? "YES" : "no")
              << "  anomalies=" << adversarial_detector.anomaly_count()
              << "  tokens=" << adversarial_detector.total_tokens() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
    std::cout << "  Signal CI (95%)  : "
              << "mean=" << std::showpos << std::fixed << std::setprecision(4) << signal_ci.mean()
              << "  hw=" << std::noshowpos << signal_ci.half_width()
              << "  [" << signal_ci.lower() << ", " << signal_ci.upper() << "]"
              << "  narrow=" << (signal_ci.is_narrow() ? "yes" : "no") << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
    std::cout << "  Markov chain     : "
              << "state=" << sentiment_persistence.current_state()
              << "  predicted=" << sentiment_persistence.predicted_state()
              << "  stickiness=" << std::fixed << std::setprecision(4) << sentiment_persistence.stickiness()
              << "  transitions=" << sentiment_persistence.state_changes()
              << "  records=" << sentiment_persistence.total_records() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
    std::cout << "  Phase portrait   : "
              << "cell=(" << phase_portrait.current_row() << "," << phase_portrait.current_col() << ")"
              << "  attractor=(" << phase_portrait.attractor_row() << "," << phase_portrait.attractor_col() << ")"
              << "  cycle=" << (phase_portrait.cycle_detected() ? "YES" : "no")
              << "  divergence=" << std::fixed << std::setprecision(4) << phase_portrait.divergence_index()
              << "  transitions=" << phase_portrait.cell_transitions()
              << "  records=" << phase_portrait.total_records() << "\n";
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
    std::cout << "  Causal impact    : "
              << "cusum=" << std::fixed << std::setprecision(4) << causal_impact.cusum_stat()
              << "  break=" << (causal_impact.break_detected() ? "YES" : "no")
              << "  breaks=" << causal_impact.break_count()
              << "  obs=" << causal_impact.observation_count() << "\n";
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
    {
        const char* div_label = "none";
        auto dk = options_flow_bridge.last_divergence();
        if (dk == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBear)
            div_label = "SMART_MONEY_BEAR";
        else if (dk == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBull)
            div_label = "SMART_MONEY_BULL";
        std::cout << "  Options flow     : "
                  << "divergence=" << div_label
                  << "  score=" << std::fixed << std::setprecision(4) << options_flow_bridge.divergence_score()
                  << "  vel_ema=" << options_flow_bridge.sentiment_velocity_ema()
                  << "  skew_ema=" << options_flow_bridge.skew_ema()
                  << "  events=" << options_flow_bridge.divergence_count() << "\n";
    }
#endif
    std::cout << "  Latency summary  : " << latency_ctrl.format_stats() << "\n";
    {
        std::cout << "  OMS adapter      : " << oms_adapter->description() << "\n";
        std::cout << "  OMS updates      : " << oms_adapter->update_count()
                  << "  errors=" << oms_adapter->error_count();
        if (oms_adapter->reconnect_count() > 0)
            std::cout << "  reconnects=" << oms_adapter->reconnect_count();
        std::cout << "\n";
    }
    {
        auto top = llm_adapter.top_tokens_by_frequency(5);
        if (!top.empty()) {
            std::cout << "  Top tokens (hits): ";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << top[i].first << "(" << top[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    {
        auto top_bias = llm_adapter.top_tokens_by_directional_bias(5);
        if (!top_bias.empty()) {
            std::cout << "  Top bias tokens  : ";
            for (size_t i = 0; i < top_bias.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << top_bias[i].first
                          << "(" << std::fixed << std::setprecision(3) << top_bias[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    {
        // Hot tokens: composite score = 0.5*(hit_rate) + 0.5*(|directional_bias|)
        auto hot = llm_adapter.export_hot_tokens(5);
        if (!hot.empty()) {
            std::cout << "  Hot tokens       : ";
            for (size_t i = 0; i < hot.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << hot[i].first
                          << "(" << std::fixed << std::setprecision(3) << hot[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    std::cout << "  ---------------------------------------------------------\n\n";

#ifdef LLMQUANT_JSON_STATS_SUMMARY
    // Emit structured JSON summaries for all subsystems.
    if (!quiet) {
        std::cout << "  [json:risk]    " << risk_mgr.to_stats_json() << "\n";
        std::cout << "  [json:engine]  " << trade_engine.to_stats_json() << "\n";
        std::cout << "  [json:adapter] " << llm_adapter.to_stats_json() << "\n";
        std::cout << "  [json:latency] " << latency_ctrl.to_stats_json() << "\n";
#ifdef LLMQUANT_DEDUP_ENABLED
        std::cout << "  [json:dedup]   " << deduplicator.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
        std::cout << "  [json:lbe]     " << latency_budget_enforcer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
        std::cout << "  [json:pnl_attr] " << pnl_attribution.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
        std::cout << "  [json:pheat]   " << portfolio_heat.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
        std::cout << "  [json:ctx]     " << context_budget.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        std::cout << "  [json:fractal] " << fractal_dim.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
        std::cout << "  [json:microstr]" << microstructure_filter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
        std::cout << "  [json:ensemble]" << signal_ensemble.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
        std::cout << "  [json:smo]     " << signal_momentum_osc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CVAR_ENABLED
        std::cout << "  [json:cvar]    " << cvar_calc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
        std::cout << "  [json:tpl]     " << tpl.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
        std::cout << "  [json:fbl]     " << feedback_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
        std::cout << "  [json:cycle]   " << sentiment_cycle.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
        std::cout << "  [json:sampler] " << adaptive_sampler.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
        std::cout << "  [json:mi]      " << mi_estimator.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
        std::cout << "  [json:bspot]   " << blind_spot.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
        std::cout << "  [json:surprise]" << signal_surprise.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        std::cout << "  [json:health]  " << stream_health.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
        std::cout << "  [json:rsizer]  " << regime_sizer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
        std::cout << "  [json:cdecay]  " << conf_decay.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
        std::cout << "  [json:xcorr]   " << cross_asset_corr.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
        std::cout << "  [json:vel]     " << velocity_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
        std::cout << "  [json:clock]   " << narrative_clock.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        std::cout << "  [json:vbreaker] " << velocity_breaker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
        std::cout << "  [json:sigcal]   " << signal_calibration.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
        std::cout << "  [json:heatmap]  " << token_bias_heatmap.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        std::cout << "  [json:oflow]    " << order_flow_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
        std::cout << "  [json:session]  " << cross_session_mem.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
        std::cout << "  [json:regime_hmm] " << regime_prob_est.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
        std::cout << "  [json:replay]   " << signal_replay.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
        std::cout << "  [json:ngram]    " << ngram_profiler.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
        std::cout << "  [json:execqual] " << exec_quality.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
        std::cout << "  [json:dispersion] " << sentiment_dispersion.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
        std::cout << "  [json:divergence] " << sentiment_divergence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
        std::cout << "  [json:influence]  " << token_influence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
        std::cout << "  [json:adversarial] " << adversarial_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
        std::cout << "  [json:signal_ci]  " << signal_ci.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
        std::cout << "  [json:markov]     " << sentiment_persistence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
        std::cout << "  [json:phase]      " << phase_portrait.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
        std::cout << "  [json:causal]     " << causal_impact.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
        std::cout << "  [json:optflow]    " << options_flow_bridge.to_stats_json() << "\n";
#endif
    }
#endif // LLMQUANT_JSON_STATS_SUMMARY

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
