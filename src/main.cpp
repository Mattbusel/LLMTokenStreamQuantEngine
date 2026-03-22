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
#ifdef LLMQUANT_REST_OMS_ENABLED
#  include "RestOmsAdapter.h"
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
#  include "FixOmsAdapter.h"
#endif
#include "MockOmsAdapter.h"
#ifdef LLMQUANT_PROMETHEUS_ENABLED
#  include "PrometheusExporter.h"
#endif
#include "llmquant_version.h"
#include <spdlog/spdlog.h>
#include <iostream>
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
    bool        no_dedup       = false;         // disable deduplication
    bool        no_hot_reload  = false;         // skip config file watcher
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
                "  --no-dedup        Disable token deduplication (all tokens treated as novel)\n"
                "  --no-hot-reload   Disable config file hot-reload watcher\n"
                "  --version         Print version and exit\n"
                "  --help            Print this help and exit\n"
                "\n"
                "Environment:\n"
                "  LLMQUANT_API_KEY        LLM API key (fallback when --stream has no key)\n"
                "  LLMQUANT_NO_PROMETHEUS  Set to 1/true/yes to disable Prometheus endpoint\n"
                "  LLMQUANT_NO_DEDUP       Set to 1/true/yes to disable token deduplication\n"
                "  LLMQUANT_NO_HOT_RELOAD  Set to 1/true/yes to disable config hot-reload\n"
                "  LLMQUANT_DRY_RUN        Set to 1/true/yes for dry-run (signal only, no OMS)\n"
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
        } else if (arg == "--stream") {
            stream_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                stream_api_key = argv[++i];  // explicit key provided on CLI
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
            try { stats_port_override = static_cast<uint16_t>(std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --stats-port requires an integer\n"; return 1; }
        } else if (arg == "--token-interval" && i + 1 < argc) {
            try { token_interval_override = std::max(1, std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --token-interval requires an integer\n"; return 1; }
        } else if (arg == "--log-level" && i + 1 < argc) {
            log_level_str = argv[++i];
        } else if (arg == "--stats-interval" && i + 1 < argc) {
            try { stats_interval_ms = std::max(100, std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --stats-interval requires an integer\n"; return 1; }
        } else if (arg == "--no-prometheus") {
            no_prometheus = true;
        } else if (arg == "--no-dedup") {
            no_dedup = true;
        } else if (arg == "--no-hot-reload") {
            no_hot_reload = true;
        } else if (arg == "--oms" && i + 1 < argc) {
            oms_address = argv[++i];
        } else if (arg == "--fix" && i + 1 < argc) {
            fix_address = argv[++i];
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
        if (!no_dedup       && env_flag("LLMQUANT_NO_DEDUP"))       no_dedup       = true;
        if (!no_hot_reload  && env_flag("LLMQUANT_NO_HOT_RELOAD"))  no_hot_reload  = true;
        if (!dry_run        && env_flag("LLMQUANT_DRY_RUN"))        dry_run        = true;
        if (!quiet          && env_flag("LLMQUANT_QUIET"))          quiet          = true;
        if (!backtest_mode  && env_flag("LLMQUANT_BACKTEST"))       backtest_mode  = true;
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

    // Deduplication layer: skip repeated tokens within a sliding TTL window.
    auto dedup_backend = std::make_shared<llmquant::InProcessDeduplicator>();
    // Dedup TTL: use config value when set (> 0), else default to 10× the token interval.
    int dedup_ttl_ms = (sys_config.token_stream.dedup_ttl_ms > 0)
        ? sys_config.token_stream.dedup_ttl_ms
        : sys_config.token_stream.token_interval_ms * 10;
    llmquant::Deduplicator deduplicator(dedup_backend,
        std::chrono::milliseconds(dedup_ttl_ms));
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

    // --list-tokens: dump the full semantic dictionary and exit immediately.
    if (list_tokens) {
        auto keys = llm_adapter.get_all_token_keys();
        std::cout << "token\tsentiment\tconfidence\tvolatility\tbias\n";
        for (const auto& k : keys) {
            SemanticWeight w;
            llm_adapter.get_token_mapping(k, w);
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
            try { fix_cfg.port = static_cast<uint16_t>(std::stoi(fix_address.substr(colon + 1))); }
            catch (...) { spdlog::error("--fix: invalid port in '{}'", fix_address); return 1; }
        } else {
            fix_cfg.host = fix_address;
        }
        oms_adapter = std::make_unique<llmquant::FixOmsAdapter>(fix_cfg);
#else
        spdlog::error("--fix requested but FIX OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_FIX_OMS=OFF). Falling back to MockOmsAdapter.");
#endif
    } else if (!oms_address.empty()) {
#ifdef LLMQUANT_REST_OMS_ENABLED
        std::string endpoint = oms_address;
        llmquant::RestOmsAdapter::Config oms_cfg;
        size_t colon = endpoint.find(':');
        if (colon != std::string::npos) {
            oms_cfg.host = endpoint.substr(0, colon);
            try { oms_cfg.port = static_cast<uint16_t>(std::stoi(endpoint.substr(colon + 1))); }
            catch (...) { spdlog::error("--oms: invalid port in '{}'", endpoint); return 1; }
        } else {
            oms_cfg.host = endpoint;
        }
        oms_adapter = std::make_unique<llmquant::RestOmsAdapter>(oms_cfg);
#else
        spdlog::error("--oms requested but REST OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_REST_OMS=OFF). Falling back to MockOmsAdapter.");
#endif
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

    // Start config hot-reload watcher now that all pipeline objects exist.
    // The callback can update every subsystem live, including reloading the
    // token file when token_stream.data_file_path changes at runtime.
    // Disabled when --no-hot-reload is passed (useful in CI/embedded contexts).
    if (no_hot_reload) {
        spdlog::info("--no-hot-reload: config file watcher disabled");
    } else if (!config.start_watching(config_file, [&risk_mgr, &trade_engine, &token_sim,
                                              &logger, &config_file,
                                              &sem_mult_sentiment, &sem_mult_confidence,
                                              &sem_mult_volatility, &sem_mult_bias](const llmquant::SystemConfig& updated) {
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
        new_eng_cfg.max_signal_age_us      = updated.trading.max_signal_age_us;
        new_eng_cfg.min_bias_threshold     = updated.trading.min_bias_threshold;
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

    // Shared token processing lambda used by both the simulator and the
    // LLMStreamClient paths.  Encapsulates dedup, latency, logging, and
    // semantic-weight pipeline so neither call site duplicates logic.
    auto process_token = [&](const std::string& text, uint64_t seq_id) {
        // Skip duplicate tokens within the dedup window (unless --no-dedup).
        if (!no_dedup) {
            auto dedup_result = deduplicator.check(text);
            logger.log_dedup_event(text, dedup_result == llmquant::DedupResult::Duplicate);
            if (dedup_result == llmquant::DedupResult::Duplicate) {
                return;
            }
        }

        latency_ctrl.start_measurement();

        logger.log_token_received(text, seq_id);

        auto weight = llm_adapter.map_token_to_weight(text);

        // Apply per-category semantic weight multipliers (hot-reloadable).
        // Read atomically so hot-reload updates are visible without a mutex.
        weight.sentiment_score  *= sem_mult_sentiment.load(std::memory_order_relaxed);
        weight.confidence_score *= sem_mult_confidence.load(std::memory_order_relaxed);
        weight.volatility_score *= sem_mult_volatility.load(std::memory_order_relaxed);
        weight.directional_bias *= sem_mult_bias.load(std::memory_order_relaxed);

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
                                       << signal.delta_bias_shift  << "  "
                      << std::setw(8)  << signal.volatility_adjustment << "  "
                      << std::setw(6)  << latency_us << "μs"
                      << gate_str
                      << std::flush;
        }

        if (passed) {
            logger.log_trade_signal(
                signal.delta_bias_shift,
                signal.volatility_adjustment,
                signal.confidence,
                static_cast<double>(latency_us),
                signal.signal_quality);
        } else {
            logger.log_risk_rejection(block_reason_copy,
                                      signal.delta_bias_shift,
                                      signal.confidence);
        }
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
        stream_client->connect();
    } else {
        token_sim.start();
    }

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

    // Main monitoring loop — prints a rolling stats bar every second.
    uint64_t last_tick = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(stats_interval_ms));

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

        // Colour the P99 value: green < 10μs, yellow < 50μs, red otherwise.
        auto p99 = stats.p99_latency.count();
        const char* p99_colour =
            (p99 < 10)  ? C("\033[32m") :
            (p99 < 50)  ? C("\033[33m") : C("\033[31m");

        // Colour the pressure bar.
        const char* press_colour =
            (pressure.composite < 0.5) ? C("\033[32m") :
            (pressure.composite < 0.8) ? C("\033[33m") : C("\033[31m");

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
                 << "llmquant_oms_update_count_total " << [&]() -> uint64_t {
#ifdef LLMQUANT_REST_OMS_ENABLED
                        if (auto* rest = dynamic_cast<llmquant::RestOmsAdapter*>(oms_adapter.get()))
                            return rest->update_count();
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
                        if (auto* fix = dynamic_cast<llmquant::FixOmsAdapter*>(oms_adapter.get()))
                            return fix->update_count();
#endif
                        return 0;
                    }() << "\n"
                 << "# HELP llmquant_oms_error_count_total Total OMS connection errors\n"
                 << "# TYPE llmquant_oms_error_count_total counter\n"
                 << "llmquant_oms_error_count_total " << [&]() -> uint64_t {
#ifdef LLMQUANT_REST_OMS_ENABLED
                        if (auto* rest = dynamic_cast<llmquant::RestOmsAdapter*>(oms_adapter.get()))
                            return rest->error_count();
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
                        if (auto* fix = dynamic_cast<llmquant::FixOmsAdapter*>(oms_adapter.get()))
                            return fix->error_count();
#endif
                        return 0;
                    }() << "\n"
                 << "# HELP llmquant_oms_reconnect_count_total Total FIX session reconnect attempts\n"
                 << "# TYPE llmquant_oms_reconnect_count_total counter\n"
                 << "llmquant_oms_reconnect_count_total " << [&]() -> uint64_t {
#ifdef LLMQUANT_FIX_OMS_ENABLED
                        if (auto* fix = dynamic_cast<llmquant::FixOmsAdapter*>(oms_adapter.get()))
                            return fix->get_reconnect_count();
#endif
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
                 << "# HELP llmquant_dedup_duplicate_rate Fraction of checked tokens that were duplicates [0,1]\n"
                 << "# TYPE llmquant_dedup_duplicate_rate gauge\n"
                 << "llmquant_dedup_duplicate_rate " << [&]() -> double {
                        uint64_t dupes = dedup_backend->total_duplicates();
                        uint64_t novel = dedup_backend->total_novel();
                        uint64_t total = novel + dupes;
                        return (total > 0) ? (static_cast<double>(dupes) / static_cast<double>(total)) : 0.0;
                    }() << "\n";
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
                 << "# HELP llmquant_risk_healthy Whether all risk gates are nominally healthy (1=yes)\n"
                 << "# TYPE llmquant_risk_healthy gauge\n"
                 << "llmquant_risk_healthy " << (risk_mgr.is_healthy() ? 1 : 0) << "\n"
                 << "# HELP llmquant_dedup_dup_rate_pct Duplicate token rate as percentage [0,100]\n"
                 << "# TYPE llmquant_dedup_dup_rate_pct gauge\n"
                 << "llmquant_dedup_dup_rate_pct " << [&]() -> double {
                        uint64_t nov = dedup_backend->total_novel();
                        uint64_t dup = dedup_backend->total_duplicates();
                        uint64_t tot = nov + dup;
                        return (tot > 0) ? (static_cast<double>(dup) * 100.0 / static_cast<double>(tot)) : 0.0;
                    }() << "\n";
            // Top-5 influential tokens as labeled gauges for Grafana dashboards.
            {
                snap << "# HELP llmquant_top_influence_token Composite influence score (freq+bias blend) [0,1]\n"
                     << "# TYPE llmquant_top_influence_token gauge\n";
                for (const auto& [tok, score] : llm_adapter.export_hot_tokens(5)) {
                    std::string safe_tok;
                    for (char c : tok) safe_tok += (c == '"') ? '\'' : c;
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
                      << "  DEDUP:" << dedup_backend->total_duplicates()
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
                      << std::flush;

            // Alert if P99 exceeds budget.
            if (p99 > sys_config.latency.target_latency_us && last_tick != stats.measurements) {
                std::cout << "  " << C("\033[31m") << "[!] P99 > target" << C("\033[0m") << std::flush;
            }
        }
        last_tick = stats.measurements;
    }

    token_sim.stop();
    if (stream_client) stream_client->stop();
    oms_adapter->stop();
#ifdef LLMQUANT_PROMETHEUS_ENABLED
    prom_exporter.stop();
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
    {
        auto ds = dedup_backend->get_stats();
        uint64_t total_dedup = ds.total_novel + ds.total_duplicates;
        double dup_rate = (total_dedup > 0)
            ? (static_cast<double>(ds.total_duplicates) * 100.0 / static_cast<double>(total_dedup))
            : 0.0;
        std::cout << "  Dedup novel      : " << ds.total_novel << "\n";
        std::cout << "  Dedup duplicates : " << ds.total_duplicates
                  << "  (" << std::fixed << std::setprecision(1) << dup_rate << "% dup rate)\n";
    }
    {
        auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - engine_start_time).count();
        std::cout << "  Uptime           : " << uptime_s << "s\n";
    }
    std::cout << "  Log entries      : " << logger.get_log_entry_count() << "\n";
    std::cout << "  Latency summary  : " << latency_ctrl.format_stats() << "\n";
    {
        std::cout << "  OMS adapter      : " << oms_adapter->description() << "\n";
#ifdef LLMQUANT_REST_OMS_ENABLED
        if (auto* rest = dynamic_cast<llmquant::RestOmsAdapter*>(oms_adapter.get())) {
            std::cout << "  OMS updates      : " << rest->update_count()
                      << "  errors=" << rest->error_count() << "\n";
        } else
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
        if (auto* fix = dynamic_cast<llmquant::FixOmsAdapter*>(oms_adapter.get())) {
            std::cout << "  OMS updates      : " << fix->update_count()
                      << "  errors=" << fix->error_count()
                      << "  reconnects=" << fix->get_reconnect_count() << "\n";
        }
#endif
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
        auto top_inf = llm_adapter.export_hot_tokens(5);
        if (!top_inf.empty()) {
            std::cout << "  Top influence    : ";
            for (size_t i = 0; i < top_inf.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << top_inf[i].first
                          << "(" << std::fixed << std::setprecision(3) << top_inf[i].second << ")";
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
        std::cout << "  [json:dedup]   " << dedup_backend->to_stats_json() << "\n";
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
