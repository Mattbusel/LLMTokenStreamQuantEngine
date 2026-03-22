#include "Config.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace llmquant {

bool Config::load_from_file(const std::string& filepath) {
    try {
        YAML::Node yaml = YAML::LoadFile(filepath);
        return load_from_yaml_string(YAML::Dump(yaml));
    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to load config file {}: {}", filepath, e.what());
        set_defaults();
        return false;
    }
}

bool Config::load_from_yaml_string(const std::string& yaml_content) {
    try {
        YAML::Node yaml = YAML::Load(yaml_content);

        // Parse into a temporary so the mutex is held only while writing config_.
        SystemConfig tmp{};

        // Token stream settings
        if (yaml["token_stream"]) {
            auto ts = yaml["token_stream"];
            if (ts["data_file_path"]) tmp.token_stream.data_file_path = ts["data_file_path"].as<std::string>();
            if (ts["token_interval_ms"]) tmp.token_stream.token_interval_ms = ts["token_interval_ms"].as<int>();
            if (ts["buffer_size"]) tmp.token_stream.buffer_size = ts["buffer_size"].as<size_t>();
            if (ts["use_memory_stream"]) tmp.token_stream.use_memory_stream = ts["use_memory_stream"].as<bool>();
            if (ts["dedup_ttl_ms"]) tmp.token_stream.dedup_ttl_ms = ts["dedup_ttl_ms"].as<int>();
        }

        // Trading settings
        if (yaml["trading"]) {
            auto t = yaml["trading"];
            if (t["bias_sensitivity"]) tmp.trading.bias_sensitivity = t["bias_sensitivity"].as<double>();
            if (t["volatility_sensitivity"]) tmp.trading.volatility_sensitivity = t["volatility_sensitivity"].as<double>();
            if (t["signal_decay_rate"]) tmp.trading.signal_decay_rate = t["signal_decay_rate"].as<double>();
            if (t["signal_cooldown_us"]) tmp.trading.signal_cooldown_us = t["signal_cooldown_us"].as<int>();
            if (t["max_signal_age_us"])      tmp.trading.max_signal_age_us      = t["max_signal_age_us"].as<double>();
            if (t["min_bias_threshold"])     tmp.trading.min_bias_threshold     = t["min_bias_threshold"].as<double>();
            if (t["max_accumulated_bias"])   tmp.trading.max_accumulated_bias   = t["max_accumulated_bias"].as<double>();
        }

        // Latency settings
        if (yaml["latency"]) {
            auto l = yaml["latency"];
            if (l["target_latency_us"]) tmp.latency.target_latency_us = l["target_latency_us"].as<int>();
            if (l["sample_window"]) tmp.latency.sample_window = l["sample_window"].as<size_t>();
            if (l["enable_profiling"]) tmp.latency.enable_profiling = l["enable_profiling"].as<bool>();
        }

        // Risk threshold parameters
        if (yaml["risk_thresholds"]) {
            auto rt = yaml["risk_thresholds"];
            if (rt["max_bias_magnitude"])       tmp.risk_thresholds.max_bias_magnitude       = rt["max_bias_magnitude"].as<double>();
            if (rt["max_volatility_magnitude"]) tmp.risk_thresholds.max_volatility_magnitude = rt["max_volatility_magnitude"].as<double>();
            if (rt["max_spread_magnitude"])     tmp.risk_thresholds.max_spread_magnitude     = rt["max_spread_magnitude"].as<double>();
            if (rt["min_confidence"])           tmp.risk_thresholds.min_confidence           = rt["min_confidence"].as<double>();
            if (rt["max_signals_per_second"])   tmp.risk_thresholds.max_signals_per_second   = rt["max_signals_per_second"].as<size_t>();
            if (rt["max_drawdown"])             tmp.risk_thresholds.max_drawdown             = rt["max_drawdown"].as<double>();
            if (rt["drawdown_window_s"])        tmp.risk_thresholds.drawdown_window_s        = rt["drawdown_window_s"].as<int>();
            if (rt["position_warn_fraction"])   tmp.risk_thresholds.position_warn_fraction   = rt["position_warn_fraction"].as<double>();
        }

        // Risk gate override settings
        if (yaml["risk"]) {
            auto r = yaml["risk"];
            if (r["disable_magnitude_gate"])  tmp.risk_overrides.disable_magnitude_gate  = r["disable_magnitude_gate"].as<bool>();
            if (r["disable_confidence_gate"]) tmp.risk_overrides.disable_confidence_gate = r["disable_confidence_gate"].as<bool>();
            if (r["disable_rate_gate"])       tmp.risk_overrides.disable_rate_gate       = r["disable_rate_gate"].as<bool>();
            if (r["disable_drawdown_gate"])   tmp.risk_overrides.disable_drawdown_gate   = r["disable_drawdown_gate"].as<bool>();
            if (r["disable_position_gate"])   tmp.risk_overrides.disable_position_gate   = r["disable_position_gate"].as<bool>();
        }

        // Metrics / observability endpoint settings
        if (yaml["metrics"]) {
            auto m = yaml["metrics"];
            if (m["stats_port"])    tmp.metrics.stats_port    = m["stats_port"].as<uint16_t>();
            if (m["bind_address"])  tmp.metrics.bind_address  = m["bind_address"].as<std::string>();
        }

        // Pressure settings
        if (yaml["pressure"]) {
            auto pr = yaml["pressure"];
            if (pr["max_ingestion_rate_tps"]) tmp.pressure.max_ingestion_rate_tps = pr["max_ingestion_rate_tps"].as<double>();
            if (pr["backoff_scale_factor"])   tmp.pressure.backoff_scale_factor   = pr["backoff_scale_factor"].as<double>();
        }

        // Semantic weight multipliers (optional — all default to 1.0).
        if (yaml["semantic_weights"]) {
            auto sw = yaml["semantic_weights"];
            if (sw["sentiment_multiplier"])  tmp.semantic_weights.sentiment_multiplier  = sw["sentiment_multiplier"].as<double>();
            if (sw["confidence_multiplier"]) tmp.semantic_weights.confidence_multiplier = sw["confidence_multiplier"].as<double>();
            if (sw["volatility_multiplier"]) tmp.semantic_weights.volatility_multiplier = sw["volatility_multiplier"].as<double>();
            if (sw["bias_multiplier"])       tmp.semantic_weights.bias_multiplier       = sw["bias_multiplier"].as<double>();
        }

        // Logging settings
        if (yaml["logging"]) {
            auto log = yaml["logging"];
            if (log["log_file_path"]) tmp.logging.log_file_path = log["log_file_path"].as<std::string>();
            if (log["format"]) tmp.logging.format = log["format"].as<std::string>();
            // Normalise format to uppercase so comparisons in main.cpp are case-insensitive.
            auto& fmt = tmp.logging.format;
            std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                           [](unsigned char c){ return std::toupper(c); });
            if (log["enable_console"]) tmp.logging.enable_console = log["enable_console"].as<bool>();
            if (log["flush_interval_ms"]) tmp.logging.flush_interval_ms = log["flush_interval_ms"].as<int>();
        }

        // Validate ranges — reject configs that would produce nonsensical behaviour.
        const auto& ts  = tmp.token_stream;
        const auto& tr  = tmp.trading;
        const auto& lat = tmp.latency;
        const auto& log = tmp.logging;
        if (ts.token_interval_ms <= 0 || ts.token_interval_ms > 60000) {
            spdlog::error("Config validation failed: token_interval_ms out of range [1, 60000]");
            set_defaults();
            return false;
        }
        if (ts.buffer_size == 0) {
            spdlog::error("Config validation failed: buffer_size must be >= 1");
            set_defaults();
            return false;
        }
        if (ts.dedup_ttl_ms < 0) {
            spdlog::error("Config validation failed: dedup_ttl_ms must be >= 0 (0 = auto)");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.bias_sensitivity) || tr.bias_sensitivity <= 0.0 || tr.bias_sensitivity > 10.0) {
            spdlog::error("Config: bias_sensitivity must be in (0, 10]");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.volatility_sensitivity) || tr.volatility_sensitivity <= 0.0 || tr.volatility_sensitivity > 10.0) {
            spdlog::error("Config: volatility_sensitivity must be in (0, 10]");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.signal_decay_rate) || tr.signal_decay_rate <= 0.0 || tr.signal_decay_rate > 1.0) {
            spdlog::error("Config validation failed: signal_decay_rate must be in (0, 1]");
            set_defaults();
            return false;
        }
        if (tr.signal_cooldown_us < 0) {
            spdlog::error("Config validation failed: signal_cooldown_us must be >= 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.max_signal_age_us) || tr.max_signal_age_us < 0.0) {
            spdlog::error("Config validation failed: trading.max_signal_age_us must be >= 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.min_bias_threshold) || tr.min_bias_threshold < 0.0) {
            spdlog::error("Config validation failed: trading.min_bias_threshold must be >= 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tr.max_accumulated_bias) || tr.max_accumulated_bias < 0.0) {
            spdlog::error("Config validation failed: trading.max_accumulated_bias must be >= 0");
            set_defaults();
            return false;
        }
        if (lat.target_latency_us <= 0) {
            spdlog::error("Config validation failed: target_latency_us must be > 0");
            set_defaults();
            return false;
        }
        if (lat.sample_window == 0) {
            spdlog::error("Config validation failed: sample_window must be >= 1");
            set_defaults();
            return false;
        }
        if (log.flush_interval_ms <= 0) {
            spdlog::error("Config validation failed: flush_interval_ms must be > 0");
            set_defaults();
            return false;
        }
        if (tmp.metrics.stats_port == 0) {
            spdlog::error("Config validation failed: metrics.stats_port must be > 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tmp.pressure.max_ingestion_rate_tps) || tmp.pressure.max_ingestion_rate_tps <= 0.0) {
            spdlog::error("Config validation failed: pressure.max_ingestion_rate_tps must be > 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(tmp.pressure.backoff_scale_factor) || tmp.pressure.backoff_scale_factor <= 0.0) {
            spdlog::error("Config validation failed: pressure.backoff_scale_factor must be > 0");
            set_defaults();
            return false;
        }
        const auto& rt = tmp.risk_thresholds;
        if (!std::isfinite(rt.max_bias_magnitude) || rt.max_bias_magnitude < 0.0) {
            spdlog::error("Config validation failed: risk_thresholds.max_bias_magnitude must be >= 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(rt.min_confidence) || rt.min_confidence < 0.0 || rt.min_confidence > 1.0) {
            spdlog::error("Config validation failed: risk_thresholds.min_confidence must be in [0, 1]");
            set_defaults();
            return false;
        }
        if (rt.max_signals_per_second == 0) {
            spdlog::error("Config validation failed: risk_thresholds.max_signals_per_second must be > 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(rt.max_drawdown) || rt.max_drawdown < 0.0) {
            spdlog::error("Config validation failed: risk_thresholds.max_drawdown must be >= 0");
            set_defaults();
            return false;
        }
        if (rt.drawdown_window_s <= 0) {
            spdlog::error("Config validation failed: risk_thresholds.drawdown_window_s must be > 0");
            set_defaults();
            return false;
        }
        if (!std::isfinite(rt.position_warn_fraction) || rt.position_warn_fraction < 0.0 || rt.position_warn_fraction > 1.0) {
            spdlog::error("Config validation failed: risk_thresholds.position_warn_fraction must be in [0, 1]");
            set_defaults();
            return false;
        }
        // Validate semantic weight multipliers — must all be finite numbers.
        const auto& sw = tmp.semantic_weights;
        if (!std::isfinite(sw.sentiment_multiplier)) {
            spdlog::error("Config validation failed: semantic_weights.sentiment_multiplier must be a finite number");
            set_defaults();
            return false;
        }
        if (!std::isfinite(sw.confidence_multiplier)) {
            spdlog::error("Config validation failed: semantic_weights.confidence_multiplier must be a finite number");
            set_defaults();
            return false;
        }
        if (!std::isfinite(sw.volatility_multiplier)) {
            spdlog::error("Config validation failed: semantic_weights.volatility_multiplier must be a finite number");
            set_defaults();
            return false;
        }
        if (!std::isfinite(sw.bias_multiplier)) {
            spdlog::error("Config validation failed: semantic_weights.bias_multiplier must be a finite number");
            set_defaults();
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(config_mutex_);
            config_ = tmp;
        }
        return true;
    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to parse YAML config: {}", e.what());
        set_defaults();
        return false;
    }
}

std::string Config::to_yaml_string() const {
    SystemConfig snap;
    {
        std::lock_guard<std::mutex> lk(config_mutex_);
        snap = config_;
    }
    YAML::Node yaml;

    // Token stream
    yaml["token_stream"]["data_file_path"]   = snap.token_stream.data_file_path;
    yaml["token_stream"]["token_interval_ms"] = snap.token_stream.token_interval_ms;
    yaml["token_stream"]["buffer_size"]       = snap.token_stream.buffer_size;
    yaml["token_stream"]["use_memory_stream"] = snap.token_stream.use_memory_stream;
    yaml["token_stream"]["dedup_ttl_ms"]      = snap.token_stream.dedup_ttl_ms;

    // Trading
    yaml["trading"]["bias_sensitivity"]      = snap.trading.bias_sensitivity;
    yaml["trading"]["volatility_sensitivity"] = snap.trading.volatility_sensitivity;
    yaml["trading"]["signal_decay_rate"]     = snap.trading.signal_decay_rate;
    yaml["trading"]["signal_cooldown_us"]    = snap.trading.signal_cooldown_us;
    yaml["trading"]["max_signal_age_us"]     = snap.trading.max_signal_age_us;
    yaml["trading"]["min_bias_threshold"]    = snap.trading.min_bias_threshold;
    yaml["trading"]["max_accumulated_bias"]  = snap.trading.max_accumulated_bias;

    // Latency
    yaml["latency"]["target_latency_us"] = snap.latency.target_latency_us;
    yaml["latency"]["sample_window"]     = snap.latency.sample_window;
    yaml["latency"]["enable_profiling"]  = snap.latency.enable_profiling;

    // Logging
    yaml["logging"]["log_file_path"]    = snap.logging.log_file_path;
    yaml["logging"]["format"]           = snap.logging.format;
    yaml["logging"]["enable_console"]   = snap.logging.enable_console;
    yaml["logging"]["flush_interval_ms"] = snap.logging.flush_interval_ms;

    // Metrics endpoint
    yaml["metrics"]["stats_port"]   = snap.metrics.stats_port;
    yaml["metrics"]["bind_address"] = snap.metrics.bind_address;

    // Pressure
    yaml["pressure"]["max_ingestion_rate_tps"] = snap.pressure.max_ingestion_rate_tps;
    yaml["pressure"]["backoff_scale_factor"]   = snap.pressure.backoff_scale_factor;

    // Risk thresholds
    yaml["risk_thresholds"]["max_bias_magnitude"]       = snap.risk_thresholds.max_bias_magnitude;
    yaml["risk_thresholds"]["max_volatility_magnitude"] = snap.risk_thresholds.max_volatility_magnitude;
    yaml["risk_thresholds"]["max_spread_magnitude"]     = snap.risk_thresholds.max_spread_magnitude;
    yaml["risk_thresholds"]["min_confidence"]           = snap.risk_thresholds.min_confidence;
    yaml["risk_thresholds"]["max_signals_per_second"]   = snap.risk_thresholds.max_signals_per_second;
    yaml["risk_thresholds"]["max_drawdown"]             = snap.risk_thresholds.max_drawdown;
    yaml["risk_thresholds"]["drawdown_window_s"]        = snap.risk_thresholds.drawdown_window_s;
    yaml["risk_thresholds"]["position_warn_fraction"]   = snap.risk_thresholds.position_warn_fraction;

    // Risk overrides (gate bypass flags — testing/debugging only)
    yaml["risk"]["disable_magnitude_gate"]  = snap.risk_overrides.disable_magnitude_gate;
    yaml["risk"]["disable_confidence_gate"] = snap.risk_overrides.disable_confidence_gate;
    yaml["risk"]["disable_rate_gate"]       = snap.risk_overrides.disable_rate_gate;
    yaml["risk"]["disable_drawdown_gate"]   = snap.risk_overrides.disable_drawdown_gate;
    yaml["risk"]["disable_position_gate"]   = snap.risk_overrides.disable_position_gate;

    // Semantic weight multipliers
    yaml["semantic_weights"]["sentiment_multiplier"]  = snap.semantic_weights.sentiment_multiplier;
    yaml["semantic_weights"]["confidence_multiplier"] = snap.semantic_weights.confidence_multiplier;
    yaml["semantic_weights"]["volatility_multiplier"] = snap.semantic_weights.volatility_multiplier;
    yaml["semantic_weights"]["bias_multiplier"]       = snap.semantic_weights.bias_multiplier;

    return YAML::Dump(yaml);
}

bool Config::save_to_file(const std::string& filepath) const {
    std::string content = to_yaml_string();
    std::ofstream f(filepath);
    if (!f.is_open()) {
        spdlog::error("[config] Failed to open '{}' for writing", filepath);
        return false;
    }
    f << content;
    if (!f.good()) {
        spdlog::error("[config] Write error for '{}'", filepath);
        return false;
    }
    return true;
}

void Config::set_defaults() {
    std::lock_guard<std::mutex> lk(config_mutex_);
    config_ = SystemConfig{};
}

bool Config::start_watching(const std::string& filepath,
                            std::function<void(const SystemConfig&)> on_reload,
                            int poll_interval_ms) {
    bool expected = false;
    if (!watching_.compare_exchange_strong(expected, true,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
        return false;  // Another thread already started the watcher.
    }

    try {
    watcher_thread_ = std::thread([this, filepath, on_reload, poll_interval_ms]() {
        namespace fs = std::filesystem;
        std::filesystem::file_time_type last_mtime{};

        // Capture the initial mtime so we don't fire immediately.
        try {
            last_mtime = fs::last_write_time(filepath);
        } catch (...) {}

        auto next_check = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(poll_interval_ms);

        while (watching_.load()) {
            // Interruptible compensating sleep: wake every 50 ms to check
            // watching_ so that stop_watching() returns promptly.
            auto now_s = std::chrono::steady_clock::now();
            while (watching_.load() && now_s < next_check) {
                auto remaining = next_check - now_s;
                auto slice = std::chrono::milliseconds{50};
                std::this_thread::sleep_for(remaining < slice ? remaining : slice);
                now_s = std::chrono::steady_clock::now();
            }
            if (!watching_.load()) break;
            next_check += std::chrono::milliseconds(poll_interval_ms);

            try {
                auto mtime = fs::last_write_time(filepath);
                if (mtime != last_mtime) {
                    last_mtime = mtime;
                    bool loaded = load_from_file(filepath);
                    if (loaded) {
                        // Take a snapshot under the lock; call on_reload outside
                        // so the callback can safely call get_config().
                        SystemConfig snapshot;
                        {
                            std::lock_guard<std::mutex> lk(config_mutex_);
                            snapshot = config_;
                        }
                        on_reload(snapshot);
                    }
                }
            } catch (...) {
                // File temporarily unavailable — retry next poll.
            }
        }
    });
    } catch (const std::exception& e) {
        spdlog::error("Config: failed to start hot-reload watcher thread: {}", e.what());
        watching_ = false;
        return false;
    }
    return true;
}

void Config::stop_watching() {
    watching_ = false;
    if (watcher_thread_.joinable()) {
        watcher_thread_.join();
    }
}

int Config::load_from_env() {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4996)  // 'getenv': safe in single-process use; _dupenv_s requires free()
#endif
    // Helper lambdas for type-safe env-var parsing.
    auto get_double = [](const char* name, double& out) -> bool {
        const char* val = std::getenv(name);
        if (!val || val[0] == '\0') return false;
        try {
            size_t pos;
            double d = std::stod(val, &pos);
            if (pos == 0 || !std::isfinite(d)) return false;
            out = d;
            return true;
        } catch (...) { return false; }
    };
    auto get_int = [](const char* name, int& out) -> bool {
        const char* val = std::getenv(name);
        if (!val || val[0] == '\0') return false;
        try {
            size_t pos;
            long l = std::stol(val, &pos);
            if (pos == 0) return false;
            out = static_cast<int>(l);
            return true;
        } catch (...) { return false; }
    };
    auto get_size_t = [](const char* name, size_t& out) -> bool {
        const char* val = std::getenv(name);
        if (!val || val[0] == '\0') return false;
        try {
            size_t pos;
            unsigned long ul = std::stoul(val, &pos);
            if (pos == 0) return false;
            out = static_cast<size_t>(ul);
            return true;
        } catch (...) { return false; }
    };
    auto get_uint16 = [](const char* name, uint16_t& out) -> bool {
        const char* val = std::getenv(name);
        if (!val || val[0] == '\0') return false;
        try {
            size_t pos;
            unsigned long ul = std::stoul(val, &pos);
            if (pos == 0 || ul > 65535) return false;
            out = static_cast<uint16_t>(ul);
            return true;
        } catch (...) { return false; }
    };

    int applied = 0;
    std::lock_guard<std::mutex> lk(config_mutex_);

    double d{}; int i{}; size_t sz{}; uint16_t u16{};

    if (get_double("LLMQUANT_BIAS_SENSITIVITY", d) && d > 0.0)
        { config_.trading.bias_sensitivity = d; ++applied; }
    if (get_double("LLMQUANT_VOL_SENSITIVITY", d) && d > 0.0)
        { config_.trading.volatility_sensitivity = d; ++applied; }
    if (get_double("LLMQUANT_SIGNAL_DECAY", d) && d > 0.0 && d <= 1.0)
        { config_.trading.signal_decay_rate = d; ++applied; }
    if (get_int("LLMQUANT_SIGNAL_COOLDOWN_US", i) && i >= 0)
        { config_.trading.signal_cooldown_us = i; ++applied; }
    if (get_double("LLMQUANT_MAX_SIGNAL_AGE_US", d) && d >= 0.0)
        { config_.trading.max_signal_age_us = d; ++applied; }
    if (get_double("LLMQUANT_MIN_BIAS_THRESHOLD", d) && d >= 0.0)
        { config_.trading.min_bias_threshold = d; ++applied; }
    if (get_double("LLMQUANT_MAX_ACCUMULATED_BIAS", d) && d >= 0.0)
        { config_.trading.max_accumulated_bias = d; ++applied; }
    if (get_double("LLMQUANT_MAX_DRAWDOWN", d) && d >= 0.0)
        { config_.risk_thresholds.max_drawdown = d; ++applied; }
    if (get_size_t("LLMQUANT_MAX_SIGNALS_PER_SECOND", sz))
        { config_.risk_thresholds.max_signals_per_second = sz; ++applied; }
    if (get_uint16("LLMQUANT_STATS_PORT", u16) && u16 > 0)
        { config_.metrics.stats_port = u16; ++applied; }

    {
        const char* log_file = std::getenv("LLMQUANT_LOG_FILE");
        if (log_file && log_file[0] != '\0') {
            config_.logging.log_file_path = log_file;
            ++applied;
        }
    }
    {
        const char* log_fmt = std::getenv("LLMQUANT_LOG_FORMAT");
        if (log_fmt && log_fmt[0] != '\0') {
            std::string fmt = log_fmt;
            std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                           [](unsigned char c){ return std::toupper(c); });
            config_.logging.format = fmt;
            ++applied;
        }
    }

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
    return applied;
}

std::vector<std::string> Config::validate() const {
    std::vector<std::string> errors;
    SystemConfig snap;
    {
        std::lock_guard<std::mutex> lk(config_mutex_);
        snap = config_;
    }
    const auto& ts  = snap.token_stream;
    const auto& tr  = snap.trading;
    const auto& lat = snap.latency;
    const auto& log = snap.logging;
    const auto& rt  = snap.risk_thresholds;
    const auto& pr  = snap.pressure;
    const auto& m   = snap.metrics;

    if (ts.token_interval_ms <= 0 || ts.token_interval_ms > 60000)
        errors.emplace_back("token_interval_ms out of range [1, 60000]");
    if (ts.buffer_size == 0)
        errors.emplace_back("buffer_size must be >= 1");
    if (!std::isfinite(tr.bias_sensitivity) || tr.bias_sensitivity <= 0.0 || tr.bias_sensitivity > 10.0)
        errors.emplace_back("bias_sensitivity must be in (0, 10]");
    if (!std::isfinite(tr.volatility_sensitivity) || tr.volatility_sensitivity <= 0.0 || tr.volatility_sensitivity > 10.0)
        errors.emplace_back("volatility_sensitivity must be in (0, 10]");
    if (!std::isfinite(tr.signal_decay_rate) || tr.signal_decay_rate <= 0.0 || tr.signal_decay_rate > 1.0)
        errors.emplace_back("signal_decay_rate must be in (0, 1]");
    if (tr.signal_cooldown_us < 0)
        errors.emplace_back("signal_cooldown_us must be >= 0");
    if (!std::isfinite(tr.max_signal_age_us) || tr.max_signal_age_us < 0.0)
        errors.emplace_back("trading.max_signal_age_us must be >= 0");
    if (!std::isfinite(tr.min_bias_threshold) || tr.min_bias_threshold < 0.0)
        errors.emplace_back("trading.min_bias_threshold must be >= 0");
    if (!std::isfinite(tr.max_accumulated_bias) || tr.max_accumulated_bias < 0.0)
        errors.emplace_back("trading.max_accumulated_bias must be >= 0");
    if (lat.target_latency_us <= 0)
        errors.emplace_back("target_latency_us must be > 0");
    if (lat.sample_window == 0)
        errors.emplace_back("sample_window must be >= 1");
    if (log.flush_interval_ms <= 0)
        errors.emplace_back("flush_interval_ms must be > 0");
    if (m.stats_port == 0)
        errors.emplace_back("metrics.stats_port must be > 0");
    if (!std::isfinite(pr.max_ingestion_rate_tps) || pr.max_ingestion_rate_tps <= 0.0)
        errors.emplace_back("pressure.max_ingestion_rate_tps must be > 0");
    if (!std::isfinite(pr.backoff_scale_factor) || pr.backoff_scale_factor <= 0.0)
        errors.emplace_back("pressure.backoff_scale_factor must be > 0");
    if (!std::isfinite(rt.max_bias_magnitude) || rt.max_bias_magnitude < 0.0)
        errors.emplace_back("risk_thresholds.max_bias_magnitude must be >= 0");
    if (!std::isfinite(rt.min_confidence) || rt.min_confidence < 0.0 || rt.min_confidence > 1.0)
        errors.emplace_back("risk_thresholds.min_confidence must be in [0, 1]");
    if (rt.max_signals_per_second == 0)
        errors.emplace_back("risk_thresholds.max_signals_per_second must be > 0");
    if (!std::isfinite(rt.max_drawdown) || rt.max_drawdown < 0.0)
        errors.emplace_back("risk_thresholds.max_drawdown must be >= 0");
    if (rt.drawdown_window_s <= 0)
        errors.emplace_back("risk_thresholds.drawdown_window_s must be > 0");
    if (!std::isfinite(rt.position_warn_fraction) || rt.position_warn_fraction < 0.0 || rt.position_warn_fraction > 1.0)
        errors.emplace_back("risk_thresholds.position_warn_fraction must be in [0, 1]");
    const auto& sw = snap.semantic_weights;
    if (!std::isfinite(sw.sentiment_multiplier))
        errors.emplace_back("semantic_weights.sentiment_multiplier must be a finite number");
    if (!std::isfinite(sw.confidence_multiplier))
        errors.emplace_back("semantic_weights.confidence_multiplier must be a finite number");
    if (!std::isfinite(sw.volatility_multiplier))
        errors.emplace_back("semantic_weights.volatility_multiplier must be a finite number");
    if (!std::isfinite(sw.bias_multiplier))
        errors.emplace_back("semantic_weights.bias_multiplier must be a finite number");
    // Validate logging format string — only "CSV" and "JSON" are supported.
    if (log.format != "CSV" && log.format != "JSON")
        errors.emplace_back("logging.format must be \"CSV\" or \"JSON\"; got \"" + log.format + "\"");
    // dedup_ttl_ms == 0 means auto (10× token_interval_ms); negative is invalid.
    if (ts.dedup_ttl_ms < 0)
        errors.emplace_back("token_stream.dedup_ttl_ms must be >= 0 (0 = auto)");
    return errors;
}

std::string Config::to_summary_string() const {
    std::lock_guard<std::mutex> lk(config_mutex_);
    std::ostringstream ss;
    ss << "=== Engine Configuration Summary ===\n"
       << "[trading]  bias_sensitivity=" << config_.trading.bias_sensitivity
       << "  vol_sensitivity=" << config_.trading.volatility_sensitivity
       << "  decay=" << config_.trading.signal_decay_rate
       << "  cooldown_us=" << config_.trading.signal_cooldown_us << "\n"
       << "[trading]  max_signal_age_us=" << config_.trading.max_signal_age_us
       << "  min_bias_threshold=" << config_.trading.min_bias_threshold
       << "  max_accumulated_bias=" << config_.trading.max_accumulated_bias << "\n"
       << "[risk]     max_bias=" << config_.risk_thresholds.max_bias_magnitude
       << "  max_vol=" << config_.risk_thresholds.max_volatility_magnitude
       << "  min_conf=" << config_.risk_thresholds.min_confidence
       << "  max_drawdown=" << config_.risk_thresholds.max_drawdown
       << "  rate_cap=" << config_.risk_thresholds.max_signals_per_second << "/s\n"
       << "[latency]  target_us=" << config_.latency.target_latency_us
       << "  window=" << config_.latency.sample_window
       << "  profiling=" << (config_.latency.enable_profiling ? "on" : "off") << "\n"
       << "[metrics]  port=" << config_.metrics.stats_port
       << "  bind=" << config_.metrics.bind_address << "\n"
       << "[pressure] max_tps=" << config_.pressure.max_ingestion_rate_tps
       << "  backoff_scale=" << config_.pressure.backoff_scale_factor << "\n"
       << "[logging]  file=" << config_.logging.log_file_path
       << "  format=" << config_.logging.format
       << "  flush_ms=" << config_.logging.flush_interval_ms << "\n"
       << "[stream]   interval_ms=" << config_.token_stream.token_interval_ms
       << "  buffer=" << config_.token_stream.buffer_size
       << "  mem=" << (config_.token_stream.use_memory_stream ? "true" : "false")
       << "  dedup_ttl_ms=" << config_.token_stream.dedup_ttl_ms
       << (config_.token_stream.dedup_ttl_ms == 0 ? " (auto)" : "") << "\n"
       << "[sem_wts]  sentiment=" << config_.semantic_weights.sentiment_multiplier
       << "  confidence=" << config_.semantic_weights.confidence_multiplier
       << "  volatility=" << config_.semantic_weights.volatility_multiplier
       << "  bias=" << config_.semantic_weights.bias_multiplier << "\n"
       << "[overrides]"
       << " magnitude=" << (config_.risk_overrides.disable_magnitude_gate ? "DISABLED" : "on")
       << "  confidence=" << (config_.risk_overrides.disable_confidence_gate ? "DISABLED" : "on")
       << "  rate=" << (config_.risk_overrides.disable_rate_gate ? "DISABLED" : "on")
       << "  drawdown=" << (config_.risk_overrides.disable_drawdown_gate ? "DISABLED" : "on")
       << "  position=" << (config_.risk_overrides.disable_position_gate ? "DISABLED" : "on") << "\n";
    return ss.str();
}

std::vector<std::string> Config::diff_from_defaults() const {
    SystemConfig snap;
    {
        std::lock_guard<std::mutex> lk(config_mutex_);
        snap = config_;
    }
    const SystemConfig def{};
    std::vector<std::string> diffs;

    auto dbl = [&](const char* key, double cur, double dflt) {
        if (cur != dflt) {
            std::ostringstream oss;
            oss << key << " = " << cur << "  (default: " << dflt << ")";
            diffs.push_back(oss.str());
        }
    };
    auto i32 = [&](const char* key, int cur, int dflt) {
        if (cur != dflt) {
            diffs.push_back(std::string(key) + " = " + std::to_string(cur)
                            + "  (default: " + std::to_string(dflt) + ")");
        }
    };
    auto u64 = [&](const char* key, size_t cur, size_t dflt) {
        if (cur != dflt) {
            diffs.push_back(std::string(key) + " = " + std::to_string(cur)
                            + "  (default: " + std::to_string(dflt) + ")");
        }
    };
    auto str = [&](const char* key, const std::string& cur, const std::string& dflt) {
        if (cur != dflt) {
            diffs.push_back(std::string(key) + " = \"" + cur
                            + "\"  (default: \"" + dflt + "\")");
        }
    };
    auto bl = [&](const char* key, bool cur, bool dflt) {
        if (cur != dflt) {
            diffs.push_back(std::string(key) + " = " + (cur ? "true" : "false")
                            + "  (default: " + (dflt ? "true" : "false") + ")");
        }
    };

    // token_stream
    str("token_stream.data_file_path",    snap.token_stream.data_file_path,    def.token_stream.data_file_path);
    i32("token_stream.token_interval_ms", snap.token_stream.token_interval_ms, def.token_stream.token_interval_ms);
    u64("token_stream.buffer_size",       snap.token_stream.buffer_size,       def.token_stream.buffer_size);
    bl ("token_stream.use_memory_stream", snap.token_stream.use_memory_stream, def.token_stream.use_memory_stream);
    i32("token_stream.dedup_ttl_ms",      snap.token_stream.dedup_ttl_ms,      def.token_stream.dedup_ttl_ms);

    // trading
    dbl("trading.bias_sensitivity",      snap.trading.bias_sensitivity,      def.trading.bias_sensitivity);
    dbl("trading.volatility_sensitivity",snap.trading.volatility_sensitivity,def.trading.volatility_sensitivity);
    dbl("trading.signal_decay_rate",     snap.trading.signal_decay_rate,     def.trading.signal_decay_rate);
    i32("trading.signal_cooldown_us",    snap.trading.signal_cooldown_us,    def.trading.signal_cooldown_us);
    dbl("trading.max_signal_age_us",     snap.trading.max_signal_age_us,     def.trading.max_signal_age_us);
    dbl("trading.min_bias_threshold",    snap.trading.min_bias_threshold,    def.trading.min_bias_threshold);
    dbl("trading.max_accumulated_bias",  snap.trading.max_accumulated_bias,  def.trading.max_accumulated_bias);

    // latency
    i32("latency.target_latency_us", snap.latency.target_latency_us, def.latency.target_latency_us);
    u64("latency.sample_window",     snap.latency.sample_window,     def.latency.sample_window);
    bl ("latency.enable_profiling",  snap.latency.enable_profiling,  def.latency.enable_profiling);

    // logging
    str("logging.log_file_path",   snap.logging.log_file_path,   def.logging.log_file_path);
    str("logging.format",          snap.logging.format,          def.logging.format);
    bl ("logging.enable_console",  snap.logging.enable_console,  def.logging.enable_console);
    i32("logging.flush_interval_ms",snap.logging.flush_interval_ms,def.logging.flush_interval_ms);

    // metrics
    if (snap.metrics.stats_port != def.metrics.stats_port) {
        diffs.push_back(std::string("metrics.stats_port = ")
                        + std::to_string(snap.metrics.stats_port)
                        + "  (default: " + std::to_string(def.metrics.stats_port) + ")");
    }
    str("metrics.bind_address", snap.metrics.bind_address, def.metrics.bind_address);

    // pressure
    dbl("pressure.max_ingestion_rate_tps", snap.pressure.max_ingestion_rate_tps, def.pressure.max_ingestion_rate_tps);
    dbl("pressure.backoff_scale_factor",   snap.pressure.backoff_scale_factor,   def.pressure.backoff_scale_factor);

    // risk_thresholds
    dbl("risk_thresholds.max_bias_magnitude",      snap.risk_thresholds.max_bias_magnitude,      def.risk_thresholds.max_bias_magnitude);
    dbl("risk_thresholds.max_volatility_magnitude",snap.risk_thresholds.max_volatility_magnitude,def.risk_thresholds.max_volatility_magnitude);
    dbl("risk_thresholds.max_spread_magnitude",    snap.risk_thresholds.max_spread_magnitude,    def.risk_thresholds.max_spread_magnitude);
    dbl("risk_thresholds.min_confidence",          snap.risk_thresholds.min_confidence,          def.risk_thresholds.min_confidence);
    u64("risk_thresholds.max_signals_per_second",  snap.risk_thresholds.max_signals_per_second,  def.risk_thresholds.max_signals_per_second);
    dbl("risk_thresholds.max_drawdown",            snap.risk_thresholds.max_drawdown,            def.risk_thresholds.max_drawdown);
    i32("risk_thresholds.drawdown_window_s",       snap.risk_thresholds.drawdown_window_s,       def.risk_thresholds.drawdown_window_s);
    dbl("risk_thresholds.position_warn_fraction",  snap.risk_thresholds.position_warn_fraction,  def.risk_thresholds.position_warn_fraction);

    // risk_overrides
    bl("risk_overrides.disable_magnitude_gate",  snap.risk_overrides.disable_magnitude_gate,  def.risk_overrides.disable_magnitude_gate);
    bl("risk_overrides.disable_confidence_gate", snap.risk_overrides.disable_confidence_gate, def.risk_overrides.disable_confidence_gate);
    bl("risk_overrides.disable_rate_gate",       snap.risk_overrides.disable_rate_gate,       def.risk_overrides.disable_rate_gate);
    bl("risk_overrides.disable_drawdown_gate",   snap.risk_overrides.disable_drawdown_gate,   def.risk_overrides.disable_drawdown_gate);
    bl("risk_overrides.disable_position_gate",   snap.risk_overrides.disable_position_gate,   def.risk_overrides.disable_position_gate);

    // semantic_weights
    dbl("semantic_weights.sentiment_multiplier",  snap.semantic_weights.sentiment_multiplier,  def.semantic_weights.sentiment_multiplier);
    dbl("semantic_weights.confidence_multiplier", snap.semantic_weights.confidence_multiplier, def.semantic_weights.confidence_multiplier);
    dbl("semantic_weights.volatility_multiplier", snap.semantic_weights.volatility_multiplier, def.semantic_weights.volatility_multiplier);
    dbl("semantic_weights.bias_multiplier",       snap.semantic_weights.bias_multiplier,       def.semantic_weights.bias_multiplier);

    return diffs;
}

} // namespace llmquant
