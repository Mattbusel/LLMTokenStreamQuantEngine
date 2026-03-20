#include "Config.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
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

bool Config::save_to_file(const std::string& filepath) const {
    SystemConfig snap;
    {
        std::lock_guard<std::mutex> lk(config_mutex_);
        snap = config_;
    }
    const auto& snap_cfg = snap;  // local alias — avoids shadowing the config_ member
    YAML::Node yaml;

    // Token stream
    yaml["token_stream"]["data_file_path"] = snap_cfg.token_stream.data_file_path;
    yaml["token_stream"]["token_interval_ms"] = snap_cfg.token_stream.token_interval_ms;
    yaml["token_stream"]["buffer_size"] = snap_cfg.token_stream.buffer_size;
    yaml["token_stream"]["use_memory_stream"] = snap_cfg.token_stream.use_memory_stream;

    // Trading
    yaml["trading"]["bias_sensitivity"] = snap_cfg.trading.bias_sensitivity;
    yaml["trading"]["volatility_sensitivity"] = snap_cfg.trading.volatility_sensitivity;
    yaml["trading"]["signal_decay_rate"] = snap_cfg.trading.signal_decay_rate;
    yaml["trading"]["signal_cooldown_us"]  = snap_cfg.trading.signal_cooldown_us;
    yaml["trading"]["max_signal_age_us"]    = snap_cfg.trading.max_signal_age_us;
    yaml["trading"]["min_bias_threshold"]  = snap_cfg.trading.min_bias_threshold;
    yaml["trading"]["max_accumulated_bias"] = snap_cfg.trading.max_accumulated_bias;

    // Latency
    yaml["latency"]["target_latency_us"] = snap_cfg.latency.target_latency_us;
    yaml["latency"]["sample_window"] = snap_cfg.latency.sample_window;
    yaml["latency"]["enable_profiling"] = snap_cfg.latency.enable_profiling;

    // Logging
    yaml["logging"]["log_file_path"] = snap_cfg.logging.log_file_path;
    yaml["logging"]["format"] = snap_cfg.logging.format;
    yaml["logging"]["enable_console"] = snap_cfg.logging.enable_console;
    yaml["logging"]["flush_interval_ms"] = snap_cfg.logging.flush_interval_ms;

    // Metrics endpoint
    yaml["metrics"]["stats_port"]   = snap_cfg.metrics.stats_port;
    yaml["metrics"]["bind_address"] = snap_cfg.metrics.bind_address;

    // Pressure
    yaml["pressure"]["max_ingestion_rate_tps"] = snap_cfg.pressure.max_ingestion_rate_tps;
    yaml["pressure"]["backoff_scale_factor"]   = snap_cfg.pressure.backoff_scale_factor;

    // Risk thresholds
    yaml["risk_thresholds"]["max_bias_magnitude"]       = snap_cfg.risk_thresholds.max_bias_magnitude;
    yaml["risk_thresholds"]["max_volatility_magnitude"] = snap_cfg.risk_thresholds.max_volatility_magnitude;
    yaml["risk_thresholds"]["max_spread_magnitude"]     = snap_cfg.risk_thresholds.max_spread_magnitude;
    yaml["risk_thresholds"]["min_confidence"]           = snap_cfg.risk_thresholds.min_confidence;
    yaml["risk_thresholds"]["max_signals_per_second"]   = snap_cfg.risk_thresholds.max_signals_per_second;
    yaml["risk_thresholds"]["max_drawdown"]             = snap_cfg.risk_thresholds.max_drawdown;
    yaml["risk_thresholds"]["drawdown_window_s"]        = snap_cfg.risk_thresholds.drawdown_window_s;
    yaml["risk_thresholds"]["position_warn_fraction"]   = snap_cfg.risk_thresholds.position_warn_fraction;

    std::ofstream f(filepath);
    if (!f.is_open()) {
        spdlog::error("[config] Failed to open '{}' for writing", filepath);
        return false;
    }
    f << yaml;
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
            // Compensating sleep: drift-free polling.
            auto now_s = std::chrono::steady_clock::now();
            if (next_check > now_s) {
                std::this_thread::sleep_for(next_check - now_s);
            }
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

} // namespace llmquant
