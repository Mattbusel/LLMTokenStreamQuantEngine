#include "Config.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

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
        }

        // Latency settings
        if (yaml["latency"]) {
            auto l = yaml["latency"];
            if (l["target_latency_us"]) tmp.latency.target_latency_us = l["target_latency_us"].as<int>();
            if (l["sample_window"]) tmp.latency.sample_window = l["sample_window"].as<size_t>();
            if (l["enable_profiling"]) tmp.latency.enable_profiling = l["enable_profiling"].as<bool>();
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
        if (ts.token_interval_ms <= 0 || ts.token_interval_ms > 60000 || ts.buffer_size == 0 ||
            tr.bias_sensitivity <= 0.0 || tr.volatility_sensitivity <= 0.0 ||
            tr.signal_decay_rate <= 0.0 || tr.signal_decay_rate > 1.0 ||
            tr.signal_cooldown_us < 0 ||
            lat.target_latency_us <= 0 || lat.sample_window == 0 ||
            log.flush_interval_ms <= 0) {
            spdlog::error("Config validation failed: one or more fields out of range");
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
    const auto& config_ = snap;  // shadow member so the rest of the function is unchanged
    YAML::Node yaml;

    // Token stream
    yaml["token_stream"]["data_file_path"] = config_.token_stream.data_file_path;
    yaml["token_stream"]["token_interval_ms"] = config_.token_stream.token_interval_ms;
    yaml["token_stream"]["buffer_size"] = config_.token_stream.buffer_size;
    yaml["token_stream"]["use_memory_stream"] = config_.token_stream.use_memory_stream;

    // Trading
    yaml["trading"]["bias_sensitivity"] = config_.trading.bias_sensitivity;
    yaml["trading"]["volatility_sensitivity"] = config_.trading.volatility_sensitivity;
    yaml["trading"]["signal_decay_rate"] = config_.trading.signal_decay_rate;
    yaml["trading"]["signal_cooldown_us"] = config_.trading.signal_cooldown_us;

    // Latency
    yaml["latency"]["target_latency_us"] = config_.latency.target_latency_us;
    yaml["latency"]["sample_window"] = config_.latency.sample_window;
    yaml["latency"]["enable_profiling"] = config_.latency.enable_profiling;

    // Logging
    yaml["logging"]["log_file_path"] = config_.logging.log_file_path;
    yaml["logging"]["format"] = config_.logging.format;
    yaml["logging"]["enable_console"] = config_.logging.enable_console;
    yaml["logging"]["flush_interval_ms"] = config_.logging.flush_interval_ms;

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

void Config::start_watching(const std::string& filepath,
                            std::function<void(const SystemConfig&)> on_reload,
                            int poll_interval_ms) {
    bool expected = false;
    if (!watching_.compare_exchange_strong(expected, true,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
        return;  // Another thread already started the watcher.
    }

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
}

void Config::stop_watching() {
    watching_ = false;
    if (watcher_thread_.joinable()) {
        watcher_thread_.join();
    }
}

} // namespace llmquant
