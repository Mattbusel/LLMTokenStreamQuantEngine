#include "Config.h"
#include <fstream>
#include <iostream>

namespace llmquant {

bool Config::load_from_file(const std::string& filepath) {
    try {
        YAML::Node yaml = YAML::LoadFile(filepath);
        return load_from_yaml_string(YAML::Dump(yaml));
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load config file " << filepath << ": " << e.what() << std::endl;
        set_defaults();
        return false;
    }
}

bool Config::load_from_yaml_string(const std::string& yaml_content) {
    try {
        YAML::Node yaml = YAML::Load(yaml_content);
        
        // Token stream settings
        if (yaml["token_stream"]) {
            auto ts = yaml["token_stream"];
            if (ts["data_file_path"]) config_.token_stream.data_file_path = ts["data_file_path"].as<std::string>();
            if (ts["token_interval_ms"]) config_.token_stream.token_interval_ms = ts["token_interval_ms"].as<int>();
            if (ts["buffer_size"]) config_.token_stream.buffer_size = ts["buffer_size"].as<size_t>();
            if (ts["use_memory_stream"]) config_.token_stream.use_memory_stream = ts["use_memory_stream"].as<bool>();
        }
        
        // Trading settings
        if (yaml["trading"]) {
            auto t = yaml["trading"];
            if (t["bias_sensitivity"]) config_.trading.bias_sensitivity = t["bias_sensitivity"].as<double>();
            if (t["volatility_sensitivity"]) config_.trading.volatility_sensitivity = t["volatility_sensitivity"].as<double>();
            if (t["signal_decay_rate"]) config_.trading.signal_decay_rate = t["signal_decay_rate"].as<double>();
            if (t["signal_cooldown_us"]) config_.trading.signal_cooldown_us = t["signal_cooldown_us"].as<int>();
        }
        
        // Latency settings
        if (yaml["latency"]) {
            auto l = yaml["latency"];
            if (l["target_latency_us"]) config_.latency.target_latency_us = l["target_latency_us"].as<int>();
            if (l["sample_window"]) config_.latency.sample_window = l["sample_window"].as<size_t>();
            if (l["enable_profiling"]) config_.latency.enable_profiling = l["enable_profiling"].as<bool>();
        }
        
        // Logging settings
        if (yaml["logging"]) {
            auto log = yaml["logging"];
            if (log["log_file_path"]) config_.logging.log_file_path = log["log_file_path"].as<std::string>();
            if (log["format"]) config_.logging.format = log["format"].as<std::string>();
            if (log["enable_console"]) config_.logging.enable_console = log["enable_console"].as<bool>();
            if (log["flush_interval_ms"]) config_.logging.flush_interval_ms = log["flush_interval_ms"].as<int>();
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to parse YAML config: " << e.what() << std::endl;
        set_defaults();
        return false;
    }
}

void Config::save_to_file(const std::string& filepath) const {
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
    
    std::ofstream file(filepath);
    file << yaml;
}

void Config::set_defaults() {
    // Defaults are already set in SystemConfig struct initialization
}

} // namespace llmquant
