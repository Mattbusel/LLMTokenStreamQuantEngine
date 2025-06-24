#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

namespace llmquant {

struct SystemConfig {
    // Token stream settings
    struct {
        std::string data_file_path{"data/mock_token_streams/sample.txt"};
        int token_interval_ms{10};
        size_t buffer_size{1024};
        bool use_memory_stream{true};
    } token_stream;
    
    // Trading engine settings
    struct {
        double bias_sensitivity{1.0};
        double volatility_sensitivity{1.0};
        double signal_decay_rate{0.95};
        int signal_cooldown_us{1000};
    } trading;
    
    // Latency settings
    struct {
        int target_latency_us{10};
        size_t sample_window{1000};
        bool enable_profiling{true};
    } latency;
    
    // Logging settings
    struct {
        std::string log_file_path{"logs/metrics.log"};
        std::string format{"CSV"};
        bool enable_console{true};
        int flush_interval_ms{100};
    } logging;
};

class Config {
public:
    Config() = default;
    ~Config() = default;

    bool load_from_file(const std::string& filepath);
    bool load_from_yaml_string(const std::string& yaml_content);
    void save_to_file(const std::string& filepath) const;
    
    const SystemConfig& get_config() const { return config_; }
    SystemConfig& get_mutable_config() { return config_; }

private:
    void set_defaults();
    
    SystemConfig config_;
};

} // namespace llmquant
