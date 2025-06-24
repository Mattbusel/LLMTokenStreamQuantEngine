#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <string>
#include <memory>
#include <atomic>

namespace llmquant {

class MetricsLogger {
public:
    enum class OutputFormat {
        CSV,
        JSON,
        BINARY
    };

    struct Config {
        std::string log_file_path{"metrics.log"};
        OutputFormat format{OutputFormat::CSV};
        bool enable_console_output{true};
        std::chrono::milliseconds flush_interval{100};
    };

    explicit MetricsLogger(const Config& config);
    ~MetricsLogger();

    // Logging interface
    void log_token_received(const std::string& token, uint64_t sequence_id);
    void log_signal_generated(double bias, double volatility, uint64_t latency_us);
    void log_latency_measurement(uint64_t latency_us);
    void log_system_stats(uint64_t memory_usage, double cpu_usage);
    
    // Performance metrics
    void log_performance_summary();
    void flush();

private:
    void initialize_loggers();
    void write_csv_header();
    
    Config config_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::shared_ptr<spdlog::logger> console_logger_;
    std::atomic<uint64_t> log_entries_{0};
};

} // namespace llmquant
