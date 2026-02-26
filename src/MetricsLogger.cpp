#include "MetricsLogger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace llmquant {

MetricsLogger::MetricsLogger(const Config& config) : config_(config) {
    initialize_loggers();
    if (config_.format == OutputFormat::CSV) {
        write_csv_header();
    }
}

MetricsLogger::~MetricsLogger() {
    flush();
}

void MetricsLogger::initialize_loggers() {
    // File logger — drop gracefully if path is empty or creation fails.
    if (!config_.log_file_path.empty()) {
        try {
            // Use a unique logger name to survive multiple MetricsLogger instances.
            static std::atomic<int> inst_id{0};
            std::string name = "file_logger_" + std::to_string(inst_id.fetch_add(1));
            // Drop any stale logger of the same name before creating a new one.
            spdlog::drop(name);
            file_logger_ = spdlog::basic_logger_mt(name, config_.log_file_path);
            file_logger_->set_pattern("[%H:%M:%S.%f] %v");
            file_logger_->flush_on(spdlog::level::info);
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "[warn] MetricsLogger: file logger skipped: " << ex.what() << "\n";
            file_logger_.reset();
        }
    }

    // Console logger
    if (config_.enable_console_output) {
        try {
            static std::atomic<int> cons_id{0};
            std::string name = "console_logger_" + std::to_string(cons_id.fetch_add(1));
            spdlog::drop(name);
            console_logger_ = spdlog::stdout_color_mt(name);
            console_logger_->set_pattern("[%H:%M:%S.%f] %v");
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "[warn] MetricsLogger: console logger skipped: " << ex.what() << "\n";
            console_logger_.reset();
        }
    }
}

void MetricsLogger::write_csv_header() {
    if (file_logger_) {
        file_logger_->info("timestamp,event_type,token,sequence_id,bias,volatility,latency_us,memory_mb,cpu_pct");
    }
}

void MetricsLogger::log_token_received(const std::string& token, uint64_t sequence_id) {
    log_entries_++;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",TOKEN_RECEIVED," << token << "," << sequence_id << ",,,,,";
        
        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_) console_logger_->info("Token received: \"{}\"", token);
    }
}

void MetricsLogger::log_signal_generated(double bias, double volatility, uint64_t latency_us) {
    log_entries_++;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",SIGNAL_GENERATED,," << std::fixed << std::setprecision(3) 
            << ",," << bias << "," << volatility << "," << latency_us << ",,";
        
        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_) {
            console_logger_->info("Mapped signal: BIAS {:+.3f} | Volatility {:+.3f}", bias, volatility);
        }
    }
}

void MetricsLogger::log_latency_measurement(uint64_t latency_us) {
    if (config_.format == OutputFormat::CSV) {
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        std::ostringstream oss;
        oss << timestamp << ",LATENCY_MEASUREMENT,,,,,," << latency_us << ",,";
        
        if (file_logger_) file_logger_->info(oss.str());
    }
}

void MetricsLogger::log_system_stats(uint64_t memory_usage, double cpu_usage) {
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",SYSTEM_STATS,,,,,," << std::fixed << std::setprecision(1)
            << ",," << (memory_usage / 1024 / 1024) << "," << cpu_usage;
        
        if (file_logger_) file_logger_->info(oss.str());
    }
}

void MetricsLogger::log_performance_summary() {
    if (console_logger_) {
        console_logger_->info("=== Performance Summary ===");
        console_logger_->info("Total log entries: {}", log_entries_.load());
        console_logger_->info("Log file: {}", config_.log_file_path);
    }
}

void MetricsLogger::flush() {
    if (file_logger_) file_logger_->flush();
    if (console_logger_) console_logger_->flush();
}

} // namespace llmquant
