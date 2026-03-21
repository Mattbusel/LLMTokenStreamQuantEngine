#include "MetricsLogger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iomanip>
#include <sstream>
#include <spdlog/spdlog.h>

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
            // Flush on warn/error only; periodic flush handled by flush_every below.
            file_logger_->flush_on(spdlog::level::warn);
        } catch (const spdlog::spdlog_ex& ex) {
            // Cannot use file logger to report its own failure; use spdlog default sink.
            spdlog::warn("[MetricsLogger] file logger skipped: {}", ex.what());
            file_logger_.reset();
        }
    }

    // Enable periodic flush at the configured interval so data reaches disk
    // without flushing on every INFO message (which would harm throughput).
    spdlog::flush_every(config_.flush_interval);

    // Console logger
    if (config_.enable_console_output) {
        try {
            static std::atomic<int> cons_id{0};
            std::string name = "console_logger_" + std::to_string(cons_id.fetch_add(1));
            spdlog::drop(name);
            console_logger_ = spdlog::stdout_color_mt(name);
            console_logger_->set_pattern("[%H:%M:%S.%f] %v");
        } catch (const spdlog::spdlog_ex& ex) {
            spdlog::warn("[MetricsLogger] console logger skipped: {}", ex.what());
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
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"token_received","token":"{}","sequence_id":{},"timestamp":{}}})",
                token, sequence_id, timestamp);
        if (console_logger_) console_logger_->info("Token received: \"{}\"", token);
    }
}

void MetricsLogger::log_signal_generated(double bias, double volatility, uint64_t latency_us) {
    log_entries_++;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",SIGNAL_GENERATED,,," << std::fixed << std::setprecision(3)
            << bias << "," << volatility << "," << latency_us << ",,";

        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_) {
            console_logger_->info("Mapped signal: BIAS {:+.3f} | Volatility {:+.3f}", bias, volatility);
        }
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"signal_generated","bias":{:.3f},"volatility":{:.3f},"latency_us":{},"timestamp":{}}})",
                bias, volatility, latency_us, timestamp);
        if (console_logger_) {
            console_logger_->info("Mapped signal: BIAS {:+.3f} | Volatility {:+.3f}", bias, volatility);
        }
    }
}

void MetricsLogger::log_latency_measurement(uint64_t latency_us) {
    log_entries_++;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",LATENCY_MEASUREMENT,,,,," << latency_us << ",,";

        if (file_logger_) file_logger_->info(oss.str());
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"latency_measurement","latency_us":{},"timestamp":{}}})",
                latency_us, timestamp);
    }
}

void MetricsLogger::log_system_stats(uint64_t memory_usage, double cpu_usage) {
    log_entries_++;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",SYSTEM_STATS,,,,,," << std::fixed << std::setprecision(1)
            << (memory_usage / 1024 / 1024) << "," << cpu_usage;

        if (file_logger_) file_logger_->info(oss.str());
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"system_stats","memory_mb":{:.1f},"cpu_pct":{:.1f},"timestamp":{}}})",
                static_cast<double>(memory_usage) / 1024.0 / 1024.0, cpu_usage, timestamp);
    }
}

void MetricsLogger::log_risk_rejection(const std::string& reason, double bias, double confidence) {
    log_entries_++;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        // Columns: timestamp,RISK_REJECTION,reason,,bias,,,, (9 cols matching header).
        // confidence has no dedicated CSV column; it is written to JSON only.
        // Previously confidence was placed in column 7 (latency_us), which was wrong.
        oss << timestamp << ",RISK_REJECTION," << reason << ",,"
            << std::fixed << std::setprecision(3) << bias << ",,,,";
        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_)
            console_logger_->warn("Risk rejection: {} bias={:+.3f} conf={:.3f}", reason, bias, confidence);
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"risk_rejection","reason":"{}","bias":{:.3f},"confidence":{:.3f},"timestamp":{}}})",
                reason, bias, confidence, timestamp);
        if (console_logger_)
            console_logger_->warn("Risk rejection: {} bias={:+.3f} conf={:.3f}", reason, bias, confidence);
    }
}

void MetricsLogger::log_trade_signal(double bias, double volatility,
                                      double confidence, double latency_us,
                                      double quality) {
    log_entries_++;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        // Columns: timestamp,TRADE_SIGNAL,,<empty>,bias,volatility,latency_us,<empty>,<empty>
        // confidence and quality are written to JSON only; they have no column in
        // the 9-column CSV header and were previously causing a column-count mismatch.
        oss << timestamp << ",TRADE_SIGNAL,,,"
            << std::fixed << std::setprecision(3)
            << bias << "," << volatility << "," << latency_us << ",,";
        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_)
            console_logger_->info("Signal bias={:+.3f} vol={:.3f} conf={:.3f} lat={:.1f}μs q={:.3f}",
                                  bias, volatility, confidence, latency_us, quality);
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"trade_signal","bias":{:.3f},"volatility":{:.3f},"confidence":{:.3f},"latency_us":{:.1f},"quality":{:.3f},"timestamp":{}}})",
                bias, volatility, confidence, latency_us, quality, timestamp);
        if (console_logger_)
            console_logger_->info("Signal bias={:+.3f} vol={:.3f} conf={:.3f} lat={:.1f}μs q={:.3f}",
                                  bias, volatility, confidence, latency_us, quality);
    }
}

void MetricsLogger::log_config_reload(const std::string& source_path, bool success) {
    log_entries_++;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const char* status = success ? "ok" : "failed";
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << timestamp << ",CONFIG_RELOAD," << status << "," << source_path << ",,,,,";
        if (file_logger_) file_logger_->info(oss.str());
        if (console_logger_)
            console_logger_->info("Config reload {}: {}", status, source_path);
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"config_reload","status":"{}","source":"{}","timestamp":{}}})",
                status, source_path, timestamp);
        if (console_logger_)
            console_logger_->info("Config reload {}: {}", status, source_path);
    }
}

void MetricsLogger::log_pipeline_health(bool healthy, double slo_breach_rate,
                                         double backoff_multiplier) {
    log_entries_++;
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const char* status = healthy ? "ok" : "degraded";
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << ts << ",PIPELINE_HEALTH," << status << ",,"
            << std::fixed << std::setprecision(4)
            << slo_breach_rate << "," << backoff_multiplier << ",,,";
        if (file_logger_) file_logger_->info(oss.str());
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"pipeline_health","status":"{}","slo_breach_rate":{},"backoff_multiplier":{},"timestamp":{}}})",
                status, slo_breach_rate, backoff_multiplier, ts);
    }
}

void MetricsLogger::log_dedup_event(const std::string& key, bool is_duplicate) {
    log_entries_++;
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const char* kind = is_duplicate ? "duplicate" : "novel";
    if (config_.format == OutputFormat::CSV) {
        std::ostringstream oss;
        oss << ts << ",DEDUP," << kind << "," << key << ",,,,,";
        if (file_logger_) file_logger_->info(oss.str());
    } else if (config_.format == OutputFormat::JSON) {
        if (file_logger_)
            file_logger_->info(
                R"({{"event":"dedup","kind":"{}","key":"{}","timestamp":{}}})",
                kind, key, ts);
    }
}

void MetricsLogger::log_performance_summary() {
    if (console_logger_) {
        double uptime_ms  = get_uptime_ms();
        double uptime_s   = uptime_ms / 1000.0;
        double entry_rate = get_log_rate();
        uint64_t entries  = log_entries_.load(std::memory_order_relaxed);
        console_logger_->info("=== Performance Summary ===");
        console_logger_->info("  Log entries  : {}", entries);
        console_logger_->info("  Uptime       : {:.1f} s", uptime_s);
        console_logger_->info("  Log rate     : {:.1f} entries/s", entry_rate);
        console_logger_->info("  Log file     : {}", config_.log_file_path.empty() ? "(none)" : config_.log_file_path);
        console_logger_->info("  Log format   : {}", config_.format == OutputFormat::CSV ? "CSV" : "JSON");
    }
}

void MetricsLogger::flush() {
    // spdlog's internal mutex makes this safe to call concurrently with the
    // background flush timer. Redundant flushes are a no-op at the OS level.
    if (file_logger_) file_logger_->flush();
    if (console_logger_) console_logger_->flush();
}

} // namespace llmquant
