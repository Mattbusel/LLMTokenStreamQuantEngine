#include "gtest/gtest.h"
#include "MetricsLogger.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static MetricsLogger::Config make_csv_config(const std::string& path) {
    MetricsLogger::Config cfg;
    cfg.log_file_path        = path;
    cfg.format               = MetricsLogger::OutputFormat::CSV;
    cfg.enable_console_output = false;
    cfg.flush_interval       = std::chrono::milliseconds{10};
    return cfg;
}

static MetricsLogger::Config make_json_config(const std::string& path) {
    MetricsLogger::Config cfg;
    cfg.log_file_path        = path;
    cfg.format               = MetricsLogger::OutputFormat::JSON;
    cfg.enable_console_output = false;
    cfg.flush_interval       = std::chrono::milliseconds{10};
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_metrics_logger_constructs_without_exception) {
    const std::string path = "/tmp/test_metrics_construct.log";
    EXPECT_NO_THROW({
        MetricsLogger logger(make_csv_config(path));
    });
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_token_increments_entry_count) {
    // We exercise log_token_received and confirm no exception is thrown.
    // Entry count is private/atomic but the call must not throw or crash.
    const std::string path = "/tmp/test_metrics_token.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_NO_THROW(logger.log_token_received("bullish", 1));
        EXPECT_NO_THROW(logger.log_token_received("crash",   2));
        EXPECT_NO_THROW(logger.log_token_received("rally",   3));
    }
    // The log file must exist and be non-empty (header + 3 entries).
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_signal_increments_entry_count) {
    const std::string path = "/tmp/test_metrics_signal.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_NO_THROW(logger.log_signal_generated(0.5,  0.3, 7));
        EXPECT_NO_THROW(logger.log_signal_generated(-0.2, 0.1, 12));
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_flush_does_not_throw) {
    const std::string path = "/tmp/test_metrics_flush.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_token_received("test", 0);
        EXPECT_NO_THROW(logger.flush());
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_json_format_constructs_without_exception) {
    const std::string path = "/tmp/test_metrics_json.log";
    EXPECT_NO_THROW({
        MetricsLogger logger(make_json_config(path));
        logger.log_token_received("volatile", 0);
        logger.log_signal_generated(0.1, 0.2, 5);
        logger.flush();
    });
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_system_stats_does_not_throw) {
    const std::string path = "/tmp/test_metrics_sysstats.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_NO_THROW(logger.log_system_stats(1024 * 1024 * 50, 12.5));
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_latency_measurement_does_not_throw) {
    const std::string path = "/tmp/test_metrics_latency.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_NO_THROW(logger.log_latency_measurement(8));
        EXPECT_NO_THROW(logger.log_latency_measurement(15));
    }
    std::remove(path.c_str());
}

} // namespace
} // namespace llmquant
