#include "gtest/gtest.h"
#include "MetricsLogger.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read non-empty lines from a text file into a vector.
static std::vector<std::string> read_log_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

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

// ---------------------------------------------------------------------------
// New: inaccessible log path must not crash (improvement #17)
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_metrics_logger_inaccessible_path_does_not_crash) {
    // On most systems, writing to a path under /nonexistent/ will fail.
    // The logger must not throw or crash; it should silently degrade.
    MetricsLogger::Config cfg;
    cfg.log_file_path         = "/nonexistent/directory/llmquant_should_fail.log";
    cfg.format                = MetricsLogger::OutputFormat::CSV;
    cfg.enable_console_output = false;
    cfg.flush_interval        = std::chrono::milliseconds(100);

    // Construction itself must not throw.
    ASSERT_NO_THROW({
        MetricsLogger logger(cfg);
        // Basic operations must not crash even with an inaccessible file.
        logger.log_token_received("crash", 1);
        logger.log_signal_generated(0.5, 0.3, 10);
        logger.flush();
    });
}

} // namespace
} // namespace llmquant

// ---------------------------------------------------------------------------
// Improvement 15: MetricsLogger with non-ASCII token strings
// ---------------------------------------------------------------------------
using namespace llmquant;  // NOLINT: test-only using directive outside namespace

TEST(MetricsLoggerTest, test_metrics_logger_json_output_contains_expected_keys) {
    const std::string path = "/tmp/test_metrics_json_content.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_token_received("rally", 7);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // JSON output must contain the token name and known JSON keys.
    EXPECT_NE(content.find("rally"), std::string::npos)
        << "JSON output must contain the logged token";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, MetricsLoggerNonAsciiToken) {
    MetricsLogger::Config cfg;
    cfg.log_file_path         = "";  // no file output
    cfg.format                = MetricsLogger::OutputFormat::CSV;
    cfg.enable_console_output = false;
    cfg.flush_interval        = std::chrono::milliseconds(100);

    ASSERT_NO_THROW({
        MetricsLogger logger(cfg);
        // Valid UTF-8 (e with acute) and invalid UTF-8 bytes.
        logger.log_token_received("\xc3\xa9", 1);
        logger.log_token_received("\xff\xfe", 2);
        logger.log_signal_generated(0.1, 0.2, 5);
        logger.flush();
    });
}

// ---------------------------------------------------------------------------
// Cycle 16: get_log_entry_count, log_risk_rejection, log_performance_summary
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_metrics_logger_get_entry_count_increments) {
    const std::string path = "/tmp/test_metrics_entry_count.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{0});
        logger.log_token_received("bullish", 1);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{1});
        logger.log_signal_generated(0.5, 0.2, 7);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{2});
        logger.log_risk_rejection("magnitude_exceeded", 1.5, 0.9);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{3});
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_risk_rejection_does_not_throw) {
    const std::string path = "/tmp/test_metrics_risk_rejection.log";
    {
        MetricsLogger logger(make_csv_config(path));
        EXPECT_NO_THROW(logger.log_risk_rejection("magnitude_exceeded", 2.0, 0.95));
        EXPECT_NO_THROW(logger.log_risk_rejection("confidence_too_low",  0.1, 0.15));
        EXPECT_NO_THROW(logger.log_risk_rejection("rate_limit",          0.5, 0.7));
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{3});
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_risk_rejection_json_contains_reason) {
    const std::string path = "/tmp/test_metrics_risk_json.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_risk_rejection("drawdown_exceeded", 0.8, 0.6);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("drawdown_exceeded"), std::string::npos)
        << "JSON output must contain the rejection reason";
    EXPECT_NE(content.find("risk_rejection"), std::string::npos)
        << "JSON output must contain the event type";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_log_performance_summary_does_not_throw) {
    const std::string path = "/tmp/test_metrics_perf_summary.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_token_received("momentum", 1);
        logger.log_signal_generated(0.3, 0.1, 5);
        EXPECT_NO_THROW(logger.log_performance_summary());
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_entry_count_not_incremented_by_non_counting_methods) {
    // log_latency_measurement and log_system_stats DO count as log entries
    // (they call log_entries_++ since they write structured log rows).
    // log_performance_summary does NOT write to the file sink and does NOT increment.
    const std::string path = "/tmp/test_metrics_no_count.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_latency_measurement(10);   // increments
        logger.log_system_stats(1024 * 1024, 5.0); // increments
        logger.log_performance_summary();     // does NOT increment (console only)
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{2});
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_metrics_logger_reset_counters_zeroes_entry_count) {
    const std::string path = "/tmp/test_metrics_reset_counters.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_token_received("bullish", 1);
        logger.log_token_received("crash",   2);
        logger.log_signal_generated(0.5, 0.2, 8);
        ASSERT_EQ(logger.get_log_entry_count(), uint64_t{3});

        logger.reset_counters();
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{0})
            << "reset_counters() must zero the entry count";

        // Logging after reset must start counting from zero again.
        logger.log_token_received("rally", 3);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{1})
            << "Entry count must increment from 0 after reset";
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// log_trade_signal
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_log_trade_signal_increments_entry_count_csv) {
    const std::string path = "/tmp/test_metrics_trade_signal_csv.log";
    {
        MetricsLogger logger(make_csv_config(path));
        ASSERT_EQ(logger.get_log_entry_count(), uint64_t{0});
        logger.log_trade_signal(0.6, 0.2, 0.85, 3.5, 0.7);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{1})
            << "log_trade_signal must increment the entry count";
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_log_trade_signal_increments_entry_count_json) {
    const std::string path = "/tmp/test_metrics_trade_signal_json.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_trade_signal(-0.4, 0.1, 0.9, 2.1, 0.5);
        EXPECT_EQ(logger.get_log_entry_count(), uint64_t{1})
            << "log_trade_signal must increment the entry count in JSON mode";
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_log_trade_signal_json_contains_event_key) {
    const std::string path = "/tmp/test_metrics_trade_signal_json_content.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_trade_signal(0.3, 0.15, 0.75, 5.0, 0.4);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("trade_signal"), std::string::npos)
        << "JSON output must contain the trade_signal event key";
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Cycle 34: log_config_reload()
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_log_config_reload_increments_entry_count_csv) {
    const std::string path = "tmp_log_config_reload_csv.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path  = path;
        cfg.format         = MetricsLogger::OutputFormat::CSV;
        cfg.enable_console_output = false;
        MetricsLogger logger(cfg);
        uint64_t before = logger.get_log_entry_count();
        logger.log_config_reload("/etc/engine/config.yaml", true);
        EXPECT_EQ(logger.get_log_entry_count(), before + 1)
            << "log_config_reload must increment entry count (CSV)";
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_log_config_reload_increments_entry_count_json) {
    const std::string path = "tmp_log_config_reload_json.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path  = path;
        cfg.format         = MetricsLogger::OutputFormat::JSON;
        cfg.enable_console_output = false;
        MetricsLogger logger(cfg);
        uint64_t before = logger.get_log_entry_count();
        logger.log_config_reload("/etc/engine/config.yaml", false);
        EXPECT_EQ(logger.get_log_entry_count(), before + 1)
            << "log_config_reload must increment entry count (JSON)";
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_log_config_reload_json_contains_event_key) {
    const std::string path = "tmp_log_config_reload_event.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path  = path;
        cfg.format         = MetricsLogger::OutputFormat::JSON;
        cfg.enable_console_output = false;
        MetricsLogger logger(cfg);
        logger.log_config_reload("/cfg/config.yaml", true);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("config_reload"), std::string::npos)
        << "JSON output must contain the config_reload event key";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_get_uptime_ms_non_negative) {
    MetricsLogger::Config cfg;
    cfg.log_file_path = "tmp_uptime_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    double uptime = logger.get_uptime_ms();
    EXPECT_GE(uptime, 0.0) << "get_uptime_ms must be >= 0";
    std::remove("tmp_uptime_test.log");
}

TEST(MetricsLoggerTest, test_get_log_rate_zero_initially) {
    MetricsLogger::Config cfg;
    cfg.log_file_path = "tmp_log_rate_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    // No entries logged — rate should be 0 (no entries) or near-0.
    double rate = logger.get_log_rate();
    EXPECT_GE(rate, 0.0) << "get_log_rate must be >= 0";
    std::remove("tmp_log_rate_test.log");
}

TEST(MetricsLoggerTest, test_get_log_rate_positive_after_entries) {
    MetricsLogger::Config cfg;
    cfg.log_file_path = "tmp_log_rate_entries_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    for (int i = 0; i < 5; ++i)
        logger.log_latency_measurement(10);
    double rate = logger.get_log_rate();
    EXPECT_GE(rate, 0.0) << "get_log_rate must be >= 0 after entries";
    std::remove("tmp_log_rate_entries_test.log");
}

TEST(MetricsLoggerTest, test_log_pipeline_health_increments_entry_count) {
    MetricsLogger::Config cfg;
    cfg.log_file_path = "tmp_pipeline_health_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    uint64_t before = logger.get_log_entry_count();
    logger.log_pipeline_health(true, 0.02, 1.0);
    EXPECT_EQ(logger.get_log_entry_count(), before + 1);
    std::remove("tmp_pipeline_health_test.log");
}

TEST(MetricsLoggerTest, test_log_dedup_event_increments_entry_count) {
    MetricsLogger::Config cfg;
    cfg.log_file_path = "tmp_dedup_event_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    uint64_t before = logger.get_log_entry_count();
    logger.log_dedup_event("token_key_abc", false);
    logger.log_dedup_event("token_key_abc", true);
    EXPECT_EQ(logger.get_log_entry_count(), before + 2);
    std::remove("tmp_dedup_event_test.log");
}

TEST(MetricsLoggerTest, test_log_pipeline_health_json_contains_event_key) {
    MetricsLogger::Config cfg;
    cfg.format = MetricsLogger::OutputFormat::JSON;
    cfg.log_file_path = "tmp_pipeline_health_json_test.log";
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    logger.log_pipeline_health(false, 0.15, 2.5);
    logger.flush();
    std::ifstream file("tmp_pipeline_health_json_test.log");
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("pipeline_health"), std::string::npos);
    std::remove("tmp_pipeline_health_json_test.log");
}

// ---------------------------------------------------------------------------
// Regression: log_risk_rejection CSV column alignment
// Previously confidence was placed in column 7 (latency_us column), which was
// wrong. It should only appear in JSON; the CSV row must have exactly 9 columns
// and column 7 must be empty (not filled with the confidence value).
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_log_risk_rejection_csv_has_nine_columns) {
    const std::string path = "tmp_risk_rejection_csv_cols.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path         = path;
        cfg.format                = MetricsLogger::OutputFormat::CSV;
        cfg.enable_console_output = false;
        MetricsLogger logger(cfg);
        logger.log_risk_rejection("magnitude_exceeded", 1.5, 0.9);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string line;
    // Skip CSV header line.
    ASSERT_TRUE(std::getline(f, line));
    // Read the first data row.
    ASSERT_TRUE(std::getline(f, line));
    // Count commas: a 9-column row has exactly 8 commas.
    // Note: std::getline stops before the trailing empty field after the last comma,
    // so we count commas directly rather than counting fields.
    int commas = 0;
    for (char c : line) if (c == ',') ++commas;
    EXPECT_EQ(commas, 8)
        << "RISK_REJECTION CSV row must have exactly 9 columns (8 commas); got "
        << commas << " commas in: " << line;
    // Column 7 (index 6, latency_us) must be empty — not filled with confidence.
    // Split by comma and check field index 6.
    std::vector<std::string> fields;
    {
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
    }
    // std::getline yields N fields for N commas (trailing comma → trailing empty dropped).
    // We need at least 7 fields to check index 6.
    ASSERT_GE(fields.size(), 7u);
    EXPECT_TRUE(fields[6].empty())
        << "CSV column 7 (latency_us) must be empty for RISK_REJECTION; got: '"
        << fields[6] << "'";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_log_risk_rejection_csv_reason_in_token_column) {
    const std::string path = "tmp_risk_rejection_csv_reason.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path         = path;
        cfg.format                = MetricsLogger::OutputFormat::CSV;
        cfg.enable_console_output = false;
        MetricsLogger logger(cfg);
        logger.log_risk_rejection("confidence_below_minimum", 0.5, 0.05);
        logger.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string line;
    ASSERT_TRUE(std::getline(f, line)); // header
    ASSERT_TRUE(std::getline(f, line)); // data row
    // The rejection reason must appear in column 3 (token column, 0-indexed col 2).
    std::vector<std::string> fields;
    {
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
    }
    ASSERT_GE(fields.size(), 3u);
    EXPECT_EQ(fields[2], "confidence_below_minimum")
        << "Rejection reason must be in the token column (index 2)";
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Cycle 35: JSON string escaping — token/key/reason/path must be escaped
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_json_token_with_double_quote_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_token_escape.log";
    {
        MetricsLogger logger(make_json_config(path));
        // Token with embedded double-quote — without escaping this would break JSON.
        logger.log_token_received("say \"hi\"", 1);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // The escaped form must appear; a raw unescaped '"' between "token":"..."
    // would make the JSON unparseable.
    EXPECT_NE(content.find("say \\\"hi\\\""), std::string::npos)
        << "double-quote in token must be escaped as \\\" in JSON output";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_json_token_with_backslash_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_token_bs.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_token_received("C:\\foo", 1);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("C:\\\\foo"), std::string::npos)
        << "backslash in token must be doubled in JSON output";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_json_dedup_key_with_quote_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_dedup_escape.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_dedup_event("key\"val", false);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("key\\\"val"), std::string::npos)
        << "double-quote in dedup key must be escaped in JSON output";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_json_config_reload_path_with_backslash_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_reload_escape.log";
    {
        MetricsLogger logger(make_json_config(path));
        logger.log_config_reload("C:\\config\\llmquant.yaml", true);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("C:\\\\config\\\\llmquant.yaml"), std::string::npos)
        << "backslash in config reload path must be doubled in JSON output";
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// CSV field quoting — token with comma must be wrapped in double-quotes
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_csv_token_with_comma_is_quoted) {
    const std::string path = "/tmp/test_metrics_csv_comma.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_token_received("bull,ish", 1);
        logger.flush();
    }
    auto lines = read_log_lines(path);
    ASSERT_FALSE(lines.empty());
    // The token field (column index 2) must be quoted: "bull,ish"
    EXPECT_NE(lines.back().find("\"bull,ish\""), std::string::npos)
        << "token with comma must be RFC-4180 quoted in CSV; line: " << lines.back();
    // Verify TOKEN_RECEIVED appears before the quoted token (not split by the interior comma).
    auto tr_pos   = lines.back().find("TOKEN_RECEIVED");
    auto quot_pos = lines.back().find("\"bull,ish\"");
    EXPECT_NE(tr_pos,   std::string::npos);
    EXPECT_NE(quot_pos, std::string::npos);
    EXPECT_LT(tr_pos,   quot_pos) << "TOKEN_RECEIVED must precede the quoted token field";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_csv_token_with_double_quote_is_escaped) {
    const std::string path = "/tmp/test_metrics_csv_dquote.log";
    {
        MetricsLogger logger(make_csv_config(path));
        logger.log_token_received("say \"hi\"", 2);
        logger.flush();
    }
    auto lines = read_log_lines(path);
    ASSERT_FALSE(lines.empty());
    // RFC 4180: embedded " must be doubled inside a quoted field → ""hi""
    EXPECT_NE(lines.back().find("\"say \"\"hi\"\"\""), std::string::npos)
        << "embedded double-quotes must be doubled in CSV; line: " << lines.back();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Cycle 32: control character escaping in JSON output
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_json_token_with_tab_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_tab.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path        = path;
        cfg.format               = MetricsLogger::OutputFormat::JSON;
        cfg.enable_token_logging = true;
        MetricsLogger logger(cfg);
        logger.log_token_received("bull\tish", 99);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // Raw '\t' must not appear in the JSON token field.
    EXPECT_EQ(content.find("bull\tish"), std::string::npos)
        << "raw tab in token must be escaped in JSON output";
    // Escaped form '\t' must appear.
    EXPECT_NE(content.find("bull\\tish"), std::string::npos)
        << "tab must be escaped as \\t in JSON output; content=" << content;
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_json_token_with_newline_is_escaped) {
    const std::string path = "/tmp/test_metrics_json_newline_token.log";
    {
        MetricsLogger::Config cfg;
        cfg.log_file_path        = path;
        cfg.format               = MetricsLogger::OutputFormat::JSON;
        cfg.enable_token_logging = true;
        MetricsLogger logger(cfg);
        logger.log_token_received("line1\nline2", 7);
        logger.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(content.find("line1\nline2"), std::string::npos)
        << "raw newline in token must be escaped in JSON output";
    EXPECT_NE(content.find("line1\\nline2"), std::string::npos)
        << "newline must be escaped as \\n; content=" << content;
    std::remove(path.c_str());
}

// ----------------------------------------------------------------
// enable_token_logging = false must suppress log_token_received()
// ----------------------------------------------------------------
TEST(MetricsLoggerTest, test_enable_token_logging_false_suppresses_entries) {
    const std::string path = "/tmp/test_metrics_token_logging_disabled.log";
    {
        MetricsLogger::Config cfg = make_csv_config(path);
        cfg.enable_token_logging = false;
        MetricsLogger logger(cfg);
        logger.log_token_received("bullish", 1);
        logger.log_token_received("bearish", 2);
        logger.flush();
    }
    auto lines = read_log_lines(path);
    // Only the CSV header line should be present — no TOKEN_RECEIVED rows.
    for (const auto& line : lines) {
        EXPECT_EQ(line.find("TOKEN_RECEIVED"), std::string::npos)
            << "TOKEN_RECEIVED must not appear when enable_token_logging=false; line=" << line;
    }
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_enable_token_logging_true_writes_entries) {
    const std::string path = "/tmp/test_metrics_token_logging_enabled.log";
    {
        MetricsLogger::Config cfg = make_csv_config(path);
        cfg.enable_token_logging = true;
        MetricsLogger logger(cfg);
        logger.log_token_received("bullish", 1);
        logger.flush();
    }
    auto lines = read_log_lines(path);
    bool found = false;
    for (const auto& line : lines) {
        if (line.find("TOKEN_RECEIVED") != std::string::npos) { found = true; break; }
    }
    EXPECT_TRUE(found) << "TOKEN_RECEIVED must appear when enable_token_logging=true";
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Cycle 45: enable_token_logging=false does not increment log_entry_count
// ---------------------------------------------------------------------------

TEST(MetricsLoggerTest, test_enable_token_logging_false_does_not_increment_entry_count) {
    const std::string path = "/tmp/test_metrics_token_logging_counter.log";
    MetricsLogger::Config cfg = make_csv_config(path);
    cfg.enable_token_logging = false;
    MetricsLogger logger(cfg);

    logger.log_token_received("bullish", 1);
    logger.log_token_received("bearish", 2);

    EXPECT_EQ(logger.get_log_entry_count(), 0u)
        << "log_entry_count must stay 0 when enable_token_logging=false";
    std::remove(path.c_str());
}

TEST(MetricsLoggerTest, test_enable_token_logging_true_increments_entry_count) {
    const std::string path = "/tmp/test_metrics_token_logging_counter2.log";
    MetricsLogger::Config cfg = make_csv_config(path);
    cfg.enable_token_logging = true;
    MetricsLogger logger(cfg);

    logger.log_token_received("bullish", 1);
    logger.log_token_received("bearish", 2);

    EXPECT_EQ(logger.get_log_entry_count(), 2u)
        << "log_entry_count must increment once per log_token_received call";
    std::remove(path.c_str());
}
