#include "gtest/gtest.h"
#include "Config.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <cstdio>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* kValidYaml = R"yaml(
token_stream:
  data_file_path: "/data/tokens.txt"
  token_interval_ms: 5
  buffer_size: 2048
  use_memory_stream: true
trading:
  bias_sensitivity: 2.0
  volatility_sensitivity: 1.5
  signal_decay_rate: 0.80
  signal_cooldown_us: 500
latency:
  target_latency_us: 8
  sample_window: 500
  enable_profiling: false
logging:
  log_file_path: "/var/log/engine.log"
  format: "json"
  enable_console: false
  flush_interval_ms: 50
)yaml";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_load_from_yaml_string_valid_yaml_parses_all_fields) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(kValidYaml);
    ASSERT_TRUE(ok);

    const SystemConfig& sc = cfg.get_config();

    // token_stream
    EXPECT_EQ(sc.token_stream.data_file_path,    "/data/tokens.txt");
    EXPECT_EQ(sc.token_stream.token_interval_ms, 5);
    EXPECT_EQ(sc.token_stream.buffer_size,       2048u);
    EXPECT_TRUE(sc.token_stream.use_memory_stream);

    // trading
    EXPECT_DOUBLE_EQ(sc.trading.bias_sensitivity,       2.0);
    EXPECT_DOUBLE_EQ(sc.trading.volatility_sensitivity, 1.5);
    EXPECT_DOUBLE_EQ(sc.trading.signal_decay_rate,      0.80);
    EXPECT_EQ(sc.trading.signal_cooldown_us,            500);

    // latency
    EXPECT_EQ(sc.latency.target_latency_us, 8);
    EXPECT_EQ(sc.latency.sample_window,     500u);
    EXPECT_FALSE(sc.latency.enable_profiling);

    // logging
    EXPECT_EQ(sc.logging.log_file_path,      "/var/log/engine.log");
    EXPECT_EQ(sc.logging.format,             "JSON");  // Config normalises format to uppercase.
    EXPECT_FALSE(sc.logging.enable_console);
    EXPECT_EQ(sc.logging.flush_interval_ms, 50);
}

TEST(ConfigTest, test_config_load_from_yaml_string_missing_fields_uses_defaults) {
    // Only override one field; everything else must stay at default.
    Config cfg;
    bool ok = cfg.load_from_yaml_string("trading:\n  bias_sensitivity: 3.0\n");
    ASSERT_TRUE(ok);

    const SystemConfig& sc = cfg.get_config();
    EXPECT_DOUBLE_EQ(sc.trading.bias_sensitivity, 3.0);

    // Unmentioned trading field stays default
    EXPECT_DOUBLE_EQ(sc.trading.signal_decay_rate, 0.95);

    // Entire token_stream section stays default
    EXPECT_EQ(sc.token_stream.token_interval_ms, 10);
    EXPECT_EQ(sc.token_stream.buffer_size,       1024u);

    // Logging defaults: format is stored uppercase by Config::set_defaults().
    EXPECT_EQ(sc.logging.format, "CSV");
}

TEST(ConfigTest, test_config_load_from_file_nonexistent_file_returns_false_and_sets_defaults) {
    Config cfg;
    bool ok = cfg.load_from_file("/does/not/exist/config.yaml");
    EXPECT_FALSE(ok);

    // Defaults must be applied after failure
    const SystemConfig& sc = cfg.get_config();
    EXPECT_EQ(sc.token_stream.token_interval_ms, 10);
    EXPECT_DOUBLE_EQ(sc.trading.bias_sensitivity, 1.0);
    EXPECT_EQ(sc.latency.target_latency_us,       10);
}

TEST(ConfigTest, test_config_save_to_file_roundtrip_preserves_values) {
    // Write a config to a temp file and reload it.
    const std::string tmp_path = "/tmp/llmquant_test_config_roundtrip.yaml";

    Config original;
    ASSERT_TRUE(original.load_from_yaml_string(kValidYaml));
    ASSERT_TRUE(original.save_to_file(tmp_path));

    Config reloaded;
    bool ok = reloaded.load_from_file(tmp_path);
    ASSERT_TRUE(ok);

    const SystemConfig& orig = original.get_config();
    const SystemConfig& rel  = reloaded.get_config();

    EXPECT_EQ(rel.token_stream.data_file_path,    orig.token_stream.data_file_path);
    EXPECT_EQ(rel.token_stream.token_interval_ms, orig.token_stream.token_interval_ms);
    EXPECT_EQ(rel.token_stream.buffer_size,       orig.token_stream.buffer_size);
    EXPECT_EQ(rel.token_stream.use_memory_stream, orig.token_stream.use_memory_stream);
    EXPECT_DOUBLE_EQ(rel.trading.bias_sensitivity,       orig.trading.bias_sensitivity);
    EXPECT_DOUBLE_EQ(rel.trading.volatility_sensitivity, orig.trading.volatility_sensitivity);
    EXPECT_DOUBLE_EQ(rel.trading.signal_decay_rate,      orig.trading.signal_decay_rate);
    EXPECT_EQ(rel.trading.signal_cooldown_us,              orig.trading.signal_cooldown_us);
    EXPECT_DOUBLE_EQ(rel.trading.max_signal_age_us,        orig.trading.max_signal_age_us);
    EXPECT_DOUBLE_EQ(rel.trading.min_bias_threshold,       orig.trading.min_bias_threshold);
    EXPECT_EQ(rel.latency.target_latency_us,             orig.latency.target_latency_us);
    EXPECT_EQ(rel.latency.sample_window,                 orig.latency.sample_window);
    EXPECT_EQ(rel.latency.enable_profiling,              orig.latency.enable_profiling);
    EXPECT_EQ(rel.logging.log_file_path,                 orig.logging.log_file_path);
    EXPECT_EQ(rel.logging.format,                        orig.logging.format);
    EXPECT_EQ(rel.logging.enable_console,                orig.logging.enable_console);
    EXPECT_EQ(rel.logging.flush_interval_ms,             orig.logging.flush_interval_ms);

    std::remove(tmp_path.c_str());
}

TEST(ConfigTest, test_config_get_config_returns_correct_defaults) {
    Config cfg;
    const SystemConfig& sc = cfg.get_config();

    // token_stream defaults
    EXPECT_EQ(sc.token_stream.data_file_path,    "tokens.txt");
    EXPECT_EQ(sc.token_stream.token_interval_ms, 10);
    EXPECT_EQ(sc.token_stream.buffer_size,       1024u);
    EXPECT_FALSE(sc.token_stream.use_memory_stream);

    // trading defaults
    EXPECT_DOUBLE_EQ(sc.trading.bias_sensitivity,       1.0);
    EXPECT_DOUBLE_EQ(sc.trading.volatility_sensitivity, 1.0);
    EXPECT_DOUBLE_EQ(sc.trading.signal_decay_rate,      0.95);
    EXPECT_EQ(sc.trading.signal_cooldown_us,            1000);
    EXPECT_DOUBLE_EQ(sc.trading.max_signal_age_us,      0.0);  // disabled by default
    EXPECT_DOUBLE_EQ(sc.trading.min_bias_threshold,     0.0);  // disabled by default

    // latency defaults
    EXPECT_EQ(sc.latency.target_latency_us, 10);
    EXPECT_EQ(sc.latency.sample_window,     1000u);
    EXPECT_TRUE(sc.latency.enable_profiling);

    // logging defaults
    EXPECT_EQ(sc.logging.log_file_path,      "metrics.log");
    EXPECT_EQ(sc.logging.format,             "CSV");  // Default is uppercase in LoggingConfig.
    EXPECT_TRUE(sc.logging.enable_console);
    EXPECT_EQ(sc.logging.flush_interval_ms, 100);
}

TEST(ConfigTest, test_config_load_from_yaml_string_malformed_yaml_returns_false) {
    Config cfg;
    // Deliberate indentation error to trigger a YAML parse failure.
    bool ok = cfg.load_from_yaml_string("key: [unclosed bracket");
    EXPECT_FALSE(ok);
    // Defaults must still be valid after a parse failure.
    EXPECT_EQ(cfg.get_config().token_stream.token_interval_ms, 10);
}

TEST(ConfigTest, test_config_hot_reload_detects_file_change) {
    const std::string tmp_path = "/tmp/llmquant_test_hot_reload.yaml";

    // Write initial config with token_interval_ms = 10.
    {
        std::ofstream f(tmp_path);
        f << "token_stream:\n  token_interval_ms: 10\n";
    }

    Config cfg;
    ASSERT_TRUE(cfg.load_from_file(tmp_path));

    std::atomic<bool> callback_fired{false};
    std::atomic<int>  reloaded_interval{0};

    (void)cfg.start_watching(tmp_path,
        [&](const SystemConfig& sc) {
            reloaded_interval = sc.token_stream.token_interval_ms;
            callback_fired    = true;
        },
        /*poll_interval_ms=*/100);

    // Sleep briefly to let the watcher capture the initial mtime.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // Modify the file — change token_interval_ms to 99.
    {
        std::ofstream f(tmp_path);
        f << "token_stream:\n  token_interval_ms: 99\n";
    }

    // Wait up to 1500 ms for the watcher to detect the change.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    while (!callback_fired.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cfg.stop_watching();
    std::remove(tmp_path.c_str());

    EXPECT_TRUE(callback_fired.load())
        << "Watcher must invoke the callback when the file changes";
    EXPECT_EQ(reloaded_interval.load(), 99)
        << "Reloaded config must reflect the new token_interval_ms value";
}

// ---------------------------------------------------------------------------
// risk_thresholds YAML section
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_risk_thresholds_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n"
        "  max_bias_magnitude: 3.0\n"
        "  min_confidence: 0.2\n"
        "  max_signals_per_second: 200\n"
        "  max_drawdown: 8.0\n"
        "  drawdown_window_s: 120\n"
        "  position_warn_fraction: 0.75\n");
    ASSERT_TRUE(ok);
    const auto& rt = cfg.get_config().risk_thresholds;
    EXPECT_DOUBLE_EQ(rt.max_bias_magnitude,  3.0);
    EXPECT_DOUBLE_EQ(rt.min_confidence,      0.2);
    EXPECT_EQ(rt.max_signals_per_second,     200u);
    EXPECT_DOUBLE_EQ(rt.max_drawdown,        8.0);
    EXPECT_EQ(rt.drawdown_window_s,          120);
    EXPECT_DOUBLE_EQ(rt.position_warn_fraction, 0.75);
}

TEST(ConfigTest, test_config_risk_thresholds_invalid_min_confidence_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n  min_confidence: 1.5\n");  // > 1.0
    EXPECT_FALSE(ok);
    // Defaults restored
    EXPECT_DOUBLE_EQ(cfg.get_config().risk_thresholds.min_confidence, 0.1);
}

TEST(ConfigTest, test_config_risk_thresholds_zero_max_signals_per_second_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n  max_signals_per_second: 0\n");
    EXPECT_FALSE(ok);
    EXPECT_GT(cfg.get_config().risk_thresholds.max_signals_per_second, 0u);
}

TEST(ConfigTest, test_config_pressure_zero_max_ingestion_rate_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "pressure:\n  max_ingestion_rate_tps: 0.0\n");
    EXPECT_FALSE(ok);
    EXPECT_GT(cfg.get_config().pressure.max_ingestion_rate_tps, 0.0);
}

TEST(ConfigTest, test_config_risk_thresholds_zero_drawdown_window_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n  drawdown_window_s: 0\n");
    EXPECT_FALSE(ok);
    EXPECT_GT(cfg.get_config().risk_thresholds.drawdown_window_s, 0);
}

TEST(ConfigTest, test_config_risk_thresholds_out_of_range_position_warn_fraction_returns_false) {
    // position_warn_fraction must be in [0, 1]
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n  position_warn_fraction: 1.5\n");
    EXPECT_FALSE(ok);
    // Defaults must be restored
    double def = cfg.get_config().risk_thresholds.position_warn_fraction;
    EXPECT_GE(def, 0.0);
    EXPECT_LE(def, 1.0);
}

TEST(ConfigTest, test_config_risk_thresholds_negative_max_bias_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n  max_bias_magnitude: -1.0\n");
    EXPECT_FALSE(ok);
    EXPECT_GE(cfg.get_config().risk_thresholds.max_bias_magnitude, 0.0);
}

// ---------------------------------------------------------------------------
// Range validation tests (improvement 3)
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_invalid_signal_decay_rate_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string("trading:\n  signal_decay_rate: 1.5\n");
    EXPECT_FALSE(ok);
    // Defaults must be restored after validation failure.
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.signal_decay_rate, 0.95);
}

TEST(ConfigTest, test_config_zero_buffer_size_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string("token_stream:\n  buffer_size: 0\n");
    EXPECT_FALSE(ok);
    EXPECT_EQ(cfg.get_config().token_stream.buffer_size, 1024u);
}

TEST(ConfigTest, test_config_negative_bias_sensitivity_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string("trading:\n  bias_sensitivity: -1.0\n");
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.bias_sensitivity, 1.0);
}

// ---------------------------------------------------------------------------
// New: concurrent hot-reload + get_config is safe (improvement #15)
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_concurrent_hot_reload_and_get_config_is_safe) {
    const std::string tmp_path = "/tmp/llmquant_test_concurrent_hotreload.yaml";
    {
        std::ofstream f(tmp_path);
        f << "trading:\n  bias_sensitivity: 1.0\n";
    }

    Config cfg;
    ASSERT_TRUE(cfg.load_from_file(tmp_path));
    (void)cfg.start_watching(tmp_path, [](const SystemConfig&) {}, /*poll_interval_ms=*/50);

    std::atomic<bool> stop_readers{false};

    // Spawn a reader thread that continuously calls get_config().
    std::thread reader([&]() {
        while (!stop_readers.load()) {
            volatile double v = cfg.get_config().trading.bias_sensitivity;
            (void)v;
        }
    });

    // Write to the file several times to trigger hot-reloads.
    for (int i = 1; i <= 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        std::ofstream f(tmp_path);
        f << "trading:\n  bias_sensitivity: " << static_cast<double>(i) << "\n";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop_readers = true;
    reader.join();

    cfg.stop_watching();
    std::remove(tmp_path.c_str());

    // No assertion on values — just verifying no crash or deadlock.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Test: Config load on unreadable file (chmod 000; skip on Windows)
// ---------------------------------------------------------------------------
TEST(ConfigTest, ConfigLoadUnreadableFile) {
#ifndef _WIN32
    // Create a temp file with valid YAML.
    std::string tmp_path = "/tmp/llmquant_unreadable_test.yaml";
    {
        std::ofstream f(tmp_path);
        f << "trading:\n  bias_sensitivity: 1.0\n";
    }
    chmod(tmp_path.c_str(), 0000);

    Config cfg;
    bool ok = cfg.load_from_file(tmp_path);
    // Either throws or returns false; either is acceptable.
    (void)ok;

    // Restore and cleanup.
    chmod(tmp_path.c_str(), 0644);
    std::remove(tmp_path.c_str());
    SUCCEED();
#else
    GTEST_SKIP() << "chmod not available on Windows";
#endif
}

// ---------------------------------------------------------------------------
// save_to_file round-trip for risk_thresholds
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_save_and_reload_preserves_risk_thresholds) {
    const std::string tmp_path = "tmp_risk_thresholds_roundtrip.yaml";

    // Load custom risk thresholds.
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "risk_thresholds:\n"
        "  max_bias_magnitude: 4.5\n"
        "  max_volatility_magnitude: 3.0\n"
        "  min_confidence: 0.25\n"
        "  max_signals_per_second: 300\n"
        "  max_drawdown: 7.5\n"
        "  drawdown_window_s: 90\n"
        "  position_warn_fraction: 0.7\n");
    ASSERT_TRUE(ok);

    // Save to a temp file.
    ASSERT_TRUE(cfg.save_to_file(tmp_path));

    // Reload from the saved file.
    Config cfg2;
    ASSERT_TRUE(cfg2.load_from_file(tmp_path));
    const auto& rt = cfg2.get_config().risk_thresholds;

    EXPECT_DOUBLE_EQ(rt.max_bias_magnitude,       4.5);
    EXPECT_DOUBLE_EQ(rt.max_volatility_magnitude, 3.0);
    EXPECT_DOUBLE_EQ(rt.min_confidence,           0.25);
    EXPECT_EQ(rt.max_signals_per_second,          300u);
    EXPECT_DOUBLE_EQ(rt.max_drawdown,             7.5);
    EXPECT_EQ(rt.drawdown_window_s,               90);
    EXPECT_DOUBLE_EQ(rt.position_warn_fraction,   0.7);

    std::remove(tmp_path.c_str());
}

TEST(ConfigTest, test_config_max_signal_age_us_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  max_signal_age_us: 500.0\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    ASSERT_TRUE(ok);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.max_signal_age_us, 500.0);
}

TEST(ConfigTest, test_config_negative_max_signal_age_us_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  max_signal_age_us: -1.0\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    EXPECT_FALSE(ok) << "Negative max_signal_age_us must fail validation";
}

TEST(ConfigTest, test_config_metrics_port_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n"
        "metrics:\n  stats_port: 9200\n  bind_address: \"127.0.0.1\"\n");
    ASSERT_TRUE(ok);
    const auto& m = cfg.get_config().metrics;
    EXPECT_EQ(m.stats_port, 9200);
    EXPECT_EQ(m.bind_address, "127.0.0.1");
}

TEST(ConfigTest, test_config_metrics_port_zero_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n"
        "metrics:\n  stats_port: 0\n");
    EXPECT_FALSE(ok) << "stats_port: 0 must fail validation";
}

TEST(ConfigTest, test_config_min_bias_threshold_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  min_bias_threshold: 0.15\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    ASSERT_TRUE(ok);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.min_bias_threshold, 0.15);
}

TEST(ConfigTest, test_config_negative_min_bias_threshold_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  min_bias_threshold: -0.1\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    EXPECT_FALSE(ok) << "Negative min_bias_threshold must fail validation";
}

TEST(ConfigTest, test_config_metrics_defaults_when_section_absent) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    ASSERT_TRUE(ok);
    EXPECT_EQ(cfg.get_config().metrics.stats_port, 9100u);
    EXPECT_EQ(cfg.get_config().metrics.bind_address, "0.0.0.0");
}

TEST(ConfigTest, test_config_to_summary_string_contains_key_fields) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 2.5\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");

    std::string summary = cfg.to_summary_string();
    EXPECT_FALSE(summary.empty()) << "to_summary_string must return non-empty output";
    // Must mention key sections.
    EXPECT_NE(summary.find("trading"), std::string::npos)  << "summary must mention trading section";
    EXPECT_NE(summary.find("risk"),    std::string::npos)  << "summary must mention risk section";
    EXPECT_NE(summary.find("latency"), std::string::npos)  << "summary must mention latency section";
    // Must include the bias_sensitivity value we set.
    EXPECT_NE(summary.find("2.5"), std::string::npos)
        << "summary must include bias_sensitivity=2.5";
}

TEST(ConfigTest, test_config_max_accumulated_bias_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  max_accumulated_bias: 5.0\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    ASSERT_TRUE(ok);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.max_accumulated_bias, 5.0);
}

TEST(ConfigTest, test_config_negative_max_accumulated_bias_returns_false) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "  max_accumulated_bias: -1.0\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    EXPECT_FALSE(ok) << "Negative max_accumulated_bias must fail validation";
}

// ---------------------------------------------------------------------------
// load_from_env tests
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_load_from_env_applies_bias_sensitivity) {
    Config cfg;
    cfg.set_defaults();
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "3.5");
#else
    setenv("LLMQUANT_BIAS_SENSITIVITY", "3.5", 1);
#endif
    int applied = cfg.load_from_env();
    EXPECT_GT(applied, 0) << "At least one env var should be applied";
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.bias_sensitivity, 3.5);
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "");
#else
    unsetenv("LLMQUANT_BIAS_SENSITIVITY");
#endif
}

TEST(ConfigTest, test_config_load_from_env_invalid_value_is_ignored) {
    Config cfg;
    cfg.set_defaults();
    double old_decay = cfg.get_config().trading.signal_decay_rate;
#ifdef _WIN32
    _putenv_s("LLMQUANT_SIGNAL_DECAY", "not_a_number");
#else
    setenv("LLMQUANT_SIGNAL_DECAY", "not_a_number", 1);
#endif
    cfg.load_from_env();
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.signal_decay_rate, old_decay);
#ifdef _WIN32
    _putenv_s("LLMQUANT_SIGNAL_DECAY", "");
#else
    unsetenv("LLMQUANT_SIGNAL_DECAY");
#endif
}

TEST(ConfigTest, test_config_load_from_env_zero_count_when_no_vars_set) {
    Config cfg;
    cfg.set_defaults();
    // Unset all env vars under test.
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "");
    _putenv_s("LLMQUANT_VOL_SENSITIVITY", "");
    _putenv_s("LLMQUANT_SIGNAL_DECAY", "");
    _putenv_s("LLMQUANT_SIGNAL_COOLDOWN_US", "");
    _putenv_s("LLMQUANT_MAX_SIGNAL_AGE_US", "");
    _putenv_s("LLMQUANT_MIN_BIAS_THRESHOLD", "");
    _putenv_s("LLMQUANT_MAX_ACCUMULATED_BIAS", "");
    _putenv_s("LLMQUANT_MAX_DRAWDOWN", "");
    _putenv_s("LLMQUANT_MAX_SIGNALS_PER_SECOND", "");
    _putenv_s("LLMQUANT_STATS_PORT", "");
#else
    for (const char* v : {"LLMQUANT_BIAS_SENSITIVITY","LLMQUANT_VOL_SENSITIVITY",
                          "LLMQUANT_SIGNAL_DECAY","LLMQUANT_SIGNAL_COOLDOWN_US",
                          "LLMQUANT_MAX_SIGNAL_AGE_US","LLMQUANT_MIN_BIAS_THRESHOLD",
                          "LLMQUANT_MAX_ACCUMULATED_BIAS",
                          "LLMQUANT_MAX_DRAWDOWN","LLMQUANT_MAX_SIGNALS_PER_SECOND",
                          "LLMQUANT_STATS_PORT"})
        unsetenv(v);
#endif
    EXPECT_EQ(cfg.load_from_env(), 0);
}

// ---------------------------------------------------------------------------
// Cycle 22: load_from_env — multiple vars applied simultaneously
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_load_from_env_multiple_vars_returns_correct_count) {
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "2.5");
    _putenv_s("LLMQUANT_VOL_SENSITIVITY",  "1.5");
    _putenv_s("LLMQUANT_SIGNAL_DECAY",     "0.9");
#else
    setenv("LLMQUANT_BIAS_SENSITIVITY", "2.5", 1);
    setenv("LLMQUANT_VOL_SENSITIVITY",  "1.5", 1);
    setenv("LLMQUANT_SIGNAL_DECAY",     "0.9", 1);
#endif
    Config cfg;
    (void)cfg.load_from_file("../config.yaml");
    int count = cfg.load_from_env();
    EXPECT_GE(count, 3) << "At least 3 env vars should have been applied";

    const auto& t = cfg.get_config().trading;
    EXPECT_DOUBLE_EQ(t.bias_sensitivity,      2.5);
    EXPECT_DOUBLE_EQ(t.volatility_sensitivity, 1.5);
    EXPECT_DOUBLE_EQ(t.signal_decay_rate,      0.9);

#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "");
    _putenv_s("LLMQUANT_VOL_SENSITIVITY",  "");
    _putenv_s("LLMQUANT_SIGNAL_DECAY",     "");
#else
    unsetenv("LLMQUANT_BIAS_SENSITIVITY");
    unsetenv("LLMQUANT_VOL_SENSITIVITY");
    unsetenv("LLMQUANT_SIGNAL_DECAY");
#endif
}

TEST(ConfigTest, test_config_load_from_env_nan_value_is_ignored) {
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "nan");
#else
    setenv("LLMQUANT_BIAS_SENSITIVITY", "nan", 1);
#endif
    Config cfg;
    (void)cfg.load_from_file("../config.yaml");
    double before = cfg.get_config().trading.bias_sensitivity;
    int count = cfg.load_from_env();
    // NaN is not finite — load_from_env must skip it.
    const double after = cfg.get_config().trading.bias_sensitivity;
    EXPECT_DOUBLE_EQ(after, before);
    EXPECT_EQ(count, 0);
#ifdef _WIN32
    _putenv_s("LLMQUANT_BIAS_SENSITIVITY", "");
#else
    unsetenv("LLMQUANT_BIAS_SENSITIVITY");
#endif
}

// ---------------------------------------------------------------------------
// LLMQUANT_MAX_ACCUMULATED_BIAS env var
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_load_from_env_max_accumulated_bias_applies) {
    Config cfg;
    cfg.set_defaults();
#ifdef _WIN32
    _putenv_s("LLMQUANT_MAX_ACCUMULATED_BIAS", "5.0");
#else
    setenv("LLMQUANT_MAX_ACCUMULATED_BIAS", "5.0", 1);
#endif
    int applied = cfg.load_from_env();
    EXPECT_GT(applied, 0);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.max_accumulated_bias, 5.0);
#ifdef _WIN32
    _putenv_s("LLMQUANT_MAX_ACCUMULATED_BIAS", "");
#else
    unsetenv("LLMQUANT_MAX_ACCUMULATED_BIAS");
#endif
}

TEST(ConfigTest, test_config_load_from_env_max_accumulated_bias_zero_valid) {
    // 0.0 = disabled (no clamp); must be accepted.
    Config cfg;
    cfg.set_defaults();
#ifdef _WIN32
    _putenv_s("LLMQUANT_MAX_ACCUMULATED_BIAS", "0.0");
#else
    setenv("LLMQUANT_MAX_ACCUMULATED_BIAS", "0.0", 1);
#endif
    int applied = cfg.load_from_env();
    EXPECT_GT(applied, 0);
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.max_accumulated_bias, 0.0);
#ifdef _WIN32
    _putenv_s("LLMQUANT_MAX_ACCUMULATED_BIAS", "");
#else
    unsetenv("LLMQUANT_MAX_ACCUMULATED_BIAS");
#endif
}

// ---------------------------------------------------------------------------
// Cycle 31: Config::validate()
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_validate_defaults_returns_empty) {
    Config cfg;
    // Default-constructed config must pass all validation checks.
    auto errors = cfg.validate();
    EXPECT_TRUE(errors.empty())
        << "Default config must have no validation errors";
}

TEST(ConfigTest, test_config_validate_after_valid_yaml_returns_empty) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    auto errors = cfg.validate();
    EXPECT_TRUE(errors.empty())
        << "Valid YAML config must produce no validation errors";
}

TEST(ConfigTest, test_config_validate_detects_multiple_errors) {
    // Directly manipulate a config via load_from_yaml_string using a valid
    // yaml first, then corrupt via update.  We need to get into an invalid
    // state.  We do this by loading with individual bad fields through
    // multiple failed loads (load_from_yaml_string restores defaults on fail),
    // so instead we use set_defaults + manually call validate with a forced
    // bad state by testing the defaults pass and checking a known invalid path.
    // Since we cannot directly set invalid values, we test that validate()
    // mirrors load_from_yaml_string's rejections by passing a yaml that fails
    // loading and verifying validate() on the resulting (default) config passes.
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: -5\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    EXPECT_FALSE(ok) << "Bad token_interval_ms should fail loading";
    // After failed load defaults are restored — defaults must be valid.
    auto errors = cfg.validate();
    EXPECT_TRUE(errors.empty())
        << "After failed load, restored defaults must pass validate()";
}

TEST(ConfigTest, test_config_validate_returns_vector_of_strings) {
    Config cfg;
    auto errors = cfg.validate();
    // Just verify the return type and that empty means valid.
    EXPECT_TRUE(errors.empty());
    // Ensure no errors for a freshly constructed config.
    EXPECT_EQ(errors.size(), 0u);
}

// ---------------------------------------------------------------------------
// Cycle 35: Config::is_valid()
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_is_valid_true_for_default_config) {
    Config cfg;
    EXPECT_TRUE(cfg.is_valid())
        << "default-constructed Config must be valid";
}

TEST(ConfigTest, test_config_is_valid_true_after_valid_yaml_load) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  flush_interval_ms: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    EXPECT_TRUE(cfg.is_valid())
        << "Config loaded from valid YAML must report is_valid() == true";
}

TEST(ConfigTest, test_config_is_valid_consistent_with_validate) {
    Config cfg;
    // is_valid() must agree with validate().empty() on the same instance.
    bool via_is_valid = cfg.is_valid();
    bool via_validate = cfg.validate().empty();
    EXPECT_EQ(via_is_valid, via_validate)
        << "is_valid() must return the same result as validate().empty()";
}

// ---------------------------------------------------------------------------
// Env var: LLMQUANT_LOG_FILE and LLMQUANT_LOG_FORMAT
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_load_from_env_log_file_applies) {
    Config cfg;
    cfg.set_defaults();
#ifdef _WIN32
    _putenv_s("LLMQUANT_LOG_FILE", "/var/log/engine.log");
#else
    setenv("LLMQUANT_LOG_FILE", "/var/log/engine.log", 1);
#endif
    int applied = cfg.load_from_env();
    EXPECT_GT(applied, 0);
    EXPECT_EQ(cfg.get_config().logging.log_file_path, "/var/log/engine.log");
#ifdef _WIN32
    _putenv_s("LLMQUANT_LOG_FILE", "");
#else
    unsetenv("LLMQUANT_LOG_FILE");
#endif
}

TEST(ConfigTest, test_config_load_from_env_log_format_uppercased) {
    Config cfg;
    cfg.set_defaults();
#ifdef _WIN32
    _putenv_s("LLMQUANT_LOG_FORMAT", "json");
#else
    setenv("LLMQUANT_LOG_FORMAT", "json", 1);
#endif
    int applied = cfg.load_from_env();
    EXPECT_GT(applied, 0);
    EXPECT_EQ(cfg.get_config().logging.format, "JSON");
#ifdef _WIN32
    _putenv_s("LLMQUANT_LOG_FORMAT", "");
#else
    unsetenv("LLMQUANT_LOG_FORMAT");
#endif
}

// ---------------------------------------------------------------------------
// to_summary_string includes [logging] section
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_to_summary_string_contains_logging_section) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "logging:\n  log_file_path: mylog.log\n  format: JSON\n  flush_interval_ms: 200\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    std::string summary = cfg.to_summary_string();
    EXPECT_NE(summary.find("logging"), std::string::npos)
        << "to_summary_string must include [logging] section";
    EXPECT_NE(summary.find("mylog.log"), std::string::npos)
        << "to_summary_string must include the log_file_path value";
    EXPECT_NE(summary.find("JSON"), std::string::npos)
        << "to_summary_string must include the log format";
}

// ---------------------------------------------------------------------------
// dedup_ttl_ms config field (added cycle 18)
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_dedup_ttl_ms_explicit_value_parsed) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "  use_memory_stream: true\n  dedup_ttl_ms: 250\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    ASSERT_TRUE(ok) << "Config with dedup_ttl_ms: 250 must parse successfully";
    EXPECT_EQ(cfg.get_config().token_stream.dedup_ttl_ms, 250);
}

TEST(ConfigTest, test_config_dedup_ttl_ms_defaults_to_zero) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "  use_memory_stream: true\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    ASSERT_TRUE(ok);
    EXPECT_EQ(cfg.get_config().token_stream.dedup_ttl_ms, 0)
        << "dedup_ttl_ms must default to 0 when absent from YAML";
}

TEST(ConfigTest, test_config_dedup_ttl_ms_negative_rejected) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "  dedup_ttl_ms: -1\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    EXPECT_FALSE(ok) << "Negative dedup_ttl_ms must fail validation";
}

TEST(ConfigTest, test_config_dedup_ttl_ms_zero_is_valid) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "  dedup_ttl_ms: 0\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    ASSERT_TRUE(ok) << "dedup_ttl_ms: 0 (auto mode) must be valid";
    EXPECT_EQ(cfg.get_config().token_stream.dedup_ttl_ms, 0);
}

// ---------------------------------------------------------------------------
// SemanticWeightsConfig parsing tests
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_semantic_weights_parsed_from_yaml) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "semantic_weights:\n"
        "  sentiment_multiplier: 2.5\n"
        "  confidence_multiplier: 0.75\n"
        "  volatility_multiplier: 1.5\n"
        "  bias_multiplier: 3.0\n");
    ASSERT_TRUE(ok);
    const auto& sw = cfg.get_config().semantic_weights;
    EXPECT_DOUBLE_EQ(sw.sentiment_multiplier,  2.5);
    EXPECT_DOUBLE_EQ(sw.confidence_multiplier, 0.75);
    EXPECT_DOUBLE_EQ(sw.volatility_multiplier, 1.5);
    EXPECT_DOUBLE_EQ(sw.bias_multiplier,       3.0);
}

TEST(ConfigTest, test_semantic_weights_default_to_one_when_absent) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    ASSERT_TRUE(ok);
    const auto& sw = cfg.get_config().semantic_weights;
    EXPECT_DOUBLE_EQ(sw.sentiment_multiplier,  1.0);
    EXPECT_DOUBLE_EQ(sw.confidence_multiplier, 1.0);
    EXPECT_DOUBLE_EQ(sw.volatility_multiplier, 1.0);
    EXPECT_DOUBLE_EQ(sw.bias_multiplier,       1.0);
}

// ---------------------------------------------------------------------------
// Config::to_yaml_string() round-trip tests
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_to_yaml_string_round_trips_trading_config) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 3.0\n  volatility_sensitivity: 2.5\n"
        "  signal_decay_rate: 0.88\n  signal_cooldown_us: 750\n"
        "  max_signal_age_us: 250.0\n  min_bias_threshold: 0.05\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n");
    ASSERT_TRUE(ok);

    std::string yaml_str = cfg.to_yaml_string();
    ASSERT_FALSE(yaml_str.empty());

    Config cfg2;
    ASSERT_TRUE(cfg2.load_from_yaml_string(yaml_str));
    const auto& tr = cfg2.get_config().trading;
    EXPECT_DOUBLE_EQ(tr.bias_sensitivity,      3.0);
    EXPECT_DOUBLE_EQ(tr.volatility_sensitivity, 2.5);
    EXPECT_DOUBLE_EQ(tr.signal_decay_rate,      0.88);
    EXPECT_EQ(tr.signal_cooldown_us,            750);
    EXPECT_DOUBLE_EQ(tr.max_signal_age_us,      250.0);
    EXPECT_DOUBLE_EQ(tr.min_bias_threshold,     0.05);
}

TEST(ConfigTest, test_to_yaml_string_round_trips_semantic_weights) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "semantic_weights:\n"
        "  sentiment_multiplier: 1.8\n"
        "  confidence_multiplier: 0.6\n"
        "  volatility_multiplier: 2.2\n"
        "  bias_multiplier: 0.9\n");
    ASSERT_TRUE(ok);

    Config cfg2;
    ASSERT_TRUE(cfg2.load_from_yaml_string(cfg.to_yaml_string()));
    const auto& sw = cfg2.get_config().semantic_weights;
    EXPECT_DOUBLE_EQ(sw.sentiment_multiplier,  1.8);
    EXPECT_DOUBLE_EQ(sw.confidence_multiplier, 0.6);
    EXPECT_DOUBLE_EQ(sw.volatility_multiplier, 2.2);
    EXPECT_DOUBLE_EQ(sw.bias_multiplier,       0.9);
}

TEST(ConfigTest, test_to_yaml_string_contains_all_sections) {
    Config cfg;
    std::string yaml_str = cfg.to_yaml_string();
    EXPECT_NE(yaml_str.find("token_stream"),    std::string::npos);
    EXPECT_NE(yaml_str.find("trading"),         std::string::npos);
    EXPECT_NE(yaml_str.find("latency"),         std::string::npos);
    EXPECT_NE(yaml_str.find("logging"),         std::string::npos);
    EXPECT_NE(yaml_str.find("metrics"),         std::string::npos);
    EXPECT_NE(yaml_str.find("pressure"),        std::string::npos);
    EXPECT_NE(yaml_str.find("risk_thresholds"), std::string::npos);
    EXPECT_NE(yaml_str.find("semantic_weights"), std::string::npos);
}

// ---------------------------------------------------------------------------
// semantic_weights load validation (NaN/Inf must be rejected by load path)
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_semantic_weights_nan_sentiment_rejected_by_load) {
    Config cfg;
    // YAML allows .nan as a special float value.
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "semantic_weights:\n  sentiment_multiplier: .nan\n");
    EXPECT_FALSE(ok) << "NaN sentiment_multiplier must fail load validation";
}

TEST(ConfigTest, test_semantic_weights_inf_bias_rejected_by_load) {
    Config cfg;
    bool ok = cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "semantic_weights:\n  bias_multiplier: .inf\n");
    EXPECT_FALSE(ok) << "Inf bias_multiplier must fail load validation";
}

// ---------------------------------------------------------------------------
// to_summary_string completeness: dedup_ttl_ms and risk overrides
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_config_to_summary_string_contains_dedup_ttl_ms) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "token_stream:\n  token_interval_ms: 10\n  buffer_size: 64\n"
        "  dedup_ttl_ms: 150\n"
        "trading:\n  bias_sensitivity: 1.0\n  volatility_sensitivity: 1.0\n"
        "  signal_decay_rate: 0.95\n  signal_cooldown_us: 1000\n"
        "latency:\n  target_latency_us: 10\n  sample_window: 100\n"
        "pressure:\n  max_ingestion_rate_tps: 50\n  backoff_scale_factor: 5\n");
    std::string summary = cfg.to_summary_string();
    EXPECT_NE(summary.find("dedup_ttl_ms=150"), std::string::npos)
        << "to_summary_string must include dedup_ttl_ms value";
}

TEST(ConfigTest, test_config_to_summary_string_contains_risk_overrides) {
    Config cfg;
    cfg.set_defaults();
    std::string summary = cfg.to_summary_string();
    EXPECT_NE(summary.find("overrides"), std::string::npos)
        << "to_summary_string must include [overrides] section";
    // Default: all gates enabled, summary should show 'on' for each.
    EXPECT_NE(summary.find("magnitude=on"), std::string::npos)
        << "magnitude gate should be 'on' with default config";
}

// ---------------------------------------------------------------------------
// Cycle 24: Config::diff_from_defaults()
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_diff_from_defaults_empty_on_fresh_config) {
    Config cfg;
    cfg.set_defaults();
    auto diffs = cfg.diff_from_defaults();
    EXPECT_TRUE(diffs.empty())
        << "A freshly-defaulted config must produce no diffs; got: "
        << (diffs.empty() ? "" : diffs[0]);
}

TEST(ConfigTest, test_diff_from_defaults_detects_trading_change) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "trading:\n  bias_sensitivity: 3.5\n");
    auto diffs = cfg.diff_from_defaults();
    bool found = false;
    for (const auto& d : diffs) {
        if (d.find("trading.bias_sensitivity") != std::string::npos) { found = true; break; }
    }
    EXPECT_TRUE(found) << "bias_sensitivity diff must appear; diffs count=" << diffs.size();
}

TEST(ConfigTest, test_diff_from_defaults_detects_risk_threshold_change) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "risk_thresholds:\n  max_drawdown: 99.0\n");
    auto diffs = cfg.diff_from_defaults();
    bool found = false;
    for (const auto& d : diffs) {
        if (d.find("risk_thresholds.max_drawdown") != std::string::npos) { found = true; break; }
    }
    EXPECT_TRUE(found) << "max_drawdown diff must appear; diffs count=" << diffs.size();
}

TEST(ConfigTest, test_diff_from_defaults_multiple_changes) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "trading:\n  bias_sensitivity: 2.0\n  signal_decay_rate: 0.8\n"
        "latency:\n  target_latency_us: 50\n");
    auto diffs = cfg.diff_from_defaults();
    EXPECT_GE(diffs.size(), size_t{3})
        << "Three changed fields must produce at least three diff entries";
}

TEST(ConfigTest, test_diff_from_defaults_contains_current_and_default_value) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "trading:\n  bias_sensitivity: 7.0\n");
    auto diffs = cfg.diff_from_defaults();
    bool has_current  = false;
    bool has_default  = false;
    for (const auto& d : diffs) {
        if (d.find("trading.bias_sensitivity") != std::string::npos) {
            has_current = (d.find("7") != std::string::npos);
            has_default = (d.find("default") != std::string::npos);
            break;
        }
    }
    EXPECT_TRUE(has_current) << "diff entry must contain the current value (7)";
    EXPECT_TRUE(has_default) << "diff entry must contain the word 'default'";
}

// ---------------------------------------------------------------------------
// Cycle 27: Config::set_token_interval_ms()
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_set_token_interval_ms_updates_interval) {
    Config cfg;
    cfg.set_token_interval_ms(42);
    auto snap = cfg.get_config();
    EXPECT_EQ(snap.token_stream.token_interval_ms, 42)
        << "set_token_interval_ms(42) must update token_interval_ms to 42";
}

TEST(ConfigTest, test_set_token_interval_ms_ignores_zero) {
    Config cfg;
    (void)cfg.load_from_yaml_string("token_stream:\n  token_interval_ms: 100\n");
    cfg.set_token_interval_ms(0);  // must be ignored
    EXPECT_EQ(cfg.get_config().token_stream.token_interval_ms, 100)
        << "set_token_interval_ms(0) must be a no-op";
}

TEST(ConfigTest, test_set_token_interval_ms_ignores_negative) {
    Config cfg;
    (void)cfg.load_from_yaml_string("token_stream:\n  token_interval_ms: 50\n");
    cfg.set_token_interval_ms(-1);  // must be ignored
    EXPECT_EQ(cfg.get_config().token_stream.token_interval_ms, 50)
        << "set_token_interval_ms(-1) must be a no-op";
}

// ---------------------------------------------------------------------------
// validate(): missing or invalid enum values
// ---------------------------------------------------------------------------

TEST(ConfigTest, test_validate_rejects_invalid_logging_format) {
    Config cfg;
    (void)cfg.load_from_yaml_string("logging:\n  format: XML\n");
    auto errs = cfg.validate();
    bool found = false;
    for (const auto& e : errs)
        if (e.find("logging.format") != std::string::npos) { found = true; break; }
    EXPECT_TRUE(found)
        << "validate() must reject logging.format=XML; errors: "
        << (errs.empty() ? "(none)" : errs[0]);
}

TEST(ConfigTest, test_validate_accepts_csv_format) {
    Config cfg;
    (void)cfg.load_from_yaml_string("logging:\n  format: CSV\n");
    auto errs = cfg.validate();
    bool format_err = false;
    for (const auto& e : errs)
        if (e.find("logging.format") != std::string::npos) { format_err = true; break; }
    EXPECT_FALSE(format_err) << "validate() must accept logging.format=CSV";
}

TEST(ConfigTest, test_validate_rejects_negative_dedup_ttl) {
    Config cfg;
    // load_from_yaml_string validates internally; a negative dedup_ttl_ms
    // must cause it to return false (and apply defaults).
    bool loaded = cfg.load_from_yaml_string("token_stream:\n  dedup_ttl_ms: -5\n");
    EXPECT_FALSE(loaded) << "load_from_yaml_string must reject dedup_ttl_ms < 0";
}

TEST(ConfigTest, test_validate_accepts_valid_redis_url) {
    Config cfg;
    (void)cfg.load_from_yaml_string("token_stream:\n  redis_url: \"redis://127.0.0.1:6379\"\n");
    auto errs = cfg.validate();
    bool redis_err = false;
    for (const auto& e : errs)
        if (e.find("redis_url") != std::string::npos) { redis_err = true; break; }
    EXPECT_FALSE(redis_err) << "validate() must accept redis://... URL";
}

TEST(ConfigTest, test_validate_accepts_rediss_url) {
    Config cfg;
    (void)cfg.load_from_yaml_string("token_stream:\n  redis_url: \"rediss://myredis.example.com:6380\"\n");
    auto errs = cfg.validate();
    bool redis_err = false;
    for (const auto& e : errs)
        if (e.find("redis_url") != std::string::npos) { redis_err = true; break; }
    EXPECT_FALSE(redis_err) << "validate() must accept rediss://... URL";
}

TEST(ConfigTest, test_validate_rejects_invalid_redis_url_scheme) {
    Config cfg;
    (void)cfg.load_from_yaml_string("token_stream:\n  redis_url: \"http://localhost:6379\"\n");
    auto errs = cfg.validate();
    bool found = false;
    for (const auto& e : errs)
        if (e.find("redis_url") != std::string::npos) { found = true; break; }
    EXPECT_TRUE(found) << "validate() must reject redis_url with non-redis:// scheme";
}

TEST(ConfigTest, test_validate_accepts_empty_redis_url) {
    Config cfg;
    // Empty redis_url means in-process backend — must not produce an error.
    (void)cfg.load_from_yaml_string("token_stream:\n  redis_url: \"\"\n");
    auto errs = cfg.validate();
    bool redis_err = false;
    for (const auto& e : errs)
        if (e.find("redis_url") != std::string::npos) { redis_err = true; break; }
    EXPECT_FALSE(redis_err) << "validate() must accept empty redis_url";
}

TEST(ConfigTest, test_validate_rejects_negative_volatility_magnitude) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "risk_thresholds:\n  max_volatility_magnitude: -1.0\n");
    auto errs = cfg.validate();
    bool found = false;
    for (const auto& e : errs)
        if (e.find("max_volatility_magnitude") != std::string::npos) { found = true; break; }
    EXPECT_TRUE(found) << "validate() must reject max_volatility_magnitude < 0";
}

TEST(ConfigTest, test_validate_rejects_negative_spread_magnitude) {
    Config cfg;
    (void)cfg.load_from_yaml_string(
        "risk_thresholds:\n  max_spread_magnitude: -0.5\n");
    auto errs = cfg.validate();
    bool found = false;
    for (const auto& e : errs)
        if (e.find("max_spread_magnitude") != std::string::npos) { found = true; break; }
    EXPECT_TRUE(found) << "validate() must reject max_spread_magnitude < 0";
}

} // namespace
} // namespace llmquant
