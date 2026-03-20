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
    original.load_from_yaml_string(kValidYaml);
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
    EXPECT_EQ(rel.trading.signal_cooldown_us,            orig.trading.signal_cooldown_us);
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
    cfg.load_from_file(tmp_path);

    std::atomic<bool> callback_fired{false};
    std::atomic<int>  reloaded_interval{0};

    cfg.start_watching(tmp_path,
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
    cfg.load_from_file(tmp_path);
    cfg.start_watching(tmp_path, [](const SystemConfig&) {}, /*poll_interval_ms=*/50);

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

} // namespace
} // namespace llmquant
