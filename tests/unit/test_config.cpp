#include "gtest/gtest.h"
#include "Config.h"

#include <fstream>
#include <string>
#include <cstdio>

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
    EXPECT_EQ(sc.logging.format,             "json");
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

    // Logging defaults
    EXPECT_EQ(sc.logging.format, "csv");
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
    original.save_to_file(tmp_path);

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
    EXPECT_EQ(sc.logging.format,             "csv");
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

} // namespace
} // namespace llmquant
