#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace llmquant {

/// Aggregated configuration for the token stream subsystem.
struct TokenStreamConfig {
    /// Path to the file containing pre-recorded token sequences.
    std::string data_file_path{"tokens.txt"};
    /// Interval between token emissions, in milliseconds.
    int token_interval_ms{10};
    /// Maximum number of tokens held in the in-memory ring buffer.
    size_t buffer_size{1024};
    /// When true the simulator reads from an in-memory vector instead of disk.
    bool use_memory_stream{false};
};

/// Configuration for the trade signal generation subsystem.
struct TradingConfig {
    /// Multiplier applied to the directional-bias component of each SemanticWeight.
    double bias_sensitivity{1.0};
    /// Multiplier applied to the volatility component of each SemanticWeight.
    double volatility_sensitivity{1.0};
    /// Multiplicative decay applied to accumulated signal on every token (0 < rate < 1).
    double signal_decay_rate{0.95};
    /// Minimum time that must elapse between consecutive signal emissions, in microseconds.
    int signal_cooldown_us{1000};
};

/// Configuration for the latency measurement and profiling subsystem.
struct LatencyConfig {
    /// Desired p99 latency target in microseconds; used for profiling alerts.
    int target_latency_us{10};
    /// Number of most-recent samples retained for percentile calculation.
    size_t sample_window{1000};
    /// When true, raw latency samples are stored so p95/p99 can be computed.
    bool enable_profiling{true};
};

/// Configuration for the structured-logging subsystem.
struct LoggingConfig {
    /// Path to the output log file.
    std::string log_file_path{"metrics.log"};
    /// Output serialisation format: "csv" or "json".
    std::string format{"csv"};
    /// When true a coloured console sink is attached in addition to the file sink.
    bool enable_console{true};
    /// How often the logger should flush buffered entries to disk, in milliseconds.
    int flush_interval_ms{100};
};

/// Top-level configuration object that aggregates all subsystem configs.
struct SystemConfig {
    TokenStreamConfig token_stream;
    TradingConfig     trading;
    LatencyConfig     latency;
    LoggingConfig     logging;
};

/// Loads, validates and exposes a SystemConfig for the entire engine.
///
/// Thread safety: all const methods are safe to call concurrently after
/// construction. Mutable methods (load_*, save_*, set_defaults) must be
/// called from a single thread before the engine starts.
class Config {
public:
    Config() { set_defaults(); }

    /// Destructor stops any running file-watcher thread before the object is
    /// destroyed.
    ~Config() { stop_watching(); }

    /// Load configuration from a YAML file on disk.
    ///
    /// # Arguments
    /// * `filepath` — Absolute or relative path to the YAML config file.
    ///
    /// # Returns
    /// `true` on success, `false` if the file cannot be opened or parsed
    /// (in which case defaults are applied).
    bool load_from_file(const std::string& filepath);

    /// Parse configuration from a YAML string already held in memory.
    ///
    /// # Arguments
    /// * `yaml_content` — A valid YAML document as a std::string.
    ///
    /// # Returns
    /// `true` on success, `false` if parsing fails (defaults are applied).
    bool load_from_yaml_string(const std::string& yaml_content);

    /// Serialise the current configuration to a YAML file on disk.
    ///
    /// # Arguments
    /// * `filepath` — Destination path; the file is created or overwritten.
    void save_to_file(const std::string& filepath) const;

    /// Reset all fields to their compiled-in defaults.
    void set_defaults();

    /// Return a read-only reference to the loaded SystemConfig.
    const SystemConfig& get_config() const { return config_; }

    /// Start watching the config file for changes and reload automatically.
    ///
    /// Spawns a background thread that polls the file's mtime every
    /// `poll_interval_ms` milliseconds. On change, reloads and invokes
    /// `on_reload` with the new SystemConfig.
    ///
    /// # Arguments
    /// * `filepath`         — Path to watch (same file passed to load_from_file).
    /// * `on_reload`        — Callback invoked on the watcher thread after reload.
    /// * `poll_interval_ms` — How often to check for changes (default 500ms).
    void start_watching(const std::string& filepath,
                        std::function<void(const SystemConfig&)> on_reload,
                        int poll_interval_ms = 500);

    /// Stop the background file-watcher thread. Blocks until the thread exits.
    void stop_watching();

private:
    SystemConfig config_;
    std::thread watcher_thread_;
    std::atomic<bool> watching_{false};
};

} // namespace llmquant
