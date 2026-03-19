#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace llmquant {

/**
 * @brief Aggregated configuration for the token stream subsystem.
 */
struct TokenStreamConfig {
    /** @brief Path to the file containing pre-recorded token sequences. */
    std::string data_file_path{"tokens.txt"};
    /** @brief Interval between token emissions, in milliseconds. */
    int token_interval_ms{10};
    /** @brief Maximum number of tokens held in the in-memory ring buffer. */
    size_t buffer_size{1024};
    /** @brief When true the simulator reads from an in-memory vector instead of disk. */
    bool use_memory_stream{false};
};

/**
 * @brief Configuration for the trade signal generation subsystem.
 */
struct TradingConfig {
    /** @brief Multiplier applied to the directional-bias component of each SemanticWeight. */
    double bias_sensitivity{1.0};
    /** @brief Multiplier applied to the volatility component of each SemanticWeight. */
    double volatility_sensitivity{1.0};
    /** @brief Multiplicative decay applied to accumulated signal on every token (0 < rate <= 1). */
    double signal_decay_rate{0.95};
    /** @brief Minimum time that must elapse between consecutive signal emissions, in microseconds. */
    int signal_cooldown_us{1000};
};

/**
 * @brief Configuration for the latency measurement and profiling subsystem.
 */
struct LatencyConfig {
    /** @brief Desired p99 latency target in microseconds; used for profiling alerts. */
    int target_latency_us{10};
    /** @brief Number of most-recent samples retained for percentile calculation. */
    size_t sample_window{1000};
    /** @brief When true, raw latency samples are stored so p95/p99 can be computed. */
    bool enable_profiling{true};
};

/**
 * @brief Configuration for the structured-logging subsystem.
 */
struct LoggingConfig {
    /** @brief Path to the output log file. */
    std::string log_file_path{"metrics.log"};
    /** @brief Output serialisation format: "CSV" or "JSON" (stored uppercase). */
    std::string format{"CSV"};
    /** @brief When true a coloured console sink is attached in addition to the file sink. */
    bool enable_console{true};
    /** @brief How often the logger should flush buffered entries to disk, in milliseconds. */
    int flush_interval_ms{100};
};

/**
 * @brief Risk gate override flags (for testing / debugging only).
 */
struct RiskOverrideConfig {
    bool disable_magnitude_gate{false};   ///< Bypass magnitude gate (testing only).
    bool disable_confidence_gate{false};  ///< Bypass confidence gate (testing only).
    bool disable_rate_gate{false};        ///< Bypass rate-limit gate (testing only).
    bool disable_drawdown_gate{false};    ///< Bypass drawdown gate (testing only).
    bool disable_position_gate{false};    ///< Bypass position/PnL gate (testing only).
};

/**
 * @brief Configuration for the pressure / back-pressure tuning subsystem.
 */
struct PressureConfig {
    /** @brief Maximum expected token ingestion rate (tokens/s); used to scale ingestion pressure. */
    double max_ingestion_rate_tps{50.0};
    /** @brief Multiplier applied to compute the exponential backoff ceiling. */
    double backoff_scale_factor{5.0};
};

/**
 * @brief Top-level configuration object that aggregates all subsystem configs.
 */
struct SystemConfig {
    TokenStreamConfig  token_stream;
    TradingConfig      trading;
    LatencyConfig      latency;
    LoggingConfig      logging;
    PressureConfig     pressure;
    RiskOverrideConfig risk_overrides;
};

/**
 * @brief Loads, validates and exposes a SystemConfig for the entire engine.
 *
 * Thread safety: all const methods are safe to call concurrently after
 * construction. Mutable methods (load_*, save_*, set_defaults) must be
 * called from a single thread before the engine starts.
 */
class Config {
public:
    Config() { set_defaults(); }

    /**
     * @brief Destructor stops any running file-watcher thread before the object is destroyed.
     */
    ~Config() { stop_watching(); }

    /**
     * @brief Load configuration from a YAML file on disk.
     *
     * @param filepath Absolute or relative path to the YAML config file.
     * @return true on success; false if the file cannot be opened or parsed
     *         (defaults are applied on failure).
     */
    bool load_from_file(const std::string& filepath);

    /**
     * @brief Parse configuration from a YAML string already held in memory.
     *
     * Field values that fail range validation cause the method to return false
     * and restore compiled-in defaults.
     *
     * @param yaml_content A valid YAML document as a std::string.
     * @return true on success; false if parsing or validation fails (defaults applied).
     */
    bool load_from_yaml_string(const std::string& yaml_content);

    /**
     * @brief Save current config to a YAML file.
     *
     * @param filepath Destination path; the file is created or overwritten.
     * @return true on success; false if the file could not be opened or a write error occurred.
     */
    bool save_to_file(const std::string& filepath) const;

    /**
     * @brief Reset all fields to their compiled-in defaults.
     */
    void set_defaults();

    /**
     * @brief Return a snapshot of the loaded config.
     *
     * Thread-safe (acquires config_mutex_ internally).
     *
     * @return A copy of the current SystemConfig.
     */
    SystemConfig get_config() const {
        std::lock_guard<std::mutex> lk(config_mutex_);
        return config_;
    }

    /**
     * @brief Set the use_memory_stream flag.
     *
     * Thread-safe (acquires config_mutex_).
     *
     * @param val New value for TokenStreamConfig::use_memory_stream.
     */
    void set_use_memory_stream(bool val) {
        std::lock_guard<std::mutex> lk(config_mutex_);
        config_.token_stream.use_memory_stream = val;
    }

    /**
     * @brief Start watching the config file for changes and reload automatically.
     *
     * Spawns a background thread that polls the file's mtime every
     * poll_interval_ms milliseconds. On change, reloads and invokes
     * on_reload with the new SystemConfig.
     *
     * No-op if a watcher is already running.
     *
     * @param filepath         Path to watch (same file passed to load_from_file).
     * @param on_reload        Callback invoked on the watcher thread after reload.
     * @param poll_interval_ms How often to check for changes (default 500 ms).
     * @return true if the watcher thread was successfully started; false otherwise.
     */
    bool start_watching(const std::string& filepath,
                        std::function<void(const SystemConfig&)> on_reload,
                        int poll_interval_ms = 500);

    /**
     * @brief Stop the background file-watcher thread.
     *
     * Blocks until the thread exits. Safe to call if start_watching() was
     * never called.
     */
    void stop_watching();

private:
    SystemConfig config_;
    mutable std::mutex config_mutex_;  ///< Guards config_ during hot-reload.
    std::thread watcher_thread_;
    std::atomic<bool> watching_{false};
};

} // namespace llmquant
